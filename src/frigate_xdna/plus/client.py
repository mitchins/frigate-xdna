"""Narrow read-only Frigate+ client (SPEC §5.3).

Implements exactly the read-only flow reflected in pinned rc2 `plus.py`
(blob d528aa17...): API key -> token, model metadata, signed download URL,
model bytes. No upload/training/annotation/write endpoints exist here.

Verified constants (rc2 tag):
  PLUS_API_HOST = "https://api.frigate.video"  (rc2 frigate/const.py)
  PLUS_ENV_VAR  = "PLUS_API_KEY"
  key format    = [a-z0-9]{8}(-[a-z0-9]{4}){3}-[a-z0-9]{12}:[a-z0-9]{40}
  token         = GET {host}/v1/auth/token (basic auth id:secret)
  refresh       = when expires missing or expires-now < 60 s
  metadata      = GET {host}/v1/model/{id}
  signed URL    = GET {host}/v1/model/{id}/signed_url -> {"url": ...}

Security rules enforced here, not just documented:
- bearer tokens go only to the API host, never to a download host;
- signed downloads allow only https, refuse private/loopback/link-local
  targets and metadata-service addresses after redirect resolution;
- tokens and signed URLs live in memory only; logs/DB receive redacted
  errors.
"""
from __future__ import annotations

import ipaddress
import re
import socket
import time
import urllib.parse

import requests

API_HOST = "https://api.frigate.video"
KEY_RE = re.compile(
    r"[a-z0-9]{8}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{12}"
    r":[a-z0-9]{40}"
)
TOKEN_SKEW_S = 60.0
CONNECT_TIMEOUT_S = 10.0
READ_TIMEOUT_S = 30.0
MAX_DOWNLOAD_BYTES = 256 * 1024 * 1024  # INTERFACES model cap 256 MiB
# Bounded retry (SPEC §5.3): GETs are idempotent; 3 attempts max, backoff
# 1 s / 2 s, Retry-After honored up to 30 s. Deterministic 400/401/403/404
# never retry.
MAX_ATTEMPTS = 3
BACKOFF_S = (1.0, 2.0)
MAX_RETRY_AFTER_S = 30.0
RETRYABLE_STATUS = frozenset({408, 425, 429, 502, 503, 504})

from ..errors import ACQUISITION_FAILED, FxdnaError  # noqa: E402


def check_key_format(key: str) -> bool:
    return KEY_RE.fullmatch(key or "") is not None


def _redacted_error(action: str, status: int | None) -> FxdnaError:
    # Never include tokens, URLs, or response bodies (may carry secrets).
    detail = f"status={status}" if status is not None else "network/timeout"
    return FxdnaError(ACQUISITION_FAILED, "ACQUISITION_FAILED",
                      f"Plus {action} failed ({detail})")


def _retry_delay(attempt: int, response) -> float | None:
    """Delay before retry, or None when the failure is deterministic."""
    if response is not None and response.status_code not in RETRYABLE_STATUS:
        return None
    if response is not None:
        try:
            after = float(response.headers.get("Retry-After", ""))
            return min(max(after, 0.0), MAX_RETRY_AFTER_S)
        except (ValueError, TypeError):
            pass
    return BACKOFF_S[min(attempt, len(BACKOFF_S) - 1)]


def _host_ips(host: str) -> list:
    try:
        return [ipaddress.ip_address(host)]
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except OSError:
        return []
    out = []
    for info in infos:
        try:
            out.append(ipaddress.ip_address(info[4][0]))
        except ValueError:
            continue
    return out


class _PinnedHTTPSAdapter(requests.adapters.HTTPAdapter):
    """HTTPS adapter pinned to one validated IP (closes DNS-rebinding TOCTOU).

    TCP connects to `ip`, while TLS SNI/certificate validation (via
    conn_kw server_hostname + assert_hostname) and the HTTP Host header
    keep using `hostname`. Plain-http URLs never reach this adapter: only
    validated https targets are pinned.

    Overrides `get_connection_with_tls_context`, the hook `requests`
    >= 2.32.2 `send()` actually uses (`get_connection` is deprecated and
    uncalled — verified against requests 2.34.2 / urllib3 2.8.0).
    """

    def __init__(self, hostname: str, ip: str):
        super().__init__()
        self._hostname = hostname
        self._ip = ip

    def send(self, request, **kwargs):
        # The pool connects to the IP, so urllib3 would emit Host: <ip>;
        # restore the validated hostname (virtual-hosted origins route on
        # Host; the header must match the TLS identity).
        parsed = urllib.parse.urlparse(request.url)
        host = parsed.hostname or self._hostname
        port = parsed.port or 443
        request.headers["Host"] = host if port == 443 else f"{host}:{port}"
        return super().send(request, **kwargs)

    def get_connection_with_tls_context(self, request, verify,
                                        proxies=None, cert=None):
        from urllib3 import HTTPSConnectionPool
        from requests.certs import where as _ca_bundle
        parsed = urllib.parse.urlparse(request.url)
        port = parsed.port or 443
        if isinstance(verify, str):
            ca, reqs = verify, "CERT_REQUIRED"
        elif verify:
            ca, reqs = _ca_bundle(), "CERT_REQUIRED"
        else:  # pragma: no cover - downloads always verify
            ca, reqs = None, "CERT_NONE"
        return HTTPSConnectionPool(
            self._ip, port=port, cert_reqs=reqs, ca_certs=ca,
            assert_hostname=self._hostname,
            server_hostname=self._hostname)


def _url_target_ok(url: str, allow_private_hosts: tuple[str, ...] = ()) -> bool:
    """https only; resolved IPs must be global unless explicitly allowlisted.

    Allowlisted hosts (tests only) may use http or https; everything else
    about the URL is still validated.
    """
    try:
        parts = urllib.parse.urlparse(url)
    except ValueError:
        return False
    if not parts.hostname:
        return False
    host = parts.hostname.lower()
    if host in allow_private_hosts:
        return parts.scheme in ("https", "http")
    if parts.scheme != "https":
        return False
    if host == "localhost" or host.endswith(
            (".localhost", ".internal", ".local", ".lan", ".home", ".corp")):
        return False
    ips = _host_ips(host)
    if not ips:
        return False
    return all(ip.is_global for ip in ips)


class PlusClient:
    """Authenticated read-only Plus access. Secrets stay in this object."""

    def __init__(self, api_key: str, host: str = API_HOST,
                 session: requests.Session | None = None):
        if not check_key_format(api_key):
            raise FxdnaError(ACQUISITION_FAILED, "ACQUISITION_FAILED",
                             "Plus API key is not formatted correctly")
        self._key_id, self._key_secret = api_key.split(":", 1)
        self._host = host.rstrip("/")
        self._session = session or requests.Session()
        self._token: str | None = None
        self._expires: float | None = None

    def _get_with_retry(self, url: str, action: str,
                        headers: dict | None = None,
                        auth=None) -> requests.Response:
        last_exc = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                # Redirects are never followed on authenticated API calls:
                # the Plus API does not legitimately redirect, and a 3xx
                # to an attacker host must be rejected, not followed (a
                # cross-host redirect would also strip the bearer and
                # return attacker JSON to the token parser).
                r = self._session.get(
                    url, headers=headers, auth=auth,
                    allow_redirects=False,
                    timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S))
            except requests.RequestException as e:
                last_exc = e
                delay = BACKOFF_S[min(attempt, len(BACKOFF_S) - 1)]
                time.sleep(delay)
                continue
            if r.ok or _retry_delay(attempt, r) is None:
                return r
            if attempt < MAX_ATTEMPTS - 1:
                delay = _retry_delay(attempt, r)
                time.sleep(delay)
        if last_exc is not None:
            raise _redacted_error(action, None) from None
        raise _redacted_error(action, r.status_code)

    def _refresh_token_if_needed(self) -> None:
        if (self._token is not None and self._expires is not None
                and self._expires - time.time() >= TOKEN_SKEW_S):
            return
        r = self._get_with_retry(
            f"{self._host}/v1/auth/token", "token refresh",
            auth=(self._key_id, self._key_secret))
        if not r.ok:
            raise _redacted_error("token refresh", r.status_code)
        try:
            data = r.json()
            self._token = str(data["accessToken"])
            self._expires = float(data.get("expires", 0) or 0)
        except (ValueError, KeyError, TypeError):
            raise _redacted_error("token refresh", r.status_code) from None

    def _api_get(self, path: str) -> requests.Response:
        self._refresh_token_if_needed()
        try:
            return self._get_with_retry(
                f"{self._host}/v1/{path}", f"GET {path}",
                headers={"authorization": f"Bearer {self._token}"})
        except requests.RequestException:
            raise _redacted_error(f"GET {path}", None) from None

    def get_model_info(self, model_id: str) -> dict:
        r = self._api_get(f"model/{model_id}")
        if not r.ok:
            raise _redacted_error("model metadata", r.status_code)
        try:
            data = r.json()
        except ValueError:
            raise _redacted_error("model metadata", r.status_code) from None
        if not isinstance(data, dict):
            raise _redacted_error("model metadata", r.status_code)
        return data

    def get_model_download_url(self, model_id: str,
                                 allow_private_hosts: tuple[str, ...] = ()
                                 ) -> str:
        r = self._api_get(f"model/{model_id}/signed_url")
        if not r.ok:
            raise _redacted_error("signed URL", r.status_code)
        try:
            url = str(r.json().get("url"))
        except (ValueError, AttributeError):
            raise _redacted_error("signed URL", r.status_code) from None
        if not _url_target_ok(url, allow_private_hosts):
            raise FxdnaError(ACQUISITION_FAILED, "ACQUISITION_FAILED",
                             "Plus signed URL target rejected (not public https)")
        return url

    def download_model(self, url: str, max_bytes: int = MAX_DOWNLOAD_BYTES,
                       chunk_size: int = 1 << 20,
                       allow_private_hosts: tuple[str, ...] = ()) -> bytes:
        """Fetch model bytes with NO auth headers and redirect validation.

        Redirects are followed manually (max 5); every hop must be https
        with globally-routable resolved IPs unless explicitly allowlisted
        (tests use loopback fakes via this parameter only).
        """
        if not _url_target_ok(url, allow_private_hosts):
            raise FxdnaError(ACQUISITION_FAILED, "ACQUISITION_FAILED",
                             "download target rejected (not public https)")
        try:
            # Fresh session: no bearer, no cookies leak from the API session.
            with requests.Session() as dl:
                current = url
                mounted: set[str] = set()
                for _ in range(6):
                    parts = urllib.parse.urlparse(current)
                    host = (parts.hostname or "").lower()
                    if parts.scheme == "https" and \
                            host not in allow_private_hosts:
                        # Pin this hop to its validated IP: validation and
                        # connection can no longer see different addresses.
                        ips = [i for i in _host_ips(host) if i.is_global]
                        if not ips:
                            raise FxdnaError(
                                ACQUISITION_FAILED, "ACQUISITION_FAILED",
                                "download target no longer resolves public")
                        prefix = f"https://{host}"
                        if prefix not in mounted:
                            dl.mount(prefix, _PinnedHTTPSAdapter(
                                host, str(ips[0])))
                            mounted.add(prefix)
                    with dl.get(current, headers={}, stream=True,
                                allow_redirects=False,
                                timeout=(CONNECT_TIMEOUT_S,
                                         READ_TIMEOUT_S)) as r:
                        if r.status_code in (301, 302, 303, 307, 308):
                            nxt = r.headers.get("location", "")
                            nxt = urllib.parse.urljoin(current, nxt)
                            if not _url_target_ok(nxt, allow_private_hosts):
                                raise FxdnaError(
                                    ACQUISITION_FAILED, "ACQUISITION_FAILED",
                                    "download redirect target rejected")
                            current = nxt
                            continue
                        if not r.ok:
                            raise _redacted_error("download", r.status_code)
                        total = 0
                        chunks: list[bytes] = []
                        for chunk in r.iter_content(chunk_size):
                            if not chunk:
                                continue
                            total += len(chunk)
                            if total > max_bytes:
                                raise FxdnaError(
                                    ACQUISITION_FAILED, "ACQUISITION_FAILED",
                                    f"download exceeds {max_bytes} byte bound")
                            chunks.append(chunk)
                        return b"".join(chunks)
                raise FxdnaError(ACQUISITION_FAILED, "ACQUISITION_FAILED",
                                 "too many download redirects")
        except requests.RequestException:
            raise _redacted_error("download", None) from None
