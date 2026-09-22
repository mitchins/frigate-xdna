"""External-observability privacy (Task 05 addendum).

Frigate+ model IDs are non-secret but security-relevant object locators:
public/status/log/report representations use locally keyed HMAC
pseudonyms (`plus:<12 hex>`) by default. API keys, bearer tokens and
signed-URL query strings are never emitted — removed entirely, never
hashed.

The per-install redaction key lives at `<data_dir>/redaction.key`
(0600, created on first use) and is NEVER included in diagnostic
bundles. Use `fxdna status --show-identifiers` (or `diagnose
--show-identifiers`) to reveal raw IDs on the owning machine only.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re

KEY_NAME = "redaction.key"
ALIAS_PREFIX = "plus:"

# plus://<id> occurrences in free text (refs.py PLUS_ID_RE shape).
_PLUS_REF_RE = re.compile(
    r"plus://([A-Za-z0-9][A-Za-z0-9_.:-]{0,127})")
# Authorization headers / bearer tokens in any casing.
_BEARER_RE = re.compile(
    r"(?i)(authorization\s*[:=]\s*(?:bearer\s+)?)[^\s\"',;}]+")
# JSON-ish token fields: "accessToken": "...", 'api_key': '...', etc.
_TOKEN_FIELD_RE = re.compile(
    r"(?i)(\"|'|)(access[_-]?token|api[_-]?key|secret|signed[_-]?url)(\"|'|)"
    r"(\s*[:=]\s*)(\"|'|)[^\"',}\s]+")
# Signed-URL query strings: keep scheme/host/path, drop the query.
_SIGNED_QS_RE = re.compile(
    r"(https?://[^\s\"',;?]+)\?[^\s\"',;]*"
    r"(sig|signature|x-amz-signature|token|expires|se)=[^\s\"',;]*",
    re.IGNORECASE)
# Any other http(s) query string in bundle content is suspect too.
_ANY_QS_RE = re.compile(r"(https?://[^\s\"',;?]+)\?[^\s\"',;]+")
# Standalone Plus key material (key_id:secret, 40-char secrets).
_KEYPAIR_RE = re.compile(
    r"\b[A-Za-z0-9-]{8,}:[A-Za-z0-9]{32,}\b")


def load_or_create_key(data_dir: str) -> bytes:
    """Return the install-local redaction key, creating it (0600) once."""
    path = os.path.join(data_dir, KEY_NAME)
    try:
        with open(path, "rb") as f:
            key = f.read()
        if len(key) >= 16:
            return key
    except FileNotFoundError:
        pass
    key = os.urandom(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, key)
    finally:
        os.close(fd)
    return key


def plus_alias(model_id: str, key: bytes) -> str:
    """Stable local pseudonym for a Plus model ID (HMAC-SHA256, 12 hex)."""
    digest = hmac.new(key, model_id.encode("utf-8"),
                      hashlib.sha256).hexdigest()[:12]
    return f"{ALIAS_PREFIX}{digest}"


def sanitize_text(text: str, key: bytes) -> str:
    """Redact IDs/credentials/URLs in free text. Idempotent."""
    def _alias(m: re.Match) -> str:
        return plus_alias(m.group(1), key)
    out = _PLUS_REF_RE.sub(_alias, text)
    out = _BEARER_RE.sub(r"\1<redacted>", out)
    out = _TOKEN_FIELD_RE.sub(r"\1\2\3\4\5<redacted>", out)
    out = _SIGNED_QS_RE.sub(r"\1?<redacted>", out)
    out = _ANY_QS_RE.sub(r"\1?<redacted>", out)
    out = _KEYPAIR_RE.sub("<redacted-keypair>", out)
    return out


def sanitize_obj(obj, key: bytes):
    """Recursively sanitize a JSON-like structure (dict/list/str)."""
    if isinstance(obj, str):
        return sanitize_text(obj, key)
    if isinstance(obj, dict):
        return {k: sanitize_obj(v, key) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_obj(v, key) for v in obj]
    return obj


def abbreviate_digest(digest: str, length: int = 16) -> str:
    """Short digest for human-readable reports (full stays in the DB)."""
    if not digest or len(digest) <= length:
        return digest or ""
    return f"{digest[:length]}…"


def sanitize_ref(ref: str, key: bytes) -> str:
    """Alias a model ref for external display (`plus://ID` -> alias)."""
    if ref.startswith("plus://"):
        return plus_alias(ref[len("plus://"):], key)
    return ref
