"""Integration tests: Plus client against scripted fake servers.

Covers token exchange/refresh/expiry, credential failures, HTTP failure
modes, redirect/SSRF guards, size bounds, metadata validation and the
no-bearer-to-download-host guarantee. Loopback only; no real credentials.
"""
import os
import threading
import unittest

from frigate_xdna.errors import FxdnaError
from frigate_xdna.plus.client import (
    PlusClient,
    _url_target_ok,
    check_key_format,
)
from tests.integration.fake_plus import (
    TEST_KEY,
    TEST_KEY_ID,
    FakeDownloadHandler,
    FakeDownloadState,
    FakePlusHandler,
    FakePlusState,
    start_server,
)

LOOP = ("127.0.0.1",)


def api_url(srv):
    return f"http://127.0.0.1:{srv.server_address[1]}"


class PlusTestBase(unittest.TestCase):
    def setUp(self):
        self.plus_state = FakePlusState()
        self.dl_state = FakeDownloadState()
        self.api = start_server(FakePlusHandler, self.plus_state)
        self.dl = start_server(FakeDownloadHandler, self.dl_state)
        self.dl_url = f"http://127.0.0.1:{self.dl.server_address[1]}/m.onnx"
        self.dl_state.download_bodies["/m.onnx"] = b"MODEL-BYTES-1234"
        self.plus_state.models["MODEL_A"] = {
            "id": "MODEL_A", "labelMap": {"0": "person"}}
        self.plus_state.signed_urls["MODEL_A"] = self.dl_url

    def tearDown(self):
        self.api.shutdown()
        self.dl.shutdown()
        self.api.server_close()
        self.dl.server_close()

    def client(self, key=TEST_KEY, **kw):
        return PlusClient(key, host=api_url(self.api), **kw)


class TestKeyFormat(unittest.TestCase):
    def test_valid_and_invalid(self):
        self.assertTrue(check_key_format(TEST_KEY))
        for bad in ("", "short", TEST_KEY_ID, TEST_KEY.upper(),
                    TEST_KEY.replace(":", ""), "x" * 100):
            self.assertFalse(check_key_format(bad), bad)

    def test_bad_key_rejected_without_network(self):
        with self.assertRaises(FxdnaError):
            PlusClient("not-a-key", host="http://127.0.0.1:1")


class TestTokenFlow(PlusTestBase):
    def test_exchange_and_reuse(self):
        c = self.client()
        c.get_model_info("MODEL_A")
        c.get_model_info("MODEL_A")
        self.assertEqual(self.plus_state.token_calls, 1)

    def test_expiry_triggers_refresh(self):
        self.plus_state.token_expires_in_s = -10
        c = self.client()
        c.get_model_info("MODEL_A")
        c.get_model_info("MODEL_A")
        self.assertEqual(self.plus_state.token_calls, 2)

    def test_bad_credentials(self):
        bad = TEST_KEY_ID + ":" + "b" * 40
        with self.assertRaises(FxdnaError) as ctx:
            self.client(key=bad).get_model_info("MODEL_A")
        self.assertEqual(ctx.exception.error_code, "ACQUISITION_FAILED")
        # secret must not leak into the message
        self.assertNotIn("b" * 40, str(ctx.exception))

    def test_malformed_token_response(self):
        self.plus_state.token_mode = "malformed"
        with self.assertRaises(FxdnaError):
            self.client().get_model_info("MODEL_A")

    def test_api_302_rejected_not_parsed(self):
        # A 3xx API response has r.ok == True in requests: it must be
        # rejected explicitly, never parsed as API data.
        self.plus_state.model_status["MODEL_A"] = 302
        with self.assertRaises(FxdnaError) as ctx:
            self.client().get_model_info("MODEL_A")
        self.assertEqual(ctx.exception.error_code, "ACQUISITION_FAILED")

    def test_token_redirect_rejected_not_followed(self):
        self.plus_state.token_mode = "redirect"
        self.plus_state.redirect_target = (
            f"http://127.0.0.1:{self.dl.server_address[1]}/evil-follow")
        with self.assertRaises(FxdnaError) as ctx:
            self.client().get_model_info("MODEL_A")
        self.assertEqual(ctx.exception.error_code, "ACQUISITION_FAILED")
        # the redirect target was never fetched (no bearer, no follow)
        self.assertEqual(self.dl_state.download_auth_headers, [])


class TestModelFetch(PlusTestBase):
    def test_metadata_and_bytes(self):
        c = self.client()
        info = c.get_model_info("MODEL_A")
        self.assertEqual(info["labelMap"], {"0": "person"})
        url = c.get_model_download_url("MODEL_A", allow_private_hosts=LOOP)
        data = c.download_model(url, allow_private_hosts=LOOP)
        self.assertEqual(data, b"MODEL-BYTES-1234")

    def test_no_bearer_to_download_host(self):
        c = self.client()
        url = c.get_model_download_url("MODEL_A", allow_private_hosts=LOOP)
        c.download_model(url, allow_private_hosts=LOOP)
        self.assertEqual(self.dl_state.download_auth_headers, [None])

    def test_404_and_429_surface_visibly(self):
        self.plus_state.model_status["MODEL_A"] = 404
        with self.assertRaises(FxdnaError):
            self.client().get_model_info("MODEL_A")
        self.plus_state.model_status["MODEL_A"] = 429
        with self.assertRaises(FxdnaError):
            self.client().get_model_info("MODEL_A")

    def test_malformed_metadata(self):
        self.plus_state.models["MODEL_A"] = "MALFORMED"
        with self.assertRaises(FxdnaError):
            self.client().get_model_info("MODEL_A")

    def test_download_404(self):
        self.dl_state.download_modes["/m.onnx"] = "404"
        c = self.client()
        url = c.get_model_download_url("MODEL_A", allow_private_hosts=LOOP)
        with self.assertRaises(FxdnaError):
            c.download_model(url, allow_private_hosts=LOOP)

    def test_oversize_download_bounded(self):
        self.dl_state.download_modes["/m.onnx"] = "huge"
        c = self.client()
        url = c.get_model_download_url("MODEL_A", allow_private_hosts=LOOP)
        with self.assertRaises(FxdnaError) as ctx:
            c.download_model(url, max_bytes=1024, allow_private_hosts=LOOP)
        self.assertIn("exceeds", str(ctx.exception))

    def test_http_redirect_rejected(self):
        self.dl_state.download_modes["/m.onnx"] = "redirect-http"
        self.dl_state.redirect_target = "http://example.com/m.onnx"
        c = self.client()
        url = c.get_model_download_url("MODEL_A", allow_private_hosts=LOOP)
        # the loopback initial URL is allowed; rejection must happen at
        # the non-allowlisted http redirect hop
        with self.assertRaises(FxdnaError):
            c.download_model(url, allow_private_hosts=LOOP)

    def test_private_ip_redirect_rejected(self):
        self.dl_state.download_modes["/m.onnx"] = "redirect-private"
        self.dl_state.redirect_target = "https://169.254.169.254/x"
        c = self.client()
        url = c.get_model_download_url("MODEL_A", allow_private_hosts=LOOP)
        with self.assertRaises(FxdnaError):
            c.download_model(url, allow_private_hosts=LOOP)

    def test_429_then_ok_recovers_with_backoff(self):
        import frigate_xdna.plus.client as pc
        # unit: Retry-After honored and capped; deterministic codes no-retry
        r429 = type("R", (), {"status_code": 429,
                              "headers": {"Retry-After": "0"}})()
        self.assertEqual(pc._retry_delay(0, r429), 0)
        r404 = type("R", (), {"status_code": 404, "headers": {}})()
        self.assertIsNone(pc._retry_delay(0, r404))
        # end to end: one 429 then success, single client call
        self.plus_state.model_status["MODEL_A"] = "FLAKY_ONCE"
        info = self.client().get_model_info("MODEL_A")
        self.assertEqual(info["id"], "MODEL_A")
        self.assertEqual(self.plus_state.flaky_counts["MODEL_A"], 2)

    def test_pinned_adapter_targets_validated_ip(self):
        import requests as _rq

        from frigate_xdna.plus.client import _PinnedHTTPSAdapter
        s = _rq.Session()
        s.mount("https://cdn.example.com",
                _PinnedHTTPSAdapter("cdn.example.com", "93.184.216.34"))
        req = _rq.Request("GET", "https://cdn.example.com/x").prepare()
        pool = s.get_adapter("https://cdn.example.com/x") \
            .get_connection_with_tls_context(req, True)
        self.assertEqual(pool.host, "93.184.216.34")
        self.assertEqual(pool.assert_hostname, "cdn.example.com")
        self.assertEqual(pool.conn_kw, {"server_hostname": "cdn.example.com"})

    def test_pinned_https_loopback_sni_and_verify(self):
        """End-to-end: TCP to 127.0.0.1, SNI + cert validation vs the name."""
        import shutil
        import socket as _socket
        import ssl as _ssl
        import subprocess as _sp
        import tempfile as _tf
        if shutil.which("openssl") is None:
            self.skipTest("openssl unavailable")
        tmp = _tf.mkdtemp(prefix="fxdna-tls-")
        cert = os.path.join(tmp, "cert.pem")
        key = os.path.join(tmp, "key.pem")
        _sp.run(["openssl", "req", "-x509", "-newkey", "rsa:2048",
                 "-keyout", key, "-out", cert, "-days", "1", "-nodes",
                 "-subj", "/CN=test.local",
                 "-addext", "subjectAltName=DNS:test.local,IP:127.0.0.1"],
                check=True, capture_output=True, timeout=60)
        seen = {}
        ready = threading.Event()

        def serve_once():
            ctx = _ssl.SSLContext(_ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(cert, key)

            def sni_cb(sock, name, ctx2):
                seen["sni"] = name

            ctx.sni_callback = sni_cb
            raw = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
            raw.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
            raw.bind(("127.0.0.1", 0))
            ready.port = raw.getsockname()[1]
            raw.listen(1)
            ready.set()
            for _ in range(2):
                conn, _ = raw.accept()
                try:
                    tls = ctx.wrap_socket(conn, server_side=True)
                    data = tls.recv(4096)
                    seen["host_header"] = [
                        line for line in data.decode().split("\r\n")
                        if line.lower().startswith("host:")]
                    body = b"pinned-ok"
                    tls.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 9\r\n"
                                b"Connection: close\r\n\r\n" + body)
                    try:
                        tls.shutdown(_socket.SHUT_RDWR)
                    except OSError:
                        pass
                    tls.close()
                except (OSError, _ssl.SSLError):
                    pass
            raw.close()

        import threading as _threading
        t = _threading.Thread(target=serve_once, daemon=True)
        t.start()
        self.assertTrue(ready.wait(timeout=10))
        import requests as _rq

        from frigate_xdna.plus.client import _PinnedHTTPSAdapter
        s = _rq.Session()
        s.mount("https://test.local",
                _PinnedHTTPSAdapter("test.local", "127.0.0.1"))
        # verify=False: SNI + Host header only (no cert check)
        r = s.get(f"https://test.local:{ready.port}/x", verify=False,
                  timeout=10)
        self.assertEqual(r.content, b"pinned-ok")
        self.assertEqual(seen.get("sni"), "test.local")
        self.assertTrue(any("test.local" in h
                            for h in seen.get("host_header", [])),
                        seen)
        # verify=<cert> (self-signed CA): full chain + hostname check pass
        r2 = s.get(f"https://test.local:{ready.port}/y", verify=cert,
                   timeout=10)
        self.assertEqual(r2.content, b"pinned-ok")

    def test_stalled_download_times_out(self):
        from unittest import mock

        import frigate_xdna.plus.client as pc
        self.dl_state.download_modes["/m.onnx"] = "slow"
        c = self.client()
        url = c.get_model_download_url("MODEL_A", allow_private_hosts=LOOP)
        with mock.patch.object(pc, "READ_TIMEOUT_S", 1.0):
            with self.assertRaises(FxdnaError) as ctx:
                c.download_model(url, allow_private_hosts=LOOP)
        self.assertEqual(ctx.exception.error_code, "ACQUISITION_FAILED")

    def test_signed_url_must_be_public_https_by_default(self):
        self.assertFalse(_url_target_ok("http://example.com/m"))
        self.assertFalse(_url_target_ok("https://169.254.169.254/m"))
        self.assertFalse(_url_target_ok("https://localhost/m"))
        # a public DNS name resolving to global IPs passes (resolution
        # mocked: no external DNS in test environments)
        import socket
        from unittest import mock
        real = socket.getaddrinfo

        def fake(host, port, **kw):
            if host == "cdn.example.com":
                return [(socket.AF_INET, socket.SOCK_STREAM, 6, "",
                         ("93.184.216.34", 443))]
            return real(host, port, **kw)

        with mock.patch("socket.getaddrinfo", side_effect=fake):
            self.assertTrue(_url_target_ok("https://cdn.example.com/m"))


if __name__ == "__main__":
    unittest.main()
