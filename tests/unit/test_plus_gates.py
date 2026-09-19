"""Hardware-free tests for Plus download safety gates (SSRF/redirect).

Only IP-literal cases run here (no DNS, no network): the redirect and
pinning paths that need sockets are covered by integration tests against
the loopback fake-Plus server.
"""
import unittest
from types import SimpleNamespace

from frigate_xdna.plus.client import _retry_delay, _url_target_ok


class TestUrlTargetOk(unittest.TestCase):
    def test_public_https_ip_ok(self):
        self.assertTrue(_url_target_ok("https://8.8.8.8/model"))

    def test_plain_http_rejected(self):
        self.assertFalse(_url_target_ok("http://8.8.8.8/model"))

    def test_private_ips_rejected(self):
        for url in ("https://10.0.0.1/m", "https://192.168.1.5/m",
                    "https://172.16.0.9/m", "https://127.0.0.1/m",
                    "https://[::1]/m", "https://[fc00::1]/m",
                    "https://169.254.169.254/latest/meta-data"):
            self.assertFalse(_url_target_ok(url), url)

    def test_local_names_rejected(self):
        for url in ("https://localhost/m", "https://x.internal/m",
                    "https://x.local/m", "https://x.lan/m",
                    "https://x.home/m", "https://x.corp/m",
                    "https://x.localhost/m"):
            self.assertFalse(_url_target_ok(url), url)

    def test_allowlisted_test_host(self):
        self.assertTrue(
            _url_target_ok("http://testhost/m", ("testhost",)))
        self.assertTrue(
            _url_target_ok("https://testhost/m", ("testhost",)))
        # allowlist is exact: other hosts still gated
        self.assertFalse(
            _url_target_ok("http://8.8.8.8/m", ("testhost",)))

    def test_malformed(self):
        self.assertFalse(_url_target_ok("not a url"))
        self.assertFalse(_url_target_ok("https://"))


class TestRetryDelay(unittest.TestCase):
    def _resp(self, status, headers=None):
        return SimpleNamespace(status_code=status, headers=headers or {})

    def test_non_retryable_none(self):
        self.assertIsNone(_retry_delay(0, self._resp(400)))
        self.assertIsNone(_retry_delay(0, self._resp(404)))

    def test_retry_after_capped(self):
        # Retry-After: 120 clamps to MAX_RETRY_AFTER_S (30.0)
        self.assertEqual(
            _retry_delay(0, self._resp(429, {"Retry-After": "120"})),
            30.0)
        self.assertEqual(
            _retry_delay(0, self._resp(429, {"Retry-After": "5"})),
            5.0)

    def test_bad_retry_after_falls_back(self):
        d = _retry_delay(0, self._resp(503, {"Retry-After": "junk"}))
        self.assertIsNotNone(d)

    def test_no_response_backoff(self):
        d0 = _retry_delay(0, None)
        d1 = _retry_delay(1, None)
        self.assertIsNotNone(d0)
        self.assertIsNotNone(d1)
        self.assertGreaterEqual(d1, d0)


if __name__ == "__main__":
    unittest.main()
