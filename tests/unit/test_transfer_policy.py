"""Unit tests: transfer gate + default Compose network exposure."""
import os
import unittest

from frigate_xdna.errors import FxdnaError
from frigate_xdna.transport.policy import check_transfer

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


class TestTransferGate(unittest.TestCase):
    def test_default_mode_accepts_bounded_transfer(self):
        check_transfer(True, 1024)  # must not raise

    def test_hardening_mode_rejects_cleanly(self):
        with self.assertRaises(FxdnaError) as ctx:
            check_transfer(False, 1024)
        self.assertEqual(ctx.exception.error_code, "INVALID_MODEL")

    def test_size_bounds_enforced(self):
        with self.assertRaises(FxdnaError):
            check_transfer(True, 0)
        with self.assertRaises(FxdnaError):
            check_transfer(True, 256 * 1024 * 1024 + 1)


class TestComposeExposure(unittest.TestCase):
    def _read(self, name):
        with open(os.path.join(REPO, "examples", name)) as f:
            return f.read()

    def test_base_compose_publishes_no_ports(self):
        body = self._read("compose.yaml")
        self.assertNotIn("ports:", body)
        self.assertNotIn("PLUS_API_KEY_FILE", body)

    def test_host_port_override_defaults_to_loopback(self):
        body = self._read("compose.host-port.yaml")
        self.assertIn("${FXDNA_BIND_IP:-127.0.0.1}:5555:5555", body)

    def test_plus_key_lives_only_in_overlay(self):
        base = self._read("compose.yaml")
        overlay = self._read("compose.plus.yaml")
        self.assertNotIn("PLUS_API_KEY", base)
        self.assertIn("PLUS_API_KEY", overlay)
        self.assertNotIn("PLUS_API_KEY_FILE", overlay)
        self.assertNotIn("secrets:", overlay)


if __name__ == "__main__":
    unittest.main()
