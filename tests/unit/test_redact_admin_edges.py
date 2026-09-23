"""Unit tests: redaction passthrough edges + admin retry exhaustion.

Short/empty digests and non-Plus refs must pass through unchanged;
a daemon that never starts listening must surface DAEMON_UNREACHABLE
after the bounded retry window — never hang.
"""
import socket
import tempfile
import unittest
from unittest import mock

from frigate_xdna import admin as admin_mod
from frigate_xdna.admin import admin_call
from frigate_xdna.errors import FxdnaError
from frigate_xdna.observability.redact import (
    abbreviate_digest,
    sanitize_ref,
)


class TestRedactEdges(unittest.TestCase):
    def test_short_digest_passes_through(self):
        self.assertEqual(abbreviate_digest("abc123"), "abc123")
        self.assertEqual(abbreviate_digest(""), "")

    def test_non_plus_ref_passes_through(self):
        ref = "local:models/yolo.onnx"
        self.assertEqual(sanitize_ref(ref, b"0" * 32), ref)


class TestAdminRetry(unittest.TestCase):
    def test_never_listening_reports_unreachable_fast(self):
        with tempfile.TemporaryDirectory() as d:
            bound = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            bound.bind(admin_mod.socket_path(d))
            try:
                ticks = iter([100.0, 100.0, 103.0])
                with mock.patch.object(admin_mod.time, "monotonic",
                                       side_effect=lambda: next(ticks)):
                    with self.assertRaises(FxdnaError) as ctx:
                        admin_call(d, {"command": "status"})
                self.assertEqual(ctx.exception.error_code,
                                 "DAEMON_UNREACHABLE")
            finally:
                bound.close()


if __name__ == "__main__":
    unittest.main()
