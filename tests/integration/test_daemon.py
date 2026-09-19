"""Integration tests: live daemon, admin socket, CLI online/offline paths.

The daemon is the only registry writer; online CLI commands go through the
private Unix socket; standalone commands refuse a daemon-owned directory.
"""
import json
import os
import tempfile
import time as _time
import unittest

from frigate_xdna.admin import admin_call
from frigate_xdna.config import Config
from frigate_xdna.errors import FxdnaError
from frigate_xdna.supervisor import Supervisor
from tests.integration.onnx_builders import make_raw_yolo


def make_config(**kw):
    base = {"data_dir": tempfile.mkdtemp(prefix="fxdna-dmn-"),
            "endpoint": "tcp://127.0.0.1:5555"}
    base.update(kw)
    return Config(**base)


class TestDaemon(unittest.TestCase):
    def setUp(self):
        self.cfg = make_config()
        self.sup = Supervisor(self.cfg)
        self.sup.start_admin()
        self.addCleanup(self.sup.stop)
        # the server thread binds asynchronously; wait for the socket
        from frigate_xdna.admin import socket_path as _sp
        deadline = _time.monotonic() + 5.0
        while not os.path.exists(_sp(self.cfg.data_dir)):
            if _time.monotonic() > deadline:
                raise RuntimeError("admin socket never appeared")
            _time.sleep(0.05)

    def test_online_prepare_status_wait_via_socket(self):
        path = os.path.join(self.cfg.data_dir, "a.onnx")
        make_raw_yolo(path)
        resp = admin_call(self.cfg.data_dir,
                          {"command": "prepare", "ref": path})
        self.assertTrue(resp["ok"])
        job = resp["job"]
        self.assertIn("job_uuid", job)
        # Drive progress like the serve loop would; a WAIT_TIMEOUT here
        # means "not yet", never a failure — keep pumping.
        from frigate_xdna.errors import FxdnaError as _FxdnaError
        deadline = _time.monotonic() + 10.0
        while True:
            self.sup.pump(0.2)
            try:
                done = self.sup.wait_job(job["job_uuid"], 0.05,
                                         pump=False)
            except _FxdnaError as e:
                if e.error_code != "WAIT_TIMEOUT":
                    raise
                if _time.monotonic() > deadline:
                    raise RuntimeError("job never reached PREPARED")
                continue
            if done["stage"] == "PREPARED":
                break
            if _time.monotonic() > deadline:
                raise RuntimeError("job never reached PREPARED")
        resp = admin_call(self.cfg.data_dir, {"command": "status"})
        by_ref = {m["ref"]: m["state"] for m in resp["status"]["models"]}
        self.assertEqual(by_ref[path], "PREPARED")
        resp = admin_call(self.cfg.data_dir, {"command": "cache_list"})
        # the fake backend publishes through the real atomic path, so the
        # completed job leaves exactly one committed artifact row
        self.assertEqual(len(resp["entries"]), 1)
        self.assertEqual(resp["entries"][0]["recipe"], "bf16-vaiml-v1")

    def test_standalone_refuses_daemon_lock(self):
        with self.assertRaises(FxdnaError) as ctx:
            Supervisor(self.cfg)
        self.assertEqual(ctx.exception.exit_code, 9)

    def test_unknown_admin_command_rejected(self):
        resp = admin_call(self.cfg.data_dir, {"command": "frobnicate"})
        self.assertFalse(resp["ok"])

    def test_malformed_admin_frame_rejected(self):
        import socket as _socket

        from frigate_xdna.admin import socket_path
        s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        s.settimeout(5)
        try:
            s.connect(socket_path(self.cfg.data_dir))
            s.sendall(b"not json\n")
            buf = b""
            while b"\n" not in buf:
                buf += s.recv(4096)
            resp = json.loads(buf.decode())
            self.assertFalse(resp["ok"])
        finally:
            s.close()


if __name__ == "__main__":
    unittest.main()
