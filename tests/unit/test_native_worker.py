"""Unit tests: NativeWorker supervision against a fake worker binary.

The fixture speaks the real IPC framing over a real socketpair child
process (spawn, fd inheritance, bounded exchange, retire, death) with
zero hardware. Covers runtime/native.py end to end.
"""
import os
import stat
import unittest

from frigate_xdna.runtime.native import (
    RESULT_BYTES,
    NativeWorker,
    WorkerError,
    worker_binary,
    worker_lib_dirs,
    worker_xrt_root,
)

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "fixtures",
                       "fake_worker.py")


def spawn(*argv):
    return NativeWorker.spawn(FIXTURE, [], worker_argv=argv)


class TestNativeWorker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        st = os.stat(FIXTURE)
        os.chmod(FIXTURE, st.st_mode | stat.S_IXUSR)

    def test_spawn_missing_binary(self):
        with self.assertRaises(WorkerError) as ctx:
            NativeWorker.spawn("/nonexistent-worker-xyz", [])
        self.assertEqual(ctx.exception.code, "WORKER_MISSING")

    def test_load_and_infer_roundtrip(self):
        w = spawn()
        try:
            self.assertTrue(w.alive())
            w.load("/tmp/x.rai", 7, "srv", 3)
            self.assertTrue(w.loaded)
            self.assertEqual(w.generation, 7)
            out = w.infer(b"\x00" * 48, [1, 3, 2, 2], 7, timeout_s=10.0)
            self.assertEqual(len(out), RESULT_BYTES)
            self.assertEqual(out, b"\x01" * RESULT_BYTES)
        finally:
            w.retire()
        self.assertFalse(w.loaded)

    def test_load_refused(self):
        w = spawn("--fail", "load")
        try:
            with self.assertRaises(WorkerError) as ctx:
                w.load("/tmp/x.rai", 1, "srv", 3)
            self.assertEqual(ctx.exception.code, "INVALID_MODEL")
            self.assertFalse(w.loaded)
        finally:
            w.retire()

    def test_load_requires_class_count(self):
        w = spawn()
        try:
            with self.assertRaises(WorkerError) as ctx:
                w.load("/tmp/x.rai", 1, "srv", 0)
            self.assertEqual(ctx.exception.code, "INVALID_ARGS")
        finally:
            w.retire()

    def test_infer_wrong_generation(self):
        w = spawn()
        try:
            w.load("/tmp/x.rai", 7, "srv", 3)
            with self.assertRaises(WorkerError):
                w.infer(b"\x00" * 48, [1, 3, 2, 2], 8, timeout_s=10.0)
        finally:
            w.retire()

    def test_infer_failure_surfaces(self):
        w = spawn("--fail", "infer")
        try:
            w.load("/tmp/x.rai", 7, "srv", 3)
            with self.assertRaises(WorkerError) as ctx:
                w.infer(b"\x00" * 48, [1, 3, 2, 2], 7, timeout_s=10.0)
            self.assertEqual(ctx.exception.code, "INVALID_ARGS")
        finally:
            w.retire()

    def test_dead_child_raises_and_retire_safe(self):
        w = spawn("--exit", "3")
        import time as _time
        deadline = _time.monotonic() + 10.0
        while w.alive() and _time.monotonic() < deadline:
            _time.sleep(0.05)
        self.assertFalse(w.alive())
        with self.assertRaises(WorkerError) as ctx:
            w.infer(b"\x00" * 4, [1, 1, 1, 1], 1, timeout_s=5.0)
        self.assertEqual(ctx.exception.code, "WORKER_DEAD")
        w.retire()  # safe on a dead child

    def test_paths_and_lib_dirs(self):
        from unittest import mock
        self.assertTrue(worker_binary().endswith("fxdna-worker"))
        self.assertIn("/opt/xilinx-xrt/lib", worker_lib_dirs())
        self.assertEqual(worker_xrt_root(), "/opt/xilinx-xrt")
        with mock.patch.dict(os.environ,
                             {"FXDNA_XRT_ROOT": "/tmp/xrt-test"}):
            self.assertEqual(worker_xrt_root(), "/tmp/xrt-test")


if __name__ == "__main__":
    unittest.main()
