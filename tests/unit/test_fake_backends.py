"""Unit tests: fake compiler/worker lifecycle interfaces."""
import os
import unittest

import numpy as np

from frigate_xdna.compiler.fake import SLOW_COMPILE_S, FakeCompileJob
from frigate_xdna.runtime.fake import FakeNativeWorker, WorkerRequest


class TestFakeCompiler(unittest.TestCase):
    def test_happy_path(self):
        job = FakeCompileJob("a" * 64, "b" * 64, duration_s=1.0,
                             device_required=False)
        self.assertEqual(job.poll(0.5), "COMPILING")
        self.assertEqual(job.poll(0.6), "PREPARED")
        self.assertEqual(job.artifact_sha256, "f" * 64)

    def test_device_lease_blocks_compile(self):
        job = FakeCompileJob("a" * 64, "b" * 64, duration_s=5.0,
                             device_required=True,
                             device_held_by_worker=True)
        self.assertEqual(job.poll(10.0), "WAITING_FOR_DEVICE")
        job.device_held_by_worker = False
        self.assertEqual(job.poll(0.0), "COMPILING")

    def test_slow_compile_exceeds_frigate_model_wait(self):
        # Frigate rc2 model-operation wait is 30 s; a cold compile must be
        # honestly longer so not-ready tests are meaningful.
        self.assertGreater(SLOW_COMPILE_S, 30.0)

    def test_deterministic_failure_terminal(self):
        job = FakeCompileJob("a" * 64, "b" * 64, duration_s=0.0,
                             succeed=False, fail_state="VALIDATION_FAILED",
                             device_required=False)
        self.assertEqual(job.poll(0.1), "VALIDATION_FAILED")
        self.assertEqual(job.poll(5.0), "VALIDATION_FAILED")  # no auto-retry


class TestFakeWorker(unittest.TestCase):
    def _req(self, gen=3, nbytes=1228800):
        return WorkerRequest(message_type="infer", request_id=7,
                             worker_generation=gen, tensor_nbytes=nbytes)

    def test_generation_mismatch_returns_zeros(self):
        w = FakeNativeWorker()
        self.assertTrue(w.load(3, "digest-A"))
        out = w.infer(self._req(gen=2), b"\x00" * 1228800)
        self.assertEqual(out, bytes(480))

    def test_unloaded_or_retired_returns_zeros(self):
        w = FakeNativeWorker()
        self.assertEqual(w.infer(self._req(), b"\x00" * 1228800), bytes(480))
        w.load(3, "d")
        w.retire()
        self.assertEqual(w.infer(self._req(gen=3), b"\x00" * 1228800),
                         bytes(480))

    def test_wrong_frame_size_returns_zeros(self):
        w = FakeNativeWorker()
        w.load(3, "d")
        self.assertEqual(w.infer(self._req(nbytes=8), b"\x00" * 8), bytes(480))

    def test_fixture_frame_served(self):
        import os
        fix = os.path.join(os.path.dirname(__file__), "..", "fixtures",
                           "raw-yolo-320.npz")
        w = FakeNativeWorker()
        w.load(3, "d")
        req = self._req()
        req.artifact_path = os.path.abspath(fix)
        out = w.infer(req, b"\x00" * 1228800)
        arr = np.frombuffer(out, dtype="<f4").reshape(20, 6)
        self.assertEqual(arr.shape, (20, 6))
        self.assertEqual(int(arr[0, 0]), 5)  # bus on top
        self.assertEqual(w.requests_seen, 1)

    def test_corrupt_npz_returns_zeros(self):
        import tempfile as _tf
        with _tf.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            f.write(b"not a zip at all" * 64)
            path = f.name
        w = FakeNativeWorker()
        w.load(3, "d")
        req = self._req()
        req.artifact_path = path
        out = w.infer(req, b"\x00" * 1228800)
        self.assertEqual(out, bytes(480))
        os.unlink(path)


if __name__ == "__main__":
    unittest.main()
