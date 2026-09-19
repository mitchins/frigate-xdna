"""Unit tests: real-backend lifecycle + backend invalidation (no NPU).

Drives RealCompileJob with a stubbed run_compile, and proves stale
(fake) rows invalidate when the backend identity flips to the audited
real one.
"""
import os
import tempfile
import unittest

from frigate_xdna.compiler.launcher import CompileResult
from frigate_xdna.compiler.real import BACKEND_ID, RealCompileJob
from frigate_xdna.config import Config
from frigate_xdna.supervisor import Supervisor


class TestRealBackendLifecycle(unittest.TestCase):
    def test_poll_transitions(self):
        import frigate_xdna.compiler.real as real_mod
        orig = real_mod.run_compile
        real_mod.run_compile = lambda *a, **k: CompileResult(
            0, 1.0, 100, rai_path="/tmp/x.rai", rai_sha256="ab" * 32,
            rai_bytes=10)
        try:
            with tempfile.TemporaryDirectory() as d:
                job = RealCompileJob("s" * 64, "c" * 64, "/s.onnx", d,
                                     prefixes=None)
                self.assertEqual(job.poll(0.1), "COMPILING")
                while job.poll(0.1) == "COMPILING":
                    pass
                self.assertEqual(job.state, "PREPARED")
        finally:
            real_mod.run_compile = orig

    def test_failure_terminal(self):
        import frigate_xdna.compiler.real as real_mod
        orig = real_mod.run_compile
        real_mod.run_compile = lambda *a, **k: CompileResult(
            1, 1.0, 100, error="boom")
        try:
            with tempfile.TemporaryDirectory() as d:
                job = RealCompileJob("s" * 64, "c" * 64, "/s.onnx", d,
                                     prefixes=None)
                job.poll(0.1)
                while job.poll(0.1) == "COMPILING":
                    pass
                self.assertEqual(job.state, "COMPILE_FAILED")
        finally:
            real_mod.run_compile = orig

    def test_backend_id_is_audited_recipe(self):
        self.assertEqual(BACKEND_ID, "bf16-vaiml-v1")


class TestBackendInvalidation(unittest.TestCase):
    def test_fake_row_invalidates_under_real_backend(self):
        from tests.integration.onnx_builders import make_raw_yolo
        data_dir = tempfile.mkdtemp(prefix="fxdna-inv-")
        cfg = Config(data_dir=data_dir, endpoint="tcp://127.0.0.1:5555")
        sup = Supervisor(cfg)  # fake backend: publishes fake row
        try:
            path = os.path.join(data_dir, "a.onnx")
            make_raw_yolo(path)
            out = sup.prepare(path)
            sup.pump(2.0)
            self.assertTrue(sup.prepare(path).get("cache_hit"))
        finally:
            sup.stop()
        # Same data dir, real backend identity: fake row must invalidate
        # into a fresh job, never activate as a real artifact.
        sup2 = Supervisor(cfg, compiler_backend_id="bf16-vaiml-v1")
        try:
            out = sup2.prepare(path)
            self.assertFalse(out.get("cache_hit"))
            self.assertIn("job_uuid", out)
        finally:
            sup2.stop()


if __name__ == "__main__":
    unittest.main()
