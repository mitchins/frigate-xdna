"""Unit tests: scripted native-runtime boundary (v0.1.1 checkpoint 3).

The REAL NativeWorker control plane (spawn, fd inheritance, bounded
LOAD/INFER exchanges, retire) runs against the scripted fixture, and
the REAL supervisor handlers (worker_infer incl. inhibition, probe
exchange) consume the outcomes. Covers load failure, vendor
exception codes, shape validation, valid/invalid/nonfinite output,
inference failure, delayed reply, worker exit/crash, and retirement
failure — all without hardware.
"""
import os
import stat
import tempfile
import time
import unittest

from frigate_xdna.compiler.launcher import _run_probe_exchange
from frigate_xdna.runtime.native import (
    RESULT_BYTES,
    NativeWorker,
    WorkerError,
)

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "fixtures",
                       "fake_worker.py")
SHAPE = [1, 3, 320, 320]
ELEMS = 1 * 3 * 320 * 320


def spawn(*argv):
    return NativeWorker.spawn(FIXTURE, [], worker_argv=argv)


def make_supervisor(data_dir):
    from frigate_xdna.config import Config
    from frigate_xdna.supervisor import Supervisor
    return Supervisor(Config(data_dir=data_dir))


class TestVendorOutcomes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        st = os.stat(FIXTURE)
        os.chmod(FIXTURE, st.st_mode | stat.S_IXUSR)

    def test_vendor_exception_code_preserved_on_infer(self):
        w = spawn("--fail", "infer", "--fail-code", "DEVICE_FAULT")
        try:
            w.load("/tmp/x.rai", 7, "srv", 3)
            with self.assertRaises(WorkerError) as ctx:
                w.infer(b"\x00" * 48, [1, 3, 2, 2], 7, timeout_s=10.0)
            self.assertEqual(ctx.exception.code, "DEVICE_FAULT")
        finally:
            w.retire()

    def test_load_refusal_code_override(self):
        w = spawn("--fail", "load", "--fail-code", "UNSUPPORTED_CONTRACT")
        try:
            with self.assertRaises(WorkerError) as ctx:
                w.load("/tmp/x.rai", 1, "srv", 3)
            self.assertEqual(ctx.exception.code, "UNSUPPORTED_CONTRACT")
            self.assertFalse(w.loaded)
        finally:
            w.retire()

    def test_shape_mismatch_refused_at_boundary(self):
        w = spawn("--expect-shape", "1,3,320,320")
        try:
            w.load("/tmp/x.rai", 7, "srv", 3)
            with self.assertRaises(WorkerError) as ctx:
                w.infer(b"\x00" * 48, [1, 3, 2, 2], 7, timeout_s=10.0)
            self.assertEqual(ctx.exception.code, "INVALID_TENSOR")
            out = w.infer(b"\x00" * (ELEMS * 4), SHAPE, 7, timeout_s=10.0)
            self.assertEqual(len(out), RESULT_BYTES)
        finally:
            w.retire()

    def test_nonfinite_output_detected_by_real_probe(self):
        w = spawn("--frame", "nan")
        try:
            detail = _run_probe_exchange(w, "model.rai", 8, SHAPE,
                                         ELEMS, 10.0)
            self.assertEqual(detail, "non-finite probe output")
        finally:
            w.retire()

    def test_short_frame_detected_by_real_probe(self):
        # The length check fires in the real NativeWorker exchange
        # (INFER_SHORT), before the probe's own frame check.
        w = spawn("--frame", "short:16")
        try:
            detail = _run_probe_exchange(w, "model.rai", 8, SHAPE,
                                         ELEMS, 10.0)
            self.assertIn("INFER_SHORT", detail)
        finally:
            w.retire()

    def test_valid_output_passes_real_probe(self):
        w = spawn("--frame", "zeros")
        try:
            detail = _run_probe_exchange(w, "model.rai", 8, SHAPE,
                                         ELEMS, 10.0)
            self.assertIsNone(detail)
        finally:
            w.retire()

    def test_delayed_reply_is_io_failure(self):
        w = spawn("--sleep", "5")
        try:
            w.load("/tmp/x.rai", 7, "srv", 3)
            t0 = time.monotonic()
            with self.assertRaises(WorkerError) as ctx:
                w.infer(b"\x00" * 48, [1, 3, 2, 2], 7, timeout_s=0.5)
            self.assertEqual(ctx.exception.code, "WORKER_IO")
            self.assertLess(time.monotonic() - t0, 5.0)
        finally:
            w.retire()

    def test_crash_during_infer_then_dead(self):
        w = spawn("--die-on", "infer")
        try:
            w.load("/tmp/x.rai", 7, "srv", 3)
            with self.assertRaises(WorkerError) as ctx:
                w.infer(b"\x00" * 48, [1, 3, 2, 2], 7, timeout_s=10.0)
            self.assertEqual(ctx.exception.code, "WORKER_IO")
            time.sleep(0.5)
            with self.assertRaises(WorkerError) as ctx2:
                w.infer(b"\x00" * 48, [1, 3, 2, 2], 7, timeout_s=5.0)
            self.assertEqual(ctx2.exception.code, "WORKER_DEAD")
        finally:
            w.retire()

    def test_retirement_failure_terminates_bounded(self):
        w = spawn("--ignore-shutdown")
        try:
            w.load("/tmp/x.rai", 7, "srv", 3)
            t0 = time.monotonic()
            w.retire()
            wall = time.monotonic() - t0
            self.assertFalse(w.alive())
            self.assertFalse(w.loaded)
            self.assertGreaterEqual(wall, 5.0)  # waited the grace period
            self.assertLess(wall, 20.0)  # ... then SIGTERM, bounded
        finally:
            w.retire()


class TestSupervisorHandling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        st = os.stat(FIXTURE)
        os.chmod(FIXTURE, st.st_mode | stat.S_IXUSR)

    def test_infer_ok_through_real_worker(self):
        with tempfile.TemporaryDirectory() as d:
            sup = make_supervisor(d)
            try:
                w = spawn()
                try:
                    w.load("/tmp/x.rai", 1, "srv", 3)
                    sup._worker = w
                    status, out = sup.worker_infer(
                        b"\x00" * (ELEMS * 4), SHAPE, 5.0)
                    self.assertEqual(status, "ok")
                    self.assertEqual(len(out), RESULT_BYTES)
                    self.assertIsNone(
                        sup.registry.get_state("inhibition"))
                finally:
                    sup._worker = None
                    w.retire()
            finally:
                sup.stop()

    def test_vendor_fault_inhibits_through_real_worker(self):
        with tempfile.TemporaryDirectory() as d:
            sup = make_supervisor(d)
            try:
                w = spawn("--fail", "infer", "--fail-code",
                          "DEVICE_FAULT")
                try:
                    w.load("/tmp/x.rai", 1, "srv", 3)
                    sup._worker = w
                    status, out = sup.worker_infer(
                        b"\x00" * (ELEMS * 4), SHAPE, 5.0)
                    self.assertEqual(status, "failed")
                    self.assertEqual(out, b"")
                    inh = sup.registry.get_state("inhibition")
                    self.assertIsNotNone(inh)
                    self.assertIn("WORKER_DEVICE_FAULT",
                                  inh["reason"])
                finally:
                    sup._worker = None
                    w.retire()
            finally:
                sup.stop()

    def test_exited_worker_drops_and_inhibits(self):
        with tempfile.TemporaryDirectory() as d:
            sup = make_supervisor(d)
            try:
                w = spawn("--exit", "3")
                deadline = time.monotonic() + 10.0
                while w.alive() and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertFalse(w.alive())
                # Pretend this exiting child was the resident worker:
                # the handler must drop it and inhibit, never serve.
                w.loaded = True
                sup._worker = w
                try:
                    status, _out = sup.worker_infer(
                        b"\x00" * 4, [1, 1, 1, 1], 5.0)
                    self.assertEqual(status, "failed")
                    self.assertIsNone(sup._worker)
                    inh = sup.registry.get_state("inhibition")
                    self.assertIsNotNone(inh)
                    self.assertIn("WORKER_DIED", inh["reason"])
                finally:
                    sup._worker = None
                    w.retire()
            finally:
                sup.stop()

    def test_no_worker_is_not_a_fault(self):
        with tempfile.TemporaryDirectory() as d:
            sup = make_supervisor(d)
            try:
                status, out = sup.worker_infer(b"\x00" * 4, [1, 1, 1, 1],
                                               5.0)
                self.assertEqual(status, "no_worker")
                self.assertEqual(out, b"")
                self.assertIsNone(sup.registry.get_state("inhibition"))
            finally:
                sup.stop()


if __name__ == "__main__":
    unittest.main()
