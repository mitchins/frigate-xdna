"""Unit tests: launcher probe + phased-compile outcomes (no vendor, no NPU).

Covers the out-of-child probe exchange (worker refusal, short frames),
the VAIML phase timeout/failure exits, and the locked run's terminal
error paths — all with injected worker factories and stubbed spawn.
"""
import hashlib
import os
import tempfile
import time
import unittest
from unittest import mock

from frigate_xdna.compiler import launcher
from frigate_xdna.compiler.launcher import (
    CompilerPrefixes,
    _probe_spec,
    _run_locked,
    _run_probe_exchange,
    _run_vaiml_phase,
    probe_artifact,
    run_compile,
)
from frigate_xdna.runtime.native import RESULT_BYTES, WorkerError
from tests.integration.onnx_builders import make_raw_yolo


def prefixes(workdir):
    sp = os.path.join(workdir, "sp")
    os.makedirs(sp, exist_ok=True)
    return CompilerPrefixes(
        quant_python="/usr/bin/python3",
        compile_python="/usr/bin/python3",
        compile_lib=sp,
        xrt_lib=os.path.join(workdir, "xrt"),
        xrt_root=os.path.join(workdir, "xrtroot"),
        recipe_dir=os.path.join(workdir, "recipe"),
        calib_dir=os.path.join(workdir, "calib"),
        vaiml_config=os.path.join(workdir, "vaiml_config.json"))


class ProbeChild:
    """NativeWorker probe interface over canned behavior."""

    def __init__(self, fail_load=None, frame=None, retire_fails=False):
        self.fail_load = fail_load
        self.frame = frame if frame is not None else bytes(RESULT_BYTES)
        self.retire_fails = retire_fails
        self.retired = False

    def load(self, artifact_path, generation, serving_digest,
             class_count, timeout_s=25.0):
        if self.fail_load is not None:
            raise WorkerError(*self.fail_load)

    def infer(self, payload, shape, generation, timeout_s):
        return self.frame

    def retire(self):
        self.retired = True
        if self.retire_fails:
            raise RuntimeError("retire exploded")


def yolo_source(workdir, classes=8, seed=5):
    path = os.path.join(workdir, "model.onnx")
    make_raw_yolo(path, res=320, classes=classes, seed=seed)
    with open(path, "rb") as f:
        return path, f.read()


class TestProbeExchange(unittest.TestCase):
    def test_probe_passes_on_finite_frame(self):
        child = ProbeChild()
        self.assertIsNone(_run_probe_exchange(
            child, "model.rai", 8, [1, 3, 320, 320],
            1 * 3 * 320 * 320, 5.0))

    def test_worker_refusal_reports_code(self):
        child = ProbeChild(fail_load=("LOAD_REFUSED", "nope"))
        detail = _run_probe_exchange(
            child, "model.rai", 8, [1, 3, 320, 320],
            1 * 3 * 320 * 320, 5.0)
        self.assertTrue(detail.startswith("LOAD_REFUSED: "))

    def test_short_frame_reported(self):
        child = ProbeChild(frame=b"\x00" * 16)
        detail = _run_probe_exchange(
            child, "model.rai", 8, [1, 3, 320, 320],
            1 * 3 * 320 * 320, 5.0)
        self.assertEqual(detail, "short frame 16")

    def test_nonfinite_frame_rejected(self):
        import struct
        frame = struct.pack("<120f", *[float("nan")] * 120)
        child = ProbeChild(frame=frame)
        detail = _run_probe_exchange(
            child, "model.rai", 8, [1, 3, 320, 320],
            1 * 3 * 320 * 320, 5.0)
        self.assertEqual(detail, "non-finite probe output")

    def test_probe_retire_failure_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            child = ProbeChild(retire_fails=True)
            rai = os.path.join(d, "model.rai")
            with open(rai, "wb") as f:
                f.write(b"RAI")
            status, _detail = probe_artifact(
                d, rai, 8, [1, 3, 320, 320],
                worker_factory=lambda: child, timeout_s=5.0)
            self.assertEqual(status, "ok")
            self.assertTrue(child.retired)

    def test_probe_spec_reads_class_count_and_shape(self):
        with tempfile.TemporaryDirectory() as d:
            path, _data = yolo_source(d, classes=8, seed=5)
            classes, shape = _probe_spec(path)
            self.assertEqual(classes, 8)
            self.assertEqual(shape, [1, 3, 320, 320])

    def test_probe_spec_rejects_non_yolo(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "soft.onnx")
            make_softmax(path)
            with self.assertRaises(ValueError) as ctx:
                _probe_spec(path)
            self.assertIn("unsupported profile", str(ctx.exception))


def make_softmax(path):
    """Valid single-output ONNX outside the raw-YOLO contract."""
    from onnx import TensorProto, helper
    inp = helper.make_tensor_value_info("images", TensorProto.FLOAT,
                                        [1, 3, 32, 32])
    out = helper.make_tensor_value_info("probs", TensorProto.FLOAT, [1, 5])
    node = helper.make_node(
        "Constant", [], ["probs"], name="p",
        value=helper.make_tensor("c", TensorProto.FLOAT, [1, 5],
                                 [0.2] * 5))
    graph = helper.make_graph([node], "soft", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid(
        "", 17)])
    model.ir_version = 8
    data = model.SerializeToString()
    with open(path, "wb") as f:
        f.write(data)
    return data


class TestVaimlPhase(unittest.TestCase):
    def test_past_deadline_is_timeout_without_spawning(self):
        calls = []
        with tempfile.TemporaryDirectory() as d:
            res = _run_vaiml_phase(
                prefixes(d), "bf16.onnx", d, "ck",
                time.monotonic() - 1.0, time.monotonic())
            self.assertEqual(calls, [])
        self.assertEqual(res.returncode, 124)
        self.assertIn("timeout", res.error)

    def test_nonzero_rc_is_terminal(self):
        with mock.patch.object(launcher, "spawn", return_value=(3, "", 0)):
            with tempfile.TemporaryDirectory() as d:
                res = _run_vaiml_phase(
                    prefixes(d), "bf16.onnx", d, "ck", time.monotonic() + 9999.0,
                    time.monotonic())
        self.assertEqual(res.returncode, 3)
        self.assertEqual(res.error, "vaiml-compile failed")

    def test_ok_log_parses_rai_coordinates(self):
        sha = hashlib.sha256(b"rai-bytes").hexdigest()
        with mock.patch.object(launcher, "spawn", return_value=(0, "", 0)):
            with tempfile.TemporaryDirectory() as d:
                os.makedirs(os.path.join(d, "cache"), exist_ok=True)
                with open(os.path.join(
                        d, "phase2-compile.stdout.log"), "w") as f:
                    f.write(f"COMPILE_OK {sha} 12345\n")
                out = _run_vaiml_phase(
                    prefixes(d), "bf16.onnx", d, "ck", time.monotonic() + 9999.0,
                    time.monotonic())
        self.assertEqual(out[0], sha)
        self.assertEqual(out[1], 12345)
        self.assertTrue(out[2].endswith(os.path.join("ck", "ck.rai")))


class TestLockedRun(unittest.TestCase):
    def test_zero_budget_is_timeout(self):
        with tempfile.TemporaryDirectory() as d:
            path, _data = yolo_source(d)
            res = _run_locked(
                prefixes(d), path, d, "ck", 0.0, time.monotonic())
        self.assertEqual(res.returncode, 124)

    def test_quant_failure_is_terminal(self):
        with mock.patch.object(launcher, "spawn", return_value=(2, "", 0)):
            with tempfile.TemporaryDirectory() as d:
                path, _data = yolo_source(d)
                res = _run_locked(
                    prefixes(d), path, d, "ck", time.monotonic() + 9999.0,
                    time.monotonic())
        self.assertEqual(res.returncode, 2)
        self.assertEqual(res.error, "bf16-prepare failed")

    def test_full_pipeline_success_with_fakes(self):
        # A coherent success: markers name real coordinates and the
        # artifact exists with exactly the claimed size. (Marker and
        # artifact enforcement now fail anything less; see
        # tests/unit/test_phase_dummies.py for the scripted cases.)
        import hashlib
        rai = b"R" * 64
        sha = hashlib.sha256(rai).hexdigest()
        with mock.patch.object(launcher, "spawn", return_value=(0, "", 0)):
            with tempfile.TemporaryDirectory() as d:
                path, _data = yolo_source(d, classes=8, seed=5)
                with open(os.path.join(
                        d, "phase1-quant.stdout.log"), "w") as f:
                    f.write(f"BF16_PREPARE_OK {sha}\n")
                with open(os.path.join(
                        d, "phase2-compile.stdout.log"), "w") as f:
                    f.write(f"COMPILE_OK {sha} {len(rai)}\n")
                rai_dir = os.path.join(d, "cache", "ck")
                os.makedirs(rai_dir, exist_ok=True)
                with open(os.path.join(rai_dir, "ck.rai"), "wb") as f:
                    f.write(rai)
                res = _run_locked(
                    prefixes(d), path, d, "ck", time.monotonic() + 9999.0,
                    time.monotonic(),
                    worker_factory=ProbeChild)
        self.assertEqual(res.returncode, 0)
        self.assertTrue(res.rai_path.endswith("ck.rai"))

    def _coherent_markers(self, d, rai=b"RAI"):
        """Pre-write marker logs + artifact so a mocked rc-0 run
        reaches the probe stage (marker/artifact enforcement)."""
        import hashlib
        sha = hashlib.sha256(rai).hexdigest()
        with open(os.path.join(d, "phase1-quant.stdout.log"), "w") as f:
            f.write(f"BF16_PREPARE_OK {sha}\n")
        with open(os.path.join(d, "phase2-compile.stdout.log"), "w") as f:
            f.write(f"COMPILE_OK {sha} {len(rai)}\n")
        rai_dir = os.path.join(d, "cache", "ck")
        os.makedirs(rai_dir, exist_ok=True)
        with open(os.path.join(rai_dir, "ck.rai"), "wb") as f:
            f.write(rai)

    def test_probe_inspection_failure_is_terminal(self):
        with mock.patch.object(launcher, "spawn", return_value=(0, "", 0)):
            with tempfile.TemporaryDirectory() as d:
                bad = os.path.join(d, "bad.onnx")
                with open(bad, "wb") as f:
                    f.write(b"not onnx at all")
                self._coherent_markers(d)
                res = _run_locked(
                    prefixes(d), bad, d, "ck", time.monotonic() + 9999.0,
                    time.monotonic(),
                    data_dir=d, worker_factory=ProbeChild)
        self.assertEqual(res.returncode, 7)
        self.assertTrue(res.error.startswith("probe inspection failed"))

    def test_probe_failure_is_terminal(self):
        with mock.patch.object(launcher, "spawn", return_value=(0, "", 0)):
            with tempfile.TemporaryDirectory() as d:
                path, _data = yolo_source(d, classes=8, seed=5)
                self._coherent_markers(d)

                def refusing():
                    return ProbeChild(
                        fail_load=("DEVICE_FAULT", "boom"))
                res = _run_locked(
                    prefixes(d), path, d, "ck", time.monotonic() + 9999.0,
                    time.monotonic(),
                    data_dir=d, worker_factory=refusing)
        self.assertEqual(res.returncode, 7)
        self.assertTrue(res.error.startswith("probe failed: "))

    def test_vaiml_timeout_merges_peaks(self):
        with mock.patch.object(launcher, "spawn", return_value=(0, "", 0)):
            now = time.monotonic()
            jumps = [now] + [now + 9999.0] * 10
            with tempfile.TemporaryDirectory() as d:
                path, _data = yolo_source(d)
                self._coherent_markers(d)
                with mock.patch.object(launcher.time, "monotonic",
                                       side_effect=jumps):
                    res = _run_locked(
                        prefixes(d), path, d, "ck", 50.0, now)
        self.assertEqual(res.returncode, 124)
        self.assertIn("timeout", res.error)

    def test_validate_timeout(self):
        with mock.patch.object(launcher, "spawn", return_value=(0, "", 0)):
            now = time.monotonic()
            jumps = [now, now] + [now + 9999.0] * 10
            with tempfile.TemporaryDirectory() as d:
                path, _data = yolo_source(d)
                self._coherent_markers(d)
                with mock.patch.object(launcher.time, "monotonic",
                                       side_effect=jumps):
                    res = _run_locked(
                        prefixes(d), path, d, "ck", 50.0, now)
        self.assertEqual(res.returncode, 124)
        self.assertIn("timeout", res.error)

    def test_lock_contention_refuses_second_compile(self):
        held = launcher._compile_lock.acquire(blocking=False)
        self.assertTrue(held)
        try:
            with tempfile.TemporaryDirectory() as d:
                path, _data = yolo_source(d)
                res = run_compile(
                    prefixes(d), path, d, "ck", timeout_s=1.0)
        finally:
            launcher._compile_lock.release()
        self.assertEqual(res.returncode, 98)
        self.assertIn("another compile", res.error)


if __name__ == "__main__":
    unittest.main()
