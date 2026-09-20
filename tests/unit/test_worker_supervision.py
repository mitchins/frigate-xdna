"""Unit tests: resident worker supervision (no hardware).

A fake child implementing the NativeWorker control interface drives
Supervisor activation, A->B switching, inference forwarding and the
inhibit-on-death policy (never respawn). Sources/artifacts are real
ONNX bytes + registry rows in a temp data dir.
"""
import hashlib
import os
import tempfile
import time
import unittest

from frigate_xdna.config import Config
from frigate_xdna.errors import FxdnaError
from frigate_xdna.runtime.native import RESULT_BYTES, WorkerError
from frigate_xdna.supervisor import Supervisor
from tests.integration.onnx_builders import make_raw_yolo


class FakeChild:
    """NativeWorker control interface over canned frames."""

    def __init__(self, fail_load=None, fail_infer=None, frame=None):
        self.fail_load = fail_load
        self.fail_infer = fail_infer
        self.frame = frame if frame is not None else bytes(RESULT_BYTES)
        self.loaded = False
        self.generation = 0
        self.retired = False
        self.loads = []
        self.infers = 0

    def alive(self):
        return not self.retired

    def load(self, artifact_path, generation, serving_digest,
             class_count, timeout_s=25.0):
        if self.fail_load is not None:
            raise WorkerError(*self.fail_load)
        if class_count <= 0:
            raise WorkerError("INVALID_ARGS", "no class count")
        self.loads.append({"artifact_path": artifact_path,
                           "generation": generation,
                           "serving_digest": serving_digest,
                           "class_count": class_count})
        self.generation = generation
        self.loaded = True

    def infer(self, payload, shape, generation, timeout_s):
        self.infers += 1
        if self.fail_infer is not None:
            raise WorkerError(*self.fail_infer)
        if generation != self.generation:
            raise WorkerError("INVALID_ARGS", "generation mismatch")
        return self.frame

    def retire(self):
        self.retired = True
        self.loaded = False


def make_config(tmp):
    return Config(data_dir=tmp, endpoint="tcp://127.0.0.1:5555")


def plant(sup, classes=46, seed=11):
    """Prepared ONNX-sourced artifact; returns (ref, compile_key, sha)."""
    data = make_raw_yolo(os.path.join(sup.data_dir, "tmp.onnx"),
                         res=320, classes=classes, seed=seed)
    sha = hashlib.sha256(data).hexdigest()
    src_dir = os.path.join(sup.data_dir, "sources", sha)
    os.makedirs(src_dir, exist_ok=True)
    with open(os.path.join(src_dir, "model.onnx"), "wb") as f:
        f.write(data)
    ref = f"plus://model-{seed}"
    ckey = f"ck-{seed}-{classes}"
    sup.registry.upsert_ref(ref, "plus", f"model-{seed}")
    sup.registry.add_source(sha, len(data), "model.onnx", "test")
    sup.registry.set_ref_source(ref, sha, "md", "PREPARED")
    art_dir = os.path.join(sup.data_dir, "artifacts", ckey)
    os.makedirs(art_dir, exist_ok=True)
    with open(os.path.join(art_dir, "model.rai"), "wb") as f:
        f.write(b"RAI")
    sup.registry.add_artifact(ckey, sha, "a" * 64, 3, "bf16-vaiml-v1",
                              "stx-npu")
    return ref, ckey, sha


def add_job(sup, ref, ckey, stage="PREPARED"):
    sup.registry.execute(
        "INSERT INTO jobs(uuid, ref, compile_key, stage, attempt,"
        " created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
        (f"job-{ckey}", ref, ckey, stage, 1, time.time(), time.time()))


class TestWorkerSupervision(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.children = []
        self.cfg = make_config(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _sup(self, **kw):
        def factory():
            child = FakeChild(**kw)
            self.children.append(child)
            return child
        return Supervisor(self.cfg, worker_factory=factory)

    def test_activate_loads_with_inspected_class_count(self):
        sup = self._sup()
        try:
            ref, ckey, _sha = plant(sup)
            self.assertTrue(sup.activate_worker(ckey))
            self.assertEqual(sup._worker_generation, 1)
            child = self.children[0]
            self.assertTrue(child.loaded)
            self.assertEqual(child.loads[0]["class_count"], 46)
            self.assertTrue(
                child.loads[0]["artifact_path"].endswith("model.rai"))
            active = sup.registry.get_state("active")
            self.assertEqual(active["compile_key"], ckey)
            self.assertEqual(active["worker_generation"], 1)
            self.assertTrue(active["serving_digest"])
        finally:
            sup.stop()

    def test_switch_retires_old_child(self):
        sup = self._sup()
        try:
            _r1, ck1, _s1 = plant(sup, classes=46, seed=11)
            _r2, ck2, _s2 = plant(sup, classes=10, seed=12)
            self.assertTrue(sup.activate_worker(ck1))
            self.assertTrue(sup.activate_worker(ck2))
            self.assertEqual(sup._worker_generation, 2)
            self.assertTrue(self.children[0].retired)
            self.assertFalse(self.children[1].retired)
            self.assertEqual(self.children[1].loads[0]["class_count"], 10)
        finally:
            sup.stop()

    def test_refused_load_inhibits_without_generation(self):
        sup = self._sup(fail_load=("INVALID_MODEL", "bad bytes"))
        try:
            _ref, ckey, _sha = plant(sup)
            self.assertFalse(sup.activate_worker(ckey))
            self.assertEqual(sup._worker_generation, 0)
            self.assertIsNone(sup._worker)
            inh = sup.registry.get_state("inhibition")
            self.assertEqual(inh["reason"], "WORKER_LOAD:INVALID_MODEL")
            self.assertTrue(self.children[0].retired)
        finally:
            sup.stop()

    def test_spawn_failure_inhibits(self):
        def factory():
            raise WorkerError("WORKER_SPAWN", "noexec")
        sup = Supervisor(self.cfg, worker_factory=factory)
        try:
            _ref, ckey, _sha = plant(sup)
            self.assertFalse(sup.activate_worker(ckey))
            inh = sup.registry.get_state("inhibition")
            self.assertEqual(inh["reason"], "WORKER_SPAWN")
        finally:
            sup.stop()

    def test_infer_ok_no_worker_and_failure(self):
        sup = self._sup()
        try:
            status, _out = sup.worker_infer(b"xy", [1, 3, 2, 2], 1.0)
            self.assertEqual(status, "no_worker")
            _ref, ckey, _sha = plant(sup)
            self.assertTrue(sup.activate_worker(ckey))
            status, out = sup.worker_infer(bytes(16 * 4), [1, 3, 2, 2],
                                           5.0)
            self.assertEqual(status, "ok")
            self.assertEqual(len(out), RESULT_BYTES)
        finally:
            sup.stop()

    def test_dead_worker_inhibits_and_never_respawns(self):
        sup = self._sup()
        try:
            _ref, ckey, _sha = plant(sup)
            self.assertTrue(sup.activate_worker(ckey))
            child = self.children[0]
            child.retired = True  # simulate unexpected death
            status, _out = sup.worker_infer(bytes(4), [1, 3, 1, 1], 1.0)
            self.assertEqual(status, "failed")
            self.assertIsNone(sup._worker)
            inh = sup.registry.get_state("inhibition")
            self.assertEqual(inh["reason"], "WORKER_DIED")
            # no respawn: still no worker afterwards
            status, _out = sup.worker_infer(bytes(4), [1, 3, 1, 1], 1.0)
            self.assertEqual(status, "no_worker")
            self.assertEqual(len(self.children), 1)
        finally:
            sup.stop()

    def test_device_fault_inhibits_but_bad_tensor_does_not(self):
        sup = self._sup(fail_infer=("DEVICE_FAULT", "npu gone"))
        try:
            _ref, ckey, _sha = plant(sup)
            self.assertTrue(sup.activate_worker(ckey))
            status, _out = sup.worker_infer(bytes(4), [1, 3, 1, 1], 1.0)
            self.assertEqual(status, "failed")
            inh = sup.registry.get_state("inhibition")
            self.assertEqual(inh["reason"], "WORKER_DEVICE_FAULT")
        finally:
            sup.stop()
        # Fresh data dir: the inhibition above must not leak across.
        self.tmp2 = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp2.cleanup)
        self.cfg = make_config(self.tmp2.name)
        sup2 = self._sup(fail_infer=("INVALID_ARGS", "bad shape"))
        try:
            _ref, ckey, _sha = plant(sup2)
            self.assertTrue(sup2.activate_worker(ckey))
            status, _out = sup2.worker_infer(bytes(4), [1, 3, 1, 1], 1.0)
            self.assertEqual(status, "failed")
            self.assertIsNone(sup2.registry.get_state("inhibition"))
            self.assertIsNotNone(sup2._worker)
        finally:
            sup2.stop()

    def test_activate_ref_and_unknown_ref(self):
        sup = self._sup()
        try:
            ref, ckey, _sha = plant(sup)
            add_job(sup, ref, ckey)
            doc = sup.activate(ref)
            self.assertEqual(doc["state"], "ACTIVE")
            self.assertEqual(doc["compile_key"], ckey)
            self.assertEqual(doc["worker_generation"], 1)
            with self.assertRaises(FxdnaError) as ctx:
                sup.activate("plus://nope")
            self.assertEqual(ctx.exception.error_code, "NOT_PREPARED")
        finally:
            sup.stop()


if __name__ == "__main__":
    unittest.main()
