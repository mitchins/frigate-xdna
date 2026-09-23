"""Unit tests: supervisor activation failure guards (no hardware).

Inspection failures, refused artifacts, failed publishes and dying
children must inhibit or roll back explicitly — never leave a live
worker with no published identity, and never raise out of a
best-effort guard.
"""
import hashlib
import os
import tempfile
import time
import unittest
from unittest import mock

from frigate_xdna.config import Config
from frigate_xdna.errors import FxdnaError
from frigate_xdna.runtime.native import RESULT_BYTES, WorkerError
from frigate_xdna.supervisor import Supervisor
from tests.integration.onnx_builders import make_raw_yolo


class FakeChild:
    def __init__(self, fail_load=None, retire_fails=False):
        self.fail_load = fail_load
        self.retire_fails = retire_fails
        self.retired = False
        self.loaded = False
        self.generation = 0

    def load(self, artifact_path, generation, serving_digest,
             class_count, timeout_s=25.0):
        if self.fail_load is not None:
            raise WorkerError(*self.fail_load)
        self.generation = generation
        self.loaded = True

    def infer(self, payload, shape, generation, timeout_s):
        return bytes(RESULT_BYTES)

    def alive(self):
        return not self.retired

    def retire(self):
        # A real retire can fail before the child is dead (e.g.
        # terminate() raising after a wait timeout): raise first so
        # the child is still live and loaded at the failure point.
        if self.retire_fails:
            raise RuntimeError("retire exploded")
        self.retired = True
        self.loaded = False


def make_config(tmp):
    return Config(data_dir=tmp, endpoint="tcp://127.0.0.1:5555")


def plant(sup, classes=8, seed=21, with_rai=True):
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
    if with_rai:
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


class TestSupervisorGuards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.children = []
        self.cfg = make_config(self.tmp.name)
        self.fail_load = None
        self.retire_fails = False

    def tearDown(self):
        self.tmp.cleanup()

    def _sup(self):
        def factory():
            child = FakeChild(fail_load=self.fail_load,
                              retire_fails=self.retire_fails)
            self.children.append(child)
            return child
        return Supervisor(self.cfg, worker_factory=factory)

    def test_stop_tolerates_retire_failure(self):
        sup = self._sup()
        try:
            _ref, ckey, _sha = plant(sup)
            self.assertTrue(sup.activate_worker(ckey))
            self.assertTrue(self.children[0].loaded)
            sup._worker.retire_fails = True
            sup.stop()
            self.assertIsNone(sup._worker)
        except Exception:
            sup.stop()
            raise

    def test_ingest_zmq_bytes_binds_content_ref(self):
        sup = self._sup()
        try:
            seen = []
            sup._ingest_source = lambda *a: seen.append(a) or {
                "compile_key": "ck"}
            out = sup.ingest_zmq_bytes("alias.onnx", b"bytes",
                                       {"ok": True}, {"profile": "yolo-raw"},
                                       "deadbeef")
            self.assertEqual(out, {"compile_key": "ck"})
            self.assertEqual(seen[0][0], "zmq-upload:deadbeef")
            rec = sup.registry.get_ref("zmq-upload:deadbeef")
            self.assertIsNotNone(rec)
        finally:
            sup.stop()

    def test_activate_refused_artifact_reports_activation_failed(self):
        sup = self._sup()
        try:
            ref, ckey, _sha = plant(sup)
            add_job(sup, ref, ckey)
            sup.activate_worker = lambda _ck: False
            with self.assertRaises(FxdnaError) as ctx:
                sup.activate(ref, maintenance=True)
            self.assertEqual(ctx.exception.error_code, "ACTIVATION_FAILED")
        finally:
            sup.stop()

    def test_unexpected_inspection_error_maps_to_validation_failed(self):
        sup = self._sup()
        try:
            _ref, ckey, _sha = plant(sup, with_rai=True)
            from frigate_xdna import supervisor as sup_mod
            with mock.patch.object(sup_mod._inspect, "load_graph_bytes",
                                   side_effect=RuntimeError("odd")):
                with self.assertRaises(FxdnaError) as ctx:
                    sup._inspect_activation(ckey)
            self.assertEqual(ctx.exception.error_code, "VALIDATION_FAILED")
        finally:
            sup.stop()

    def test_corrupt_source_surfaces_invalid_model(self):
        sup = self._sup()
        try:
            _ref, ckey, sha = plant(sup, with_rai=True)
            with open(os.path.join(sup.data_dir, "sources", sha,
                                   "model.onnx"), "wb") as f:
                f.write(b"garbage-not-onnx")
            with self.assertRaises(FxdnaError) as ctx:
                sup._inspect_activation(ckey)
            self.assertEqual(ctx.exception.error_code, "INVALID_MODEL")
        finally:
            sup.stop()

    def test_unsupported_contract_refused(self):
        sup = self._sup()
        try:
            _ref, ckey, _sha = plant(sup, classes=0, seed=22)
            with self.assertRaises(FxdnaError) as ctx:
                sup._inspect_activation(ckey)
            self.assertEqual(ctx.exception.error_code,
                             "UNSUPPORTED_CONTRACT")
        finally:
            sup.stop()

    def test_missing_artifact_bytes_not_prepared(self):
        sup = self._sup()
        try:
            _ref, ckey, _sha = plant(sup, with_rai=False)
            with self.assertRaises(FxdnaError) as ctx:
                sup._inspect_activation(ckey)
            self.assertEqual(ctx.exception.error_code, "NOT_PREPARED")
        finally:
            sup.stop()

    def test_inhibit_quiet_never_raises(self):
        sup = self._sup()
        try:
            sup.registry.close()
            sup._inhibit_quiet("REASON", "ref")
        finally:
            sup.stop()

    def test_load_failure_with_exploding_retire_still_inhibits(self):
        self.fail_load = ("DEVICE_FAULT", "dead")
        self.retire_fails = True
        sup = self._sup()
        try:
            _ref, ckey, _sha = plant(sup)
            self.assertFalse(sup.activate_worker(ckey))
            self.assertIsNotNone(sup.registry.get_state("inhibition"))
        finally:
            sup.stop()

    def test_publish_failure_rolls_back_and_inhibits(self):
        self.retire_fails = True
        sup = self._sup()
        try:
            _ref, ckey, _sha = plant(sup)
            real_set_state = sup.registry.set_state

            def guarded(key, value):
                if key == "active":
                    raise RuntimeError("publish down")
                return real_set_state(key, value)

            with mock.patch.object(sup.registry, "set_state",
                                   side_effect=guarded):
                self.assertFalse(sup.activate_worker(ckey))
            child = self.children[0]
            self.assertTrue(child.loaded)
            self.assertIsNone(sup._worker)
            self.assertEqual(sup._worker_generation, 0)
            self.assertIsNotNone(sup.registry.get_state("inhibition"))
        finally:
            sup.stop()

    def test_publish_failure_never_leaves_untracked_live_child(self):
        """A new child whose retire explodes during publish rollback
        must be confirmed dead or remain explicitly tracked."""
        self.retire_fails = True
        sup = self._sup()
        try:
            _ref, ckey, _sha = plant(sup)
            real_set_state = sup.registry.set_state

            def guarded(key, value):
                if key == "active":
                    raise RuntimeError("publish down")
                return real_set_state(key, value)

            with mock.patch.object(sup.registry, "set_state",
                                   side_effect=guarded):
                self.assertFalse(sup.activate_worker(ckey))
            child = self.children[0]
            self.assertTrue(child.alive())
            self.assertTrue(child.loaded)
            self.assertTrue(child.retired or sup._worker is child
                            or child in sup._orphans)
        finally:
            sup.stop()

    def test_old_child_retire_failure_ignored_on_switch(self):
        sup = self._sup()
        try:
            _r1, ck1, _s1 = plant(sup, classes=8, seed=23)
            _r2, ck2, _s2 = plant(sup, classes=6, seed=24)
            self.assertTrue(sup.activate_worker(ck1))
            self.assertTrue(self.children[0].loaded)
            self.children[0].retire_fails = True
            self.assertTrue(sup.activate_worker(ck2))
            self.assertEqual(sup._worker_generation, 2)
        finally:
            sup.stop()

    def test_load_failure_with_live_child_tracks_orphan(self):
        self.fail_load = ("DEVICE_FAULT", "dead")
        self.retire_fails = True
        sup = self._sup()
        try:
            _ref, ckey, _sha = plant(sup)
            self.assertFalse(sup.activate_worker(ckey))
            child = self.children[0]
            self.assertTrue(child.alive())
            self.assertIn(child, sup._orphans)
            self.assertIsNone(sup._worker)
            self.assertIsNotNone(sup.registry.get_state("inhibition"))
        finally:
            sup.stop()

    def test_old_child_retire_failure_on_switch_tracks_orphan(self):
        sup = self._sup()
        try:
            _r1, ck1, _s1 = plant(sup, classes=8, seed=23)
            _r2, ck2, _s2 = plant(sup, classes=6, seed=24)
            self.assertTrue(sup.activate_worker(ck1))
            old = self.children[0]
            self.assertTrue(old.loaded)
            old.retire_fails = True
            self.assertTrue(sup.activate_worker(ck2))
            self.assertEqual(sup._worker_generation, 2)
            self.assertIn(old, sup._orphans)
        finally:
            sup.stop()

    def test_drop_worker_with_live_child_tracks_orphan(self):
        sup = self._sup()
        try:
            _ref, ckey, _sha = plant(sup)
            self.assertTrue(sup.activate_worker(ckey))
            child = self.children[0]
            self.assertTrue(child.loaded)
            child.retire_fails = True
            sup._drop_worker("TEST")
            self.assertIsNone(sup._worker)
            self.assertTrue(child.alive())
            self.assertIn(child, sup._orphans)
            self.assertIsNotNone(sup.registry.get_state("inhibition"))
        finally:
            sup.stop()

    def test_drop_worker_tolerates_retire_failure(self):
        sup = self._sup()
        try:
            _ref, ckey, _sha = plant(sup)
            self.assertTrue(sup.activate_worker(ckey))
            self.assertTrue(self.children[0].loaded)
            sup._worker.retire_fails = True
            sup._drop_worker("TEST")
            self.assertIsNone(sup._worker)
            self.assertIsNotNone(sup.registry.get_state("inhibition"))
        finally:
            sup.stop()


if __name__ == "__main__":
    unittest.main()
