"""Unit tests: desired-state/cache/state correctness (v0.1.2 C5).

Local models get every appliance behaviour Plus has: shared bytes
converge on one artifact, revisions prepare without worker switches,
absent sources never destroy caches, failures stay bounded, and a
fresh Frigate transfer rebinds by content — never by basename.
"""
import hashlib
import os
import tempfile
import time
import unittest

import numpy as np
from onnx import TensorProto, helper

from frigate_xdna.config import load_config
from frigate_xdna.errors import FxdnaError
from frigate_xdna.models import inspect as _inspect
from frigate_xdna.observability.progress import RecordingReporter
from frigate_xdna.supervisor import Supervisor
from tests.integration.onnx_builders import make_raw_yolo


def put(d, name, seed=3, classes=4, res=320):
    path = os.path.join(d, name + ".onnx")
    return path, make_raw_yolo(path, res=res, classes=classes, seed=seed)


def digest_of(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def pump_until(sup, ref, stages, timeout=15.0):
    deadline = time.monotonic() + timeout
    while True:
        sup.pump(0.05)
        state = (sup.registry.get_ref(ref) or {}).get("state")
        if state in stages:
            return state
        if time.monotonic() >= deadline:
            raise AssertionError(f"{ref} stuck at {state}")
        time.sleep(0.02)


def zmq_ingest(sup, basename, raw):
    """Mirror the transport: inspect shared code, then content ingest."""
    sha = hashlib.sha256(raw).hexdigest()
    model, _ = _inspect.load_graph_bytes(raw)
    contract = _inspect.inspect_model(model)
    cls = _inspect.classify_output(contract["outputs"])
    assert cls["profile"] is not None
    return sup.ingest_zmq_bytes(basename, raw, contract, cls, sha), sha


def make_dynamic(path):
    inp = helper.make_tensor_value_info("images", TensorProto.FLOAT,
                                        [1, 3, "h", "w"])
    out = helper.make_tensor_value_info("output0", TensorProto.FLOAT,
                                        [1, 84, 100])
    node = helper.make_node(
        "Constant", [], ["output0"], name="c",
        value=helper.make_tensor("t", TensorProto.FLOAT, [1, 84, 100],
                                 np.zeros((1, 84, 100),
                                          dtype=np.float32
                                          ).flatten().tolist()))
    model = helper.make_model(
        helper.make_graph([node], "dyn", [inp], [out]),
        opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    with open(path, "wb") as f:
        f.write(model.SerializeToString())


class FakeChild:
    """Minimal NativeWorker control interface (load-only)."""

    def __init__(self):
        self.loaded = False
        self.generation = 0
        self.loads = []

    def alive(self):
        return True

    def load(self, artifact_path, generation, serving_digest,
             class_count, timeout_s=25.0):
        self.loads.append(artifact_path)
        self.generation = generation
        self.loaded = True

    def retire(self):
        self.loaded = False


class StubPlus:
    """Fake Plus client serving fixture bytes (no network)."""

    def __init__(self, raw, calls):
        self.raw = raw
        self.calls = calls

    def get_model_info(self, model_id):
        self.calls.append(("info", model_id))
        return {"id": model_id, "width": 320, "height": 320}

    def get_model_download_url(self, model_id, allow_private_hosts=()):
        self.calls.append(("url", model_id))
        return "https://plus.test/m.onnx"

    def download_model(self, url, allow_private_hosts=()):
        self.calls.append(("download", url))
        return self.raw


class TestDesiredState(unittest.TestCase):
    def _sup(self, cfg, **kw):
        return Supervisor(cfg, reporter=RecordingReporter(), **kw)

    def test_shared_bytes_one_artifact(self):
        with tempfile.TemporaryDirectory() as data, \
                tempfile.TemporaryDirectory() as models:
            src, raw = put(models, "orig")
            with open(os.path.join(models, "alias.onnx"), "wb") as f:
                f.write(raw)
            cfg = load_config({"FXDNA_DATA_DIR": data,
                               "FXDNA_MODEL_DIR": models,
                               "FXDNA_MODELS": "local://orig\n"
                                               "local://alias"})
            sup = self._sup(cfg)
            try:
                outs = sup.reconcile_configured()
                self.assertTrue(all(o["ok"] for o in outs), outs)
                for ref in ("local://orig", "local://alias"):
                    self.assertEqual(
                        pump_until(sup, ref, ("PREPARED",)), "PREPARED")
                keys = {sup.registry.prepared_key_for_ref(r)
                        for r in ("local://orig", "local://alias")}
                self.assertEqual(len(keys), 1)
                self.assertEqual(sup.registry.query(
                    "SELECT COUNT(*) FROM sources"), [(1,)])
                self.assertEqual(sup.registry.query(
                    "SELECT COUNT(*) FROM artifacts"), [(1,)])
            finally:
                sup.stop()

    def test_zmq_transfer_converges_no_second_compile(self):
        with tempfile.TemporaryDirectory() as data, \
                tempfile.TemporaryDirectory() as models:
            _, raw = put(models, "yolov9-t-320")
            cfg = load_config({"FXDNA_DATA_DIR": data,
                               "FXDNA_MODEL_DIR": models,
                               "FXDNA_MODELS": "local://yolov9-t-320"})
            sup = self._sup(cfg)
            try:
                self.assertTrue(sup.reconcile_configured()[0]["ok"])
                self.assertEqual(pump_until(sup, "local://yolov9-t-320",
                                            ("PREPARED",)), "PREPARED")
                key = sup.registry.prepared_key_for_ref(
                    "local://yolov9-t-320")
                out, sha = zmq_ingest(sup, "yolov9-t-320.onnx", raw)
                self.assertTrue(out["cache_hit"])
                self.assertEqual(out["compile_key"], key)
                self.assertEqual(out["source_sha256"], sha)
                # The transfer queued no compilation: every job for the
                # synthetic ref is already terminal-PREPARED.
                stages = sup.registry.query(
                    "SELECT DISTINCT stage FROM jobs WHERE ref=?",
                    (f"zmq-upload:{sha}",))
                self.assertEqual(stages, [("PREPARED",)])
                self.assertEqual(sup.registry.query(
                    "SELECT COUNT(*) FROM artifacts"), [(1,)])
            finally:
                sup.stop()

    def test_rename_identical_bytes(self):
        with tempfile.TemporaryDirectory() as data, \
                tempfile.TemporaryDirectory() as models:
            src, raw = put(models, "before")
            cfg = load_config({"FXDNA_DATA_DIR": data,
                               "FXDNA_MODEL_DIR": models})
            sup = self._sup(cfg)
            try:
                out1 = sup.prepare(src)
                self.assertEqual(pump_until(sup, src, ("PREPARED",)),
                                 "PREPARED")
                dst = os.path.join(models, "after.onnx")
                with open(dst, "wb") as f:
                    f.write(raw)
                out2 = sup.prepare(dst)
                self.assertTrue(out2["cache_hit"])
                self.assertEqual(out2["compile_key"],
                                 out1["compile_key"])
            finally:
                sup.stop()

    def test_reconcile_prepares_revision_keeps_old(self):
        with tempfile.TemporaryDirectory() as data, \
                tempfile.TemporaryDirectory() as models:
            put(models, "driveway", seed=3)
            cfg = load_config({"FXDNA_DATA_DIR": data,
                               "FXDNA_MODEL_DIR": models,
                               "FXDNA_MODELS": "local://driveway"})
            sup = self._sup(cfg)
            try:
                self.assertTrue(sup.reconcile_configured()[0]["ok"])
                self.assertEqual(pump_until(sup, "local://driveway",
                                            ("PREPARED",)), "PREPARED")
                old_key = sup.registry.prepared_key_for_ref(
                    "local://driveway")
                old_source = sup.registry.get_ref(
                    "local://driveway")["source_sha256"]
                put(models, "driveway", seed=4)
                outs = sup.reconcile_configured()
                self.assertTrue(outs[0]["ok"], outs)
                self.assertEqual(pump_until(sup, "local://driveway",
                                            ("PREPARED",)), "PREPARED")
                ref = sup.registry.get_ref("local://driveway")
                self.assertNotEqual(ref["source_sha256"], old_source)
                self.assertNotEqual(
                    sup.registry.prepared_key_for_ref("local://driveway"),
                    old_key)
                # Previous artifact preserved, not switched away
                # silently: it is still a committed row with bytes.
                self.assertIsNotNone(sup.registry.get_artifact(old_key))
                self.assertTrue(os.path.isfile(os.path.join(
                    data, "artifacts", old_key, "model.rai")))
                # Reconciled bytes re-prepare idempotently...
                again = sup.prepare("local://driveway")
                self.assertTrue(again["cache_hit"])
                # ...while a further unacknowledged change still asks.
                put(models, "driveway", seed=6)
                with self.assertRaises(FxdnaError) as ctx:
                    sup.prepare("local://driveway")
                self.assertEqual(ctx.exception.error_code,
                                 "SOURCE_CHANGED")
            finally:
                sup.stop()

    def test_reconcile_never_switches_worker(self):
        children = []

        def factory():
            child = FakeChild()
            children.append(child)
            return child

        with tempfile.TemporaryDirectory() as data, \
                tempfile.TemporaryDirectory() as models:
            put(models, "cam", seed=3)
            cfg = load_config({"FXDNA_DATA_DIR": data,
                               "FXDNA_MODEL_DIR": models,
                               "FXDNA_MODELS": "local://cam"})
            sup = self._sup(cfg, worker_factory=factory)
            try:
                sup.reconcile_configured()
                self.assertEqual(pump_until(sup, "local://cam",
                                            ("PREPARED",)), "PREPARED")
                sup.activate("local://cam")
                live_before = sup._live_worker_key()
                self.assertIsNotNone(live_before)
                self.assertEqual(len(children[0].loads), 1)
                put(models, "cam", seed=5)
                sup.reconcile_configured()
                self.assertEqual(pump_until(sup, "local://cam",
                                            ("PREPARED",)), "PREPARED")
                self.assertEqual(sup._live_worker_key(), live_before)
                self.assertEqual(len(children), 1)
                self.assertEqual(len(children[0].loads), 1)
            finally:
                sup.stop()

    def test_absent_source_restart_preserves_cache(self):
        with tempfile.TemporaryDirectory() as data, \
                tempfile.TemporaryDirectory() as models:
            _, raw = put(models, "field")
            cfg = load_config({"FXDNA_DATA_DIR": data,
                               "FXDNA_MODEL_DIR": models,
                               "FXDNA_MODELS": "local://field"})
            sup = self._sup(cfg)
            try:
                sup.reconcile_configured()
                self.assertEqual(pump_until(sup, "local://field",
                                            ("PREPARED",)), "PREPARED")
            finally:
                sup.stop()
            os.remove(os.path.join(models, "field.onnx"))
            sup2 = self._sup(cfg)
            try:
                outs = sup2.reconcile_configured()
                self.assertFalse(outs[0]["ok"])
                self.assertIn("not found", outs[0]["reason"])
                # Cache untouched: source + artifact rows and bytes.
                self.assertEqual(sup2.registry.query(
                    "SELECT COUNT(*) FROM sources"), [(1,)])
                self.assertEqual(sup2.registry.query(
                    "SELECT COUNT(*) FROM artifacts"), [(1,)])
                key = sup2.registry.prepared_key_for_ref("local://field")
                self.assertIsNotNone(key)
                self.assertTrue(os.path.isfile(os.path.join(
                    data, "artifacts", key, "model.rai")))
                # Still servable by content: a Frigate transfer of the
                # same bytes cache-hits without the mounted file.
                out, _ = zmq_ingest(sup2, "field.onnx", raw)
                self.assertTrue(out["cache_hit"])
                self.assertEqual(out["compile_key"], key)
            finally:
                sup2.stop()

    def test_bad_sources_fail_bounded(self):
        with tempfile.TemporaryDirectory() as data, \
                tempfile.TemporaryDirectory() as models:
            cfg = load_config({"FXDNA_DATA_DIR": data,
                               "FXDNA_MODEL_DIR": models})
            sup = self._sup(cfg)
            try:
                # A directory is not a file.
                os.mkdir(os.path.join(models, "dir.onnx"))
                with self.assertRaises(FxdnaError) as ctx:
                    sup.prepare("local://dir")
                self.assertEqual(ctx.exception.error_code,
                                 "INVALID_MODEL")
                # Corrupt bytes are refused, not compiled.
                bad = os.path.join(models, "bad.onnx")
                with open(bad, "wb") as f:
                    f.write(b"not onnx at all")
                with self.assertRaises(FxdnaError):
                    sup.prepare("local://bad")
                # Unsupported graph is refused, not compiled.
                dyn = os.path.join(models, "dyn.onnx")
                make_dynamic(dyn)
                with self.assertRaises(FxdnaError) as ctx:
                    sup.prepare("local://dyn")
                self.assertEqual(ctx.exception.error_code,
                                 "UNSUPPORTED_CONTRACT")
                # Escaped symlink never becomes a source.
                outside = os.path.join(data, "outside.onnx")
                with open(outside, "wb") as f:
                    f.write(b"nope")
                os.symlink(outside,
                           os.path.join(models, "evil.onnx"))
                with self.assertRaises(FxdnaError):
                    sup.prepare("local://evil")
                for ref in ("local://dir", "local://bad",
                            "local://dyn", "local://evil"):
                    self.assertEqual(sup.registry.query(
                        "SELECT uuid FROM jobs WHERE ref=?", (ref,)), [],
                        ref)
            finally:
                sup.stop()

    def test_mixed_local_plus(self):
        with tempfile.TemporaryDirectory() as data, \
                tempfile.TemporaryDirectory() as models:
            _, raw = put(models, "local-one", seed=3)
            _, plus_raw = put(models, "plus-src", seed=9)
            calls = []
            cfg = load_config({"FXDNA_DATA_DIR": data,
                               "FXDNA_MODEL_DIR": models,
                               "FXDNA_MODELS": "local://local-one\n"
                                               "plus://MODEL_P",
                               "PLUS_API_KEY": "test-key"})
            sup = self._sup(
                cfg,
                plus_client_factory=lambda secret: StubPlus(plus_raw,
                                                           calls))
            try:
                outs = sup.reconcile_configured()
                self.assertTrue(all(o["ok"] for o in outs), outs)
                for ref in ("local://local-one", "plus://MODEL_P"):
                    self.assertEqual(
                        pump_until(sup, ref, ("PREPARED",)), "PREPARED")
                kinds = {sup.registry.get_ref(r)["kind"]
                         for r in ("local://local-one", "plus://MODEL_P")}
                self.assertEqual(kinds, {"local", "plus"})
                keys = {sup.registry.prepared_key_for_ref(r)
                        for r in ("local://local-one", "plus://MODEL_P")}
                self.assertEqual(len(keys), 2)
                self.assertTrue(any(c[0] == "download" for c in calls))
            finally:
                sup.stop()

    def test_local_only_makes_zero_plus_calls(self):
        calls = []

        def factory(secret):
            calls.append(secret)
            raise AssertionError("Plus must not be touched")

        with tempfile.TemporaryDirectory() as data, \
                tempfile.TemporaryDirectory() as models:
            put(models, "solo", seed=3)
            cfg = load_config({"FXDNA_DATA_DIR": data,
                               "FXDNA_MODEL_DIR": models,
                               "FXDNA_MODELS": "local://solo"})
            sup = self._sup(cfg, plus_client_factory=factory)
            try:
                outs = sup.reconcile_configured()
                self.assertTrue(all(o["ok"] for o in outs), outs)
                self.assertEqual(pump_until(sup, "local://solo",
                                            ("PREPARED",)), "PREPARED")
                self.assertEqual(calls, [])
            finally:
                sup.stop()

    def test_reconnect_rebinds_without_recompile(self):
        with tempfile.TemporaryDirectory() as data, \
                tempfile.TemporaryDirectory() as models:
            _, raw = put(models, "porch", seed=3)
            cfg = load_config({"FXDNA_DATA_DIR": data,
                               "FXDNA_MODEL_DIR": models})
            sup = self._sup(cfg)
            try:
                sup.prepare("local://porch")
                self.assertEqual(pump_until(sup, "local://porch",
                                            ("PREPARED",)), "PREPARED")
                key = sup.registry.prepared_key_for_ref("local://porch")
                # Two Frigate connections, different basenames.
                for name in ("porch.onnx", "renamed-copy.onnx"):
                    out, _ = zmq_ingest(sup, name, raw)
                    self.assertTrue(out["cache_hit"], name)
                    self.assertEqual(out["compile_key"], key, name)
                self.assertEqual(sup.registry.query(
                    "SELECT COUNT(*) FROM artifacts"), [(1,)])
            finally:
                sup.stop()

    def test_same_basename_different_bytes_stay_distinct(self):
        with tempfile.TemporaryDirectory() as data, \
                tempfile.TemporaryDirectory() as models:
            _, raw_a = put(models, "a", seed=3)
            _, raw_b = put(models, "b", seed=7)
            cfg = load_config({"FXDNA_DATA_DIR": data,
                               "FXDNA_MODEL_DIR": models})
            sup = self._sup(cfg)
            try:
                out_a, sha_a = zmq_ingest(sup, "model.onnx", raw_a)
                out_b, sha_b = zmq_ingest(sup, "model.onnx", raw_b)
                self.assertNotEqual(sha_a, sha_b)
                self.assertNotEqual(out_a["compile_key"],
                                    out_b["compile_key"])
                for sha in (sha_a, sha_b):
                    self.assertEqual(pump_until(
                        sup, f"zmq-upload:{sha}", ("PREPARED",)),
                        "PREPARED")
                self.assertEqual(sup.registry.query(
                    "SELECT COUNT(*) FROM artifacts"), [(2,)])
            finally:
                sup.stop()


if __name__ == "__main__":
    unittest.main()
