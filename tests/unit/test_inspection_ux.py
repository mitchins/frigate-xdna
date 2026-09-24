"""Unit tests: compatibility inspection as appliance UX (v0.1.2 C3).

One inspected result drives preparation, console logs and status.
Refusals are stable, stored on the ref, and never become compiles.
No label names are ever inferred.
"""
import json
import os
import sqlite3
import tempfile
import time
import unittest

import jsonschema
import numpy as np
import onnx
from onnx import TensorProto, helper

from frigate_xdna.cache.registry import Registry
from frigate_xdna.cli import _read_status
from frigate_xdna.config import load_config
from frigate_xdna.errors import FxdnaError
from frigate_xdna.models import inspect as _inspect
from frigate_xdna.observability.progress import RecordingReporter, format_event
from frigate_xdna.supervisor import Supervisor
from tests.integration.onnx_builders import make_raw_yolo

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def load_schema(name):
    with open(os.path.join(REPO, "schemas", name)) as f:
        return json.load(f)


def make_graph(path, inputs, outputs):
    nodes = [helper.make_node(
        "Constant", [], [o[0]], name=f"c-{o[0]}",
        value=helper.make_tensor(f"t-{o[0]}", TensorProto.FLOAT,
                                 o[1], np.zeros(o[1],
                                                dtype=np.float32
                                                ).flatten().tolist()))
        for o in outputs]
    graph = helper.make_graph(
        nodes, "probe",
        [helper.make_tensor_value_info(n, TensorProto.FLOAT, s)
         for n, s in inputs],
        [helper.make_tensor_value_info(n, TensorProto.FLOAT, s)
         for n, s in outputs])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=False)
    with open(path, "wb") as f:
        f.write(model.SerializeToString())


def make_dynamic_input(path):
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
    graph = helper.make_graph([node], "dyn", [inp], [out])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    with open(path, "wb") as f:
        f.write(model.SerializeToString())


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


def check_model(path):
    with open(path, "rb") as f:
        model, digest = _inspect.load_graph_bytes(f.read())
    contract = _inspect.inspect_model(model)
    cls = _inspect.classify_output(contract["outputs"])
    return contract, cls, digest


class TestSummarize(unittest.TestCase):
    def test_compatible_raw_yolo(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "y.onnx")
            make_raw_yolo(path, res=320, classes=80, seed=3)
            contract, cls, _ = check_model(path)
            summary = _inspect.summarize_inspection(contract, cls, None)
        self.assertTrue(summary["compatible"])
        self.assertIsNone(summary["reason"])
        self.assertEqual(summary["input"], "1x3x320x320")
        self.assertEqual(summary["input_dtype"], "float32")
        self.assertEqual(summary["layout"], "NCHW")
        self.assertEqual(summary["profile"], "yolo-raw")
        self.assertEqual(summary["output"], "raw YOLO [1,84,100]")
        self.assertEqual(summary["class_count"], 80)
        self.assertEqual(summary["opset"], 17)
        # Numeric IDs only: no label vocabulary anywhere.
        blob = json.dumps(summary).lower()
        for banned in ("coco", "person", "labelmap", "labels"):
            self.assertNotIn(banned, blob)

    def test_dynamic_input_refusal_is_stable(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "dyn.onnx")
            make_dynamic_input(path)
            with open(path, "rb") as f:
                model, _ = _inspect.load_graph_bytes(f.read())
            with self.assertRaises(FxdnaError) as ctx:
                _inspect.inspect_model(model)
            self.assertIn("dynamic input dimensions",
                          ctx.exception.message)
            first = ctx.exception.message
        summary = _inspect.summarize_inspection(None, None,
                                                ctx.exception)
        self.assertFalse(summary["compatible"])
        self.assertEqual(summary["reason"], first)
        self.assertIsNone(summary["input"])
        self.assertIsNone(summary["class_count"])

    def test_multi_output_points_at_raw_exports(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "two.onnx")
            make_graph(path, [("images", [1, 3, 320, 320])],
                       [("o1", [1, 84, 100]), ("o2", [1, 4, 100])])
            contract, cls, _ = check_model(path)
            self.assertIsNone(cls["profile"])
            self.assertIn("raw predictions", cls["error"])
            summary = _inspect.summarize_inspection(contract, cls, None)
        self.assertFalse(summary["compatible"])
        self.assertIn("raw predictions", summary["reason"])
        self.assertEqual(summary["input"], "1x3x320x320")

    def test_transposed_single_output_refused(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.onnx")
            make_graph(path, [("images", [1, 3, 320, 320])],
                       [("output0", [1, 100, 84])])
            contract, cls, _ = check_model(path)
            self.assertIsNone(cls["profile"])
            summary = _inspect.summarize_inspection(contract, cls, None)
        self.assertFalse(summary["compatible"])
        self.assertIn("[1,4+C,N]", summary["reason"])


class TestPrepareRecordsInspection(unittest.TestCase):
    def test_compatible_status_carries_inspection(self):
        schema = load_schema("status.schema.json")
        with tempfile.TemporaryDirectory() as data, \
                tempfile.TemporaryDirectory() as models:
            path = os.path.join(models, "good.onnx")
            make_raw_yolo(path, res=320, classes=4, seed=3)
            cfg = load_config({"FXDNA_DATA_DIR": data,
                               "FXDNA_MODEL_DIR": models,
                               "FXDNA_MODELS": "local://good"})
            rec = RecordingReporter()
            sup = Supervisor(cfg, reporter=rec)
            try:
                sup.prepare("local://good")
                insp = next(e for e in rec.events
                            if e["kind"] == "inspection_complete")
                self.assertEqual(insp["profile"], "yolo-raw")
                self.assertEqual(insp["classes"], 4)
                self.assertEqual(insp["shape_str"], "1x3x320x320")
                line = format_event(insp)
                self.assertIn("local://good", line)
                self.assertIn("4 classes", line)
                stored = sup.registry.get_ref("local://good")["inspection"]
                self.assertTrue(stored["compatible"])
                self.assertEqual(stored["class_count"], 4)
                self.assertEqual(stored["output"], "raw YOLO [1,8,100]")
                doc = sup.status("local://good")
                jsonschema.validate(doc, schema)
                view = doc["models"][0]
                self.assertTrue(view["inspection"]["compatible"])
                self.assertEqual(view["inspection"]["class_count"], 4)
                self.assertEqual(
                    pump_until(sup, "local://good", ("PREPARED",)),
                    "PREPARED")
                # Inspection survives preparation: same summary still
                # reported once the artifact is prepared.
                again = sup.status("local://good")["models"][0]
                self.assertTrue(again["inspection"]["compatible"])
            finally:
                sup.stop()
            # Offline status carries it too (same projection path).
            doc = _read_status(cfg, "local://good")
            jsonschema.validate(doc, schema)
            self.assertTrue(doc["models"][0]["inspection"]["compatible"])

    def test_refusal_is_stored_stable_and_queues_nothing(self):
        with tempfile.TemporaryDirectory() as data, \
                tempfile.TemporaryDirectory() as models:
            path = os.path.join(models, "dyn.onnx")
            make_dynamic_input(path)
            cfg = load_config({"FXDNA_DATA_DIR": data,
                               "FXDNA_MODEL_DIR": models})
            rec = RecordingReporter()
            sup = Supervisor(cfg, reporter=rec)
            try:
                with self.assertRaises(FxdnaError) as ctx:
                    sup.prepare("local://dyn")
                self.assertEqual(ctx.exception.error_code,
                                 "UNSUPPORTED_CONTRACT")
                first = ctx.exception.message
                # State machine untouched: still NEW, refusal attached.
                ref = sup.registry.get_ref("local://dyn")
                self.assertEqual(ref["state"], "NEW")
                stored = ref["inspection"]
                self.assertFalse(stored["compatible"])
                self.assertEqual(stored["reason"], first)
                self.assertIn("dynamic", stored["reason"])
                # No compile queued for an unsupported graph.
                self.assertEqual(sup.registry.query(
                    "SELECT uuid FROM jobs WHERE ref=?",
                    ("local://dyn",)), [])
                # Stable: a second prepare refuses identically.
                with self.assertRaises(FxdnaError) as ctx2:
                    sup.prepare("local://dyn")
                self.assertEqual(ctx2.exception.error_code,
                                 "UNSUPPORTED_CONTRACT")
                self.assertEqual(ctx2.exception.message, first)
                fail = next(e for e in rec.events
                            if e["kind"] == "preparation_failed")
                self.assertEqual(fail["phase"], "INSPECTION")
                self.assertEqual(fail["code"], "UNSUPPORTED_CONTRACT")
                self.assertIn("dynamic", format_event(fail))
                doc = sup.status("local://dyn")
                self.assertFalse(
                    doc["models"][0]["inspection"]["compatible"])
            finally:
                sup.stop()


class TestInspectionMigration(unittest.TestCase):
    def test_v3_to_v4_keeps_rows(self):
        with tempfile.TemporaryDirectory() as data:
            reg = Registry(os.path.join(data, "r.db"))
            reg.upsert_ref("local://m", "local", "m")
            reg.close()
            cx = sqlite3.connect(os.path.join(data, "r.db"))
            cx.execute("ALTER TABLE model_refs DROP COLUMN inspection_json")
            cx.execute("UPDATE schema_version SET version=3")
            cx.commit()
            cx.close()
            reg2 = Registry(os.path.join(data, "r.db"))
            try:
                cols = [r[1] for r in reg2.query(
                    "PRAGMA table_info(model_refs)")]
                self.assertIn("inspection_json", cols)
                self.assertEqual(
                    reg2.query("SELECT version FROM schema_version"),
                    [(4,)])
                rec = reg2.get_ref("local://m")
                self.assertEqual(rec["state"], "NEW")
                self.assertIsNone(rec["inspection"])
            finally:
                reg2.close()


if __name__ == "__main__":
    unittest.main()
