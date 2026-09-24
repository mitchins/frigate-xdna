"""Unit tests: public YOLO contract fixtures (v0.1.2 C4).

Each claimed model family is backed by a real public graph inspected
in the evidence area; this module pins those recorded contracts (see
tests/fixtures/yolo-public-contracts-0.1.2.manifest.json) so inspector
drift can never silently narrow or widen them. ONNX binaries live only
in the evidence area — re-prove them with
tools/verify_yolo_contracts.py.
"""
import json
import os
import unittest

from frigate_xdna.models import inspect as _inspect

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
MANIFEST = os.path.join(
    REPO, "tests", "fixtures",
    "yolo-public-contracts-0.1.2.manifest.json")

REQUIRED_CASE_KEYS = {"id", "role", "onnx", "graph", "verdict"}
REQUIRED_GRAPH_KEYS = {"input", "outputs", "opset", "ir_version",
                       "node_count", "class_count"}


def load_manifest():
    with open(MANIFEST, encoding="utf-8") as f:
        return json.load(f)


class TestManifestShape(unittest.TestCase):
    def test_three_real_cases_with_provenance(self):
        manifest = load_manifest()
        self.assertEqual([c["id"] for c in manifest["cases"]],
                         ["A", "B", "C"])
        for case in manifest["cases"]:
            with self.subTest(case=case["id"]):
                self.assertTrue(REQUIRED_CASE_KEYS <= set(case))
                self.assertTrue(REQUIRED_GRAPH_KEYS <= set(case["graph"]))
                self.assertTrue(
                    "upstream" in case or "provenance" in case,
                    "every case records where its bytes came from")
                onnx = case["onnx"]
                self.assertEqual(len(onnx["sha256"]), 64)
                self.assertGreater(onnx["bytes"], 1_000_000)
                self.assertTrue(onnx["filename"].endswith(".onnx"))


class TestRecordedContracts(unittest.TestCase):
    def test_recorded_outputs_classify_as_recorded(self):
        manifest = load_manifest()
        for case in manifest["cases"]:
            with self.subTest(case=case["id"]):
                graph = case["graph"]
                cls = _inspect.classify_output(graph["outputs"])
                verdict = case["verdict"]
                if verdict["compatible"]:
                    self.assertEqual(cls["profile"],
                                     verdict["profile"])
                    self.assertEqual(cls["channels"] - 4,
                                     graph["class_count"])
                else:
                    self.assertIsNone(cls["profile"])

    def test_recorded_summaries_match_verdicts(self):
        manifest = load_manifest()
        for case in manifest["cases"]:
            with self.subTest(case=case["id"]):
                graph = case["graph"]
                contract = {"input_shape": graph["input"]["shape"],
                            "input_dtype": graph["input"]["dtype"],
                            "outputs": graph["outputs"],
                            "opset": graph["opset"],
                            "ir_version": graph["ir_version"]}
                cls = _inspect.classify_output(graph["outputs"])
                summary = _inspect.summarize_inspection(
                    contract, cls, None, "0" * 64)
                verdict = case["verdict"]
                self.assertEqual(summary["compatible"],
                                 verdict["compatible"])
                if verdict["compatible"]:
                    self.assertEqual(summary["class_count"],
                                     graph["class_count"])
                    self.assertEqual(summary["output"],
                                     verdict["serving_summary"])
                    self.assertIsNone(summary["reason"])

    def test_no_adapter_was_needed(self):
        # Checkpoint 4 exit: the only permitted shape change is an
        # explicit recorded layout adapter. All three mandatory
        # targets fit [1,4+C,N], so none exists.
        manifest = load_manifest()
        for case in manifest["cases"]:
            with self.subTest(case=case["id"]):
                for out in case["graph"]["outputs"]:
                    shape = out["shape"]
                    self.assertEqual(len(shape), 3)
                    self.assertEqual(shape[0], 1)
                    self.assertGreaterEqual(shape[1], 6)
                    self.assertEqual(shape[1] - 4,
                                     case["graph"]["class_count"])


if __name__ == "__main__":
    unittest.main()
