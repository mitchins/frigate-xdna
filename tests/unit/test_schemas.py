"""Unit tests: JSON schemas validate fixtures; invalid docs fail."""
import json
import os
import unittest

import jsonschema

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def load_schema(name):
    with open(os.path.join(REPO, "schemas", name)) as f:
        return json.load(f)


def load_fixture(name):
    with open(os.path.join(REPO, "tests", "fixtures", name)) as f:
        return json.load(f)


class TestSchemas(unittest.TestCase):
    def test_status_schema_accepts_cli_document(self):
        schema = load_schema("status.schema.json")
        doc = {"schema_version": 1, "service": "frigate-xdna",
               "version": "0.1.0.dev0", "state": "NOT_IMPLEMENTED",
               "active": None, "models": []}
        jsonschema.validate(doc, schema)

    def test_status_schema_rejects_ready_forgery(self):
        schema = load_schema("status.schema.json")
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate({"schema_version": 1,
                                 "service": "frigate-xdna",
                                 "version": "x", "state": "READY"}, schema)

    def test_descriptor_fixtures_validate(self):
        schema = load_schema("model-descriptor.schema.json")
        for name in ("yolov9s-320.descriptor.json",
                     "plus-private-example.descriptor.json"):
            with self.subTest(name=name):
                jsonschema.validate(load_fixture(name), schema)

    def test_descriptor_rejects_unknown_profile(self):
        schema = load_schema("model-descriptor.schema.json")
        bad = load_fixture("yolov9s-320.descriptor.json")
        bad["output"]["profile"] = "segmentation"
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(bad, schema)

    def test_descriptor_rejects_implicit_class_count(self):
        schema = load_schema("model-descriptor.schema.json")
        bad = load_fixture("yolov9s-320.descriptor.json")
        del bad["output"]["class_count"]
        del bad["serving"]["label_map"]
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(bad, schema)

    def test_descriptor_rejects_batch_and_channel_mismatch(self):
        import copy
        schema = load_schema("model-descriptor.schema.json")
        base = load_fixture("yolov9s-320.descriptor.json")
        bad = copy.deepcopy(base)
        bad["input"]["shape"] = [2, 3, 320, 320]
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(bad, schema)
        bad = copy.deepcopy(base)
        bad["input"]["shape"] = [1, 320, 320, 3]  # channels-last under nchw
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(bad, schema)

    def test_bundle_manifest_required_and_confined(self):
        import copy
        schema = load_schema("model-descriptor.schema.json")
        base = load_fixture("yolov9s-320.descriptor.json")
        bad = copy.deepcopy(base)
        bad["source"] = {"kind": "onnx-bundle", "sha256": "a" * 64}
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(bad, schema)
        bad["source"]["bundle_manifest"] = {"members": [
            {"path": "../escape.bin", "sha256": "b" * 64, "size_bytes": 1}]}
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(bad, schema)

    def test_artifact_schema_rejects_absolute_paths_and_secrets(self):
        schema = load_schema("artifact.schema.json")
        doc = {"schema_version": 1, "compile_key": "a" * 64,
               "source_sha256": "b" * 64, "source_size_bytes": 123,
               "artifact_sha256": "c" * 64,
               "recipe_id": "bf16-vaiml-v1",
               "recipe_config_sha256": "d" * 64,
               "compiler_payload_sha256": "e" * 64,
               "target_profile": "t",
               "artifact_compatibility_id": "fbs-v1",
               "compile_input": {"shape": [1, 3, 320, 320],
                                 "dtype": "float32"},
               "paths": {"rai": "/abs/model.rai", "descriptor": "d.json"}}
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(doc, schema)
        doc["paths"]["rai"] = "../escape/model.rai"
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(doc, schema)
        doc["paths"] = {"rai": "model.rai", "descriptor": "d.json"}
        jsonschema.validate(doc, schema)  # relative paths pass

    def test_imported_rai_requires_artifact_digest(self):
        schema = load_schema("model-descriptor.schema.json")
        doc = load_fixture("yolov9s-320.descriptor.json")
        doc["source"] = {"kind": "imported-rai", "sha256": "d" * 64}
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(doc, schema)
        doc["source"]["artifact_sha256"] = "e" * 64
        jsonschema.validate(doc, schema)


if __name__ == "__main__":
    unittest.main()
