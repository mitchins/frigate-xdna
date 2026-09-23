"""Unit tests: SBOM generator identity + validator gate (Task 8.5.2).

The RC1 publication proved an unattestable SBOM must fail before
`docker push`, not at the attestation step. These tests pin the
generator's CycloneDX 1.5 identity (deterministic serialNumber,
licence.name for internal labels) and the validator's rejections.
Hardware-free; operates on the real committed vendor.lock inputs.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
import uuid

TOOLS = os.path.join(os.path.dirname(__file__), "..", "..", "tools")


def gen_sbom(out, image="ghcr.io/mitchins/frigate-xdna",
             image_id="deadbeefcafe"):
    r = subprocess.run(
        [sys.executable, os.path.join(TOOLS, "gen_sbom.py"),
         "--out", out, "--image", image, "--image-id", image_id],
        capture_output=True, text=True, cwd=os.path.join(TOOLS, ".."))
    assert r.returncode == 0, r.stderr
    with open(out, encoding="utf-8") as f:
        return json.load(f)


def validate(path):
    return subprocess.run(
        [sys.executable, os.path.join(TOOLS, "validate_sbom.py"), path],
        capture_output=True, text=True)


class TestGeneratorIdentity(unittest.TestCase):
    def test_document_identity_fields(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "sbom.json")
            doc = gen_sbom(out)
        self.assertEqual(
            doc["$schema"],
            "http://cyclonedx.org/schema/bom-1.5.schema.json")
        self.assertEqual(doc["bomFormat"], "CycloneDX")
        self.assertEqual(doc["specVersion"], "1.5")
        self.assertEqual(doc["version"], 1)
        serial = doc["serialNumber"]
        self.assertTrue(serial.startswith("urn:uuid:"))
        parsed = uuid.UUID(serial[len("urn:uuid:"):])
        self.assertEqual(parsed.variant, uuid.RFC_4122)

    def test_serial_deterministic_per_image_identity(self):
        with tempfile.TemporaryDirectory() as d:
            first = gen_sbom(os.path.join(d, "a.json"),
                             image_id="abc123")
            second = gen_sbom(os.path.join(d, "b.json"),
                              image_id="abc123")
            third = gen_sbom(os.path.join(d, "c.json"),
                             image_id="def456")
        self.assertEqual(first["serialNumber"], second["serialNumber"])
        self.assertNotEqual(first["serialNumber"], third["serialNumber"])


class TestLicenceRepresentation(unittest.TestCase):
    def test_internal_labels_use_name_not_id(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "sbom.json")
            doc = gen_sbom(out)
        seen = set()
        for comp in doc["components"]:
            for choice in comp.get("licenses", []):
                lic = choice["license"]
                self.assertNotIn("id", lic)
                self.assertTrue(lic.get("name"))
                seen.add(lic["name"])
        for label in ("amd-eula", "apache-2.0", "onnx-mit"):
            self.assertIn(label, seen)


class TestScopeMapping(unittest.TestCase):
    def props(self, comp):
        return {p["name"]: p["value"]
                for p in comp.get("properties", [])}

    def test_runtime_is_required_and_dev_test_is_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "sbom.json")
            doc = gen_sbom(out)
        by_name = {c["name"]: c for c in doc["components"]
                   if c["bom-ref"].startswith("pypi:")}
        runtime = by_name["requests"]
        self.assertEqual(runtime["scope"], "required")
        self.assertEqual(
            self.props(runtime)["fxdna:dependency-scope"], "runtime")
        dev = by_name["coverage"]
        self.assertEqual(dev["scope"], "excluded")
        self.assertEqual(
            self.props(dev)["fxdna:dependency-scope"], "dev-test")
        for comp in by_name.values():
            self.assertIn(comp["scope"], ("required", "excluded"))

    def test_vendor_bom_ref_carries_licence_identity(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "sbom.json")
            doc = gen_sbom(out)
        refs = [c["bom-ref"] for c in doc["components"]
                if c["bom-ref"].startswith("vendor:")]
        self.assertTrue(refs)
        for ref in refs:
            _head, lic = ref.split("#", 1)
            self.assertTrue(lic)
        self.assertEqual(len(set(refs)), len(refs))


class TestValidatorGate(unittest.TestCase):
    def write(self, d, name, doc):
        path = os.path.join(d, name)
        with open(path, "w", encoding="utf-8") as f:
            if isinstance(doc, str):
                f.write(doc)
            else:
                json.dump(doc, f)
        return path

    def valid_doc(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "sbom.json")
            return gen_sbom(out)

    def test_accepts_generated_sbom(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "sbom.json")
            gen_sbom(out)
            r = validate(out)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("ACCEPT", r.stdout)

    def test_rejects_missing_serial(self):
        doc = self.valid_doc()
        del doc["serialNumber"]
        with tempfile.TemporaryDirectory() as d:
            r = validate(self.write(d, "s.json", doc))
        self.assertEqual(r.returncode, 1)
        self.assertIn("serialNumber", r.stderr)

    def test_rejects_malformed_serial(self):
        doc = self.valid_doc()
        doc["serialNumber"] = "urn:uuid:not-a-uuid"
        with tempfile.TemporaryDirectory() as d:
            r = validate(self.write(d, "s.json", doc))
        self.assertEqual(r.returncode, 1)

    def test_rejects_wrong_format(self):
        doc = self.valid_doc()
        doc["bomFormat"] = "SPDX"
        with tempfile.TemporaryDirectory() as d:
            r = validate(self.write(d, "s.json", doc))
        self.assertEqual(r.returncode, 1)
        self.assertIn("bomFormat", r.stderr)

    def test_rejects_licence_id(self):
        doc = self.valid_doc()
        vendored = next(c for c in doc["components"] if "licenses" in c)
        vendored["licenses"] = [{"license": {"id": "amd-eula"}}]
        with tempfile.TemporaryDirectory() as d:
            r = validate(self.write(d, "s.json", doc))
        self.assertEqual(r.returncode, 1)
        self.assertIn("license.id", r.stderr)

    def test_rejects_unparsable_json(self):
        with tempfile.TemporaryDirectory() as d:
            r = validate(self.write(d, "s.json", "{nope"))
        self.assertEqual(r.returncode, 1)

    def test_rejects_non_string_serial(self):
        doc = self.valid_doc()
        doc["serialNumber"] = 12345
        with tempfile.TemporaryDirectory() as d:
            r = validate(self.write(d, "s.json", doc))
        self.assertEqual(r.returncode, 1)
        self.assertIn("serialNumber", r.stderr)

    def test_rejects_non_canonical_serial(self):
        doc = self.valid_doc()
        body = doc["serialNumber"][len("urn:uuid:"):]
        doc["serialNumber"] = "urn:uuid:" + body.upper()
        with tempfile.TemporaryDirectory() as d:
            r = validate(self.write(d, "s.json", doc))
        self.assertEqual(r.returncode, 1)
        self.assertIn("canonical", r.stderr)

    def test_rejects_non_object_component(self):
        doc = self.valid_doc()
        doc["components"].append("not-an-object")
        with tempfile.TemporaryDirectory() as d:
            r = validate(self.write(d, "s.json", doc))
        self.assertEqual(r.returncode, 1)

    def test_rejects_duplicate_bom_ref(self):
        doc = self.valid_doc()
        doc["components"].append(dict(doc["components"][0]))
        with tempfile.TemporaryDirectory() as d:
            r = validate(self.write(d, "s.json", doc))
        self.assertEqual(r.returncode, 1)
        self.assertIn("bom-ref", r.stderr)

    def test_rejects_empty_bom_ref(self):
        doc = self.valid_doc()
        doc["components"][0]["bom-ref"] = ""
        with tempfile.TemporaryDirectory() as d:
            r = validate(self.write(d, "s.json", doc))
        self.assertEqual(r.returncode, 1)
        self.assertIn("bom-ref", r.stderr)

    def test_rejects_empty_metadata_bom_ref(self):
        doc = self.valid_doc()
        doc["metadata"]["component"]["bom-ref"] = ""
        with tempfile.TemporaryDirectory() as d:
            r = validate(self.write(d, "s.json", doc))
        self.assertEqual(r.returncode, 1)
        self.assertIn("bom-ref", r.stderr)


if __name__ == "__main__":
    unittest.main()
