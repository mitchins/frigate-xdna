"""Unit tests: reference provenance is self-consistent.

Asserts the imported reference sources match the hashes recorded in
native/reference/README.md. Fails loudly on drift rather than testing
research-directory contents.
"""
import hashlib
import os
import re
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class TestReferenceProvenance(unittest.TestCase):
    def test_reference_hashes_match_readme(self):
        readme = open(os.path.join(REPO, "native", "reference",
                                   "README.md")).read()
        for name in ("worker-phase7.cc", "yolo_flexml-phase5.cc"):
            path = os.path.join(REPO, "native", "reference", name)
            self.assertTrue(os.path.isfile(path), name)
            recorded = re.search(re.escape(name) + r".*`([0-9a-f]{64})`",
                                 readme)
            self.assertIsNotNone(recorded, f"no recorded hash for {name}")
            self.assertEqual(sha256_file(path), recorded.group(1), name)


if __name__ == "__main__":
    unittest.main()
