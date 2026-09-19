"""Contract tests: pinned sources are byte-identical to the rc2 blobs.

Pure-python git blob hashing (sha1 of b"blob {len}\\0" + content) so the
test needs no git binary. Fails on any drift from upstream.lock.json.
"""
import hashlib
import json
import os
import unittest

UPSTREAM = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "upstream")


def git_blob_sha1(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.sha1(b"blob %d\0" % len(content) + content).hexdigest()


class TestPinned(unittest.TestCase):
    def test_verbatim_files_match_lock(self):
        lock = json.load(open(os.path.join(UPSTREAM, "upstream.lock.json")))
        for src, meta in lock["files"].items():
            if meta.get("local") is None:
                continue
            if meta["role"].startswith("pinned-verbatim"):
                with self.subTest(src=src):
                    local = os.path.join(UPSTREAM, meta["local"])
                    self.assertTrue(os.path.isfile(local), local)
                    self.assertEqual(git_blob_sha1(local), meta["git_blob"])

    def test_reference_files_present(self):
        lock = json.load(open(os.path.join(UPSTREAM, "upstream.lock.json")))
        for src, meta in lock["files"].items():
            if meta["role"].startswith("reference-only"):
                with self.subTest(src=src):
                    self.assertTrue(
                        os.path.isfile(os.path.join(UPSTREAM, meta["local"])))


if __name__ == "__main__":
    unittest.main()
