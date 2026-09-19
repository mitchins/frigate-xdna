"""Unit tests for packaging/verify_payload.py traversal guards.

The verifier is supply-chain code: a manifest entry must never redirect
reads outside the payload root it claims to describe.
"""
import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_VERIFY = os.path.join(_HERE, "..", "..", "packaging", "verify_payload.py")


def _load():
    spec = importlib.util.spec_from_file_location(
        "fxdna_verify_payload", _VERIFY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


vp = _load()


def _write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def _manifest(entries):
    return {"files": entries}


def _run(manifest, payload, xrt):
    with tempfile.TemporaryDirectory() as tmp:
        mp = os.path.join(tmp, "manifest.json")
        with open(mp, "w") as f:
            json.dump(_manifest(manifest), f)
        argv = ["verify_payload.py", "--manifest", mp,
                "--payload", payload, "--xrt", xrt]
        old = sys.argv
        sys.argv = argv
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = vp.main()
        finally:
            sys.argv = old
        return rc, buf.getvalue()


def _entry(payload_dir, rel, data=b"abc"):
    full = os.path.join(payload_dir, rel)
    _write(full, data)
    return {"path": rel, "size_bytes": len(data),
            "sha256": vp.sha256_file(full)}


class TestSafeJoin(unittest.TestCase):
    def test_normal(self):
        self.assertEqual(vp.safe_join("/p", "a/b.bin"), "/p/a/b.bin")

    def test_absolute_rejected(self):
        self.assertIsNone(vp.safe_join("/p", "/etc/passwd"))

    def test_dotdot_escape_rejected(self):
        self.assertIsNone(vp.safe_join("/p", "../../etc/passwd"))
        self.assertIsNone(vp.safe_join("/p", "a/../../../etc"))

    def test_dotdot_inside_ok(self):
        self.assertEqual(vp.safe_join("/p", "a/../b.bin"), "/p/b.bin")


class TestVerifyPayload(unittest.TestCase):
    def test_valid_tree_passes(self):
        with tempfile.TemporaryDirectory() as payload, \
                tempfile.TemporaryDirectory() as xrt:
            ent = _entry(payload, "model.bin", b"model-bytes")
            rc, out = _run([ent], payload, xrt)
            self.assertEqual(rc, 0)
            self.assertIn("errors=0", out)

    def test_traversal_entry_fails(self):
        with tempfile.TemporaryDirectory() as payload, \
                tempfile.TemporaryDirectory() as xrt:
            evil = os.path.join(xrt, "..", "evil-outside")
            with open(os.path.join(payload, "dummy"), "wb") as f:
                f.write(b"x")
            entries = [
                {"path": "xrt/../../evil-outside",
                 "size_bytes": 1, "sha256": "00" * 32},
                {"path": "ok.bin", "size_bytes": 1,
                 "sha256": vp.sha256_file(
                     os.path.join(payload, "dummy"))},
            ]
            _ = evil
            rc, out = _run(entries, payload, xrt)
            self.assertEqual(rc, 1)
            self.assertIn("TRAVERSAL", out)

    def test_unsupplied_prefix_skipped_not_failed(self):
        # Staged Dockerfile verification: calib/legal/flexmlrt entries
        # are SKIPped (rc 0) when their base dir is not supplied; the
        # later stage that supplies them still verifies each file.
        with tempfile.TemporaryDirectory() as tmp:
            payload = os.path.join(tmp, "payload")
            xrt = os.path.join(tmp, "xrt")
            os.makedirs(payload)
            os.makedirs(xrt)
            entries = [_entry(payload, "a.py")]
            entries.append({"path": "calib/000000000009.jpg",
                            "size_bytes": 3,
                            "sha256": "abc"})
            rc, out = _run(entries, payload, xrt)
            self.assertEqual(rc, 0)
            self.assertIn("SKIP calib/000000000009.jpg", out)

    def test_hash_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as payload, \
                tempfile.TemporaryDirectory() as xrt:
            ent = _entry(payload, "model.bin", b"real")
            ent["sha256"] = "ff" * 32
            rc, out = _run([ent], payload, xrt)
            self.assertEqual(rc, 1)
            self.assertIn("HASH", out)

    def test_unexpected_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as payload, \
                tempfile.TemporaryDirectory() as xrt:
            ent = _entry(payload, "model.bin", b"real")
            os.symlink("model.bin", os.path.join(payload, "evil-link"))
            rc, out = _run([ent], payload, xrt)
            self.assertEqual(rc, 1)
            self.assertIn("UNEXPECTED_SYMLINK", out)


if __name__ == "__main__":
    unittest.main()
