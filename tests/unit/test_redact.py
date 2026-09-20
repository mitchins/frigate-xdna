"""Unit tests: external-observability privacy (Task 05 addendum).

Sentinel identifiers/secrets are planted in the registry, journal and
config surface; the exported diagnose bundle must not contain them
recursively, while stable HMAC pseudonyms must.
"""
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from frigate_xdna import cli
from frigate_xdna.cache.registry import Registry
from frigate_xdna.errors import SUCCESS
from frigate_xdna.observability import redact

MODEL_SENTINEL = "private-model-SENTINEL-123"
TOKEN_SENTINEL = "secret-SENTINEL-456"
URL_SENTINEL = ("https://dl.example.invalid/weights.bin?X-Amz-Signature="
                "SENTINEL-sig&Expires=999")
SHA_SENTINEL = "ab" * 32


def run_cli(argv, data_dir):
    out, err = io.StringIO(), io.StringIO()
    old = dict(os.environ)
    try:
        os.environ["FXDNA_DATA_DIR"] = data_dir
        with redirect_stdout(out), redirect_stderr(err):
            try:
                rc = cli.main(argv)
            except SystemExit as e:
                rc = e.code
    finally:
        os.environ.clear()
        os.environ.update(old)
    return rc, out.getvalue(), err.getvalue()


def plant_state(data_dir):
    """Registry + journal + model bytes carrying sentinel values."""
    db = os.path.join(data_dir, "registry.sqlite3")
    reg = Registry(db)
    try:
        ref = f"plus://{MODEL_SENTINEL}"
        reg.upsert_ref(ref, "plus", MODEL_SENTINEL, state="VERIFIED")
        reg.set_ref_source(ref, SHA_SENTINEL, "cd" * 32, "VERIFIED")
    finally:
        reg.close()
    ops = os.path.join(data_dir, "operations")
    os.makedirs(ops, exist_ok=True)
    with open(os.path.join(ops, "current.json"), "w") as f:
        json.dump({"operation": "verify", "ref": f"plus://{MODEL_SENTINEL}",
                   "note": f"fetched {URL_SENTINEL}"}, f)
    with open(os.path.join(ops, "history.jsonl"), "w") as f:
        f.write(json.dumps(
            {"event": "download",
             "auth": f"Authorization: Bearer {TOKEN_SENTINEL}"}) + "\n")
        f.write(json.dumps({"event": "ready",
                            "ref": f"plus://{MODEL_SENTINEL}"}) + "\n")
    # Private model bytes: must never be copied into the bundle.
    art = os.path.join(data_dir, "artifacts")
    os.makedirs(art, exist_ok=True)
    with open(os.path.join(art, "weights-SENTINEL-123.rai"), "wb") as f:
        f.write(b"RAI-SENTINEL-BYTES" * 64)


def bundle_text(out_dir):
    texts = []
    for root, _, files in os.walk(out_dir):
        for name in files:
            with open(os.path.join(root, name), "rb") as f:
                texts.append(f.read().decode("utf-8", "replace"))
    return "\n".join(texts)


class TestRedact(unittest.TestCase):
    def test_alias_stable_and_opaque(self):
        key = b"0" * 32
        a1 = redact.plus_alias(MODEL_SENTINEL, key)
        a2 = redact.plus_alias(MODEL_SENTINEL, key)
        self.assertEqual(a1, a2)
        self.assertTrue(a1.startswith("plus:"))
        self.assertEqual(len(a1), len("plus:") + 12)
        self.assertNotIn(MODEL_SENTINEL, a1)
        other = redact.plus_alias("different-id", key)
        self.assertNotEqual(a1, other)

    def test_key_created_once_with_strict_mode(self):
        with tempfile.TemporaryDirectory() as d:
            k1 = redact.load_or_create_key(d)
            st = os.stat(os.path.join(d, redact.KEY_NAME))
            self.assertEqual(oct(st.st_mode & 0o777), "0o600")
            k2 = redact.load_or_create_key(d)
            self.assertEqual(k1, k2)

    def test_diagnose_bundle_contains_no_sentinels(self):
        with tempfile.TemporaryDirectory() as d:
            plant_state(d)
            out = os.path.join(d, "bundle")
            rc, bout, _ = run_cli(["diagnose", "--out", out], d)
            self.assertEqual(rc, SUCCESS)
            doc = json.loads(bout)
            self.assertEqual(doc["identifiers"], "pseudonymized")
            text = bundle_text(out)
            for sentinel in (MODEL_SENTINEL, TOKEN_SENTINEL,
                             "SENTINEL-sig", SHA_SENTINEL,
                             "RAI-SENTINEL-BYTES",
                             f"Bearer {TOKEN_SENTINEL}"):
                self.assertNotIn(sentinel, text)
            # The bearer scheme word may remain; the secret must not.
            self.assertIn("Bearer <redacted>", text)
            # No key material, DB copy, or model bytes in the bundle.
            names = []
            for root, _, files in os.walk(out):
                names.extend(files)
            self.assertNotIn(redact.KEY_NAME, names)
            self.assertNotIn("registry.sqlite3", names)
            self.assertFalse(any(n.endswith(".rai") for n in names))
            # Pseudonym present and stable across a second export.
            self.assertIn("plus:", text)
            out2 = os.path.join(d, "bundle2")
            run_cli(["diagnose", "--out", out2], d)
            key = redact.load_or_create_key(d)
            want = redact.plus_alias(MODEL_SENTINEL, key)
            self.assertIn(want, text)
            self.assertIn(want, bundle_text(out2))

    def test_status_dict_active_passes_through(self):
        # activate() publishes a machine struct, not a ref string;
        # status/diagnose must not crash on it in either mode.
        from frigate_xdna.cli import _StatusView
        view = _StatusView(False, b"0" * 32)
        doc = {"compile_key": "ab" * 32, "worker_generation": 1,
               "serving_digest": "cd" * 32}
        self.assertEqual(view.active(doc), doc)
        raw = _StatusView(True, None).active(doc)
        self.assertEqual(raw, doc)

    def test_show_identifiers_reveals_raw_on_owner_machine(self):
        with tempfile.TemporaryDirectory() as d:
            plant_state(d)
            out = os.path.join(d, "bundle-raw")
            rc, bout, _ = run_cli(
                ["diagnose", "--out", out, "--show-identifiers"], d)
            self.assertEqual(rc, SUCCESS)
            self.assertEqual(json.loads(bout)["identifiers"], "raw")
            self.assertIn(MODEL_SENTINEL, bundle_text(out))

    def test_status_redacted_by_default(self):
        with tempfile.TemporaryDirectory() as d:
            plant_state(d)
            rc, out, _ = run_cli(["status", "--json"], d)
            self.assertEqual(rc, SUCCESS)
            doc = json.loads(out)
            self.assertNotIn(MODEL_SENTINEL, out)
            self.assertTrue(doc["models"][0]["ref"].startswith("plus:"))
            self.assertIn("…", doc["models"][0]["source_sha256"])
            rc, out, _ = run_cli(
                ["status", "--json", "--show-identifiers"], d)
            self.assertEqual(rc, SUCCESS)
            self.assertIn(MODEL_SENTINEL, out)


if __name__ == "__main__":
    unittest.main()
