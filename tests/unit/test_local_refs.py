"""Unit tests: first-class local model refs (v0.1.2 C1).

`local://ID` resolves against FXDNA_MODEL_DIR, is confined to it, and
behaves as a mutable alias: changed bytes are a new revision
(SOURCE_CHANGED, previous artifact kept), never a silent mutation.
"""
import hashlib
import os
import tempfile
import time
import unittest

from frigate_xdna import deploy_checks
from frigate_xdna.cache import gc as _gc
from frigate_xdna.config import load_config
from frigate_xdna.errors import FxdnaError
from frigate_xdna.models.refs import (
    LOCAL_ID_RE,
    parse_ref,
    resolve_local_path,
    wire_alias,
)
from frigate_xdna.supervisor import Supervisor
from tests.integration.onnx_builders import make_raw_yolo


def put_model(d, name="yolov9-t-320", classes=4, seed=3):
    path = os.path.join(d, name + ".onnx")
    make_raw_yolo(path, res=320, classes=classes, seed=seed)
    with open(path, "rb") as f:
        return path, hashlib.sha256(f.read()).hexdigest()


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


class TestLocalRefParser(unittest.TestCase):
    def test_valid_ids_resolve(self):
        for mid in ("yolov9-t-320", "A", "a.b_c-d9", "X" * 128):
            with self.subTest(mid=mid):
                parsed = parse_ref(f"local://{mid}", "/models")
                self.assertEqual(parsed["kind"], "local")
                self.assertEqual(parsed["id"], mid)
                self.assertEqual(parsed["ref"], f"local://{mid}")
                self.assertEqual(parsed["path"],
                                 f"/models/{mid}.onnx")
                self.assertEqual(wire_alias(parsed, None), mid)
                self.assertTrue(LOCAL_ID_RE.fullmatch(mid))

    def test_invalid_ids_rejected(self):
        bad = ["local://", "local://a/b", "local://..", "local://.",
               "local://-x", "local://.x", "local://a%b",
               "local://a?b", "local://a b", "local://%2e%2e",
               "local://host/id", "local://" + "Y" * 129,
               "local://a://b"]
        for ref in bad:
            with self.subTest(ref=ref):
                with self.assertRaises(FxdnaError) as ctx:
                    parse_ref(ref, "/models")
                self.assertEqual(ctx.exception.exit_code, 2)  # INVALID_ARGS

    def test_model_dir_required(self):
        for missing in (None, "", "relative/dir"):
            with self.subTest(missing=missing):
                with self.assertRaises(FxdnaError):
                    parse_ref("local://m", missing)

    def test_other_forms_unchanged(self):
        plus = parse_ref("plus://abc", "/models")
        self.assertEqual(plus["kind"], "plus")
        path = parse_ref("/models/foo.onnx", "/models")
        self.assertEqual(path["kind"], "onnx")
        with self.assertRaises(FxdnaError):
            parse_ref("http://x/y.onnx", "/models")
        with self.assertRaises(FxdnaError):
            parse_ref("ftp://x", "/models")


class TestLocalPathContainment(unittest.TestCase):
    def test_symlinked_model_dir_still_resolves(self):
        with tempfile.TemporaryDirectory() as d:
            real = os.path.join(d, "real")
            os.mkdir(real)
            link = os.path.join(d, "link")
            os.symlink(real, link)
            self.assertEqual(resolve_local_path(link, "m"),
                             os.path.join(link, "m.onnx"))

    def test_symlink_escape_refused(self):
        with tempfile.TemporaryDirectory() as d:
            models = os.path.join(d, "models")
            os.mkdir(models)
            outside = os.path.join(d, "outside.onnx")
            with open(outside, "wb") as f:
                f.write(b"not a model")
            os.symlink(outside, os.path.join(models, "evil.onnx"))
            with self.assertRaises(FxdnaError) as ctx:
                resolve_local_path(models, "evil")
            self.assertIn("escapes", ctx.exception.message)


class TestModelDirConfig(unittest.TestCase):
    def test_default_and_override(self):
        self.assertEqual(load_config({}).model_dir, "/models")
        c = load_config({"FXDNA_MODEL_DIR": "/srv/models"})
        self.assertEqual(c.model_dir, "/srv/models")
        self.assertEqual(c.redacted()["model_dir"], "/srv/models")

    def test_relative_or_empty_rejected(self):
        for bad in ("relative", "", "  "):
            with self.subTest(bad=bad):
                with self.assertRaises(FxdnaError):
                    load_config({"FXDNA_MODEL_DIR": bad})

    def test_local_only_needs_no_plus_key(self):
        with tempfile.TemporaryDirectory() as d:
            c = load_config({"FXDNA_MODELS": "local://m",
                             "FXDNA_MODEL_DIR": d})
            got = deploy_checks.check_plus_credential(c)
            self.assertTrue(got["ok"])
            c2 = load_config({"FXDNA_MODELS": "plus://abc",
                              "FXDNA_MODEL_DIR": d})
            self.assertFalse(deploy_checks.check_plus_credential(c2)["ok"])


class TestLocalAliasPrepare(unittest.TestCase):
    def _sup(self, cfg, **kw):
        return Supervisor(cfg, **kw)

    def test_prepare_cache_hit_and_second_alias(self):
        with tempfile.TemporaryDirectory() as data, \
                tempfile.TemporaryDirectory() as models:
            _, digest = put_model(models, "yolov9-t-320")
            # Same bytes under a second alias: byte-identical copy.
            src = os.path.join(models, "yolov9-t-320.onnx")
            dst = os.path.join(models, "same-bytes.onnx")
            with open(src, "rb") as f:
                raw = f.read()
            with open(dst, "wb") as f:
                f.write(raw)
            cfg = load_config({"FXDNA_DATA_DIR": data,
                               "FXDNA_MODEL_DIR": models,
                               "FXDNA_MODELS": "local://yolov9-t-320"})
            sup = self._sup(cfg)
            try:
                out = sup.prepare("local://yolov9-t-320")
                self.assertFalse(out["cache_hit"])
                self.assertEqual(
                    pump_until(sup, "local://yolov9-t-320",
                               ("PREPARED",)), "PREPARED")
                ref = sup.registry.get_ref("local://yolov9-t-320")
                self.assertEqual(ref["kind"], "local")
                self.assertEqual(ref["source_sha256"], digest)
                # Same bytes re-prepared: cache hit, no new job.
                out2 = sup.prepare("local://yolov9-t-320")
                self.assertTrue(out2["cache_hit"])
                self.assertEqual(out2["compile_key"],
                                 out["compile_key"])
                # Same bytes under a second alias converge on one key.
                out3 = sup.prepare("local://same-bytes")
                self.assertTrue(out3["cache_hit"])
                self.assertEqual(out3["compile_key"], out["compile_key"])
            finally:
                sup.stop()

    def test_changed_bytes_are_new_revision(self):
        with tempfile.TemporaryDirectory() as data, \
                tempfile.TemporaryDirectory() as models:
            _, digest_v1 = put_model(models, "driveway", seed=3)
            cfg = load_config({"FXDNA_DATA_DIR": data,
                               "FXDNA_MODEL_DIR": models,
                               "FXDNA_MODELS": "local://driveway"})
            sup = self._sup(cfg)
            try:
                out = sup.prepare("local://driveway")
                self.assertEqual(
                    pump_until(sup, "local://driveway", ("PREPARED",)),
                    "PREPARED")
                key_v1 = out["compile_key"]
                # Operator replaces the file: new bytes, same alias.
                _, digest_v2 = put_model(models, "driveway", seed=4)
                self.assertNotEqual(digest_v2, digest_v1)
                with self.assertRaises(FxdnaError) as ctx:
                    sup.prepare("local://driveway")
                self.assertEqual(ctx.exception.error_code,
                                 "SOURCE_CHANGED")
                self.assertEqual(
                    sup.registry.get_ref(
                        "local://driveway")["state"], "SOURCE_CHANGED")
                # Previous artifact kept and still usable.
                self.assertIsNotNone(sup.registry.get_artifact(key_v1))
                self.assertEqual(sup._usable_cached_key("local://driveway"),
                                 key_v1)
                # Explicit refresh accepts the new revision...
                out2 = sup.prepare("local://driveway", refresh=True)
                self.assertNotEqual(out2["compile_key"], key_v1)
                self.assertEqual(
                    pump_until(sup, "local://driveway", ("PREPARED",)),
                    "PREPARED")
                self.assertEqual(
                    sup.registry.get_ref(
                        "local://driveway")["source_sha256"], digest_v2)
                # ...while the previous artifact row survives.
                self.assertIsNotNone(sup.registry.get_artifact(key_v1))
            finally:
                sup.stop()

    def test_missing_file_and_explicit_path(self):
        with tempfile.TemporaryDirectory() as data, \
                tempfile.TemporaryDirectory() as models:
            path, _ = put_model(models, "kept")
            cfg = load_config({"FXDNA_DATA_DIR": data,
                               "FXDNA_MODEL_DIR": models})
            sup = self._sup(cfg)
            try:
                with self.assertRaises(FxdnaError) as ctx:
                    sup.prepare("local://absent")
                self.assertIn("not found", ctx.exception.message)
                # The 0.1.1 explicit-path interface is unchanged.
                out = sup.prepare(path)
                self.assertEqual(out["ref"], path)
                self.assertEqual(
                    pump_until(sup, path, ("PREPARED",)), "PREPARED")
                self.assertEqual(
                    sup.registry.get_ref(path)["kind"], "onnx")
            finally:
                sup.stop()

    def test_configured_pin_protects_local_source(self):
        with tempfile.TemporaryDirectory() as data, \
                tempfile.TemporaryDirectory() as models:
            _, digest = put_model(models, "pinned")
            cfg = load_config({"FXDNA_DATA_DIR": data,
                               "FXDNA_MODEL_DIR": models,
                               "FXDNA_MODELS": "local://pinned"})
            sup = self._sup(cfg)
            try:
                sup.prepare("local://pinned")
                self.assertEqual(
                    pump_until(sup, "local://pinned", ("PREPARED",)),
                    "PREPARED")
                plan = _gc.plan_prune(data, sup.registry)
                doomed = {c["digest"] for c in plan["candidates"]}
                self.assertNotIn(digest, doomed)
            finally:
                sup.stop()


if __name__ == "__main__":
    unittest.main()
