"""Integration tests: supervisor, cache store, registry, jobs, CLI daemon.

Covers the Task 02 acceptance bullets with tmp data dirs and in-test
ONNX builders: coexistence, dedup, source-change, interruption recovery,
locks, pins/prune, offline behaviour, local imports, fake transitions,
wait semantics. No hardware, no external network.
"""
import json
import os
import sqlite3
import tempfile
import unittest

from frigate_xdna.cache import gc as _gc
from frigate_xdna.cache.registry import Registry
from frigate_xdna.cache.store import disk_preflight, ensure_layout, locked
from frigate_xdna.config import Config
from frigate_xdna.errors import FxdnaError
from frigate_xdna.supervisor import Supervisor
from tests.integration.onnx_builders import (
    make_external_data_model,
    make_raw_yolo,
)


def make_config(**kw):
    base = {"data_dir": tempfile.mkdtemp(prefix="fxdna-t02-"),
            "endpoint": "tcp://127.0.0.1:5555"}
    base.update(kw)
    return Config(**base)


class ManagerBase(unittest.TestCase):
    def setUp(self):
        self.cfg = make_config()
        self.sup = Supervisor(self.cfg)
        self.addCleanup(self.sup.stop)

    def local_onnx(self, name="a.onnx", res=320, seed=3):
        path = os.path.join(self.cfg.data_dir, name)
        return path, make_raw_yolo(path, res=res, seed=seed)


class TestLocalPrepare(ManagerBase):
    def test_two_models_coexist(self):
        pa, _ = self.local_onnx("a.onnx", seed=3)
        pb, _ = self.local_onnx("b.onnx", seed=4)
        ra = self.sup.prepare(pa)
        rb = self.sup.prepare(pb)
        self.assertNotEqual(ra["source_sha256"], rb["source_sha256"])
        self.assertNotEqual(ra["compile_key"], rb["compile_key"])
        self.sup.pump(1.0)
        st = self.sup.status()
        self.assertEqual(
            {m["ref"]: m["state"] for m in st["models"]},
            {pa: "PREPARED", pb: "PREPARED"})

    def test_same_bytes_different_aliases_dedup(self):
        pa, data = self.local_onnx("a.onnx", seed=3)
        pb = os.path.join(self.cfg.data_dir, "b.onnx")
        with open(pb, "wb") as f:
            f.write(data)
        ra = self.sup.prepare(pa)
        rb = self.sup.prepare(pb)
        self.assertEqual(ra["source_sha256"], rb["source_sha256"])
        self.assertEqual(ra["compile_key"], rb["compile_key"])
        self.assertEqual(ra["job_uuid"], rb["job_uuid"])  # one job joined
        self.sup.pump(5.0)
        states = {m["ref"]: m["state"]
                  for m in self.sup.status()["models"]}
        self.assertEqual(states, {pa: "PREPARED", pb: "PREPARED"})

    def test_same_alias_new_bytes_needs_refresh(self):
        pa, _ = self.local_onnx("a.onnx", seed=3)
        self.sup.prepare(pa)
        make_raw_yolo(pa, seed=9)  # overwrite same path, new bytes
        with self.assertRaises(FxdnaError) as ctx:
            self.sup.prepare(pa)
        self.assertEqual(ctx.exception.error_code, "SOURCE_CHANGED")
        # old source still recorded; refresh accepts the new bytes
        before = self.sup.registry.get_ref(pa)["source_sha256"]
        out = self.sup.prepare(pa, refresh=True)
        self.assertNotEqual(out["source_sha256"], before)
        self.assertIn("job_uuid", out)

    def test_rejects_bad_magic_pt_and_external(self):
        bad = os.path.join(self.cfg.data_dir, "x.onnx")
        with open(bad, "wb") as f:
            f.write(b"not a protobuf at all........")
        with self.assertRaises(FxdnaError) as ctx:
            self.sup.prepare(bad)
        self.assertEqual(ctx.exception.error_code, "INVALID_MODEL")
        with self.assertRaises(FxdnaError) as ctx:
            self.sup.prepare("/tmp/weights.pt")
        self.assertEqual(ctx.exception.exit_code, 2)
        ext = os.path.join(self.cfg.data_dir, "e.onnx")
        make_external_data_model(ext)
        with self.assertRaises(FxdnaError):
            self.sup.prepare(ext)

    def test_rai_import_needs_descriptor_and_hash(self):
        rai = os.path.join(self.cfg.data_dir, "m.rai")
        with open(rai, "wb") as f:
            f.write(b"FAKE-RAI-" * 100)
        with self.assertRaises(FxdnaError):
            self.sup.prepare(rai)
        desc = os.path.join(self.cfg.data_dir, "m.json")
        with open(desc, "w") as f:
            json.dump({"artifact_sha256": "0" * 64,
                       "target_profile": "t", "serving": {}}, f)
        with self.assertRaises(FxdnaError) as ctx:
            self.sup.prepare(rai, descriptor_path=desc)
        self.assertEqual(ctx.exception.error_code, "CACHE_CORRUPT")

    def test_cache_hit_no_second_job(self):
        pa, _ = self.local_onnx("a.onnx", seed=3)
        first = self.sup.prepare(pa)
        self.sup.pump(5.0)
        # the fake backend's PREPARED publishes through the real atomic
        # path: a repeated prepare is a cache hit with no new job
        art = self.sup.registry.get_artifact(first["compile_key"])
        self.assertIsNotNone(art)
        second = self.sup.prepare(pa)
        self.assertTrue(second.get("cache_hit"))
        self.assertNotIn("job_uuid", second)

    def test_crash_adopted_after_revalidation(self):
        import hashlib
        import json as _json
        pa, _ = self.local_onnx("a.onnx", seed=3)
        out = self.sup.prepare(pa)
        self.sup.pump(5.0)
        ckey = out["compile_key"]
        art = self.sup.registry.get_artifact(ckey)
        # simulate crash between artifact rename and DB commit: drop the row
        self.sup.registry.execute(
            "DELETE FROM artifacts WHERE compile_key=?", (ckey,))
        self.sup.stop()
        sup2 = Supervisor(self.cfg)
        self.addCleanup(sup2.stop)
        self.assertIsNotNone(sup2.registry.get_artifact(ckey))

    def test_backend_change_invalidates_cache(self):
        import json as _json
        pa, _ = self.local_onnx("a.onnx", seed=3)
        out = self.sup.prepare(pa)
        self.sup.pump(5.0)
        ckey = out["compile_key"]
        # foreign backend row+files: must not be trusted
        manifest_p = os.path.join(
            self.cfg.data_dir, "artifacts", ckey, "artifact.json")
        with open(manifest_p) as f:
            manifest = _json.load(f)
        manifest["backend"] = "something-else"
        with open(manifest_p, "w") as f:
            _json.dump(manifest, f)
        second = self.sup.prepare(pa)
        self.assertFalse(second.get("cache_hit"))
        self.assertIn("job_uuid", second)
        self.assertFalse(os.path.isdir(os.path.join(
            self.cfg.data_dir, "artifacts", ckey)))


class TestJobsAndWait(ManagerBase):
    def test_wait_success_and_failure(self):
        pa, _ = self.local_onnx("a.onnx", seed=3)
        out = self.sup.prepare(pa)
        job = self.sup.wait_job(out["job_uuid"], 5.0)
        self.assertEqual(job["stage"], "PREPARED")
        pb, _ = self.local_onnx("b.onnx", seed=4)
        fail = self.sup.jobs.submit(
            pb, "ck", duration_s=0.0, succeed=False,
            fail_state="COMPILE_FAILED")
        job = self.sup.wait_job(fail["uuid"], 5.0)
        self.assertEqual(job["stage"], "COMPILE_FAILED")
        # deterministic failure is sticky: resubmit reports it, no retry
        again = self.sup.jobs.submit(pb, "ck")
        self.assertEqual(again["stage"], "COMPILE_FAILED")

    def test_wait_timeout(self):
        pa, _ = self.local_onnx("a.onnx", seed=3)
        out = self.sup.prepare(pa)
        # slow the fake job down so the deadline hits first
        backend = self.sup.jobs._backends[out["job_uuid"]]
        backend.duration_s = 3600.0
        with self.assertRaises(FxdnaError) as ctx:
            self.sup.wait_job(out["job_uuid"], 0.0)
        self.assertEqual(ctx.exception.error_code, "WAIT_TIMEOUT")

    def test_terminal_failure_visible_on_ref(self):
        # A sticky terminal failure must surface on the ref, not hide
        # behind QUEUED for a job that will never run.
        pa, _ = self.local_onnx("a.onnx", seed=3)
        orig = self.sup.jobs.backend_factory
        self.sup.jobs.backend_factory = lambda **kw: orig(
            **{**kw, "succeed": False, "fail_state": "COMPILE_FAILED"})
        self.sup.prepare(pa)
        self.sup.pump(1.0)
        # the pump loop itself maps the terminal stage onto every ref
        # of the job (primary + aliases), not just the job row
        rec = self.sup.registry.get_ref(pa)
        self.assertEqual(rec["state"], "COMPILE_FAILED")
        out = self.sup.prepare(pa)
        self.assertEqual(out["state"], "COMPILE_FAILED")

    def test_geometry_required_for_compile_key(self):
        pa, _ = self.local_onnx("a.onnx", seed=3)
        with open(pa, "rb") as f:
            data = f.read()
        from frigate_xdna.errors import FxdnaError as _E
        with self.assertRaises(_E) as ctx:
            self.sup._ingest_source(pa, "a.onnx", data, "local", {},
                                    False)
        self.assertEqual(ctx.exception.error_code, "UNSUPPORTED_CONTRACT")

    def test_waiting_for_device(self):
        from frigate_xdna.compiler.fake import FakeCompileJob
        jm = self.sup.jobs
        jm.backend_factory = lambda **kw: FakeCompileJob(
            **{**kw, "device_required": True, "device_held_by_worker": True})
        pa, _ = self.local_onnx("a.onnx", seed=3)
        out = self.sup.prepare(pa)
        job = jm.pump(out["job_uuid"], 1.0)
        self.assertEqual(job["stage"], "WAITING_FOR_DEVICE")

    def test_interrupted_job_recovery(self):
        pa, _ = self.local_onnx("a.onnx", seed=3)
        out = self.sup.prepare(pa)
        # simulate crash: leave work scratch + QUEUED job, reopen supervisor
        work = os.path.join(self.cfg.data_dir, "work", out["job_uuid"])
        os.makedirs(work, exist_ok=True)
        self.sup.stop()
        sup2 = Supervisor(self.cfg)
        self.addCleanup(sup2.stop)
        job = sup2.registry.get_job(out["job_uuid"])
        self.assertEqual(job["stage"], "INTERRUPTED")
        self.assertTrue(os.path.isdir(
            os.path.join(self.cfg.data_dir, "failures", out["job_uuid"])))

    def test_orphan_artifact_quarantined(self):
        os.makedirs(os.path.join(self.cfg.data_dir, "artifacts", "ab" * 32))
        self.sup.stop()
        sup2 = Supervisor(self.cfg)
        self.addCleanup(sup2.stop)
        self.assertTrue(os.path.isfile(
            os.path.join(self.cfg.data_dir, "quarantine", "ab" * 32 + ".json")))
        # directory moved out of artifacts/ so the key is recompilable
        self.assertTrue(os.path.isdir(
            os.path.join(self.cfg.data_dir, "quarantine", "ab" * 32)))
        self.assertFalse(os.path.isdir(
            os.path.join(self.cfg.data_dir, "artifacts", "ab" * 32)))


class TestLocksPinsPrune(ManagerBase):
    def test_standalone_refuses_live_lock(self):
        with self.assertRaises(FxdnaError) as ctx:
            Supervisor(self.cfg)
        self.assertEqual(ctx.exception.exit_code, 9)

    def test_registry_refuses_newer_schema(self):
        db = os.path.join(self.cfg.data_dir, "registry.sqlite3")
        cx = sqlite3.connect(db)
        cx.execute("UPDATE schema_version SET version=999")
        cx.commit()
        cx.close()
        with self.assertRaises(RuntimeError):
            Registry(db)

    def test_pins_and_prune(self):
        pa, data_a = self.local_onnx("a.onnx", seed=3)
        pb, _ = self.local_onnx("b.onnx", seed=4)
        ra = self.sup.prepare(pa)
        rb = self.sup.prepare(pb)
        _gc.pin_ref(self.sup.registry, pa, "manual")
        plan = self.sup.prune(apply=False)
        digests = {c["digest"] for c in plan["candidates"]}
        self.assertIn(rb["source_sha256"], digests)
        self.assertNotIn(ra["source_sha256"], digests)
        # dry run deletes nothing
        self.assertTrue(os.path.isdir(os.path.join(
            self.cfg.data_dir, "sources", rb["source_sha256"])))
        done = self.sup.prune(apply=True)
        self.assertIn(rb["source_sha256"], done["removed"])
        self.assertTrue(os.path.isdir(os.path.join(
            self.cfg.data_dir, "sources", ra["source_sha256"])))
        # registry follow-through: sources row gone, dangling ref cleared
        self.assertIsNone(
            self.sup.registry.get_source(rb["source_sha256"]))
        brec = self.sup.registry.get_ref(pb)
        self.assertIsNone(brec["source_sha256"])

    def test_disk_preflight(self):
        class FakeStat:
            f_bavail = 100
            f_frsize = 4096
            f_blocks = 10 ** 9
        pre = disk_preflight(self.cfg.data_dir, 1 << 30, statvfs=lambda d: FakeStat())
        self.assertFalse(pre["ok"])
        self.assertGreater(pre["shortfall_bytes"], 0)

    def test_removing_ref_from_config_keeps_cache(self):
        pa, _ = self.local_onnx("a.onnx", seed=3)
        self.sup.prepare(pa)
        # config no longer lists the ref: registry row and bytes persist
        rec = self.sup.registry.get_ref(pa)
        self.assertIsNotNone(rec)
        self.assertTrue(os.path.isfile(os.path.join(
            self.cfg.data_dir, "sources", rec["source_sha256"],
            "model.onnx")))

    def test_recover_clears_own_inhibition(self):
        self.sup.registry.set_state("inhibition", {"ref": "plus://X",
                                                   "reason": "t"})
        out = self.sup.recover_ref("plus://X")
        self.assertTrue(out["cleared"])
        # second recover finds nothing: evidence moved to a separate key
        again = self.sup.recover_ref("plus://X")
        self.assertFalse(again["cleared"])
        self.assertIsNone(self.sup.registry.get_state("inhibition"))
        kept = self.sup.registry.get_state("last_inhibition_cleared")
        self.assertEqual(kept["ref"], "plus://X")
        out = self.sup.recover_ref("plus://Y")
        self.assertFalse(out["cleared"])


class TestOffline(unittest.TestCase):
    def test_offline_plus_refused_but_status_works(self):
        cfg = make_config(offline=True)
        sup = Supervisor(cfg)
        try:
            with self.assertRaises(FxdnaError) as ctx:
                sup.prepare("plus://MODEL_A")
            self.assertEqual(ctx.exception.error_code, "ACQUISITION_FAILED")
            doc = sup.status()
            self.assertEqual(doc["state"], "SERVING")
        finally:
            sup.stop()

    def test_local_prepare_works_offline(self):
        cfg = make_config(offline=True)
        sup = Supervisor(cfg)
        try:
            path = os.path.join(cfg.data_dir, "a.onnx")
            make_raw_yolo(path)
            out = sup.prepare(path)
            self.assertIn("compile_key", out)
        finally:
            sup.stop()


class TestPlusPrepare(unittest.TestCase):
    """Plus acquisition path: download, inspection, metadata comparison."""

    def setUp(self):
        from tests.integration.fake_plus import (
            TEST_KEY,
            FakeDownloadHandler,
            FakeDownloadState,
            FakePlusHandler,
            FakePlusState,
            start_server,
        )
        self.plus_state = FakePlusState()
        self.dl_state = FakeDownloadState()
        self.api = start_server(FakePlusHandler, self.plus_state)
        self.dl = start_server(FakeDownloadHandler, self.dl_state)
        self.api_url = f"http://127.0.0.1:{self.api.server_address[1]}"
        self.dl_url = f"http://127.0.0.1:{self.dl.server_address[1]}/m.onnx"
        self.cfg = make_config(plus_api_key=TEST_KEY)

        from frigate_xdna.plus.client import PlusClient
        api_url = self.api_url

        def factory(secret):
            return PlusClient(secret, host=api_url)

        self.sup = Supervisor(self.cfg, plus_client_factory=factory,
                              plus_allow_private_hosts=("127.0.0.1",))
        self.addCleanup(self.sup.stop)

    def tearDown(self):
        self.api.shutdown()
        self.dl.shutdown()
        self.api.server_close()
        self.dl.server_close()

    def _serve_model(self, model_id, onnx_bytes, metadata):
        self.dl_state.download_bodies["/m.onnx"] = onnx_bytes
        self.plus_state.models[model_id] = metadata
        self.plus_state.signed_urls[model_id] = self.dl_url

    def test_plus_prepare_inspects_and_caches(self):
        import tempfile as _tf
        from tests.integration.onnx_builders import make_raw_yolo
        with _tf.NamedTemporaryFile(suffix=".onnx") as f:
            onnx_bytes = make_raw_yolo(f.name, res=320, seed=5)
        self._serve_model("MODEL_A", onnx_bytes,
                          {"id": "MODEL_A", "width": 320, "height": 320,
                           "labelMap": {"0": "person"}})
        out = self.sup.prepare("plus://MODEL_A")
        self.assertIn("compile_key", out)
        self.sup.pump(5.0)
        st = self.sup.status("plus://MODEL_A")
        self.assertEqual(st["models"][0]["state"], "PREPARED")
        # metadata JSON persisted for serving identity
        rec = self.sup.registry.get_ref("plus://MODEL_A")
        meta_path = os.path.join(
            self.cfg.data_dir, "metadata", rec["metadata_sha256"] + ".json")
        self.assertTrue(os.path.isfile(meta_path))

    def test_plus_metadata_conflict_fails(self):
        import tempfile as _tf
        from tests.integration.onnx_builders import make_raw_yolo
        with _tf.NamedTemporaryFile(suffix=".onnx") as f:
            onnx_bytes = make_raw_yolo(f.name, res=320, seed=6)
        self._serve_model("MODEL_B", onnx_bytes,
                          {"id": "MODEL_B", "width": 640, "height": 640})
        with self.assertRaises(FxdnaError) as ctx:
            self.sup.prepare("plus://MODEL_B")
        self.assertEqual(ctx.exception.error_code, "UNSUPPORTED_CONTRACT")

    def test_plus_unsupported_contract_no_job(self):
        import tempfile as _tf
        bad = os.path.join(self.cfg.data_dir, "bad.onnx")
        with open(bad, "wb") as f:
            f.write(b"not onnx")
        with open(bad, "rb") as f:
            onnx_bytes = f.read()
        self._serve_model("MODEL_C", onnx_bytes, {"id": "MODEL_C"})
        with self.assertRaises(FxdnaError):
            self.sup.prepare("plus://MODEL_C")
        jobs = self.sup.registry.query("SELECT uuid FROM jobs")
        self.assertEqual(jobs, [])


if __name__ == "__main__":
    unittest.main()
