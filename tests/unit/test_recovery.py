"""Unit tests: bounded recovery without data destruction (v0.1.1 C5).

Proves the headline sequence (fail -> correct environment -> same
/data restart -> resume -> PREPARED), bounded transient retries,
unsafe/safety/permanent no-retry, explicit recover semantics, legacy
migration honesty, cache preservation, and attempt visibility — all
hardware-free with scripted backends.
"""
import hashlib
import json
import os
import sqlite3
import tempfile
import time
import unittest
from unittest import mock

from frigate_xdna import retry_policy
from frigate_xdna.config import Config
from frigate_xdna.observability.progress import RecordingReporter
from frigate_xdna.supervisor import Supervisor
from tests.integration.onnx_builders import make_raw_yolo

FLEXMLRT_DETAIL = (
    "Compilation Complete / FlexMLRT Exception: "
    "mmap(... flags=8209 ...) failed (err=-11): "
    "Resource temporarily unavailable"
)


def make_config(tmp):
    return Config(data_dir=tmp, endpoint="tcp://127.0.0.1:5559")


def local_onnx(d, name="m.onnx", classes=4, seed=3):
    path = os.path.join(d, name)
    make_raw_yolo(path, res=320, classes=classes, seed=seed)
    return path


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


def job_rows(sup, ref):
    return sup.registry.query(
        "SELECT uuid, stage, attempt FROM jobs WHERE ref=?", (ref,))


def expire_backoff(sup, uuid):
    row = sup.registry.get_job(uuid)
    rec = dict(row["failure"])
    rec["not_before"] = 0.0
    sup.registry.set_failure(uuid, rec)


def expire_all_eligible(sup, ref):
    for uuid, _stage, _attempt in job_rows(sup, ref):
        row = sup.registry.get_job(uuid)
        if row["failure"] and not row["failure"].get("resumed_to"):
            expire_backoff(sup, uuid)


class TestClassify(unittest.TestCase):
    def test_kinds(self):
        # Host memlock varies (stock CI runners are constrained), so
        # pin the allowance: adequate -> unattributable, constrained
        # -> config-blocked. Never depend on the test machine.
        with mock.patch.object(retry_policy, "memlock_adequate",
                               return_value=True):
            self.assertEqual(
                retry_policy.classify("COMPILE_FAILED", "COMPILE_FAILED",
                                      FLEXMLRT_DETAIL)["kind"],
                "unknown")
        with mock.patch.object(retry_policy, "memlock_adequate",
                               return_value=False):
            c = retry_policy.classify("COMPILE_FAILED", "COMPILE_FAILED",
                                      FLEXMLRT_DETAIL)
            self.assertEqual(c["kind"], "config_blocked")
            self.assertTrue(c["retryable"])
            self.assertFalse(c["auto"])
            self.assertIn("ulimits", c["guidance"])
        bare = retry_policy.classify("COMPILE_FAILED", "COMPILE_FAILED",
                                         "")
        self.assertEqual(bare["kind"], "unknown")
        self.assertTrue(bare["retryable"])
        self.assertFalse(bare["auto"])
        self.assertEqual(
            retry_policy.classify("COMPILE_FAILED", "COMPILE_FAILED",
                                  "compile timeout after 2700s")["kind"],
            "transient")
        self.assertEqual(
            retry_policy.classify("VALIDATION_FAILED", "VALIDATION_FAILED",
                                  "")["kind"], "permanent")
        self.assertEqual(
            retry_policy.classify("QUARANTINED", "QUARANTINED",
                                  "")["kind"], "safety")
        self.assertEqual(
            retry_policy.classify("UNSUPPORTED_CONTRACT",
                                  "UNSUPPORTED_CONTRACT", "")["kind"],
            "permanent")
        self.assertEqual(
            retry_policy.classify("COMPILE_FAILED", "AUTH_FAILED",
                                  "")["kind"], "permanent")
        self.assertEqual(
            retry_policy.classify("INTERRUPTED", "INTERRUPTED",
                                  "")["kind"], "interrupted_safe")

    def test_errno_alone_never_diagnoses_memlock(self):
        with mock.patch.object(retry_policy, "memlock_adequate",
                               return_value=False):
            c = retry_policy.classify("COMPILE_FAILED", "COMPILE_FAILED",
                                      "some EAGAIN somewhere")
            self.assertEqual(c["kind"], "unknown")
            self.assertFalse(c["auto"])

    def test_backoff_and_eligibility(self):
        self.assertEqual(retry_policy.backoff_s(1), 30.0)
        self.assertEqual(retry_policy.backoff_s(2), 60.0)
        self.assertEqual(retry_policy.backoff_s(3), 120.0)
        now = 1000.0
        rec = retry_policy.new_record("COMPILE_FAILED", "COMPILE_FAILED",
                                      "compile timeout", 1, now=now)
        eligible, _ = retry_policy.auto_eligible(rec, now + 31.0)
        self.assertTrue(eligible)
        eligible, why = retry_policy.auto_eligible(rec, now + 1.0)
        self.assertFalse(eligible)
        rec3 = retry_policy.new_record("COMPILE_FAILED", "COMPILE_FAILED",
                                       "", 3, now=now)
        eligible, why = retry_policy.auto_eligible(rec3, now + 9999.0)
        self.assertFalse(eligible)
        self.assertIn("exhausted", why)
        eligible, _ = retry_policy.auto_eligible(None, now)
        self.assertFalse(eligible)

    def test_safety_inhibition_vocabulary(self):
        for reason in ("QUARANTINED:x", "SAFETY_INHIBITED",
                       "WORKER_LOAD:DEVICE_FAULT",
                       "interrupted_unsafe_operation", "unclean_operation"):
            self.assertTrue(retry_policy.is_safety_inhibition(reason),
                            reason)
        for reason in ("WORKER_DIED", "WORKER_SPAWN", "ACTIVATION_STATE"):
            self.assertFalse(retry_policy.is_safety_inhibition(reason),
                             reason)


class TestMemlockResume(unittest.TestCase):
    """Headline UX: fail -> correct -> same-/data restart -> PREPARED."""

    def test_failure_correction_restart_success(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = make_config(d)
            rec = RecordingReporter()
            # Broken environment: inadequate memlock, compile fails
            # with the locked-memory vendor pattern.
            with mock.patch.object(retry_policy, "memlock_adequate",
                                   return_value=False):
                sup1 = Supervisor(
                    cfg, fake_compile={"device_required": False,
                                       "succeed": False,
                                       "fail_state": "COMPILE_FAILED",
                                       "detail": FLEXMLRT_DETAIL},
                    reporter=rec)
                try:
                    ref = local_onnx(d)
                    out = sup1.prepare(ref)
                    state = pump_until(sup1, ref, ("COMPILE_FAILED",))
                    self.assertEqual(state, "COMPILE_FAILED")
                    row = sup1.registry.get_job(out["job_uuid"])
                    self.assertEqual(row["failure"]["kind"],
                                     "config_blocked")
                    # No automatic retry while unchanged.
                    sup1.pump(0.1)
                    self.assertEqual(len(job_rows(sup1, ref)), 1)
                finally:
                    sup1.stop()
                # Restart, still broken: same terminal row, no new work.
                sup2 = Supervisor(cfg)
                try:
                    out2 = sup2.prepare(ref)
                    self.assertEqual(out2["job_uuid"], out["job_uuid"])
                    self.assertEqual(len(job_rows(sup2, ref)), 1)
                finally:
                    sup2.stop()
            # Correction verified: same /data, new process, corrected
            # environment (default fake backend succeeds).
            with mock.patch.object(retry_policy, "memlock_adequate",
                                   return_value=True):
                sup3 = Supervisor(cfg, reporter=rec)
                try:
                    out3 = sup3.prepare(ref)
                    self.assertNotEqual(out3["job_uuid"], out["job_uuid"])
                    # Auto-resume continues the persisted count (only
                    # explicit recover resets it); the global bound
                    # holds across the failure lineage.
                    self.assertEqual(
                        sup3.registry.get_job(
                            out3["job_uuid"])["attempt"], 2)
                    state = pump_until(sup3, ref, ("PREPARED",))
                    self.assertEqual(state, "PREPARED")
                    kinds = rec.kinds()
                    self.assertIn("resumed", kinds)
                    resumed = next(e for e in rec.events
                                   if e["kind"] == "resumed")
                    self.assertIn("blocking condition corrected",
                                  resumed["reason"])
                finally:
                    sup3.stop()

    def test_projection_shows_attempts_and_guidance(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = make_config(d)
            with mock.patch.object(retry_policy, "memlock_adequate",
                                   return_value=False):
                sup = Supervisor(
                    cfg, fake_compile={"device_required": False,
                                       "succeed": False,
                                       "fail_state": "COMPILE_FAILED",
                                       "detail": FLEXMLRT_DETAIL})
                try:
                    ref = local_onnx(d)
                    sup.prepare(ref)
                    pump_until(sup, ref, ("COMPILE_FAILED",))
                    models = sup.status(ref)["models"]
                    self.assertEqual(models[0]["attempts"], 1)
                    self.assertTrue(models[0]["retryable"])
                    failure = models[0]["failure"]
                    self.assertEqual(failure["kind"], "config_blocked")
                    self.assertIn("ulimits", failure["guidance"])
                finally:
                    sup.stop()


class TestBoundedTransient(unittest.TestCase):


    def test_three_attempts_then_exhausted(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = make_config(d)
            sup = Supervisor(
                cfg, fake_compile={"device_required": False,
                                   "succeed": False,
                                   "fail_state": "COMPILE_FAILED",
                                   "detail": "compile timeout"})
            try:
                ref = local_onnx(d)
                sup.prepare(ref)
                pump_until(sup, ref, ("COMPILE_FAILED",))
                for _ in range(2):
                    expire_all_eligible(sup, ref)
                    sup.pump(0.05)
                    pump_until(sup, ref, ("COMPILE_FAILED",))
                rows = job_rows(sup, ref)
                # Three terminal attempts; history preserved on rows.
                self.assertEqual(len(rows), 3)
                self.assertEqual({r[2] for r in rows}, {1, 2, 3})
                last = sup.registry.get_job(
                    sup.registry.latest_job_for_ref(ref)["uuid"])
                self.assertEqual(len(last["failure"]["history"]), 3)
                # Budget exhausted: further pumps never resubmit.
                expire_all_eligible(sup, ref)
                sup.pump(0.05)
                sup.pump(0.05)
                self.assertEqual(len(job_rows(sup, ref)), 3)
            finally:
                sup.stop()

    def test_no_live_duplicate_on_retry(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = make_config(d)
            sup = Supervisor(cfg,
                             fake_compile={"device_required": False})
            try:
                ref = local_onnx(d)
                out = sup.prepare(ref)
                out2 = sup.prepare(ref)
                self.assertEqual(out["job_uuid"], out2["job_uuid"])
                self.assertEqual(len(job_rows(sup, ref)), 1)
            finally:
                sup.stop()


class TestNoRetryClasses(unittest.TestCase):
    def test_quarantine_never_retries(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = make_config(d)
            sup = Supervisor(
                cfg, fake_compile={"device_required": False,
                                   "succeed": False,
                                   "fail_state": "QUARANTINED"})
            try:
                ref = local_onnx(d)
                sup.prepare(ref)
                pump_until(sup, ref, ("QUARANTINED",))
                for _ in range(3):
                    sup.pump(0.05)
                self.assertEqual(len(job_rows(sup, ref)), 1)
                sup.stop()
                sup2 = Supervisor(cfg)
                try:
                    sup2.prepare(ref)
                    self.assertEqual(len(job_rows(sup2, ref)), 1)
                finally:
                    sup2.stop()
            finally:
                try:
                    sup.stop()
                except Exception:
                    pass

    def test_validation_failure_never_retries_even_on_recover(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = make_config(d)
            sup = Supervisor(
                cfg, fake_compile={"device_required": False,
                                   "succeed": False,
                                   "fail_state": "VALIDATION_FAILED"})
            try:
                ref = local_onnx(d)
                sup.prepare(ref)
                pump_until(sup, ref, ("VALIDATION_FAILED",))
                res = sup.recover_ref(ref)
                self.assertFalse(res["requeued"])
                self.assertEqual(len(job_rows(sup, ref)), 1)
            finally:
                sup.stop()


class TestRecoverRoute(unittest.TestCase):
    def test_recover_clears_and_requeues(self):
        from frigate_xdna.runtime.safety import inhibit
        with tempfile.TemporaryDirectory() as d:
            cfg = make_config(d)
            sup = Supervisor(
                cfg, fake_compile={"device_required": False,
                                   "succeed": False,
                                   "fail_state": "COMPILE_FAILED",
                                   "detail": "compile timeout"})
            try:
                ref = local_onnx(d)
                sup.prepare(ref)
                pump_until(sup, ref, ("COMPILE_FAILED",))
                # Exhaust the budget: three terminal rows.
                for _ in range(2):
                    expire_all_eligible(sup, ref)
                    sup.pump(0.05)
                    pump_until(sup, ref, ("COMPILE_FAILED",))
                self.assertEqual(len(job_rows(sup, ref)), 3)
                inhibit(d, sup.registry, "WORKER_DIED", ref)
                res = sup.recover_ref(ref)
                self.assertTrue(res["cleared"])
                self.assertTrue(res["requeued"])
                self.assertIsNone(sup.registry.get_state("inhibition"))
                newest = sup.registry.latest_job_for_ref(ref)
                self.assertEqual(newest["stage"], "QUEUED")
                self.assertEqual(newest["attempt"], 1)
            finally:
                sup.stop()

    def test_recover_refuses_safety(self):
        from frigate_xdna.runtime.safety import inhibit
        with tempfile.TemporaryDirectory() as d:
            cfg = make_config(d)
            sup = Supervisor(cfg)
            try:
                ref = local_onnx(d)
                sup.registry.upsert_ref(ref, "onnx", None)
                inhibit(d, sup.registry, "QUARANTINED:device-suspect",
                        ref)
                res = sup.recover_ref(ref)
                self.assertFalse(res["cleared"])
                self.assertFalse(res["requeued"])
                self.assertIn("QUARANTINED", res["note"])
                self.assertIsNotNone(sup.registry.get_state("inhibition"))
            finally:
                sup.stop()

    def test_recover_unknown_legacy_starts_bounded_cycle(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = make_config(d)
            sup = Supervisor(
                cfg, fake_compile={"device_required": False,
                                   "succeed": False,
                                   "fail_state": "COMPILE_FAILED"})
            try:
                ref = local_onnx(d)
                out = sup.prepare(ref)
                pump_until(sup, ref, ("COMPILE_FAILED",))
                # Age the row into legacy shape: terminal, no evidence.
                sup.registry.execute(
                    "UPDATE jobs SET failure_json=NULL WHERE uuid=?",
                    (out["job_uuid"],))
                sup.stop()
                sup2 = Supervisor(cfg)
                try:
                    # No automatic resume for evidence-free rows...
                    out2 = sup2.prepare(ref)
                    self.assertEqual(out2["job_uuid"], out["job_uuid"])
                    # ...but the acknowledged operator route works.
                    res = sup2.recover_ref(ref)
                    self.assertTrue(res["requeued"])
                    newest = sup2.registry.latest_job_for_ref(ref)
                    self.assertEqual(newest["stage"], "QUEUED")
                    self.assertNotEqual(newest["uuid"], out["job_uuid"])
                finally:
                    sup2.stop()
            finally:
                try:
                    sup.stop()
                except Exception:
                    pass


class TestLegacyMigration(unittest.TestCase):
    def _v1_db(self, d):
        """A genuine v1 store: v2 column dropped, version reset."""
        db = os.path.join(d, "registry.sqlite3")
        from frigate_xdna.cache.registry import Registry
        reg = Registry(db)
        reg.execute("INSERT INTO model_refs(ref, kind, state, updated_at)"
                    " VALUES (?,?,?,?)", ("plus://rc3", "plus",
                                          "COMPILE_FAILED",
                                          time.time()))
        reg.execute("INSERT INTO jobs(uuid, ref, compile_key, stage,"
                    " attempt, created_at, updated_at) VALUES"
                    " (?,?,?,?,?,?,?)",
                    ("rc3-job", "plus://rc3", "ck-rc3", "COMPILE_FAILED",
                     1, time.time(), time.time()))
        reg.close()
        cx = sqlite3.connect(db)
        cx.execute("ALTER TABLE jobs DROP COLUMN failure_json")
        cx.execute("ALTER TABLE jobs DROP COLUMN source_sha256")
        cx.execute("UPDATE schema_version SET version=1")
        cx.commit()
        cx.close()
        return db

    def test_migration_preserves_rows_and_refuses_silence(self):
        with tempfile.TemporaryDirectory() as d:
            from frigate_xdna.cache.registry import Registry
            db = self._v1_db(d)
            reg = Registry(db)
            try:
                row = reg.query("SELECT version FROM schema_version")
                self.assertEqual(row[0][0], 3)
                job = reg.get_job("rc3-job")
                self.assertEqual(job["stage"], "COMPILE_FAILED")
                self.assertIsNone(job["failure"])
                rec = reg.get_ref("plus://rc3")
                self.assertEqual(rec["state"], "COMPILE_FAILED")
            finally:
                reg.close()

    def test_rc3_artifact_survives_upgrade_without_recompile(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = make_config(d)
            sup = Supervisor(cfg,
                             fake_compile={"device_required": False})
            try:
                ref = local_onnx(d, name="rc3.onnx", seed=9)
                out = sup.prepare(ref)
                pump_until(sup, ref, ("PREPARED",))
                ckey = out["compile_key"]
                sup.stop()
                # "Upgrade": reopen on the same /data; prepare must be
                # a cache hit with zero compiler work.
                sup2 = Supervisor(cfg)
                try:
                    before = sup2.registry.query(
                        "SELECT COUNT(*) FROM jobs")[0][0]
                    out2 = sup2.prepare(ref)
                    self.assertTrue(out2.get("cache_hit"))
                    self.assertEqual(out2["compile_key"], ckey)
                    after = sup2.registry.query(
                        "SELECT COUNT(*) FROM jobs")[0][0]
                    # One provenance row for the hit, no compile.
                    self.assertEqual(after, before + 1)
                    newest = sup2.registry.latest_job_for_ref(ref)
                    self.assertEqual(newest["stage"], "PREPARED")
                finally:
                    sup2.stop()
            finally:
                try:
                    sup.stop()
                except Exception:
                    pass

    def test_fetch_failure_serves_usable_cache(self):
        """SPEC §4.4: a failed Plus refresh must not downgrade a
        working cached model; the failure stays visible."""
        from frigate_xdna.errors import FxdnaError
        with tempfile.TemporaryDirectory() as d:
            cfg = make_config(d)
            sup = Supervisor(cfg,
                             fake_compile={"device_required": False})
            try:
                ref = local_onnx(d, name="cached.onnx", seed=11)
                out = sup.prepare(ref)
                pump_until(sup, ref, ("PREPARED",))
                ckey = out["compile_key"]
                # The same bytes behind a Plus alias (as after a real
                # Plus fetch of identical content).
                alias = "plus://cached-model"
                sup.registry.upsert_ref(alias, "plus", "cached-model")
                sup.registry.execute(
                    "INSERT INTO jobs(uuid, ref, compile_key, stage,"
                    " attempt, created_at, updated_at) VALUES"
                    " (?,?,?,?,?,?,?)",
                    ("alias-prep", alias, ckey, "PREPARED", 1,
                     time.time(), time.time()))
                sup.registry.set_ref_state(alias, "PREPARED")
                with mock.patch.object(
                        Supervisor, "_fetch_plus",
                        side_effect=FxdnaError(4, "DOWNLOAD_FAILED",
                                               "network down")):
                    rec = RecordingReporter()
                    sup.reporter = rec
                    got = sup.prepare(alias)
                self.assertTrue(got.get("cache_hit"))
                self.assertEqual(got["compile_key"], ckey)
                kinds = rec.kinds()
                self.assertIn("fetch_failed_cached", kinds)
                # And without any cache, the same failure still raises.
                with mock.patch.object(
                        Supervisor, "_fetch_plus",
                        side_effect=FxdnaError(4, "DOWNLOAD_FAILED",
                                               "network down")):
                    with self.assertRaises(FxdnaError):
                        sup.prepare("plus://never-seen")
            finally:
                sup.stop()


class TestReviewFindings(unittest.TestCase):
    """Regression tests for the checkpoint-5 review round."""

    def test_two_eligible_rows_open_one_attempt(self):
        """retry_due serializes per row: the second eligible row for a
        key sees the first's live attempt and stands down."""
        with tempfile.TemporaryDirectory() as d:
            cfg = make_config(d)
            sup = Supervisor(
                cfg, fake_compile={"device_required": False,
                                   "succeed": False,
                                   "fail_state": "COMPILE_FAILED",
                                   "detail": "compile timeout"})
            try:
                ref = local_onnx(d)
                sup.prepare(ref)
                pump_until(sup, ref, ("COMPILE_FAILED",))
                first = sup.registry.latest_job_for_ref(ref)
                # A second eligible terminal row for the same key.
                sup.registry.execute(
                    "INSERT INTO jobs(uuid, ref, compile_key, stage,"
                    " attempt, created_at, updated_at, failure_json)"
                    " VALUES (?,?,?,?,?,?,?,?)",
                    ("second-uuid", ref, first["compile_key"],
                     "COMPILE_FAILED", 1, time.time(), time.time(),
                     json.dumps(
                         {**first["failure"], "not_before": 0.0,
                          "resumed_to": None})))
                expire_all_eligible(sup, ref)
                opened = sup.jobs.retry_due()
                self.assertEqual(len(opened), 1)
                live = [r for r in job_rows(sup, ref)
                        if r[1] not in ("COMPILE_FAILED",)]
                self.assertEqual(len(live), 1)
            finally:
                sup.stop()

    def test_resume_without_source_refuses_cleanly(self):
        from frigate_xdna.errors import FxdnaError
        with tempfile.TemporaryDirectory() as d:
            cfg = make_config(d)
            sup = Supervisor(cfg)
            try:
                # Unresolvable resume source fails loudly, never
                # sourceless.
                with self.assertRaises(FxdnaError):
                    sup._compile_source_path("c" * 64)
                # A raising factory never opens a row: the terminal
                # row stays for operator review.
                sup.registry.upsert_ref("plus://x", "plus", "x")
                sup.registry.execute(
                    "INSERT INTO jobs(uuid, ref, compile_key, stage,"
                    " attempt, created_at, updated_at, failure_json)"
                    " VALUES (?,?,?,?,?,?,?,?)",
                    ("t-uuid", "plus://x", "ck-missing",
                     "COMPILE_FAILED", 1, time.time(), time.time(),
                     json.dumps(
                         retry_policy.new_record(
                             "COMPILE_FAILED", "COMPILE_FAILED",
                             "compile timeout", 1, now=0.0))))
                sup.jobs.backend_factory = lambda **kw: (_ for _ in ()
                                                         ).throw(
                    FxdnaError(10, "CACHE_CORRUPT", "gone"))
                out = sup.jobs._maybe_resume(
                    "plus://x", "ck-missing",
                    sup.registry.get_job("t-uuid"), trigger="test")
                self.assertIsNone(out)
                # The terminal evidence remains for operator review,
                # and the refusal is stamped so retries stop burning.
                kept = sup.registry.get_job("t-uuid")
                self.assertEqual(kept["stage"], "COMPILE_FAILED")
                self.assertEqual(len(job_rows(sup, "plus://x")), 1)
                self.assertIn("resume_refused",
                              kept["failure"])
            finally:
                sup.stop()

    def test_resume_binds_failed_attempt_source(self):
        """Cache-corruption guard: ref ingests bytes X (key K1, fails),
        then --refresh ingests bytes Y (key K2). The K1 resume must
        compile X, never the ref's current bytes Y."""
        with tempfile.TemporaryDirectory() as d:
            cfg = make_config(d)
            sup = Supervisor(
                cfg, fake_compile={"device_required": False,
                                   "succeed": False,
                                   "fail_state": "COMPILE_FAILED",
                                   "detail": "compile timeout"})
            try:
                path = os.path.join(d, "mutable.onnx")
                make_raw_yolo(path, res=320, classes=4, seed=71)
                out_x = sup.prepare(path)
                pump_until(sup, path, ("COMPILE_FAILED",))
                row_x = sup.registry.get_job(out_x["job_uuid"])
                sha_x = row_x["source_sha256"]
                self.assertTrue(sha_x)
                # New bytes under the same ref: new key, new job.
                make_raw_yolo(path, res=320, classes=4, seed=72)
                out_y = sup.prepare(path, refresh=True)
                self.assertNotEqual(out_y["compile_key"],
                                    out_x["compile_key"])
                pump_until(sup, path, ("COMPILE_FAILED",))
                # Expire only the K1 backoff; its resume must bind X.
                # (The failing fake backend terminates the new attempt
                # in the same pump, so assert on the attempt-2 row.)
                expire_backoff(sup, out_x["job_uuid"])
                sup.pump(0.05)
                sup.pump(0.05)
                k1_rows = [r for r in job_rows(sup, path)
                           if sup.registry.get_job(r[0])["compile_key"]
                           == out_x["compile_key"]]
                self.assertEqual(len(k1_rows), 2)
                new_row = sup.registry.get_job(
                    max(k1_rows, key=lambda r: r[2])[0])
                self.assertEqual(new_row["attempt"], 2)
                self.assertEqual(new_row["source_sha256"], sha_x)
                # ...and the bound bytes resolve to the X file, never Y.
                resolved = sup._compile_source_path(
                    new_row["source_sha256"])
                with open(resolved, "rb") as f:
                    self.assertEqual(
                        hashlib.sha256(f.read()).hexdigest(), sha_x)
            finally:
                sup.stop()

    def test_requeue_consumes_source_row(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = make_config(d)
            sup = Supervisor(
                cfg, fake_compile={"device_required": False,
                                   "succeed": False,
                                   "fail_state": "COMPILE_FAILED",
                                   "detail": "compile timeout"})
            try:
                ref = local_onnx(d)
                out = sup.prepare(ref)
                pump_until(sup, ref, ("COMPILE_FAILED",))
                res = sup.recover_ref(ref)
                self.assertTrue(res["requeued"])
                self.assertEqual(len(job_rows(sup, ref)), 2)
                source = sup.registry.get_job(out["job_uuid"])
                self.assertIn("resumed_to", source["failure"])
                # The consumed source row never opens a second chain.
                expire_all_eligible(sup, ref)
                sup.pump(0.05)
                sup.pump(0.05)
                self.assertEqual(len(job_rows(sup, ref)), 2)
            finally:
                sup.stop()

    def test_unknown_record_recovers_explicitly(self):
        """Memlock pattern with adequate allowance: no auto resume,
        but acknowledged recover opens one bounded cycle."""
        with tempfile.TemporaryDirectory() as d:
            cfg = make_config(d)
            with mock.patch.object(retry_policy, "memlock_adequate",
                                   return_value=True):
                sup = Supervisor(
                    cfg, fake_compile={"device_required": False,
                                       "succeed": False,
                                       "fail_state": "COMPILE_FAILED",
                                       "detail": FLEXMLRT_DETAIL})
                try:
                    ref = local_onnx(d)
                    out = sup.prepare(ref)
                    pump_until(sup, ref, ("COMPILE_FAILED",))
                    row = sup.registry.get_job(out["job_uuid"])
                    self.assertEqual(row["failure"]["kind"], "unknown")
                    sup.pump(0.05)
                    self.assertEqual(len(job_rows(sup, ref)), 1)
                    res = sup.recover_ref(ref)
                    self.assertTrue(res["requeued"])
                finally:
                    sup.stop()

    def test_recover_interrupted_row_gets_record_and_resumes(self):
        """store.recover marks workdir jobs INTERRUPTED without
        records; reconcile must record them so they can resume."""
        with tempfile.TemporaryDirectory() as d:
            cfg = make_config(d)
            sup = Supervisor(
                cfg, fake_compile={"device_required": True,
                                   "device_held_by_worker": True})
            try:
                ref = local_onnx(d)
                out = sup.prepare(ref)
                sup.pump(0.05)
                work = os.path.join(d, "work", out["job_uuid"])
                os.makedirs(work, exist_ok=True)
                sup.stop()
                sup2 = Supervisor(cfg)
                try:
                    row = sup2.registry.get_job(out["job_uuid"])
                    self.assertEqual(row["stage"], "INTERRUPTED")
                    self.assertIsNotNone(row["failure"])
                    out2 = sup2.prepare(ref)
                    self.assertNotEqual(out2["job_uuid"], out["job_uuid"])
                finally:
                    sup2.stop()
            finally:
                try:
                    sup.stop()
                except Exception:
                    pass

    def test_host_reboot_token_mismatch_never_auto_resumes(self):
        """A row from a previous boot may have died with a host
        reboot mid-operation: unproven safety, no automatic resume."""
        with tempfile.TemporaryDirectory() as d:
            cfg = make_config(d)
            sup = Supervisor(
                cfg, fake_compile={"device_required": True,
                                   "device_held_by_worker": True})
            try:
                ref = local_onnx(d)
                out = sup.prepare(ref)
                sup.pump(0.05)
                row = sup.registry.get_job(out["job_uuid"])
                self.assertEqual(row["stage"], "WAITING_FOR_DEVICE")
                sup.registry.execute(
                    "UPDATE jobs SET boot_token=? WHERE uuid=?",
                    ("old-boot-token", out["job_uuid"]))
                sup.stop()
                sup2 = Supervisor(cfg)
                try:
                    row = sup2.registry.get_job(out["job_uuid"])
                    self.assertEqual(row["stage"], "INTERRUPTED")
                    self.assertEqual(row["failure"]["kind"], "unknown")
                    out2 = sup2.prepare(ref)
                    self.assertEqual(out2["job_uuid"], out["job_uuid"])
                    # ...but acknowledged recover still works.
                    res = sup2.recover_ref(ref)
                    self.assertTrue(res["requeued"])
                finally:
                    sup2.stop()
            finally:
                try:
                    sup.stop()
                except Exception:
                    pass

    def test_recover_refused_while_other_ref_inhibited(self):
        from frigate_xdna.runtime.safety import inhibit
        with tempfile.TemporaryDirectory() as d:
            cfg = make_config(d)
            sup = Supervisor(
                cfg, fake_compile={"device_required": False,
                                   "succeed": False,
                                   "fail_state": "COMPILE_FAILED",
                                   "detail": "compile timeout"})
            try:
                ref_a = local_onnx(d, name="a.onnx", seed=21)
                ref_b = local_onnx(d, name="b.onnx", seed=22)
                sup.prepare(ref_a)
                pump_until(sup, ref_a, ("COMPILE_FAILED",))
                sup.registry.upsert_ref(ref_b, "onnx", None)
                inhibit(d, sup.registry, "DEVICE_FAULT", ref_b)
                res = sup.recover_ref(ref_a)
                self.assertFalse(res["cleared"])
                self.assertFalse(res["requeued"])
                self.assertIn(ref_b, res["note"])
                self.assertEqual(len(job_rows(sup, ref_a)), 1)
                self.assertIsNotNone(sup.registry.get_state("inhibition"))
            finally:
                sup.stop()


class TestCodexFindings(unittest.TestCase):
    """Regression tests for the Codex review round."""

    def test_probe_error_text_classifies_permanent(self):
        rec = retry_policy.new_record(
            "COMPILE_FAILED", "COMPILE_FAILED", "", 1,
            error_text="probe failed: non-finite probe output")
        self.assertEqual(rec["kind"], "permanent")
        self.assertFalse(rec["retryable"])
        self.assertIn("non-finite", rec["reason"])

    def test_timeout_error_text_classifies_transient(self):
        rec = retry_policy.new_record(
            "COMPILE_FAILED", "COMPILE_FAILED", "", 1,
            error_text="vaiml-compile timeout")
        self.assertEqual(rec["kind"], "transient")
        self.assertTrue(rec["retryable"])

    def test_null_boot_token_is_unproven(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = make_config(d)
            sup = Supervisor(
                cfg, fake_compile={"device_required": True,
                                   "device_held_by_worker": True})
            try:
                ref = local_onnx(d)
                out = sup.prepare(ref)
                sup.pump(0.05)
                sup.registry.execute(
                    "UPDATE jobs SET boot_token=NULL WHERE uuid=?",
                    (out["job_uuid"],))
                sup.stop()
                sup2 = Supervisor(cfg)
                try:
                    row = sup2.registry.get_job(out["job_uuid"])
                    self.assertEqual(row["stage"], "INTERRUPTED")
                    self.assertEqual(row["failure"]["kind"], "unknown")
                    out2 = sup2.prepare(ref)
                    self.assertEqual(out2["job_uuid"], out["job_uuid"])
                finally:
                    sup2.stop()
            finally:
                try:
                    sup.stop()
                except Exception:
                    pass

    def test_aliases_follow_the_retry(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = make_config(d)
            sup1 = Supervisor(
                cfg, fake_compile={"device_required": False,
                                   "succeed": False,
                                   "fail_state": "COMPILE_FAILED",
                                   "detail": "compile timeout"})
            try:
                ref_a = local_onnx(d, name="a.onnx", seed=31)
                ref_b = local_onnx(d, name="b.onnx", seed=31)
                out_a = sup1.prepare(ref_a)
                out_b = sup1.prepare(ref_b)
                self.assertEqual(out_a["job_uuid"], out_b["job_uuid"])
                pump_until(sup1, ref_a, ("COMPILE_FAILED",))
                sup1.stop()
                sup2 = Supervisor(cfg)
                try:
                    expire_all_eligible(sup2, ref_a)
                    out2 = sup2.prepare(ref_a)
                    pump_until(sup2, ref_a, ("PREPARED",))
                    key = out2["compile_key"]
                    self.assertEqual(
                        sup2.registry.prepared_key_for_ref(ref_b), key)
                    by_ref = {m["ref"]: m
                              for m in sup2.status()["models"]}
                    self.assertEqual(by_ref[ref_b]["state"], "PREPARED")
                    self.assertEqual(by_ref[ref_b]["compile_key"], key)
                finally:
                    sup2.stop()
            finally:
                try:
                    sup1.stop()
                except Exception:
                    pass

    def test_recover_exit_codes(self):
        import io as _io
        from contextlib import redirect_stdout as _redirect
        from unittest import mock as _mock

        from frigate_xdna import cli as _cli
        from frigate_xdna.runtime.safety import inhibit

        def run_cli(argv, data_dir):
            buf = _io.StringIO()
            with _mock.patch.dict(os.environ,
                                  {"FXDNA_DATA_DIR": data_dir}):
                with _redirect(buf):
                    return _cli.main(argv), buf.getvalue()

        with tempfile.TemporaryDirectory() as d:
            cfg = make_config(d)
            sup = Supervisor(cfg)
            try:
                ref = local_onnx(d, seed=41)
                sup.registry.upsert_ref(ref, "onnx", None)
                inhibit(d, sup.registry, "DEVICE_FAULT", ref)
                sup.stop()
                # Standalone path.
                rc, _out = run_cli(["recover", ref, "--acknowledge"], d)
                self.assertEqual(rc, 8)
                rc, _out = run_cli(["recover", "plus://other",
                                    "--acknowledge"], d)
                self.assertEqual(rc, 3)
                # Daemon path (result nested under "recovered").
                sup2 = Supervisor(cfg)
                try:
                    sup2.start_admin()
                    rc, out = run_cli(["recover", ref, "--acknowledge"],
                                      d)
                    self.assertEqual(rc, 8)
                    doc = json.loads(out)
                    self.assertTrue(
                        doc["recovered"]["refused_safety"])
                    rc, _out = run_cli(["recover", "plus://other",
                                        "--acknowledge"], d)
                    self.assertEqual(rc, 3)
                finally:
                    sup2.stop()
            finally:
                try:
                    sup.stop()
                except Exception:
                    pass


class TestUpgradeFollowups(unittest.TestCase):
    def test_baseline_snapshot_stays_silent(self):
        from frigate_xdna.observability.progress import RecordingReporter
        with tempfile.TemporaryDirectory() as d:
            cfg = make_config(d)
            sup = Supervisor(
                cfg, fake_compile={"device_required": False,
                                   "succeed": False,
                                   "fail_state": "COMPILE_FAILED"})
            try:
                ref = local_onnx(d)
                sup.prepare(ref)
                pump_until(sup, ref, ("COMPILE_FAILED",))
                sup.stop()
                rec = RecordingReporter()
                sup2 = Supervisor(cfg, reporter=rec)
                try:
                    sup2.pump(0.05)
                    sup2.report_progress()
                    sup2.report_progress()
                    self.assertNotIn("preparation_failed", rec.kinds())
                finally:
                    sup2.stop()
            finally:
                try:
                    sup.stop()
                except Exception:
                    pass

    def test_legacy_retryable_projection(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = make_config(d)
            sup = Supervisor(cfg)
            try:
                for ref, stage in (("plus://a", "COMPILE_FAILED"),
                                   ("plus://q", "QUARANTINED")):
                    sup.registry.upsert_ref(ref, "plus", ref)
                    sup.registry.set_ref_state(ref, stage)
                    sup.registry.execute(
                        "INSERT INTO jobs(uuid, ref, compile_key, stage,"
                        " attempt, created_at, updated_at) VALUES"
                        " (?,?,?,?,?,?,?)",
                        (f"u-{stage}", ref, "ck", stage, 0,
                         time.time(), time.time()))
                models = {m["ref"]: m for m in sup.status()["models"]}
                self.assertTrue(models["plus://a"]["retryable"])
                self.assertFalse(models["plus://q"]["retryable"])
                res = sup.recover_ref("plus://q")
                self.assertFalse(res["requeued"])
            finally:
                sup.stop()


class TestInterruptResume(unittest.TestCase):
    def test_restart_interrupted_resumes_with_clean_safety(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = make_config(d)
            rec = RecordingReporter()
            sup = Supervisor(
                cfg, fake_compile={"device_required": True,
                                   "device_held_by_worker": True},
                reporter=rec)
            try:
                ref = local_onnx(d)
                out = sup.prepare(ref)
                sup.pump(0.05)
                job = sup.registry.get_job(out["job_uuid"])
                self.assertEqual(job["stage"], "WAITING_FOR_DEVICE")
                sup.stop()
                sup2 = Supervisor(cfg, reporter=rec)
                try:
                    row = sup2.registry.get_job(out["job_uuid"])
                    self.assertEqual(row["stage"], "INTERRUPTED")
                    self.assertEqual(row["failure"]["kind"],
                                     "interrupted_safe")
                    out2 = sup2.prepare(ref)
                    self.assertNotEqual(out2["job_uuid"], out["job_uuid"])
                    self.assertIn("resumed", rec.kinds())
                finally:
                    sup2.stop()
            finally:
                try:
                    sup.stop()
                except Exception:
                    pass

    def test_inhibition_blocks_automatic_resume(self):
        from frigate_xdna.runtime.safety import inhibit
        with tempfile.TemporaryDirectory() as d:
            cfg = make_config(d)
            sup = Supervisor(
                cfg, fake_compile={"device_required": True,
                                   "device_held_by_worker": True})
            try:
                ref = local_onnx(d)
                out = sup.prepare(ref)
                sup.pump(0.05)
                sup.stop()
                sup2 = Supervisor(cfg)
                try:
                    inhibit(d, sup2.registry, "WORKER_DIED", ref)
                    sup2.prepare(ref)
                    # Same interrupted row: no new attempt while
                    # inhibited.
                    self.assertEqual(
                        sup2.registry.latest_job_for_ref(ref)["uuid"],
                        out["job_uuid"])
                finally:
                    sup2.stop()
            finally:
                try:
                    sup.stop()
                except Exception:
                    pass


if __name__ == "__main__":
    unittest.main()
