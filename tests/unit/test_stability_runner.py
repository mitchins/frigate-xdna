"""Unit tests: stability runner (no hardware).

The runner must exercise the production Supervisor path with a fake
native child (same seam as worker-supervision tests): journal
durability contract, phase escalation, refusal to start after an
unacknowledged interrupted run, no compile, stop-on-first-failure.
Scale kwargs shrink calibration/steady durations; production
defaults are pinned separately.
"""
import json
import os
import tempfile
import time
import unittest

from frigate_xdna.config import Config
from frigate_xdna.errors import DEVICE_UNAVAILABLE, NOT_READY, SUCCESS
from frigate_xdna.observability import stability
from frigate_xdna.stability_runner import (
    CAL_MEASURED,
    CAL_WARMUP,
    DEFERRED_PROFILES,
    STEADY_DURATION_S,
    run_stability,
)
from frigate_xdna.supervisor import Supervisor
from tests.integration.onnx_builders import make_raw_yolo


class FakeChild:
    """NativeWorker control interface (same as supervision tests)."""

    def __init__(self, fail_load=None, fail_infer_after=None,
                 timeout_after=None):
        self.fail_load = fail_load
        self.fail_infer_after = fail_infer_after
        self.timeout_after = timeout_after
        self.loaded = False
        self.generation = 0
        self.retired = False
        self.infers = 0

    def alive(self):
        return not self.retired

    def load(self, artifact_path, generation, serving_digest,
             class_count, timeout_s=25.0):
        if self.fail_load is not None:
            from frigate_xdna.runtime.native import WorkerError
            raise WorkerError(*self.fail_load)
        self.generation = generation
        self.loaded = True

    def infer(self, payload, shape, generation, timeout_s):
        from frigate_xdna.runtime.native import WorkerError
        self.infers += 1
        if (self.fail_infer_after is not None
                and self.infers > self.fail_infer_after):
            raise WorkerError("WORKER_IO", "scripted failure")
        if (self.timeout_after is not None
                and self.infers > self.timeout_after):
            raise WorkerError("TIMEOUT", "scripted slow")
        from frigate_xdna.runtime.native import RESULT_BYTES
        return bytes(RESULT_BYTES)

    def retire(self):
        self.retired = True


class RunnerHarness(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="fx-stabrun-")
        self.lines: list[str] = []
        self.children: list[FakeChild] = []

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _config(self, **kw):
        return Config(data_dir=self.tmp,
                      endpoint="tcp://127.0.0.1:5555", **kw)

    def _factory(self, **child_kw):
        def factory():
            child = FakeChild(**child_kw)
            self.children.append(child)
            return child
        return factory

    def _plant(self, sup, ref="plus://model-7", classes=46):
        data = make_raw_yolo(os.path.join(sup.data_dir, "tmp.onnx"),
                             res=320, classes=classes, seed=7)
        import hashlib
        sha = hashlib.sha256(data).hexdigest()
        src_dir = os.path.join(sup.data_dir, "sources", sha)
        os.makedirs(src_dir, exist_ok=True)
        with open(os.path.join(src_dir, "model.onnx"), "wb") as f:
            f.write(data)
        ckey = "ck-runner-1"
        sup.registry.upsert_ref(ref, "plus", "model-7")
        sup.registry.add_source(sha, len(data), "model.onnx", "test")
        sup.registry.set_ref_source(ref, sha, "md", "PREPARED")
        art_dir = os.path.join(sup.data_dir, "artifacts", ckey)
        os.makedirs(art_dir, exist_ok=True)
        with open(os.path.join(art_dir, "model.rai"), "wb") as f:
            f.write(b"RAI")
        sup.registry.add_artifact(ckey, sha, "a" * 64, 3,
                                  "bf16-vaiml-v1", "stx-npu")
        sup.registry.execute(
            "INSERT INTO jobs(uuid, ref, compile_key, stage, attempt,"
            " created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
            (f"job-{ckey}", ref, ckey, "PREPARED", 1,
             time.time(), time.time()))
        return ref

    def _run(self, config, ref, profile="smoke", **kw):
        # Patch Supervisor's worker factory through a subclass-free
        # seam: run_stability constructs Supervisor(config) itself,
        # so inject via worker_bin? No — monkeypatch the spawn seam
        # the same way the supervision tests do (worker_factory is a
        # Supervisor ctor arg), by patching _spawn_worker is wrong;
        # instead patch Supervisor.__init__ default through a wrapper.
        original_init = Supervisor.__init__
        factory = kw.pop("factory", None) or self._factory()

        def init_patch(self, cfg, **init_kw):
            if factory is not None:
                init_kw.setdefault("worker_factory", factory)
            original_init(self, cfg, **init_kw)

        Supervisor.__init__ = init_patch
        try:
            return run_stability(
                config, ref=ref, profile=profile,
                console=self.lines.append,
                steady_duration_s=kw.pop("steady_duration_s", 0.6),
                cal_warmup=kw.pop("cal_warmup", 2),
                cal_measured=kw.pop("cal_measured", 4),
                **kw)
        finally:
            Supervisor.__init__ = original_init


class TestHappyPaths(RunnerHarness):
    def test_smoke_pass_writes_full_journal(self):
        cfg = self._config()
        sup = Supervisor(cfg, worker_factory=self._factory())
        try:
            ref = self._plant(sup)
        finally:
            sup.stop()
        rc = self._run(cfg, ref, profile="smoke")
        self.assertEqual(rc, SUCCESS)
        rid = stability.latest_run(self.tmp)
        self.assertIsNotNone(rid)
        d = stability.run_dir(self.tmp, rid)
        for name in (stability.METADATA, stability.CURRENT,
                     stability.TIMELINE, stability.RESULT):
            self.assertTrue(os.path.isfile(os.path.join(d, name)),
                            name)
        report = stability.load_report(self.tmp)
        self.assertIsNone(report["interrupted"])
        self.assertEqual(report["result"]["outcome"], "PASS")
        self.assertEqual(report["result"]["last_phase"], "H2")
        self.assertTrue(any("E0 ENVIRONMENT   PASS" in p
                            for p in report["result"]["phases"]))
        self.assertTrue(any("H2 CALIBRATE" in p
                            for p in report["result"]["phases"]))
        # Timeline: matched STARTED/COMPLETED for RUN and each phase.
        tl = stability.read_timeline(self.tmp, rid)
        self.assertEqual(tl[0]["step"], "RUN")
        self.assertEqual(tl[0]["state"], "STARTED")
        self.assertEqual(tl[-1]["state"], "COMPLETED")
        self.assertEqual(tl[-1]["step"], "RUN")
        # Inferences: 2 warmup + 4 measured + 1 H1 probe.
        self.assertEqual(self.children[-1].infers, 7)
        # current.json carries the §9.2 checkpoint fields.
        cur = json.load(open(os.path.join(d, stability.CURRENT)))
        for key in ("phase", "submitted", "completed", "errors",
                    "timeouts", "worker_generation", "runtime_status"):
            self.assertIn(key, cur)
        # Metadata pins model identity + host snapshot.
        meta = report["metadata"]
        self.assertEqual(meta["model"]["compile_key"], "ck-runner-1")
        self.assertIn("host_info", meta)

    def test_gentle_pass_runs_h3(self):
        cfg = self._config()
        sup = Supervisor(cfg, worker_factory=self._factory())
        try:
            ref = self._plant(sup)
        finally:
            sup.stop()
        rc = self._run(cfg, ref, profile="gentle")
        self.assertEqual(rc, SUCCESS)
        report = stability.load_report(self.tmp)
        self.assertEqual(report["result"]["last_phase"], "H3")
        self.assertTrue(any("H3 LOW_STEADY    PASS" in p
                            for p in report["result"]["phases"]))
        # Steady phase submitted at least a few paced requests.
        self.assertGreaterEqual(report["result"]["completed"], 7)
        tl = stability.read_timeline(self.tmp, report["run_id"])
        h3_start = [r for r in tl if r["phase"] == "H3"
                    and r["state"] == "STARTED"]
        self.assertTrue(h3_start)
        self.assertIn("target_req_s", h3_start[0])
        # Console output rendered phase lines.
        joined = "\n".join(self.lines)
        self.assertIn("H2 CALIBRATE", joined)
        self.assertIn("H3", joined)


class TestRefusals(RunnerHarness):
    def test_deferred_profiles_refused(self):
        for profile in DEFERRED_PROFILES:
            self.lines.clear()
            rc = self._run(self._config(), "plus://x",
                           profile=profile)
            self.assertEqual(rc, NOT_READY)
            self.assertIn("deferred", "\n".join(self.lines))

    def test_not_prepared_message(self):
        cfg = self._config()
        rc = self._run(cfg, "plus://never-prepared", profile="smoke")
        self.assertEqual(rc, NOT_READY)
        joined = "\n".join(self.lines)
        self.assertIn("MODEL_NOT_PREPARED", joined)
        self.assertIn("fxdna prepare plus://never-prepared --wait",
                      joined)
        report = stability.load_report(self.tmp)
        self.assertEqual(report["result"]["error_code"],
                         "NOT_PREPARED")

    def test_unacknowledged_interrupted_run_blocks_start(self):
        # Fabricate a prior interrupted run (STARTED, no result).
        j = stability.StabilityJournal(self.tmp,
                                       "2026-01-01T00-00-00-old")
        j.metadata({"schema_version": 1, "run_id": "old",
                    "profile": "gentle", "model": {}})
        j.timeline("RUN", "RUN", "STARTED")
        j.timeline("H3", "PHASE", "STARTED")
        j.current({"phase": "H3", "submitted": 5, "completed": 4})
        cfg = self._config()
        rc = self._run(cfg, "plus://x", profile="smoke")
        self.assertEqual(rc, DEVICE_UNAVAILABLE)
        joined = "\n".join(self.lines)
        self.assertIn("PREVIOUS STABILITY RUN INTERRUPTED", joined)
        self.assertIn("No automatic continuation", joined)
        # No new run directory appeared.
        self.assertEqual(stability.list_runs(self.tmp),
                         ["2026-01-01T00-00-00-old"])
        # Acknowledgement unblocks (run then proceeds to its own
        # E0 FAIL for the unprepared ref).
        stability.acknowledge_run(self.tmp, "reviewed")
        self.lines.clear()
        rc = self._run(cfg, "plus://x", profile="smoke")
        self.assertEqual(rc, NOT_READY)
        self.assertIn("MODEL_NOT_PREPARED", "\n".join(self.lines))

    def test_configured_needs_exactly_one_model(self):
        rc = self._run(self._config(), None, profile="smoke",
                       configured=True)
        self.assertEqual(rc, 2)
        rc = self._run(Config(data_dir=self.tmp, models=("a", "b"),
                              endpoint="tcp://127.0.0.1:5555"),
                       None, profile="smoke", configured=True)
        self.assertEqual(rc, 2)

    def test_missing_ref_is_invalid_args(self):
        rc = self._run(self._config(), None, profile="smoke")
        self.assertEqual(rc, 2)


class TestFailures(RunnerHarness):
    def test_activation_failure_stops_escalation(self):
        cfg = self._config()
        sup = Supervisor(cfg,
                         worker_factory=self._factory(
                             fail_load=("WORKER_LOAD", "scripted")))
        try:
            ref = self._plant(sup)
        finally:
            sup.stop()
        rc = self._run(cfg, ref, profile="gentle",
                       factory=self._factory(
                           fail_load=("WORKER_LOAD", "scripted")))
        self.assertEqual(rc, DEVICE_UNAVAILABLE)
        report = stability.load_report(self.tmp)
        self.assertEqual(report["result"]["outcome"], "FAIL")
        self.assertEqual(report["result"]["error_code"],
                         "ACTIVATION_FAILED")
        self.assertEqual(report["result"]["last_phase"], "H1")
        # Exactly one worker spawn: no autoloop, no escalation.
        self.assertEqual(len(self.children), 1)

    def test_infer_failure_aborts_with_counts(self):
        cfg = self._config()
        sup = Supervisor(cfg, worker_factory=self._factory())
        try:
            ref = self._plant(sup)
        finally:
            sup.stop()
        rc = self._run(cfg, ref, profile="gentle",
                       factory=self._factory(fail_infer_after=6),
                       steady_duration_s=5.0)
        self.assertEqual(rc, DEVICE_UNAVAILABLE)
        report = stability.load_report(self.tmp)
        self.assertEqual(report["result"]["outcome"], "FAIL")
        self.assertEqual(report["result"]["errors"], 1)
        self.assertGreaterEqual(report["result"]["completed"], 6)
        # Failed run still has a deliberate result (not interrupted).
        self.assertIsNone(report["interrupted"])

    def test_ipc_timeout_counts_as_timeout_not_error(self):
        cfg = self._config()
        sup = Supervisor(cfg, worker_factory=self._factory())
        try:
            ref = self._plant(sup)
        finally:
            sup.stop()
        rc = self._run(cfg, ref, profile="smoke",
                       factory=self._factory(timeout_after=1),
                       steady_duration_s=5.0)
        self.assertEqual(rc, DEVICE_UNAVAILABLE)
        report = stability.load_report(self.tmp)
        doc = report["result"]
        self.assertEqual(doc["error_code"], "TIMEOUT")
        self.assertEqual(doc["timeouts"], 1)
        self.assertEqual(doc["errors"], 0)
        # A bounded receive expiring must not drop the worker for a
        # single slow request (SPEC §6): no inhibition recorded.
        self.assertIsNone(report["metadata"].get("inhibition"))

    def test_h2_operator_stop_finishes_interrupted(self):
        from frigate_xdna.observability.stability import StabilityJournal
        from frigate_xdna.stability_runner import _Run, _StopFlag

        class SlowSup:
            config = None
            _worker = None
            _worker_generation = 0

            def __init__(self, stop):
                self.stop = stop
                self.calls = 0

            def worker_infer(self, payload, shape, timeout_s):
                self.calls += 1
                if self.calls >= 2:
                    self.stop.stop = True
                return ("ok", b"x" * 480)

        stop = _StopFlag()
        j = StabilityJournal(self.tmp, "2026-01-02T00-00-00-stop")
        run = _Run(self._config(), SlowSup(stop), j,
                   "gentle", (1, 3, 320, 320), b"t",
                   self.lines.append, stop, 0.6, 2, 8)
        err = run.phase_h2()
        self.assertEqual(err, "INTERRUPTED")
        self.assertTrue(any("operator stop" in p for p in run.phases))


class TestProductionDefaults(unittest.TestCase):
    def test_scale_constants_pinned(self):
        self.assertEqual(CAL_WARMUP, 5)
        self.assertEqual(CAL_MEASURED, 25)
        self.assertEqual(STEADY_DURATION_S, 120.0)


if __name__ == "__main__":
    unittest.main()
