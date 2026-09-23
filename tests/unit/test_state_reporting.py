"""Unit tests: truthful state, health, waits, console (v0.1.1 checkpoint 4).

Boundary tests prove: a live worker reports ready; a dead daemon
produces a failing liveness exit; prepared/verified/active waits have
real predicates; console lines describe real transitions; preflight
fails fast with corrective actions.
"""
import hashlib
import io
import json
import os
import stat
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from unittest import mock

from frigate_xdna import cli as cli_mod
from frigate_xdna.config import Config
from frigate_xdna.errors import NOT_READY, SUCCESS
from frigate_xdna.observability.progress import (
    RecordingReporter,
    format_event,
)
from frigate_xdna.supervisor import Supervisor
from tests.integration.onnx_builders import make_raw_yolo

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "fixtures",
                       "fake_worker.py")


def chmod_fixture():
    st = os.stat(FIXTURE)
    os.chmod(FIXTURE, st.st_mode | stat.S_IXUSR)


def fixture_factory(*argv):
    from frigate_xdna.runtime.native import NativeWorker

    def make():
        return NativeWorker.spawn(FIXTURE, [], worker_argv=argv)
    return make


def make_config(tmp):
    return Config(data_dir=tmp, endpoint="tcp://127.0.0.1:5559")


def plant(sup, classes=46, seed=11):
    """Prepared artifact via registry rows (no compile)."""
    data = make_raw_yolo(os.path.join(sup.data_dir, "tmp.onnx"),
                         res=320, classes=classes, seed=seed)
    sha = hashlib.sha256(data).hexdigest()
    src_dir = os.path.join(sup.data_dir, "sources", sha)
    os.makedirs(src_dir, exist_ok=True)
    with open(os.path.join(src_dir, "model.onnx"), "wb") as f:
        f.write(data)
    ref = f"plus://model-{seed}"
    ckey = f"ck-{seed}-{classes}"
    sup.registry.upsert_ref(ref, "plus", f"model-{seed}")
    sup.registry.add_source(sha, len(data), "model.onnx", "test")
    sup.registry.set_ref_source(ref, sha, "md", "PREPARED")
    art_dir = os.path.join(sup.data_dir, "artifacts", ckey)
    os.makedirs(art_dir, exist_ok=True)
    with open(os.path.join(art_dir, "model.rai"), "wb") as f:
        f.write(b"RAI")
    sup.registry.add_artifact(ckey, sha, "a" * 64, 3, "bf16-vaiml-v1",
                              "stx-npu")
    sup.registry.execute(
        "INSERT INTO jobs(uuid, ref, compile_key, stage, attempt,"
        " created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
        (f"job-{ckey}", ref, ckey, "PREPARED", 1, time.time(),
         time.time()))
    return ref, ckey, sha


def run_health(data_dir, *argv):
    buf = io.StringIO()
    with mock.patch.dict(os.environ, {"FXDNA_DATA_DIR": data_dir},
                         clear=False):
        with redirect_stdout(buf):
            rc = cli_mod.main(["health", *argv])
    return rc, json.loads(buf.getvalue())


class TestHealthReadiness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        chmod_fixture()

    def test_no_worker_live_not_ready(self):
        with tempfile.TemporaryDirectory() as d:
            sup = Supervisor(make_config(d))
            try:
                sup.start_admin()
                rc, doc = run_health(d)
                self.assertEqual(rc, SUCCESS)
                self.assertTrue(doc["alive"])
                self.assertFalse(doc["ready"])
                rc, doc = run_health(d, "--ready")
                self.assertEqual(rc, NOT_READY)
                self.assertFalse(doc["ready"])
                self.assertIn("no active worker", doc["reason"])
            finally:
                sup.stop()

    def test_live_fixture_worker_is_ready(self):
        with tempfile.TemporaryDirectory() as d:
            sup = Supervisor(make_config(d),
                             worker_factory=fixture_factory())
            try:
                ref, ckey, _sha = plant(sup)
                self.assertTrue(sup.activate_worker(ckey))
                sup.start_admin()
                rc, doc = run_health(d, "--ready")
                self.assertEqual(rc, SUCCESS)
                self.assertTrue(doc["alive"])
                self.assertTrue(doc["ready"])
                self.assertEqual(doc["worker"]["generation"], 1)
                self.assertEqual(doc["worker"]["compile_key"], ckey)
                self.assertTrue(doc["worker"]["alive"])
                # The ref projects ACTIVE with a live worker.
                models = sup.status(ref)["models"]
                self.assertEqual(models[0]["state"], "ACTIVE")
                self.assertEqual(sup.status()["active"]["compile_key"],
                                 ckey)
            finally:
                sup.stop()

    def test_dead_worker_is_not_ready_but_daemon_live(self):
        with tempfile.TemporaryDirectory() as d:
            sup = Supervisor(make_config(d),
                             worker_factory=fixture_factory(
                                 "--exit", "3"))
            try:
                ref, ckey, _sha = plant(sup)
                # Activation fails honestly (spawned child exits).
                self.assertFalse(sup.activate_worker(ckey))
                sup.start_admin()
                rc, doc = run_health(d)
                self.assertEqual(rc, SUCCESS)
                self.assertTrue(doc["alive"])
                rc, doc = run_health(d, "--ready")
                self.assertEqual(rc, NOT_READY)
                self.assertFalse(doc["ready"])
            finally:
                sup.stop()

    def test_inhibition_blocks_readiness(self):
        from frigate_xdna.runtime.safety import inhibit
        with tempfile.TemporaryDirectory() as d:
            sup = Supervisor(make_config(d),
                             worker_factory=fixture_factory())
            try:
                _ref, ckey, _sha = plant(sup)
                self.assertTrue(sup.activate_worker(ckey))
                inhibit(d, sup.registry, "WORKER_DIED", "plus://x")
                sup.start_admin()
                rc, doc = run_health(d, "--ready")
                self.assertEqual(rc, NOT_READY)
                self.assertIn("inhibited", doc["reason"])
            finally:
                sup.stop()

    def test_stale_active_record_claims_nothing(self):
        """A persisted `active` record without a live worker (crash
        simulation) must not project ACTIVE or ready."""
        with tempfile.TemporaryDirectory() as d:
            sup = Supervisor(make_config(d))
            try:
                ref, ckey, _sha = plant(sup)
                sup.registry.set_state("active", {
                    "compile_key": ckey, "worker_generation": 9,
                    "serving_digest": "s" * 64})
                doc = sup.status(ref)
                self.assertIsNone(doc["active"])
                self.assertIsNone(doc["worker"])
                self.assertFalse(doc["ready"])
                self.assertEqual(doc["models"][0]["state"], "PREPARED")
            finally:
                sup.stop()


class TestWaitPredicates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        chmod_fixture()

    def _live(self, d, **kw):
        sup = Supervisor(make_config(d), **kw)
        sup.start_admin()
        return sup

    def _wait(self, d, ref, want, timeout=10.0):
        buf = io.StringIO()
        with mock.patch.dict(os.environ, {"FXDNA_DATA_DIR": d},
                             clear=False):
            with redirect_stdout(buf):
                rc = cli_mod.main(["wait", ref, "--state", want,
                                   "--timeout", str(timeout)])
        return rc

    def test_wait_prepared_verified_active_ranking(self):
        with tempfile.TemporaryDirectory() as d:
            sup = self._live(d, worker_factory=fixture_factory())
            try:
                ref, ckey, _sha = plant(sup)
                # PREPARED now; activation records verification and
                # projects ACTIVE — each lesser wait is satisfied.
                self.assertEqual(self._wait(d, ref, "prepared"), SUCCESS)
                self.assertTrue(sup.activate_worker(ckey))
                self.assertTrue(sup.registry.is_verified(ckey))
                self.assertEqual(self._wait(d, ref, "verified"), SUCCESS)
                self.assertEqual(self._wait(d, ref, "active"), SUCCESS)
            finally:
                sup.stop()

    def test_wait_terminal_maps_exit(self):
        with tempfile.TemporaryDirectory() as d:
            sup = self._live(d)
            try:
                ref = "plus://stuck"
                sup.registry.upsert_ref(ref, "plus", "stuck")
                sup.registry.set_ref_state(ref, "COMPILE_FAILED")
                rc = self._wait(d, ref, "prepared", timeout=2.0)
                self.assertEqual(rc, 6)
            finally:
                sup.stop()

    def test_wait_timeout_is_bounded(self):
        with tempfile.TemporaryDirectory() as d:
            sup = self._live(d)
            try:
                ref = "plus://queued"
                sup.registry.upsert_ref(ref, "plus", "queued")
                sup.registry.set_ref_state(ref, "QUEUED")
                t0 = time.monotonic()
                rc = self._wait(d, ref, "prepared", timeout=1.0)
                self.assertEqual(rc, NOT_READY)
                self.assertLess(time.monotonic() - t0, 10.0)
            finally:
                sup.stop()

    def test_activate_wait_honored(self):
        with tempfile.TemporaryDirectory() as d:
            sup = self._live(d, worker_factory=fixture_factory())
            try:
                ref, _ckey, _sha = plant(sup)
                buf = io.StringIO()
                with mock.patch.dict(os.environ, {"FXDNA_DATA_DIR": d},
                                     clear=False):
                    with redirect_stdout(buf):
                        rc = cli_mod.main(["activate", ref, "--wait",
                                           "--timeout", "10"])
                self.assertEqual(rc, SUCCESS)
                models = sup.status(ref)["models"]
                self.assertEqual(models[0]["state"], "ACTIVE")
            finally:
                sup.stop()


class TestConsoleReporting(unittest.TestCase):
    def test_prepare_and_prepared_events(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = make_config(d)
            rec = RecordingReporter()
            sup = Supervisor(cfg, fake_compile={"device_required": False},
                             reporter=rec, heartbeat_s=3600.0)
            try:
                path = os.path.join(d, "m.onnx")
                make_raw_yolo(path, res=320, classes=4, seed=3)
                out = sup.prepare(path)
                sup.pump(0.2)
                sup.report_progress()
                kinds = rec.kinds()
                self.assertIn("inspection_complete", kinds)
                self.assertIn("preparing_model", kinds)
                insp = next(e for e in rec.events
                            if e["kind"] == "inspection_complete")
                self.assertEqual(insp["profile"], "yolo-raw")
                self.assertFalse(out["cache_hit"])
                # Fake backend needs a few pumps to finish.
                deadline = time.monotonic() + 10.0
                while sup.registry.get_ref(out["ref"])["state"] != \
                        "PREPARED" and time.monotonic() < deadline:
                    sup.pump(0.1)
                    sup.report_progress()
                    time.sleep(0.05)
                kinds = rec.kinds()
                self.assertIn("model_prepared", kinds)
                prep = next(e for e in rec.events
                            if e["kind"] == "model_prepared")
                self.assertIn("127.0.0.1:5559", prep["endpoint"])
            finally:
                sup.stop()

    def test_heartbeat_during_slow_compile(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = make_config(d)
            rec = RecordingReporter()
            sup = Supervisor(
                cfg, fake_compile={"device_required": False,
                                   "duration_s": 0.6},
                reporter=rec, heartbeat_s=0.05)
            try:
                path = os.path.join(d, "m.onnx")
                make_raw_yolo(path, res=320, classes=4, seed=4)
                sup.prepare(path)
                deadline = time.monotonic() + 10.0
                while sup.registry.latest_job_for_ref(path)["stage"] \
                        not in ("PREPARED", "COMPILE_FAILED") \
                        and time.monotonic() < deadline:
                    sup.pump(0.05)
                    sup.report_progress()
                    time.sleep(0.02)
                beats = [e for e in rec.events
                         if e["kind"] == "heartbeat"]
                self.assertTrue(beats, rec.kinds())
                self.assertIn("elapsed", format_event(beats[0]))
            finally:
                sup.stop()

    def test_failure_event_carries_code(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = make_config(d)
            rec = RecordingReporter()
            sup = Supervisor(
                cfg, fake_compile={"device_required": False,
                                   "succeed": False,
                                   "fail_state": "COMPILE_FAILED"},
                reporter=rec, heartbeat_s=3600.0)
            try:
                path = os.path.join(d, "m.onnx")
                make_raw_yolo(path, res=320, classes=4, seed=5)
                sup.prepare(path)
                deadline = time.monotonic() + 10.0
                while sup.registry.latest_job_for_ref(path)["stage"] \
                        not in ("PREPARED", "COMPILE_FAILED") \
                        and time.monotonic() < deadline:
                    sup.pump(0.05)
                    sup.report_progress()
                    time.sleep(0.02)
                fails = [e for e in rec.events
                         if e["kind"] == "preparation_failed"]
                self.assertTrue(fails, rec.kinds())
                self.assertEqual(fails[0]["code"], "COMPILE_FAILED")
                line = format_event(fails[0])
                self.assertIn("COMPILE_FAILED", line)
                self.assertIn("fxdna status", line)
            finally:
                sup.stop()

    def test_no_reporter_is_silent(self):
        with tempfile.TemporaryDirectory() as d:
            sup = Supervisor(make_config(d),
                             fake_compile={"device_required": False})
            try:
                path = os.path.join(d, "m.onnx")
                make_raw_yolo(path, res=320, classes=4, seed=6)
                sup.prepare(path)
                sup.pump(0.1)
                sup.report_progress()  # must not raise
            finally:
                sup.stop()


class TestPreflight(unittest.TestCase):
    def test_memlock_constrained_fails_with_action(self):
        import resource

        from frigate_xdna import deploy_checks
        with mock.patch.object(resource, "getrlimit",
                               return_value=(8 * 1024 * 1024,
                                             8 * 1024 * 1024)):
            check = deploy_checks.check_memlock()
        self.assertFalse(check["ok"])
        self.assertIn("ulimits", check["message"])
        self.assertIn("memlock", check["message"])
        self.assertEqual(check["code"], 6)

    def test_memlock_sufficient_passes(self):
        import resource

        from frigate_xdna import deploy_checks
        with mock.patch.object(
                resource, "getrlimit",
                return_value=(resource.RLIM_INFINITY,
                              resource.RLIM_INFINITY)):
            check = deploy_checks.check_memlock()
        self.assertTrue(check["ok"])

    def test_key_readability_bits(self):
        from frigate_xdna.deploy_checks import _uid_can_read
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "key")
            with open(path, "w") as f:
                f.write("k")
            os.chmod(path, 0o600)
            # Owner reads; another UID with no group/other bits fails —
            # the documented root-owned-0600 rule for UID 10001.
            self.assertTrue(_uid_can_read(path, os.geteuid(),
                                          os.getegid(), ()))
            self.assertFalse(_uid_can_read(path, 10001, 10001, ()))
            os.chmod(path, 0o644)
            self.assertTrue(_uid_can_read(path, 10001, 10001, ()))

    def test_missing_device_fails(self):
        from frigate_xdna import deploy_checks
        check = deploy_checks.check_device("/nonexistent-accel-xyz")
        self.assertFalse(check["ok"])
        self.assertEqual(check["code"], 8)
        self.assertIn("FXDNA_DEVICE", check["message"])

    def test_plus_without_credential_fails(self):
        from frigate_xdna import deploy_checks
        cfg = Config(data_dir="/tmp", models=("plus://abc",))
        check = deploy_checks.check_plus_credential(cfg)
        self.assertFalse(check["ok"])
        self.assertEqual(check["code"], 4)

    def test_config_error_not_masked_by_environment(self):
        """Constrained runners (e.g. stock CI memlock) must still
        surface the fixable credential error first, with its code."""
        import resource

        from frigate_xdna import deploy_checks
        with tempfile.TemporaryDirectory() as d:
            cfg = Config(data_dir=d, models=("plus://abc",))
            with mock.patch.object(
                    resource, "getrlimit",
                    return_value=(8 * 1024 * 1024, 8 * 1024 * 1024)):
                checks = deploy_checks.run_preflight(cfg)
            bad = deploy_checks.blocking_failures(checks)
            self.assertGreaterEqual(len(bad), 2)
            self.assertEqual(bad[0]["name"], "plus-credential")
            self.assertEqual(bad[0]["code"], 4)

    def test_doctor_lists_preflight_checks(self):
        buf = io.StringIO()
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.dict(os.environ, {"FXDNA_DATA_DIR": d},
                                 clear=False):
                with redirect_stdout(buf):
                    rc = cli_mod.main(["doctor"])
        self.assertEqual(rc, SUCCESS)
        names = [c["name"] for c in json.loads(buf.getvalue())["checks"]]
        for want in ("memlock", "data-dir", "device", "key-file",
                     "plus-credential", "models-configured"):
            self.assertIn(want, names)


if __name__ == "__main__":
    unittest.main()
