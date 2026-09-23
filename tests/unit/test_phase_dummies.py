"""Unit tests: scripted vendor-boundary outcomes (v0.1.1 checkpoint 3).

Dummy phase executables honor the exact invocation/output/artifact
boundary the real launcher consumes (same argv, same marker lines,
same artifact coordinates) while scripting success, delay, nonzero
exit, timeout, crash, malformed markers, missing/truncated artifact,
validation failure, and the field FlexMLRT mmap failure on stdout as
well as stderr. The REAL launcher subprocess path runs (no spawn
mocking here); the REAL probe worker path runs through scripted
NativeWorker IPC; one test drives the REAL Supervisor prepare/pump/
publish path with dummy prefixes.

Dummies are test-only: behavior travels in a behavior.json sidecar
(the child env is allowlisted, so no knobs pass through environ),
every generated file carries the FXDNA-TEST-FIXTURE marker, and dummy
artifacts start with FXDNA-TEST-ARTIFACT magic the real validator
would never bless.
"""
import hashlib
import json
import os
import stat
import sys
import tempfile
import time
import unittest

from frigate_xdna.compiler.launcher import CompilerPrefixes, run_compile
from tests.integration.onnx_builders import make_raw_yolo

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

FLEXMLRT_FAILURE = (
    "Compilation Complete\n"
    "some vendor chatter\n"
    "FlexMLRT Exception:\n"
    "mmap(... flags=8209 ...) failed (err=-11):\n"
    "Resource temporarily unavailable\n"
)

DUMMY_COMMON = '''#!/usr/bin/env python3
"""FXDNA-TEST-FIXTURE: scripted compiler phase (tests only, never shipped)."""
import json
import os
import signal
import sys
import time


def args(argv):
    out = {}
    i = 1
    while i < len(argv):
        if argv[i].startswith("--") and i + 1 < len(argv):
            out[argv[i][2:]] = argv[i + 1]
            i += 2
        else:
            i += 1
    return out


def main():
    me = os.path.splitext(os.path.basename(__file__))[0]
    with open(os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "behavior.json")) as f:
        behav = json.load(f).get(me, {})
    time.sleep(float(behav.get("sleep", 0)))
    if behav.get("crash") == "sigsegv":
        os.kill(os.getpid(), signal.SIGSEGV)
    for line in behav.get("stdout", []):
        print(line, flush=True)
    for line in behav.get("stderr", []):
        print(line, file=sys.stderr, flush=True)
    produce(me, behav)
    sys.exit(int(behav.get("rc", 0)))


def produce(me, behav):
    opts = args(sys.argv)
    if me == "prepare" and "out" in opts:
        with open(opts["out"], "wb") as f:
            f.write(b"FXDNA-TEST-FIXTURE-bf16")
    if me == "compile" and behav.get("write_artifact"):
        import hashlib as _hashlib
        size = int(behav.get("artifact_bytes", 64))
        dest = os.path.join(opts["cache-dir"], opts["cache-key"])
        os.makedirs(dest, exist_ok=True)
        body = b"FXDNA-TEST-ARTIFACT:" + b"T" * max(
            0, size - len(b"FXDNA-TEST-ARTIFACT:"))
        with open(os.path.join(
                dest, opts["cache-key"] + ".rai"), "wb") as f:
            f.write(body)
        if behav.get("marker") == "auto":
            print("COMPILE_OK %s %d" % (
                _hashlib.sha256(body).hexdigest(), len(body)),
                flush=True)


main()
'''

DUMMY_SCRIPT = DUMMY_COMMON


def write_recipe(workdir, behavior):
    recipe = os.path.join(workdir, "recipe")
    os.makedirs(recipe, exist_ok=True)
    for name in ("prepare.py", "compile.py", "validate.py"):
        path = os.path.join(recipe, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(DUMMY_SCRIPT)
        st = os.stat(path)
        os.chmod(path, st.st_mode | stat.S_IXUSR)
    with open(os.path.join(recipe, "behavior.json"), "w",
              encoding="utf-8") as f:
        json.dump(behavior, f)
    return recipe


def prefixes(workdir, behavior):
    recipe = write_recipe(workdir, behavior)
    sp = os.path.join(workdir, "sp")
    os.makedirs(sp, exist_ok=True)
    return CompilerPrefixes(
        quant_python=sys.executable,
        compile_python=sys.executable,
        compile_lib=sp,
        xrt_lib=os.path.join(workdir, "xrt"),
        xrt_root=os.path.join(workdir, "xrtroot"),
        recipe_dir=recipe,
        calib_dir=os.path.join(workdir, "calib"),
        vaiml_config=os.path.join(workdir, "vaiml_config.json"))


OK_SHA_BEHAVIOR = {
    "prepare": {"stdout": ["BF16_PREPARE_OK " + "b" * 64]},
    "compile": {"write_artifact": True, "artifact_bytes": 64,
                "marker": "auto"},
    "validate": {},
}


def yolo_source(workdir, classes=8, seed=5):
    path = os.path.join(workdir, "model.onnx")
    make_raw_yolo(path, res=320, classes=classes, seed=seed)
    return path


def fixture_worker_factory(*argv):
    """Real NativeWorker control plane over the scripted fixture."""
    from frigate_xdna.runtime.native import NativeWorker
    fixture = os.path.join(os.path.dirname(__file__), "..", "fixtures",
                           "fake_worker.py")
    st = os.stat(fixture)
    os.chmod(fixture, st.st_mode | stat.S_IXUSR)

    def make():
        return NativeWorker.spawn(fixture, [], worker_argv=argv)
    return make


class TestScriptedPhases(unittest.TestCase):
    def test_success_end_to_end_with_real_probe(self):
        with tempfile.TemporaryDirectory() as d:
            path = yolo_source(d)
            res = run_compile(
                prefixes(d, OK_SHA_BEHAVIOR), path, d, "ck",
                timeout_s=120.0, data_dir=d,
                worker_factory=fixture_worker_factory())
            self.assertEqual(res.returncode, 0)
            self.assertEqual(res.error, "")
            with open(res.rai_path, "rb") as f:
                body = f.read()
            self.assertEqual(hashlib.sha256(body).hexdigest(),
                             res.rai_sha256)
            self.assertEqual(len(body), res.rai_bytes)
            self.assertTrue(body.startswith(b"FXDNA-TEST-ARTIFACT:"))

    def test_delayed_completion_still_succeeds(self):
        behavior = dict(OK_SHA_BEHAVIOR)
        behavior["compile"] = dict(OK_SHA_BEHAVIOR["compile"], sleep=3)
        with tempfile.TemporaryDirectory() as d:
            path = yolo_source(d)
            res = run_compile(prefixes(d, behavior), path, d, "ck",
                              timeout_s=120.0)
        self.assertEqual(res.returncode, 0)
        self.assertGreaterEqual(res.wall_s, 3.0)

    def test_phase1_nonzero_exit(self):
        behavior = {"prepare": {"rc": 2, "stderr": ["quant blew up"]},
                    "compile": {}, "validate": {}}
        with tempfile.TemporaryDirectory() as d:
            res = run_compile(prefixes(d, behavior), yolo_source(d),
                              d, "ck", timeout_s=60.0)
        self.assertEqual(res.returncode, 2)
        self.assertEqual(res.error, "bf16-prepare failed")
        self.assertIn("quant blew up", res.detail)

    def test_phase2_nonzero_exit(self):
        behavior = dict(OK_SHA_BEHAVIOR)
        behavior["compile"] = {"rc": 3, "stderr": ["vaiml blew up"]}
        with tempfile.TemporaryDirectory() as d:
            res = run_compile(prefixes(d, behavior), yolo_source(d),
                              d, "ck", timeout_s=60.0)
        self.assertEqual(res.returncode, 3)
        self.assertEqual(res.error, "vaiml-compile failed")
        self.assertIn("vaiml blew up", res.detail)

    def test_validate_nonzero_exit(self):
        behavior = dict(OK_SHA_BEHAVIOR)
        behavior["validate"] = {"rc": 1, "stdout": ["shape mismatch"]}
        with tempfile.TemporaryDirectory() as d:
            res = run_compile(prefixes(d, behavior), yolo_source(d),
                              d, "ck", timeout_s=60.0)
        self.assertEqual(res.returncode, 1)
        self.assertEqual(res.error, "validate failed")
        self.assertIn("shape mismatch", res.detail)

    def test_timeout_is_124(self):
        behavior = dict(OK_SHA_BEHAVIOR)
        behavior["compile"] = dict(behavior["compile"], sleep=30)
        with tempfile.TemporaryDirectory() as d:
            res = run_compile(prefixes(d, behavior), yolo_source(d),
                              d, "ck", timeout_s=4.0)
        self.assertEqual(res.returncode, 124)
        self.assertIn("timeout", res.error)

    def test_crash_is_terminal(self):
        behavior = dict(OK_SHA_BEHAVIOR)
        behavior["compile"] = {"crash": "sigsegv"}
        with tempfile.TemporaryDirectory() as d:
            res = run_compile(prefixes(d, behavior), yolo_source(d),
                              d, "ck", timeout_s=60.0)
        self.assertNotEqual(res.returncode, 0)
        self.assertEqual(res.error, "vaiml-compile failed")


class TestMarkerEnforcement(unittest.TestCase):
    def test_prepare_rc0_without_marker_fails(self):
        behavior = {"prepare": {"stdout": ["all done, no marker"]},
                    "compile": {}, "validate": {}}
        with tempfile.TemporaryDirectory() as d:
            res = run_compile(prefixes(d, behavior), yolo_source(d),
                              d, "ck", timeout_s=60.0)
        self.assertNotEqual(res.returncode, 0)
        self.assertEqual(res.error, "bf16-prepare marker missing")

    def test_prepare_garbage_marker_fails(self):
        behavior = {"prepare": {"stdout": ["BF16_PREPARE_OK nope"]},
                    "compile": {}, "validate": {}}
        with tempfile.TemporaryDirectory() as d:
            res = run_compile(prefixes(d, behavior), yolo_source(d),
                              d, "ck", timeout_s=60.0)
        self.assertEqual(res.error, "bf16-prepare marker missing")

    def test_complete_line_without_coordinates_fails(self):
        behavior = dict(OK_SHA_BEHAVIOR)
        behavior["compile"] = {"stdout": ["Compilation Complete"]}
        with tempfile.TemporaryDirectory() as d:
            res = run_compile(prefixes(d, behavior), yolo_source(d),
                              d, "ck", timeout_s=60.0)
        self.assertEqual(res.error, "vaiml-compile marker missing")

    def test_malformed_coordinates_fail_without_exception(self):
        behavior = dict(OK_SHA_BEHAVIOR)
        behavior["compile"] = {
            "stdout": ["COMPILE_OK not-a-sha not-a-number"]}
        with tempfile.TemporaryDirectory() as d:
            res = run_compile(prefixes(d, behavior), yolo_source(d),
                              d, "ck", timeout_s=60.0)
        self.assertEqual(res.error, "vaiml-compile marker missing")

    def test_marker_without_artifact_fails(self):
        behavior = dict(OK_SHA_BEHAVIOR)
        behavior["compile"] = {"stdout": ["COMPILE_OK " + "c" * 64
                                          + " 64"]}
        with tempfile.TemporaryDirectory() as d:
            res = run_compile(prefixes(d, behavior), yolo_source(d),
                              d, "ck", timeout_s=60.0)
        self.assertEqual(res.error, "vaiml-compile artifact missing")

    def test_truncated_artifact_fails(self):
        behavior = dict(OK_SHA_BEHAVIOR)
        behavior["compile"] = {"stdout": ["COMPILE_OK " + "c" * 64
                                          + " 1024"],
                               "write_artifact": True, "artifact_bytes": 30}
        with tempfile.TemporaryDirectory() as d:
            res = run_compile(prefixes(d, behavior), yolo_source(d),
                              d, "ck", timeout_s=60.0)
        self.assertEqual(res.returncode, 1)
        self.assertIn("truncated", res.error)
        self.assertIn("got 30 want 1024", res.error)

    def test_complete_line_with_valid_artifact_succeeds(self):
        behavior = dict(OK_SHA_BEHAVIOR)
        behavior["compile"] = {"stdout": ["Compilation Complete"],
                               "write_artifact": True, "artifact_bytes": 64,
                               "marker": "auto"}
        with tempfile.TemporaryDirectory() as d:
            res = run_compile(prefixes(d, behavior), yolo_source(d),
                              d, "ck", timeout_s=60.0)
        self.assertEqual(res.returncode, 0)

    def test_hash_mismatch_fails(self):
        behavior = dict(OK_SHA_BEHAVIOR)
        behavior["compile"] = {"stdout": ["COMPILE_OK " + "c" * 64
                                          + " 64"],
                               "write_artifact": True, "artifact_bytes": 64}
        with tempfile.TemporaryDirectory() as d:
            res = run_compile(prefixes(d, behavior), yolo_source(d),
                              d, "ck", timeout_s=60.0)
        self.assertEqual(res.returncode, 1)
        self.assertEqual(res.error,
                         "vaiml-compile artifact hash mismatch")


class TestFlexmlrtFieldFailure(unittest.TestCase):
    def test_stdout_pattern_is_terminal_with_detail(self):
        behavior = dict(OK_SHA_BEHAVIOR)
        behavior["compile"] = {"rc": 3,
                               "stdout": FLEXMLRT_FAILURE.splitlines()}
        with tempfile.TemporaryDirectory() as d:
            res = run_compile(prefixes(d, behavior), yolo_source(d),
                              d, "ck", timeout_s=60.0)
        self.assertNotEqual(res.returncode, 0)
        self.assertEqual(res.error, "vaiml-compile failed")
        self.assertIn("flags=8209", res.detail)
        self.assertIn("Resource temporarily unavailable", res.detail)

    def test_stderr_pattern_is_terminal_with_detail(self):
        behavior = dict(OK_SHA_BEHAVIOR)
        behavior["compile"] = {"rc": 3,
                               "stderr": FLEXMLRT_FAILURE.splitlines()}
        with tempfile.TemporaryDirectory() as d:
            res = run_compile(prefixes(d, behavior), yolo_source(d),
                              d, "ck", timeout_s=60.0)
        self.assertNotEqual(res.returncode, 0)
        self.assertEqual(res.error, "vaiml-compile failed")
        self.assertIn("err=-11", res.detail)


class TestFixtureHygiene(unittest.TestCase):
    def test_all_fixtures_carry_the_marker(self):
        """Every scripted-vendor file must be greppable so the
        release sanity gate can refuse an image containing one."""
        for name in ("fake_worker.py",):
            with open(os.path.join(REPO_ROOT, "tests", "fixtures",
                                   name), encoding="utf-8") as f:
                body = f.read()
            self.assertIn("FXDNA-TEST-FIXTURE", body, name)

    def test_dummy_template_carries_the_marker(self):
        self.assertIn("FXDNA-TEST-FIXTURE", DUMMY_SCRIPT)

    def test_production_tree_has_no_fixture_marker(self):
        src = os.path.join(REPO_ROOT, "src")
        offenders = []
        for root, _dirs, files in os.walk(src):
            if "__pycache__" in root:
                continue
            for name in files:
                if not name.endswith(".py"):
                    continue
                path = os.path.join(root, name)
                with open(path, encoding="utf-8") as f:
                    if "FXDNA-TEST-FIXTURE" in f.read():
                        offenders.append(path)
        self.assertEqual(offenders, [])


class TestSupervisorWithDummyBackend(unittest.TestCase):
    def test_prepare_pump_publish_reaches_prepared(self):
        """Real launcher -> JobManager -> Supervisor -> persistence."""
        from frigate_xdna.config import Config
        from frigate_xdna.supervisor import Supervisor
        with tempfile.TemporaryDirectory() as d:
            sup = Supervisor(
                Config(data_dir=d),
                compiler_prefixes=prefixes(d, OK_SHA_BEHAVIOR),
                worker_factory=fixture_worker_factory())
            try:
                out = sup.prepare(yolo_source(d))
                self.assertFalse(out["cache_hit"])
                deadline = time.monotonic() + 120.0
                while True:
                    sup.pump(0.1)
                    rec = sup.registry.get_ref(out["ref"])
                    if rec["state"] in ("PREPARED", "COMPILE_FAILED",
                                        "RESOURCE_EXCEEDED",
                                        "VALIDATION_FAILED"):
                        break
                    if time.monotonic() >= deadline:
                        self.fail("dummy compile never reached terminal")
                    time.sleep(0.1)
                self.assertEqual(rec["state"], "PREPARED")
                art = sup.registry.get_artifact(out["compile_key"])
                self.assertIsNotNone(art)
                with open(os.path.join(
                        d, "artifacts", out["compile_key"],
                        "artifact.json"), encoding="utf-8") as f:
                    manifest = json.load(f)
                self.assertEqual(manifest["backend"], "bf16-vaiml-v1")
            finally:
                sup.stop()


if __name__ == "__main__":
    unittest.main()
