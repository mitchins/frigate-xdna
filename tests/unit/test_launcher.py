"""Unit tests: compiler launcher isolation (no vendor, no NPU).

Proves the env contract, secret stripping, timeouts, log bounds and the
one-at-a-time lock using trivial child processes — never the real
toolchain.
"""
import os
import tempfile
import unittest

from frigate_xdna.compiler.launcher import (
    FORBIDDEN_ENV_KEYS,
    CompilerPrefixes,
    build_compile_env,
    build_quant_env,
    run_compile,
    spawn,
)


def prefixes(workdir):
    sp = os.path.join(workdir, "sp")
    os.makedirs(sp, exist_ok=True)
    return CompilerPrefixes(
        quant_python="/usr/bin/python3",
        compile_python="/usr/bin/python3",
        compile_lib=sp,
        xrt_lib=os.path.join(workdir, "xrt"),
        xrt_root=os.path.join(workdir, "xrtroot"),
        recipe_dir=os.path.join(workdir, "recipe"),
        calib_dir=os.path.join(workdir, "calib"),
        vaiml_config=os.path.join(workdir, "vaiml_config.json"))


class TestEnvContract(unittest.TestCase):
    def test_xrt_precedes_vendored_libs(self):
        with tempfile.TemporaryDirectory() as d:
            env = build_compile_env(prefixes(d), d)
        parts = env["LD_LIBRARY_PATH"].split(os.pathsep)
        xrt = next(i for i, p in enumerate(parts) if p.endswith("xrt"))
        voe = next(i for i, p in enumerate(parts) if p.endswith("voe/lib"))
        self.assertLess(xrt, voe)

    def test_no_forbidden_keys_even_when_parent_poisoned(self):
        old = dict(os.environ)
        try:
            os.environ["PLUS_API_KEY"] = "secret"
            os.environ["BEARER_TOKEN"] = "secret"
            with tempfile.TemporaryDirectory() as d:
                for env in (build_compile_env(prefixes(d), d),
                            build_quant_env(d)):
                    for key in FORBIDDEN_ENV_KEYS:
                        self.assertNotIn(key, env)
                    self.assertNotIn("PLUS_API_KEY", str(list(env)))
        finally:
            os.environ.clear()
            os.environ.update(old)

    def test_pythonpath_unset(self):
        with tempfile.TemporaryDirectory() as d:
            env = build_compile_env(prefixes(d), d)
        self.assertNotIn("PYTHONPATH", env)
        self.assertEqual(env["OMP_NUM_THREADS"], "4")


class TestSpawn(unittest.TestCase):
    def test_secret_stripping_end_to_end(self):
        old = dict(os.environ)
        try:
            os.environ["PLUS_API_KEY"] = "topsecret-value"
            with tempfile.TemporaryDirectory() as d:
                env = build_quant_env(d)
                rc, _ = spawn(["/usr/bin/env"], env, d, 30.0,
                              os.path.join(d, "env"))
                self.assertEqual(rc, 0)
                out = open(os.path.join(d, "env.stdout.log")).read()
                self.assertNotIn("topsecret-value", out)
                self.assertNotIn("PLUS_API_KEY", out)
        finally:
            os.environ.clear()
            os.environ.update(old)

    def test_timeout_kills_group(self):
        with tempfile.TemporaryDirectory() as d:
            env = build_quant_env(d)
            rc, wall = spawn(["/bin/sleep", "60"], env, d, 2.0,
                             os.path.join(d, "sleep"))
            self.assertEqual(rc, 124)
            self.assertLess(wall, 30.0)

    def test_forbidden_env_refused(self):
        with tempfile.TemporaryDirectory() as d:
            env = build_quant_env(d)
            env = dict(env, PLUS_API_KEY="x")
            with self.assertRaises(RuntimeError):
                spawn(["/bin/true"], env, d, 10.0,
                      os.path.join(d, "t"))

    def test_missing_executable(self):
        with tempfile.TemporaryDirectory() as d:
            rc, _ = spawn(["/nonexistent-xyz"], build_quant_env(d), d,
                          10.0, os.path.join(d, "t"))
            self.assertEqual(rc, 127)


class TestOneAtATime(unittest.TestCase):
    def test_concurrent_run_compile_serialized(self):
        import frigate_xdna.compiler.launcher as mod
        self.assertTrue(mod._compile_lock.acquire(blocking=False))
        try:
            with tempfile.TemporaryDirectory() as d:
                res = run_compile(prefixes(d), "/nonexistent.onnx", d,
                                  "k")
                self.assertEqual(res.returncode, 98)
        finally:
            mod._compile_lock.release()


if __name__ == "__main__":
    unittest.main()
