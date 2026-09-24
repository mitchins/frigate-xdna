"""Unit tests: authoritative build identity (v0.1.1 checkpoint 2).

A published image must report its exact release version + source
revision (never 0.1.0.dev0, never reconstructed from a mutable tag);
an RC reports that RC; a source checkout reports development
explicitly. The compiled-model cache identity must not move with the
application patch version.
"""
import json
import os
import unittest

from frigate_xdna import __version__ as PACKAGE_VERSION
from frigate_xdna.build_identity import (
    channel_for,
    format_identity,
    get_build_identity,
    version_string,
)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

FAKE_SHA = "a" * 40


def write_identity(tmpdir: str, payload) -> str:
    path = os.path.join(tmpdir, "build-identity.json")
    with open(path, "w", encoding="utf-8") as f:
        if isinstance(payload, str):
            f.write(payload)
        else:
            json.dump(payload, f)
    return path


class TestChannelMapping(unittest.TestCase):
    def test_stable_rc_dev(self):
        self.assertEqual(channel_for("0.1.1"), "release")
        self.assertEqual(channel_for("0.1.1-rc.2"), "release-candidate")
        self.assertEqual(channel_for("0.1.0.dev0"), "development")
        self.assertEqual(channel_for("unknown"), "development")
        self.assertEqual(channel_for("latest"), "development")
        self.assertEqual(channel_for(""), "development")


class TestIdentityRead(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.realpath(os.path.join(
            REPO, "tests", "unit", ".tmp-ident"))
        os.makedirs(self.tmp, exist_ok=True)
        self._old = os.environ.get("FXDNA_BUILD_IDENTITY_FILE")

    def tearDown(self):
        for name in os.listdir(self.tmp):
            os.unlink(os.path.join(self.tmp, name))
        if self._old is None:
            os.environ.pop("FXDNA_BUILD_IDENTITY_FILE", None)
        else:
            os.environ["FXDNA_BUILD_IDENTITY_FILE"] = self._old

    def _point(self, payload) -> None:
        if payload is None:
            os.environ["FXDNA_BUILD_IDENTITY_FILE"] = os.path.join(
                self.tmp, "missing.json")
        else:
            os.environ["FXDNA_BUILD_IDENTITY_FILE"] = write_identity(
                self.tmp, payload)

    def test_missing_file_is_development(self):
        self._point(None)
        ident = get_build_identity()
        self.assertEqual(ident["version"], PACKAGE_VERSION)
        self.assertEqual(ident["revision"], "unknown")
        self.assertEqual(ident["channel"], "development")

    def test_stable_identity(self):
        self._point({"schema_version": 1, "version": "0.1.1",
                     "revision": FAKE_SHA})
        ident = get_build_identity()
        self.assertEqual(ident, {"schema_version": 1, "version": "0.1.1",
                                 "revision": FAKE_SHA,
                                 "channel": "release"})

    def test_dev_build_keeps_valid_revision(self):
        self._point({"schema_version": 1, "version": "0.1.1.dev0",
                     "revision": FAKE_SHA})
        ident = get_build_identity()
        self.assertEqual(ident["channel"], "development")
        self.assertEqual(ident["revision"], FAKE_SHA)
        self.assertEqual(ident["version"], "0.1.1.dev0")

    def test_rc_identity_is_not_final(self):
        self._point({"schema_version": 1, "version": "0.1.1-rc.2",
                     "revision": FAKE_SHA})
        ident = get_build_identity()
        self.assertEqual(ident["version"], "0.1.1-rc.2")
        self.assertEqual(ident["channel"], "release-candidate")

    def test_malformed_files_fall_back_without_raising(self):
        bad = ["not json{", "[1, 2]",
               {"version": "0.1.1"},
               {"version": 5, "revision": FAKE_SHA},
               {"version": "0.1.1", "revision": "xyz"},
               {"version": "0.1.1", "revision": FAKE_SHA,
                "extra": [1, 2]}]
        for payload in bad[:-1]:
            with self.subTest(payload=payload):
                self._point(payload)
                ident = get_build_identity()
                self.assertEqual(ident["channel"], "development")
                self.assertEqual(ident["revision"], "unknown")
        # A non-release version keeps a valid baked revision while
        # still reporting development explicitly.
        self._point({"version": "latest", "revision": FAKE_SHA})
        ident = get_build_identity()
        self.assertEqual(ident["channel"], "development")
        self.assertEqual(ident["revision"], FAKE_SHA)
        # Unknown extra keys are tolerated; identity still authoritative.
        self._point(bad[-1])
        ident = get_build_identity()
        self.assertEqual(ident["version"], "0.1.1")

    def test_format_identity_line(self):
        self._point({"schema_version": 1, "version": "0.1.1-rc.2",
                     "revision": FAKE_SHA})
        self.assertEqual(format_identity(get_build_identity()),
                         f"frigate-xdna 0.1.1-rc.2 revision={FAKE_SHA}")

    def test_version_string_names_channel(self):
        self._point({"schema_version": 1, "version": "0.1.1",
                     "revision": FAKE_SHA})
        out = version_string()
        self.assertIn("fxdna 0.1.1", out)
        self.assertIn(f"revision={FAKE_SHA}", out)
        self.assertIn("channel=release", out)


class TestIdentityInOutputs(unittest.TestCase):
    def test_cli_version_reports_identity(self):
        import io
        from contextlib import redirect_stdout

        from frigate_xdna.cli import main
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["--version"])
        self.assertEqual(rc, 0)
        # Single machine-parseable line: argparse's version action
        # wraps past 80 columns on non-tty output (release gate).
        lines = buf.getvalue().strip().splitlines()
        self.assertEqual(len(lines), 1)
        # A "--version" behind "--" is a positional, not the flag.
        with self.assertRaises(SystemExit) as cm2:
            main(["--", "--version"])
        self.assertNotEqual(cm2.exception.code, 0)
        out = lines[0]
        self.assertTrue(out.startswith("fxdna "))
        self.assertIn("revision=", out)
        self.assertIn("channel=development", out)  # unbaked checkout

    def test_read_status_without_registry_reports_development(self):
        import tempfile

        from frigate_xdna.cli import _read_status
        from frigate_xdna.config import Config
        with tempfile.TemporaryDirectory() as d:
            doc = _read_status(Config(data_dir=d))
        self.assertEqual(doc["state"], "STARTING")
        self.assertEqual(doc["version"], PACKAGE_VERSION)
        self.assertEqual(doc["build"]["channel"], "development")

    def test_supervisor_status_reports_baked_identity(self):
        import tempfile
        from unittest import mock

        from frigate_xdna.build_identity import IDENTITY_ENV
        from frigate_xdna.config import Config
        from frigate_xdna.supervisor import Supervisor
        with tempfile.TemporaryDirectory() as d:
            ident_path = write_identity(
                d, {"schema_version": 1, "version": "0.1.1-rc.2",
                    "revision": FAKE_SHA})
            with mock.patch.dict(os.environ, {IDENTITY_ENV: ident_path}):
                sup = Supervisor(Config(data_dir=d))
                try:
                    doc = sup.status()
                finally:
                    sup.stop()
        self.assertEqual(doc["version"], "0.1.1-rc.2")
        self.assertEqual(doc["build"]["revision"], FAKE_SHA)
        self.assertEqual(doc["build"]["channel"], "release-candidate")


class TestCacheIdentityStable(unittest.TestCase):
    def test_compile_key_ignores_build_identity(self):
        """Application patch versions must not invalidate the model
        cache: the compile key derives from source/payload/recipe/
        target only."""
        from frigate_xdna.build_identity import IDENTITY_ENV
        from frigate_xdna.cache.keys import compile_key
        kw = {"source_sha256": "s" * 64,
              "compiler_payload_sha256": "p" * 64,
              "recipe_id": "bf16-vaiml-v1", "recipe_config_sha256": "c" * 64,
              "target_profile": "t", "artifact_compatibility_id": "a",
              "compile_input": {"shape": [1, 3, 320, 320]}}
        before = compile_key(**kw)
        import tempfile
        from unittest import mock
        with tempfile.TemporaryDirectory() as d:
            ident_path = write_identity(
                d, {"schema_version": 1, "version": "9.9.9",
                    "revision": FAKE_SHA})
            with mock.patch.dict(os.environ, {IDENTITY_ENV: ident_path}):
                after = compile_key(**kw)
        self.assertEqual(before, after)

    def test_cache_keys_never_read_build_identity(self):
        with open(os.path.join(REPO, "src", "frigate_xdna", "cache",
                               "keys.py"), encoding="utf-8") as f:
            src = f.read()
        self.assertNotIn("build_identity", src)


if __name__ == "__main__":
    unittest.main()
