"""Unit tests: CLI surface, exit codes, offline status schema."""
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from frigate_xdna import cli
from frigate_xdna.errors import NOT_READY, SUCCESS


def run_cli(argv, env_extra=None):
    out, err = io.StringIO(), io.StringIO()
    old = dict(os.environ)
    try:
        os.environ.update({"FXDNA_DATA_DIR": os.environ.get(
            "FXDNA_TEST_DATA_DIR", tempfile.mkdtemp())})
        if env_extra:
            os.environ.update(env_extra)
        with redirect_stdout(out), redirect_stderr(err):
            try:
                rc = cli.main(argv)
            except SystemExit as e:
                rc = e.code
    finally:
        os.environ.clear()
        os.environ.update(old)
    return rc, out.getvalue(), err.getvalue()


class TestCli(unittest.TestCase):
    def test_help_lists_all_commands(self):
        rc, out, _ = run_cli(["--help"])
        self.assertEqual(rc, 0)
        for cmd in ("serve", "prepare", "status", "wait", "activate",
                    "cache", "doctor", "health", "recover"):
            self.assertIn(cmd, out)

    def test_install_serve_handlers_registers_signals(self):
        import signal as _signal

        from frigate_xdna import cli as _cli
        calls = []

        def _handler(signum, _frame):
            calls.append(signum)

        old_term = _signal.getsignal(_signal.SIGTERM)
        old_int = _signal.getsignal(_signal.SIGINT)
        try:
            _cli._install_serve_handlers(_handler)
            self.assertIs(_signal.getsignal(_signal.SIGTERM), _handler)
            self.assertIs(_signal.getsignal(_signal.SIGINT), _handler)
            # the handler only signals shutdown, never raises
            _signal.getsignal(_signal.SIGTERM)(_signal.SIGTERM, None)
            self.assertEqual(calls, [int(_signal.SIGTERM)])
        finally:
            _signal.signal(_signal.SIGTERM, old_term)
            _signal.signal(_signal.SIGINT, old_int)

    def test_unknown_command_rejected(self):
        rc, _, _ = run_cli(["frobnicate"])
        self.assertEqual(rc, 2)

    def test_status_json_matches_schema_shape(self):
        rc, out, _ = run_cli(["status", "--json"])
        self.assertEqual(rc, SUCCESS)
        doc = json.loads(out)
        self.assertEqual(doc["schema_version"], 1)
        self.assertEqual(doc["service"], "frigate-xdna")
        self.assertIn(doc["state"], ("STARTING", "SERVING"))

    def test_cache_list_json_on_empty_store(self):
        rc, out, _ = run_cli(["cache", "list", "--json"])
        self.assertEqual(rc, SUCCESS)
        self.assertEqual(json.loads(out)["entries"], [])

    def test_health_reports_liveness_honestly(self):
        # No daemon: liveness-false WITH a nonzero exit (D10 — success
        # with alive=false once masked dead daemons).
        rc, out, _ = run_cli(["health"])
        self.assertEqual(rc, NOT_READY)
        self.assertFalse(json.loads(out)["alive"])
        rc, out, _ = run_cli(["health", "--ready"])
        self.assertEqual(rc, NOT_READY)
        self.assertFalse(json.loads(out)["ready"])

    def test_stale_socket_is_not_alive(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "control.sock"), "w") as f:
                f.write("not a socket")
            old = os.environ.get("FXDNA_TEST_DATA_DIR")
            os.environ["FXDNA_TEST_DATA_DIR"] = d
            try:
                rc, out, _ = run_cli(["health"])
            finally:
                if old is None:
                    del os.environ["FXDNA_TEST_DATA_DIR"]
                else:
                    os.environ["FXDNA_TEST_DATA_DIR"] = old
        self.assertEqual(rc, NOT_READY)
        self.assertFalse(json.loads(out)["alive"])

    def test_doctor_hardware_refused_without_lease(self):
        rc, _, err = run_cli(["doctor", "--hardware"])
        self.assertEqual(rc, 8)
        self.assertIn("DEVICE_UNAVAILABLE", err)

    def test_doctor_passive_ok(self):
        rc, out, _ = run_cli(["doctor"])
        self.assertEqual(rc, SUCCESS)
        self.assertIn("checks", json.loads(out))

    def test_prepare_plus_without_key_fails_visibly(self):
        rc, _, err = run_cli(["prepare", "plus://SOMEID"])
        self.assertEqual(rc, 4)
        self.assertIn("ACQUISITION_FAILED", err)

    def test_config_error_exit_code(self):
        old = os.environ.get("PLUS_API_KEY"), os.environ.get("PLUS_API_KEY_FILE")
        os.environ["PLUS_API_KEY"] = "a"
        os.environ["PLUS_API_KEY_FILE"] = "b"
        try:
            rc, _, _ = run_cli(["status"])
        finally:
            for k, v in (("PLUS_API_KEY", old[0]), ("PLUS_API_KEY_FILE", old[1])):
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
