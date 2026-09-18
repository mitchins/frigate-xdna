"""Unit tests: CLI surface, exit codes, offline status schema."""
import io
import json
import unittest
from contextlib import redirect_stdout, redirect_stderr

from frigate_xdna import cli
from frigate_xdna.errors import NOT_READY, SUCCESS


def run_cli(argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        try:
            rc = cli.main(argv)
        except SystemExit as e:
            rc = e.code
    return rc, out.getvalue(), err.getvalue()


class TestCli(unittest.TestCase):
    def test_help_lists_all_commands(self):
        rc, out, _ = run_cli(["--help"])
        self.assertEqual(rc, 0)
        for cmd in ("serve", "prepare", "status", "wait", "activate",
                    "cache", "doctor", "health", "recover"):
            self.assertIn(cmd, out)

    def test_unknown_command_rejected(self):
        rc, _, _ = run_cli(["frobnicate"])
        self.assertEqual(rc, 2)

    def test_status_json_matches_schema_shape(self):
        rc, out, _ = run_cli(["status", "--json"])
        self.assertEqual(rc, SUCCESS)
        doc = json.loads(out)
        self.assertEqual(doc["schema_version"], 1)
        self.assertEqual(doc["service"], "frigate-xdna")
        # foundation honesty: never claim readiness
        self.assertNotEqual(doc["state"], "SERVING")
        self.assertIsNone(doc["active"])

    def test_unimplemented_commands_refuse_without_faking(self):
        for argv in (["serve"], ["prepare", "plus://X"], ["wait", "plus://X"],
                     ["activate", "plus://X"], ["cache", "list"],
                     ["recover", "plus://X", "--acknowledge"]):
            rc, _, err = run_cli(argv)
            self.assertEqual(rc, NOT_READY, argv)
            self.assertIn("not implemented", err, argv)

    def test_health_not_ready_without_daemon(self):
        rc, out, _ = run_cli(["health"])
        self.assertEqual(rc, NOT_READY)
        self.assertFalse(json.loads(out)["alive"])

    def test_config_error_exit_code(self):
        import os
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
