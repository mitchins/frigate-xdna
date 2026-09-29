"""Unit tests: host-info / stability CLI surface, diagnose bundle
integration and the compatibility docs contract
(docs/COMPATIBILITY.md).
"""
import json
import os
import shutil
import tempfile
import unittest

from frigate_xdna import cli

from .test_stability_report import rec, write_run

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def run_cli(argv, env_extra=None):
    import io
    from contextlib import redirect_stderr, redirect_stdout
    out, err = io.StringIO(), io.StringIO()
    old = dict(os.environ)
    data = tempfile.mkdtemp(prefix="fx-cli2-")
    try:
        os.environ["FXDNA_DATA_DIR"] = data
        if env_extra:
            os.environ.update(env_extra)
        with redirect_stdout(out), redirect_stderr(err):
            try:
                rc = cli.main(argv)
            except SystemExit as e:
                rc = e.code
        return rc, out.getvalue(), err.getvalue(), data
    finally:
        os.environ.clear()
        os.environ.update(old)
        shutil.rmtree(data, ignore_errors=True)


class HostInfoCLITest(unittest.TestCase):
    def test_json_is_valid_and_schema_stable(self):
        rc, out, err, _ = run_cli(["host-info", "--json"])
        self.assertEqual(rc, 0)
        doc = json.loads(out)
        self.assertEqual(doc["schema_version"], 1)
        for section in ("host", "npu", "power", "appliance"):
            self.assertIn(section, doc)
        emb = doc["appliance"]["embedded"]
        for key in ("xrt", "xdna_shim", "flexmlrt", "recipe"):
            self.assertIn(key, emb)
        # Appliance runtime is labelled, never confused with host XRT.
        self.assertIn("not host XRT", doc["note"])

    def test_text_output_renders(self):
        rc, out, err, _ = run_cli(["host-info"])
        self.assertEqual(rc, 0)
        self.assertIn("Appliance", out)
        self.assertIn("NPU", out)


class StabilityCLITest(unittest.TestCase):
    def test_report_with_no_runs_exits_not_ready(self):
        rc, out, err, _ = run_cli(["stability", "report", "--last"])
        self.assertEqual(rc, 3)
        self.assertIn("no stability run", err)

    def test_report_last_json(self):
        data = tempfile.mkdtemp(prefix="fx-stabcli-")
        try:
            write_run(data, "2026-09-29T09-00-00",
                      timeline=[rec(1, "H4", "PM_RESUME_REQUEST",
                                    "STARTED", cycle=3,
                                    submitted=4821, completed=4821)])
            import io
            from contextlib import redirect_stderr, redirect_stdout
            out, err = io.StringIO(), io.StringIO()
            old = dict(os.environ)
            os.environ["FXDNA_DATA_DIR"] = data
            try:
                with redirect_stdout(out), redirect_stderr(err):
                    rc = cli.main(["stability", "report", "--last",
                                   "--json"])
            finally:
                os.environ.clear()
                os.environ.update(old)
            self.assertEqual(rc, 0)
            doc = json.loads(out.getvalue())
            self.assertEqual(doc["interrupted"]["step"],
                             "PM_RESUME_REQUEST")
        finally:
            shutil.rmtree(data, ignore_errors=True)

    def test_run_not_implemented_in_this_build(self):
        rc, out, err, _ = run_cli(
            ["stability", "run", "--configured", "--profile", "gentle"])
        self.assertEqual(rc, 3)
        self.assertIn("not implemented", err)
        self.assertIn("NOT_IMPLEMENTED", err)

    def test_acknowledge_reason_required(self):
        rc, out, err, _ = run_cli(["stability", "acknowledge", "--last"])
        self.assertNotEqual(rc, 0)

    def test_acknowledge_without_interrupted_run(self):
        rc, out, err, _ = run_cli(
            ["stability", "acknowledge", "--last", "--reason", "x"])
        self.assertEqual(rc, 3)


class DiagnoseIntegrationTest(unittest.TestCase):
    def test_bundle_contains_host_info_and_stability(self):
        data = tempfile.mkdtemp(prefix="fx-diag-")
        out_dir = tempfile.mkdtemp(prefix="fx-diag-out-")
        try:
            write_run(data, "2026-09-29T08-00-00",
                      timeline=[
                          rec(1, "H3", "PHASE", "COMPLETED"),
                          rec(2, "H4", "PM_RESUME_REQUEST", "STARTED",
                              cycle=1, submitted=10, completed=10)],
                      metadata={
                          "schema_version": 1, "run_id": "r",
                          "model": {"family": "plus://ABC123",
                                    "resolution": 320}})
            import io
            from contextlib import redirect_stderr, redirect_stdout
            out, err = io.StringIO(), io.StringIO()
            old = dict(os.environ)
            os.environ["FXDNA_DATA_DIR"] = data
            try:
                with redirect_stdout(out), redirect_stderr(err):
                    rc = cli.main(["diagnose", "--out", out_dir])
            finally:
                os.environ.clear()
                os.environ.update(old)
            self.assertEqual(rc, 0)
            names = set(os.listdir(out_dir))
            self.assertIn("host-info.json", names)
            self.assertIn("stability-summary.json", names)
            self.assertIn("stability-timeline.tail.jsonl", names)
            host = json.load(open(os.path.join(out_dir,
                                               "host-info.json")))
            self.assertEqual(host["schema_version"], 1)
            summary = json.load(open(
                os.path.join(out_dir, "stability-summary.json")))
            self.assertEqual(summary["interrupted"]["step"],
                             "PM_RESUME_REQUEST")
            # Plus ref in stability metadata is pseudonymized, not raw.
            blob = json.dumps(summary)
            self.assertNotIn("plus://ABC123", blob)
            self.assertIn("plus:", blob)
        finally:
            shutil.rmtree(data, ignore_errors=True)
            shutil.rmtree(out_dir, ignore_errors=True)

    def test_bundle_without_stability_still_has_host_info(self):
        out_root = tempfile.mkdtemp(prefix="fx-diag2-")
        try:
            rc, out, err, _ = run_cli(
                ["diagnose", "--out", os.path.join(out_root, "d")])
            names = set(os.listdir(os.path.join(out_root, "d")))
            self.assertIn("host-info.json", names)
            self.assertNotIn("stability-summary.json", names)
            manifest = json.load(open(os.path.join(out_root, "d",
                                                   "manifest.json")))
            self.assertIn("host-info.json", manifest["files"])
        finally:
            shutil.rmtree(out_root, ignore_errors=True)


class CompatibilityDocTest(unittest.TestCase):
    def setUp(self):
        self.doc = open(os.path.join(REPO, "docs", "COMPATIBILITY.md"),
                        encoding="utf-8").read()

    def _summary_rows(self):
        rows = []
        in_table = False
        for line in self.doc.splitlines():
            if line.startswith("| ID |"):
                in_table = True
                continue
            if in_table:
                if not line.startswith("|"):
                    break
                cells = [c.strip() for c in line.strip("|").split("|")]
                if set(cells[0]) <= {"-", ":"} or not cells[0]:
                    continue
                rows.append(cells)
        return rows

    def test_seeded_rows_and_result_classes(self):
        rows = self._summary_rows()
        ids = [r[0] for r in rows]
        self.assertEqual(ids, ["[O-001](#o-001)", "[O-002](#o-002)",
                               "[O-003](#o-003)"])
        allowed = ("PASS", "LIMITED", "FAIL", "RESET", "UNKNOWN")
        for r in rows:
            verdict = r[-1].strip("* —-")
            head = verdict.split("—")[0].strip()
            self.assertIn(head, allowed, r)

    def test_no_banned_support_vocabulary_in_rows(self):
        for r in self._summary_rows():
            joined = " ".join(r).lower()
            for banned in ("stable", "unsupported", "supported ",
                           "certified"):
                self.assertNotIn(banned, joined)

    def test_external_section_is_present_and_labelled(self):
        self.assertIn(
            "Related platform observations — not frigate-xdna "
            "compatibility results", self.doc)

    def test_detail_records_use_required_field_vocabulary(self):
        for field in ("Evidence class", "Evidence class".lower(),
                      "amdxdna srcversion", "NPU firmware",
                      "autosuspend", "embedded XRT"):
            self.assertIn(field, self.doc)

    def test_host_evidence_script_exists_and_is_syntax_clean(self):
        script = os.path.join(REPO, "tools",
                              "collect-host-evidence.sh")
        self.assertTrue(os.access(script, os.X_OK))
        import subprocess
        rc = subprocess.call(["bash", "-n", script],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        self.assertEqual(rc, 0)

    def test_issue_template_exists(self):
        path = os.path.join(REPO, ".github", "ISSUE_TEMPLATE",
                            "compatibility-report.md")
        self.assertTrue(os.path.isfile(path))

    def test_interfaces_documents_new_commands(self):
        iface = open(os.path.join(REPO, "docs", "INTERFACES.md"),
                     encoding="utf-8").read()
        for cmd in ("host-info", "stability report",
                    "stability acknowledge", "stability run"):
            self.assertIn(cmd, iface)


if __name__ == "__main__":
    unittest.main()
