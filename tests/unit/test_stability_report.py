"""Unit tests: stability journal, interrupted-run detection,
report/acknowledge semantics (docs/COMPATIBILITY.md §9-§12).

A host reset may erase the final log line, so the journal on disk —
not console output — is the record. These tests pin: interrupted
detection reports the last unmatched STARTED without causal claims;
a finished run is never interrupted; acknowledgement is required,
idempotent and confined to the stability latch.
"""
import json
import os
import shutil
import tempfile
import unittest

from frigate_xdna.observability import stability


def write_run(data_dir, run_id, timeline=None, result=None,
              metadata=None, acknowledge=None):
    d = stability.run_dir(data_dir, run_id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, stability.METADATA), "w") as f:
        json.dump(metadata or {
            "schema_version": 1, "run_id": run_id,
            "profile": "pm",
            "model": {"family": "YOLOv9", "resolution": 320,
                      "compile_key": "3f7cf949"},
            "phases": ["E0 PASS", "H1 PASS", "H2 PASS", "H3 PASS"],
        }, f)
    with open(os.path.join(d, stability.TIMELINE), "w") as f:
        for rec in timeline or []:
            f.write(json.dumps(rec) + "\n")
    if result is not None:
        with open(os.path.join(d, stability.RESULT), "w") as f:
            json.dump(result, f)
    if acknowledge is not None:
        with open(os.path.join(d, stability.ACKNOWLEDGE), "w") as f:
            json.dump(acknowledge, f)
    return d


def rec(seq, phase, step, state, cycle=None, submitted=None,
        completed=None, wall_time=None):
    r = {"sequence": seq, "phase": phase, "step": step,
         "state": state,
         "wall_time": wall_time if wall_time is not None
         else 1_800_000_000.0 + seq}
    if cycle is not None:
        r["cycle"] = cycle
    if submitted is not None:
        r["submitted"] = submitted
    if completed is not None:
        r["completed"] = completed
    return r


class DetectInterruptedTest(unittest.TestCase):
    def test_matched_operations_are_not_interrupted(self):
        tl = [rec(1, "H4", "PM_RESUME_REQUEST", "STARTED", cycle=3,
                  submitted=4821, completed=4821),
              rec(2, "H4", "PM_RESUME_REQUEST", "COMPLETED", cycle=3,
                  submitted=4822, completed=4822)]
        self.assertIsNone(
            stability.detect_interrupted(tl, has_result=False))

    def test_unmatched_started_reports_position_only(self):
        tl = [rec(1, "H3", "PHASE", "COMPLETED"),
              rec(2, "H4", "PHASE", "COMPLETED"),
              rec(3, "H4", "PM_RESUME_REQUEST", "STARTED", cycle=3,
                  submitted=4821, completed=4821)]
        got = stability.detect_interrupted(tl, has_result=False)
        self.assertEqual(got["phase"], "H4")
        self.assertEqual(got["step"], "PM_RESUME_REQUEST")
        self.assertEqual(got["cycle"], 3)
        self.assertEqual(got["completed"], 4821)

    def test_result_present_means_not_interrupted(self):
        tl = [rec(1, "H4", "PM_RESUME_REQUEST", "STARTED", cycle=3)]
        self.assertIsNone(
            stability.detect_interrupted(tl, has_result=True))

    def test_cycles_are_tracked_separately(self):
        tl = [rec(1, "H4", "PM_RESUME_REQUEST", "STARTED", cycle=1),
              rec(2, "H4", "PM_RESUME_REQUEST", "COMPLETED", cycle=1),
              rec(3, "H4", "PM_RESUME_REQUEST", "STARTED", cycle=2)]
        got = stability.detect_interrupted(tl, has_result=False)
        self.assertEqual(got["cycle"], 2)


class ReportTest(unittest.TestCase):
    def setUp(self):
        self.data = tempfile.mkdtemp(prefix="fx-stab-")

    def tearDown(self):
        shutil.rmtree(self.data, ignore_errors=True)

    def test_no_runs(self):
        self.assertIsNone(stability.load_report(self.data))
        self.assertEqual(stability.list_runs(self.data), [])
        self.assertIsNone(stability.latest_run(self.data))

    def test_finished_run_report(self):
        write_run(self.data, "2026-09-29T10-00-00",
                  timeline=[rec(1, "E0", "PHASE", "COMPLETED")],
                  result={"outcome": "PASS", "last_phase": "H4"})
        got = stability.load_report(self.data)
        self.assertEqual(got["run_id"], "2026-09-29T10-00-00")
        self.assertEqual(got["result"]["outcome"], "PASS")
        self.assertIsNone(got["interrupted"])
        text = stability.format_report(got)
        self.assertIn("PASS", text)
        self.assertIn("not certification", text)

    def test_interrupted_run_report_text(self):
        write_run(self.data, "2026-09-29T11-00-00",
                  timeline=[
                      rec(1, "H3", "PHASE", "COMPLETED"),
                      rec(2, "H4", "PHASE", "COMPLETED"),
                      rec(3, "H4", "PM_RESUME_REQUEST", "STARTED",
                          cycle=3, submitted=4821, completed=4821)])
        got = stability.load_report(self.data)
        self.assertIsNone(got["result"])
        self.assertEqual(got["interrupted"]["step"],
                         "PM_RESUME_REQUEST")
        text = stability.format_report(got)
        self.assertIn("INTERRUPTED", text)
        self.assertIn("PM_RESUME_REQUEST STARTED", text)
        self.assertIn("No causal conclusion", text)
        self.assertIn("acknowledge --last", text)

    def test_latest_run_selected(self):
        write_run(self.data, "2026-09-29T10-00-00",
                  result={"outcome": "PASS", "last_phase": "H2"})
        write_run(self.data, "2026-09-29T12-00-00",
                  result={"outcome": "PASS", "last_phase": "H3"})
        got = stability.load_report(self.data)
        self.assertEqual(got["run_id"], "2026-09-29T12-00-00")


    def test_partial_final_line_keeps_earlier_breadcrumbs(self):
        # A host reset can truncate mid-line; the durable records
        # before it are exactly what an interrupted-run report needs.
        d = write_run(self.data, "r-partial",
                      timeline=[rec(1, "H4", "PM_RESUME_REQUEST",
                                    "STARTED", cycle=2)])
        with open(os.path.join(d, stability.TIMELINE), "a") as f:
            f.write('{"sequence": 2, "phase": "H4", "step": "PM_RE')  # truncated
        got = stability.load_report(self.data, "r-partial")
        self.assertEqual(got["interrupted"]["cycle"], 2)
        self.assertEqual(got["timeline_records"], 1)

    def test_interrupted_report_merges_last_checkpoint_counts(self):
        # The unmatched STARTED carries no counters; the last durable
        # current.json checkpoint does. The report must merge them
        # (position evidence), never leave None when a checkpoint
        # exists.
        d = write_run(self.data, "r-ckpt",
                      timeline=[rec(1, "H3", "PHASE", "STARTED")])
        with open(os.path.join(d, stability.CURRENT), "w") as f:
            json.dump({"phase": "H3", "sub_phase": "steady",
                       "submitted": 72, "completed": 72,
                       "errors": 0, "timeouts": 0,
                       "wall_time": 1790662990.0}, f)
        got = stability.load_report(self.data, "r-ckpt")
        self.assertEqual(got["interrupted"]["completed"], 72)
        self.assertEqual(got["interrupted"]["checkpoint"]["submitted"],
                         72)
        text = stability.format_report(got)
        self.assertIn("last completed request: 72", text)


class AcknowledgeTest(unittest.TestCase):
    def setUp(self):
        self.data = tempfile.mkdtemp(prefix="fx-stab-")

    def tearDown(self):
        shutil.rmtree(self.data, ignore_errors=True)

    def test_no_runs_refused(self):
        got = stability.acknowledge_run(self.data, "reviewed")
        self.assertFalse(got["acknowledged"])
        self.assertEqual(got["error_code"], "NOT_FOUND")

    def test_finished_run_refused(self):
        write_run(self.data, "r1",
                  result={"outcome": "PASS", "last_phase": "H4"})
        got = stability.acknowledge_run(self.data, "reviewed")
        self.assertEqual(got["error_code"], "NOT_INTERRUPTED")

    def test_interrupted_run_acknowledged_durably(self):
        write_run(self.data, "r1",
                  timeline=[rec(1, "H4", "PM_RESUME_REQUEST",
                                "STARTED", cycle=2)])
        got = stability.acknowledge_run(self.data, "reviewed reset")
        self.assertTrue(got["acknowledged"])
        path = os.path.join(stability.run_dir(self.data, "r1"),
                            stability.ACKNOWLEDGE)
        self.assertTrue(os.path.isfile(path))
        stored = json.load(open(path))
        self.assertEqual(stored["reason"], "reviewed reset")
        # Report now shows acknowledgement, keeps the evidence.
        report = stability.load_report(self.data)
        self.assertTrue(report["acknowledged"])
        self.assertIsNotNone(report["interrupted"])

    def test_acknowledge_idempotent(self):
        write_run(self.data, "r1",
                  timeline=[rec(1, "H4", "PM_RESUME_REQUEST",
                                "STARTED", cycle=2)])
        first = stability.acknowledge_run(self.data, "one")
        second = stability.acknowledge_run(self.data, "two")
        self.assertTrue(first["acknowledged"])
        self.assertTrue(second["acknowledged"])
        self.assertEqual(second["error_code"], "ALREADY")


if __name__ == "__main__":
    unittest.main()
