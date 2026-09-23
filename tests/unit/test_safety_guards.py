"""Unit tests: safety-journal failure paths (SPEC §10, filesystem only).

An unhealthy registry or filesystem must never turn a failure-path
boolean into a raise, lose an inhibition, or skip history rotation.
"""
import json
import os
import tempfile
import unittest
from unittest import mock

from frigate_xdna.cache.registry import Registry
from frigate_xdna.runtime import safety


def _reg(tmp):
    return Registry(os.path.join(tmp, "registry.sqlite3"))


class ExplodingRegistry:
    def get_state(self, _key):
        raise RuntimeError("registry down")

    def set_state(self, _key, _value):
        raise RuntimeError("registry down")

    def execute(self, *_a):
        raise RuntimeError("registry down")


class TestSafetyGuards(unittest.TestCase):
    def test_boot_token_falls_back_when_no_proc(self):
        with mock.patch("builtins.open",
                        side_effect=OSError("no proc")):
            self.assertEqual(safety._boot_token(), "unknown")

    def test_unclean_complete_preserves_journal_when_inhibit_fails(self):
        with tempfile.TemporaryDirectory() as d:
            reg = _reg(d)
            try:
                safety.begin_operation(d, reg, "activation", "r", "a", 1)
                cur = os.path.join(d, safety.JOURNAL_CURRENT)
                safety.complete_operation(d, ExplodingRegistry(),
                                          clean=False, error="boom")
                self.assertTrue(os.path.isfile(cur))
                self.assertIsNotNone(
                    safety.check_inhibited(d, ExplodingRegistry()))
            finally:
                reg.close()

    def test_dir_fsync_failure_still_journals(self):
        with tempfile.TemporaryDirectory() as d:
            reg = _reg(d)
            try:
                with mock.patch.object(os, "open",
                                       side_effect=OSError("ro")):
                    rec = safety.begin_operation(
                        d, reg, "activation", "r", "a", 1)
                cur = os.path.join(d, safety.JOURNAL_CURRENT)
                self.assertTrue(os.path.isfile(cur))
                self.assertFalse(rec["completed"])
            finally:
                reg.close()

    def test_unlink_failure_keeps_completion(self):
        with tempfile.TemporaryDirectory() as d:
            reg = _reg(d)
            try:
                safety.begin_operation(d, reg, "activation", "r", "a", 1)
                with mock.patch.object(os, "unlink",
                                       side_effect=OSError("ro")):
                    safety.complete_operation(d, reg, clean=True)
                cur = os.path.join(d, safety.JOURNAL_CURRENT)
                self.assertTrue(os.path.isfile(cur))
            finally:
                reg.close()

    def test_rotation_read_failure_keeps_completion(self):
        with tempfile.TemporaryDirectory() as d:
            reg = _reg(d)
            try:
                hist = os.path.join(d, safety.JOURNAL_HISTORY)
                os.makedirs(os.path.dirname(hist), exist_ok=True)
                with open(hist, "w") as f:
                    for i in range(3000):
                        f.write(json.dumps({"i": i,
                                            "pad": "x" * 700}) + "\n")
                real_open = open

                def selective_open(path, mode="r", *args, **kwargs):
                    if (isinstance(path, str)
                            and path.endswith("history.jsonl")
                            and "r" in mode and "w" not in mode
                            and "a" not in mode):
                        raise OSError("read down")
                    return real_open(path, mode, *args, **kwargs)

                safety.begin_operation(d, reg, "activation", "r", "a", 1)
                with mock.patch("builtins.open",
                                side_effect=selective_open):
                    safety.complete_operation(d, reg, clean=True)
                cur = os.path.join(d, safety.JOURNAL_CURRENT)
                self.assertFalse(os.path.exists(cur))
            finally:
                reg.close()

    def test_unwritable_history_keeps_completion(self):
        with tempfile.TemporaryDirectory() as d:
            reg = _reg(d)
            try:
                hist = os.path.join(d, safety.JOURNAL_HISTORY)
                os.makedirs(hist, exist_ok=True)
                safety.begin_operation(d, reg, "activation", "r", "a", 1)
                safety.complete_operation(d, reg, clean=True)
                cur = os.path.join(d, safety.JOURNAL_CURRENT)
                self.assertFalse(os.path.exists(cur))
            finally:
                reg.close()

    def test_check_inhibited_tolerates_dead_registry(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(
                safety.check_inhibited(d, ExplodingRegistry()))

    def test_check_inhibited_ignores_corrupt_journal(self):
        with tempfile.TemporaryDirectory() as d:
            reg = _reg(d)
            try:
                cur = os.path.join(d, safety.JOURNAL_CURRENT)
                os.makedirs(os.path.dirname(cur), exist_ok=True)
                with open(cur, "w") as f:
                    f.write("{corrupt")
                self.assertIsNone(safety.check_inhibited(d, reg))
            finally:
                reg.close()

    def test_check_inhibited_reports_interrupted_operation(self):
        with tempfile.TemporaryDirectory() as d:
            reg = _reg(d)
            try:
                safety.begin_operation(d, reg, "compile", "r", "a", 3)
                inh = safety.check_inhibited(d, reg)
                self.assertEqual(inh["reason"],
                                 "interrupted_unsafe_operation")
                self.assertEqual(inh["operation"], "compile")
            finally:
                reg.close()

    def test_history_rotates_past_size_cap(self):
        with tempfile.TemporaryDirectory() as d:
            reg = _reg(d)
            try:
                hist = os.path.join(d, safety.JOURNAL_HISTORY)
                os.makedirs(os.path.dirname(hist), exist_ok=True)
                with open(hist, "w") as f:
                    for i in range(3000):
                        f.write(json.dumps({"i": i,
                                            "pad": "x" * 700}) + "\n")
                self.assertGreater(os.path.getsize(hist), 2 * 1024 * 1024)
                safety.begin_operation(d, reg, "activation", "r", "a", 1)
                safety.complete_operation(d, reg, clean=True)
                with open(hist) as f:
                    lines = f.readlines()
                self.assertEqual(len(lines), safety.MAX_HISTORY)
                self.assertEqual(json.loads(lines[-1])["operation"],
                                 "activation")
            finally:
                reg.close()

    def test_clear_inhibition_tolerates_dead_registry(self):
        safety.clear_inhibition(ExplodingRegistry())


if __name__ == "__main__":
    unittest.main()
