"""Hardware-free tests for the safety journal/breaker (SPEC §10).

An interrupted unsafe operation must inhibit NPU activation until an
explicit `fxdna recover`. Filesystem only; no device touched.
"""
import json
import os
import tempfile
import unittest

from frigate_xdna.cache.registry import Registry
from frigate_xdna.runtime import safety


def _reg(tmp):
    return Registry(os.path.join(tmp, "registry.sqlite3"))


class TestJournal(unittest.TestCase):
    def test_begin_complete_clean(self):
        with tempfile.TemporaryDirectory() as d:
            reg = _reg(d)
            try:
                rec = safety.begin_operation(
                    d, reg, "activation", "ref", "ck", 1)
                cur = os.path.join(d, safety.JOURNAL_CURRENT)
                self.assertTrue(os.path.isfile(cur))
                self.assertFalse(rec["completed"])
                safety.complete_operation(d, reg, clean=True)
                self.assertFalse(os.path.exists(cur))
                hist = os.path.join(d, safety.JOURNAL_HISTORY)
                with open(hist) as f:
                    lines = f.readlines()
                self.assertEqual(len(lines), 1)
                stored = json.loads(lines[0])
                self.assertTrue(stored["completed"])
                self.assertTrue(stored["clean"])
                self.assertIsNone(
                    safety.check_inhibited(d, reg))
            finally:
                reg.close()

    def test_unclean_completion_inhibits(self):
        with tempfile.TemporaryDirectory() as d:
            reg = _reg(d)
            try:
                safety.begin_operation(
                    d, reg, "compile", "ref", "ck", 0)
                safety.complete_operation(
                    d, reg, clean=False, error="boom")
                inh = safety.check_inhibited(d, reg)
                self.assertIsNotNone(inh)
                self.assertEqual(
                    inh.get("reason"), "unclean_operation")
                self.assertFalse(os.path.exists(
                    os.path.join(d, safety.JOURNAL_CURRENT)))
            finally:
                reg.close()

    def test_interrupted_journal_inhibits(self):
        with tempfile.TemporaryDirectory() as d:
            reg = _reg(d)
            try:
                safety.begin_operation(
                    d, reg, "activation", "ref", "ck", 3)
                # no complete_operation: simulates crash/kill
                inh = safety.check_inhibited(d, reg)
                self.assertIsNotNone(inh)
                self.assertEqual(
                    inh.get("reason"), "interrupted_unsafe_operation")
                self.assertEqual(inh.get("operation"), "activation")
            finally:
                reg.close()

    def test_inhibit_and_clear(self):
        with tempfile.TemporaryDirectory() as d:
            reg = _reg(d)
            try:
                self.assertIsNone(safety.check_inhibited(d, reg))
                safety.inhibit(d, reg, "manual", "ref")
                inh = safety.check_inhibited(d, reg)
                self.assertIsNotNone(inh)
                self.assertEqual(inh.get("reason"), "manual")
                safety.clear_inhibition(reg)
                self.assertIsNone(safety.check_inhibited(d, reg))
            finally:
                reg.close()

    def test_complete_without_journal_is_noop(self):
        with tempfile.TemporaryDirectory() as d:
            reg = _reg(d)
            try:
                safety.complete_operation(d, reg, clean=True)
                self.assertIsNone(safety.check_inhibited(d, reg))
            finally:
                reg.close()

    def test_history_rotation_caps_size(self):
        with tempfile.TemporaryDirectory() as d:
            reg = _reg(d)
            try:
                hist = os.path.join(d, safety.JOURNAL_HISTORY)
                os.makedirs(os.path.dirname(hist), exist_ok=True)
                # pre-fill beyond the 2 MiB rotation threshold
                with open(hist, "w") as f:
                    for i in range(3000):
                        f.write(json.dumps(
                            {"i": i, "pad": "x" * 900}) + "\n")
                self.assertGreater(os.path.getsize(hist), 2 * 1024 * 1024)
                safety.begin_operation(d, reg, "activation")
                safety.complete_operation(d, reg, clean=True)
                with open(hist) as f:
                    lines = f.readlines()
                self.assertLessEqual(len(lines), safety.MAX_HISTORY)
                # newest record (the completion) survived rotation
                last = json.loads(lines[-1])
                self.assertTrue(last["completed"])
                self.assertGreaterEqual(
                    last["completed_at"], last["started_at"])
            finally:
                reg.close()

    def test_bool_compat_positional(self):
        # Old callers: complete_operation(data_dir, clean, error)
        with tempfile.TemporaryDirectory() as d:
            reg = _reg(d)
            try:
                safety.begin_operation(d, reg, "teardown")
                safety.complete_operation(d, True)
                self.assertFalse(os.path.exists(
                    os.path.join(d, safety.JOURNAL_CURRENT)))
            finally:
                reg.close()


if __name__ == "__main__":
    unittest.main()
