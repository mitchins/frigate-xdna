"""Hardware-free tests for ZMQ identity/generation binding (SPEC §6).

No inference message carries a model hash; the SessionTable binds each
ROUTER identity to source_sha256 + serving_digest + generation, and old
generations must never serve after an A->B switch.
"""
import time
import unittest

from frigate_xdna.transport.sessions import QUIESCENCE_S, SessionTable


class TestSessionTable(unittest.TestCase):
    def test_bind_get_touch(self):
        t = SessionTable()
        self.assertIsNone(t.get(b"id1"))
        b = t.bind(b"id1", "sha-a", "srv-a", "ck-a", 0)
        self.assertEqual(b.source_sha256, "sha-a")
        self.assertEqual(b.serving_digest, "srv-a")
        self.assertEqual(b.compile_key, "ck-a")
        self.assertEqual(b.generation, 0)
        self.assertIs(t.get(b"id1"), b)
        t.touch(b"id1")  # unknown identities are ignored
        t.touch(b"nope")

    def test_rebind_replaces_identity(self):
        t = SessionTable()
        t.bind(b"id1", "sha-a", "srv", "ck-a", 0)
        t.bind(b"id1", "sha-b", "srv", "ck-b", 0)
        self.assertEqual(t.get(b"id1").source_sha256, "sha-b")

    def test_invalidate_generation_isolates_ab(self):
        t = SessionTable()
        t.bind(b"a-sock", "sha-a", "srv-a", "ck-a", 0)
        t.bind(b"b-sock", "sha-b", "srv-b", "ck-b", 0)
        # Simulate A->B switch to generation 1 for b-sock only
        t.invalidate_generation(0)
        self.assertIsNone(t.get(b"a-sock"))
        self.assertIsNone(t.get(b"b-sock"))
        t.bind(b"b-sock", "sha-b", "srv-b", "ck-b", 1)
        self.assertEqual(t.get(b"b-sock").generation, 1)

    def test_quiescence(self):
        t = SessionTable()
        self.assertTrue(t.is_quiescent(0))
        self.assertFalse(t.has_active_traffic(0))
        t.bind(b"id1", "sha", "srv", "ck", 0)
        # freshly used: not quiescent
        self.assertFalse(t.is_quiescent(0))
        self.assertTrue(t.has_active_traffic(0))
        # other generations unaffected
        self.assertTrue(t.is_quiescent(1))
        # backdate past the window: quiescent again
        b = t.get(b"id1")
        b.last_used_at = time.monotonic() - QUIESCENCE_S - 1.0
        self.assertTrue(t.is_quiescent(0))
        self.assertFalse(t.has_active_traffic(0))

    def test_superseded_and_active(self):
        t = SessionTable()
        self.assertFalse(t.is_superseded("ck-a"))
        t.mark_superseded("ck-a")
        self.assertTrue(t.is_superseded("ck-a"))
        t.clear_superseded("ck-a")
        self.assertFalse(t.is_superseded("ck-a"))
        t.set_active("ck-b", "sha-b", "srv-b", 2)
        self.assertEqual(t.active_compile_key, "ck-b")
        self.assertEqual(t.active_source_sha256, "sha-b")
        self.assertEqual(t.active_serving_digest, "srv-b")
        self.assertEqual(t.active_generation, 2)


if __name__ == "__main__":
    unittest.main()
