"""Hardware-free tests for the NPU device lease (SPEC §3.1, §10).

The lease serializes compile/activation/inference inside this sidecar via
a file lock. No device is touched; the lock file lives in a temp dir.
"""
import os
import tempfile
import unittest

from frigate_xdna.runtime.device_lease import DEVICE_LOCK_NAME, DeviceLease


class TestDeviceLease(unittest.TestCase):
    def test_acquire_release(self):
        with tempfile.TemporaryDirectory() as d:
            lease = DeviceLease(d)
            self.assertFalse(lease.held())
            self.assertTrue(lease.try_acquire())
            self.assertTrue(lease.held())
            self.assertTrue(os.path.isfile(
                os.path.join(d, DEVICE_LOCK_NAME)))
            lease.release()
            self.assertFalse(lease.held())
            lease.release()  # idempotent

    def test_exclusive_contention(self):
        with tempfile.TemporaryDirectory() as d:
            a = DeviceLease(d)
            b = DeviceLease(d)
            self.assertTrue(a.try_acquire())
            self.assertFalse(b.try_acquire())
            self.assertFalse(b.acquire(timeout_s=0.1))
            a.release()
            self.assertTrue(b.try_acquire())
            b.release()

    def test_acquire_timeout_zero(self):
        with tempfile.TemporaryDirectory() as d:
            a = DeviceLease(d)
            b = DeviceLease(d)
            self.assertTrue(a.try_acquire())
            self.assertFalse(b.acquire())
            a.release()
            self.assertTrue(b.acquire())

    def test_is_locked_by_other(self):
        with tempfile.TemporaryDirectory() as d:
            a = DeviceLease(d)
            probe = DeviceLease(d)
            self.assertFalse(probe.is_locked_by_other())
            self.assertTrue(a.try_acquire())
            # own fd held: self-report is negative (self-fd guard)
            self.assertFalse(a.is_locked_by_other())
            # a different handle observes the lock
            self.assertTrue(probe.is_locked_by_other())
            a.release()
            self.assertFalse(probe.is_locked_by_other())

    def test_owner_pid_recorded(self):
        with tempfile.TemporaryDirectory() as d:
            lease = DeviceLease(d)
            self.assertTrue(lease.try_acquire())
            try:
                with open(os.path.join(d, DEVICE_LOCK_NAME)) as f:
                    self.assertEqual(f.read(), str(os.getpid()))
            finally:
                lease.release()


if __name__ == "__main__":
    unittest.main()
