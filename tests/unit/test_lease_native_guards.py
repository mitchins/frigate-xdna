"""Unit tests: device-lease and native-worker failure paths (no NPU).

Lock contention, unusable lock paths and dead-child retirement must
produce explicit booleans/codes — never hangs, respawns or escapes.
"""
import os
import socket
import subprocess
import tempfile
import threading
import unittest
from unittest import mock

from frigate_xdna.runtime import ipc
from frigate_xdna.runtime import native as native_mod
from frigate_xdna.runtime.device_lease import DeviceLease
from frigate_xdna.runtime.native import NativeWorker, WorkerError


class TestDeviceLeaseGuards(unittest.TestCase):
    def test_unusable_path_refuses(self):
        with tempfile.TemporaryDirectory() as d:
            lease = DeviceLease(os.path.join(d, "no-such-dir"))
            self.assertFalse(lease.try_acquire())
            self.assertFalse(lease.is_locked_by_other())

    def test_second_acquirer_loses_while_held(self):
        with tempfile.TemporaryDirectory() as d:
            first = DeviceLease(d)
            second = DeviceLease(d)
            self.assertTrue(first.try_acquire())
            try:
                self.assertFalse(second.try_acquire())
                self.assertTrue(second.is_locked_by_other())
                self.assertFalse(first.is_locked_by_other())
            finally:
                first.release()
            self.assertTrue(second.try_acquire())
            second.release()

    def test_pid_write_failure_still_acquires(self):
        with tempfile.TemporaryDirectory() as d:
            lease = DeviceLease(d)
            with mock.patch.object(os, "ftruncate",
                                   side_effect=OSError("ro")):
                self.assertTrue(lease.try_acquire())
            lease.release()

    def test_release_survives_dead_fd(self):
        with tempfile.TemporaryDirectory() as d:
            lease = DeviceLease(d)
            self.assertTrue(lease.try_acquire())
            os.close(lease._fd)
            lease.release()
            self.assertFalse(lease.held())

    def test_busy_probe_close_failure_still_reports_busy(self):
        with tempfile.TemporaryDirectory() as d:
            holder = DeviceLease(d)
            self.assertTrue(holder.try_acquire())
            try:
                probe = DeviceLease(d)
                real_close = os.close

                def close_then_fail(fd):
                    real_close(fd)
                    raise OSError("bad")

                with mock.patch.object(os, "close",
                                       side_effect=close_then_fail):
                    self.assertTrue(probe.is_locked_by_other())
            finally:
                holder.release()


class FakeSockUnused:
    pass


class FakeProc:
    """Popen interface with scripted wait behavior."""

    def __init__(self, waits):
        self.waits = list(waits)
        self.terminated = False
        self.killed = False

    def poll(self):
        return None

    def wait(self, timeout=None):
        outcome = self.waits.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


class FakeSock:
    def __init__(self):
        self.closed = False
        self.sent = []

    def close(self):
        self.closed = True


class TestNativeWorkerGuards(unittest.TestCase):
    def test_spawn_failure_reports_worker_spawn(self):
        with mock.patch.object(subprocess, "Popen",
                               side_effect=OSError("noexec")):
            with self.assertRaises(WorkerError) as ctx:
                NativeWorker.spawn("/bin/true", [], xrt_root="/xrt")
            self.assertEqual(ctx.exception.code, "WORKER_SPAWN")

    def test_retire_escalates_term_then_kill(self):
        proc = FakeProc([subprocess.TimeoutExpired("w", 1),
                         subprocess.TimeoutExpired("w", 1), 0])
        sock, peer = socket.socketpair()
        try:
            worker = NativeWorker(sock, proc, 0)
            worker.retire()
        finally:
            peer.close()
        self.assertTrue(proc.terminated)
        self.assertTrue(proc.killed)
        self.assertFalse(worker.loaded)

    def test_retire_tolerates_wait_errors(self):
        proc = FakeProc([OSError("gone")])
        sock, peer = socket.socketpair()
        try:
            worker = NativeWorker(sock, proc, 0)
            worker.retire()
        finally:
            peer.close()
        self.assertFalse(worker.loaded)

    def test_exchange_io_failure_maps_to_worker_io(self):
        proc = FakeProc([0])
        sock, peer = socket.socketpair()
        try:
            worker = NativeWorker(sock, proc, 0)
            with mock.patch.object(native_mod.ipc, "send_message",
                                   side_effect=OSError("broken")):
                with self.assertRaises(WorkerError) as ctx:
                    worker.load("a.rai", 1, "digest", 2)
            self.assertEqual(ctx.exception.code, "WORKER_IO")
        finally:
            peer.close()

    def test_infer_short_frame_refused(self):
        worker_sock, peer_sock = socket.socketpair()
        proc = FakeProc([0])
        worker = NativeWorker(worker_sock, proc, 0)

        def peer():
            req, _payload = ipc.recv_message(peer_sock, timeout=5.0)
            ipc.send_message(
                peer_sock,
                {"message_type": "RESULT",
                 "request_id": req.get("request_id", 1)},
                b"\x00" * 10)
            peer_sock.close()

        thread = threading.Thread(target=peer, daemon=True)
        thread.start()
        try:
            with self.assertRaises(WorkerError) as ctx:
                worker.infer(b"\x00" * 48, [1, 3, 2, 2], 0, 5.0)
            self.assertEqual(ctx.exception.code, "INFER_SHORT")
        finally:
            thread.join(timeout=5.0)

    def test_retire_tolerates_send_failure(self):
        proc = FakeProc([0])
        sock, peer = socket.socketpair()
        try:
            worker = NativeWorker(sock, proc, 0)
            with mock.patch.object(native_mod.ipc, "send_message",
                                   side_effect=OSError("broken")):
                worker.retire()
        finally:
            peer.close()
        self.assertFalse(worker.loaded)

    def test_retire_tolerates_close_failure(self):
        proc = FakeProc([0])
        sock, peer = socket.socketpair()
        try:
            worker = NativeWorker(sock, proc, 0)
            worker._sock = ClosingBoomSocket(sock)
            worker.retire()
        finally:
            peer.close()
        self.assertFalse(worker.loaded)


class ClosingBoomSocket:
    """Real socket whose close explodes (retire must not care)."""

    def __init__(self, sock):
        self._sock = sock

    def __getattr__(self, name):
        if name == "close":
            raise OSError("close exploded")
        return getattr(self._sock, name)


if __name__ == "__main__":
    unittest.main()
