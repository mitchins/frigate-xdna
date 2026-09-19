"""Device lease: serialize NPU operations (SPEC §3.1, §10).

One sidecar instance per physical NPU; lease is a file lock on
/data/device.lock. Cannot prevent unrelated software, but serializes
compile / activation / validation / inference / teardown inside this
sidecar.
"""
from __future__ import annotations

import fcntl
import os
import time

DEVICE_LOCK_NAME = "device.lock"


class DeviceLease:
    def __init__(self, data_dir: str):
        self.path = os.path.join(data_dir, DEVICE_LOCK_NAME)
        self._fd = None

    def try_acquire(self) -> bool:
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        except OSError:
            return False
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False
        self._fd = fd
        # write owner pid for diagnosis
        try:
            os.ftruncate(fd, 0)
            os.write(fd, str(os.getpid()).encode())
            os.fsync(fd)
        except OSError:
            pass
        return True

    def acquire(self, timeout_s: float = 0.0) -> bool:
        deadline = time.monotonic() + timeout_s if timeout_s > 0 else None
        while True:
            if self.try_acquire():
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
            if deadline is None:
                return False
            time.sleep(0.05)

    def release(self) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

    def held(self) -> bool:
        return self._fd is not None

    def is_locked_by_other(self) -> bool:
        """Check if another process holds the lock (non-blocking probe)."""
        if self._fd is not None:
            return False
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        except OSError:
            return False
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
            return False
        except OSError:
            try:
                os.close(fd)
            except OSError:
                pass
            return True
