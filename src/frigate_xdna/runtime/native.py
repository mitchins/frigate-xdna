"""Resident native worker supervision (SPEC §3, INTERFACES.md §4).

One worker child per active model: spawned on activation, LOADs the
verified artifact over a private socketpair (inherited fd, no public
socket), serves bounded INFER, retired on switch. A dead/failed worker
inhibits (explicit recover); it is NEVER respawned automatically —
no autoloop after a suspected device fault.

Wire protocol: JSON headers + binary payload over the socketpair
(`runtime/ipc.py` framing, `native/include/protocol.h` semantics).
"""
from __future__ import annotations

import itertools
import os
import subprocess

from . import ipc

RESULT_BYTES = 20 * 6 * 4  # one 480-byte [20,6] float32 frame

LOAD_TIMEOUT_S = 25.0  # Frigate's model wait is 30 s; LOAD must fit inside
RETIRE_GRACE_S = 5.0


class WorkerError(Exception):
    """Structured worker failure (carries the native error_code)."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


class NativeWorker:
    """A supervised fxdna-worker child. Synchronous, bounded, one at a time.

    The supervisor serializes calls (the worker serves one INFER at a
    time); the ZMQ frontend runs them off the event loop with the
    request's remaining budget as timeout.
    """

    def __init__(self, sock, proc: subprocess.Popen, generation: int):
        self._sock = sock
        self._proc = proc
        self._generation = generation
        self._requests = itertools.count(1)
        self.loaded = False

    @classmethod
    def spawn(cls, worker_bin: str, lib_dirs: list[str],
              xrt_root: str | None = None) -> NativeWorker:
        """Spawn the worker with a private socketpair on an inherited fd.

        FlexML resolves XRT via XILINX_XRT (not just LD_LIBRARY_PATH);
        without it Model creation fails before ever touching the device.
        """
        if not os.path.isfile(worker_bin) or not os.access(worker_bin,
                                                            os.X_OK):
            raise WorkerError("WORKER_MISSING",
                              f"worker binary not executable: {worker_bin}")
        sup_sock, child_sock = ipc.create_socketpair()
        try:
            fdno = child_sock.fileno()
            env = {"PATH": "/usr/bin:/bin",
                   "LD_LIBRARY_PATH": ":".join(lib_dirs),
                   "XILINX_XRT": xrt_root or worker_xrt_root()}
            proc = subprocess.Popen(
                [worker_bin, "--fd", str(fdno)],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, close_fds=True,
                pass_fds=(fdno,), env=env)
        except (OSError, ValueError) as e:
            sup_sock.close()
            child_sock.close()
            raise WorkerError("WORKER_SPAWN",
                              f"worker spawn failed: {e}") from e
        child_sock.close()
        return cls(sup_sock, proc, 0)

    def alive(self) -> bool:
        return self._proc.poll() is None

    def _exchange(self, header: dict, payload: bytes,
                  timeout_s: float) -> tuple[dict, bytes]:
        if not self.alive():
            raise WorkerError("WORKER_DEAD", "worker child exited")
        try:
            ipc.send_message(self._sock, header, payload)
            return ipc.recv_message(self._sock, timeout=timeout_s)
        except (OSError, ValueError, ConnectionError) as e:
            raise WorkerError("WORKER_IO",
                              f"worker IPC failed: {e}") from e

    def load(self, artifact_path: str, generation: int,
             serving_digest: str, class_count: int,
             timeout_s: float = LOAD_TIMEOUT_S) -> None:
        """LOAD the verified artifact; raises WorkerError on any refusal."""
        if class_count <= 0:
            raise WorkerError("INVALID_ARGS",
                              "LOAD requires inspected class_count")
        rep, _ = self._exchange(
            {"message_type": "LOAD",
             "request_id": next(self._requests),
             "worker_generation": generation,
             "artifact_path": artifact_path,
             "serving_digest": serving_digest,
             "class_count": class_count},
            b"", timeout_s)
        if rep.get("message_type") != "STATUS" or rep.get("error_code"):
            raise WorkerError(rep.get("error_code") or "LOAD_REFUSED",
                              rep.get("error_message") or "LOAD refused")
        self._generation = generation
        self.loaded = True

    def infer(self, payload: bytes, shape: list[int], generation: int,
              timeout_s: float) -> bytes:
        """Bounded INFER; returns the 480-byte result frame.

        Raises WorkerError on transport failure, generation mismatch,
        unloaded worker, or native-reported failure (incl. DEVICE_FAULT).
        """
        rep, out = self._exchange(
            {"message_type": "INFER",
             "request_id": next(self._requests),
             "worker_generation": generation,
             "tensor_spec": {"shape": list(shape), "dtype": "float32"}},
            payload, timeout_s)
        if rep.get("message_type") != "RESULT" or rep.get("error_code"):
            raise WorkerError(rep.get("error_code") or "INFER_REFUSED",
                              rep.get("error_message") or "INFER refused")
        if len(out) != RESULT_BYTES:
            raise WorkerError("INFER_SHORT",
                              f"result {len(out)} bytes, want {RESULT_BYTES}")
        return out

    def retire(self) -> None:
        """Retire exactly once: SHUTDOWN, bounded SIGTERM, then SIGKILL.

        Single pass, no loops, no respawn. Called on model switch and at
        supervisor stop; safe to call on an already-dead child.
        """
        try:
            if self.alive():
                try:
                    ipc.send_message(
                        self._sock,
                        {"message_type": "SHUTDOWN",
                         "request_id": next(self._requests),
                         "worker_generation": self._generation})
                except (OSError, ValueError):
                    pass
        finally:
            try:
                self._sock.close()
            except OSError:
                pass
        try:
            self._proc.wait(timeout=RETIRE_GRACE_S)
        except subprocess.TimeoutExpired:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=RETIRE_GRACE_S)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=RETIRE_GRACE_S)
        except (OSError, ValueError):
            pass
        self.loaded = False

    @property
    def generation(self) -> int:
        return self._generation


def worker_lib_dirs() -> list[str]:
    """Runtime library dirs for the worker child (image layout)."""
    return ["/opt/flexmlrt/lib", "/opt/xilinx-xrt/lib"]


def worker_binary() -> str:
    """Worker binary path (image layout; tests inject a factory)."""
    return os.environ.get("FXDNA_WORKER_BIN", "/opt/fxdna/native/fxdna-worker")


def worker_xrt_root() -> str:
    """XRT prefix for the worker child (image layout; native dev overrides
    via FXDNA_XRT_ROOT)."""
    return os.environ.get("FXDNA_XRT_ROOT", "/opt/xilinx-xrt")


