"""Fake native worker interface (control-plane tests only).

Defines the private supervisor<->native IPC message shapes and the worker
generation lifecycle the real C++ child (Task 04) must implement.
Operates on the golden raw-tensor fixtures; never touches hardware.
"""
from __future__ import annotations

from dataclasses import dataclass, field

PROTOCOL_VERSION = 1

MESSAGE_TYPES = ("load", "infer", "result", "status", "shutdown")

ZERO_FRAME_BYTES = 20 * 6 * 4  # one 480-byte [20,6] float32 frame


@dataclass
class WorkerRequest:
    protocol_version: int = PROTOCOL_VERSION
    message_type: str = "infer"  # load | infer | result | status | shutdown
    request_id: int = 0
    worker_generation: int = 0
    artifact_path: str = ""
    serving_digest: str = ""
    tensor_nbytes: int = 0

    def valid(self) -> bool:
        return (
            self.protocol_version == PROTOCOL_VERSION
            and self.message_type in MESSAGE_TYPES
            and self.request_id >= 0
            and self.worker_generation >= 0
        )


@dataclass
class FakeNativeWorker:
    """Serves canned [20,6] frames from a fixture; tracks generation."""
    generation: int = 0
    serving_digest: str = ""
    loaded: bool = False
    requests_seen: int = 0
    retired: bool = False

    def load(self, generation: int, serving_digest: str) -> bool:
        if self.retired:
            return False
        self.generation = generation
        self.serving_digest = serving_digest
        self.loaded = True
        return True

    def infer(self, request: WorkerRequest, frame: bytes) -> bytes:
        self.requests_seen += 1
        if (
            self.retired
            or not self.loaded
            or not request.valid()
            or request.worker_generation != self.generation
            or len(frame) != request.tensor_nbytes
        ):
            return bytes(ZERO_FRAME_BYTES)
        import numpy as np

        canned = np.load(request.artifact_path)["expected"] \
            if request.artifact_path.endswith(".npz") else None
        if canned is None:
            return bytes(ZERO_FRAME_BYTES)
        return bytes(canned.astype("<f4", copy=False).tobytes())

    def retire(self) -> None:
        self.retired = True
        self.loaded = False
