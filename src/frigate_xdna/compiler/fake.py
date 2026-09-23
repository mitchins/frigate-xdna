"""Fake compiler backend interface (control-plane tests only).

Defines the job lifecycle contract the real audited compiler job (Task 03)
must implement. No vendor imports, no NPU, no subprocesses, no network.
"""
from __future__ import annotations

from dataclasses import dataclass, field

TERMINAL_STATES = frozenset({
    "PREPARED", "COMPILE_FAILED", "RESOURCE_EXCEEDED", "VALIDATION_FAILED",
    "UNSUPPORTED_CONTRACT", "QUARANTINED", "INTERRUPTED",
})

# A fake "slow" compile must exceed Frigate's 30 s model-operation wait so
# cold-miss tests prove the truthful not-ready reply.
SLOW_COMPILE_S = 45.0


@dataclass
class FakeCompileJob:
    BACKEND_ID = "fake-v0"
    source_sha256: str
    compile_key: str
    duration_s: float = 0.0
    succeed: bool = True
    fail_state: str = "COMPILE_FAILED"
    device_required: bool = True
    device_held_by_worker: bool = True
    elapsed_s: float = 0.0
    state: str = "QUEUED"
    artifact_sha256: str | None = None
    log: list[str] = field(default_factory=list)
    job_uuid: str = ""
    # Scripted vendor tail for classification tests (mirrors the real
    # backend's result.detail; production never sets this).
    detail: str = ""

    @property
    def phase(self) -> str:
        """Fake has no sub-phases: the stage is the phase."""
        return self.state

    def poll(self, dt_s: float) -> str:
        """Advance the fake job by dt_s seconds. Pure function of inputs."""
        if self.state in TERMINAL_STATES:
            return self.state
        if self.device_required and self.device_held_by_worker:
            self.state = "WAITING_FOR_DEVICE"
            self.log.append("waiting-for-device")
            return self.state
        self.state = "COMPILING"
        self.elapsed_s += dt_s
        if self.elapsed_s >= self.duration_s:
            if self.succeed:
                self.state = "PREPARED"
                self.artifact_sha256 = "f" * 64
                self.log.append("prepared")
            else:
                self.state = self.fail_state
                self.log.append(f"failed:{self.fail_state}")
        return self.state
