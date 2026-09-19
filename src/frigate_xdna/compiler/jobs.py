"""Preparation job management (Task 02: fake compile backend).

Same-key requests join one job (no duplicates). Deterministic failures are
terminal with no auto-retry. The fake backend stands in for the audited
compiler job (Task 03) behind this exact interface.
"""
from __future__ import annotations

import time
import uuid

from ..errors import FxdnaError, NOT_READY
from .fake import TERMINAL_STATES, FakeCompileJob

_NON_TERMINAL = ("QUEUED", "WAITING_FOR_DEVICE", "COMPILING",
                 "FETCHING", "DOWNLOADED", "INSPECTED")

TERMINAL_ERROR_STATES = frozenset({
    "COMPILE_FAILED", "RESOURCE_EXCEEDED", "VALIDATION_FAILED",
    "UNSUPPORTED_CONTRACT", "QUARANTINED", "INTERRUPTED",
})


class JobManager:
    def __init__(self, registry, backend_factory=None,
                 boot_token: str | None = None):
        self.registry = registry
        self.backend_factory = backend_factory or (lambda **kw: FakeCompileJob(**kw))
        self.boot_token = boot_token
        self._backends: dict[str, FakeCompileJob] = {}

    def submit(self, ref: str, compile_key: str | None,
               duration_s: float = 0.0, succeed: bool = True,
               fail_state: str = "COMPILE_FAILED",
               device_required: bool = False) -> dict:
        """Submit or join a preparation job. Returns the job record.

        Lookup and insert run inside one exclusive transaction so
        concurrent submitters cannot create duplicate live jobs for the
        same ref/compile key.
        """
        with self.registry.transaction():
            existing = self.registry.find_job(ref, compile_key)
            if existing and existing["stage"] not in TERMINAL_STATES:
                # Join the live job; record this ref as an alias so status
                # advances for every alias sharing the compile key (S9).
                self.registry.add_alias(existing["uuid"], ref)
                return existing
            if existing and existing["stage"] in TERMINAL_ERROR_STATES:
                # No automatic retry of a deterministically failed job.
                return existing
            if existing and existing["stage"] == "PREPARED":
                # Reusable only with a committed artifact behind it;
                # otherwise the key genuinely needs work (recovery path).
                if compile_key and self.registry.get_artifact(compile_key):
                    return existing
            if compile_key:
                # Cross-ref dedup: different aliases sharing bytes share
                # the compile key; join the live job, never duplicate it.
                rows = self.registry.query(
                    "SELECT uuid, ref, compile_key, stage, attempt,"
                    " error_code, progress FROM jobs WHERE compile_key=?"
                    f" AND stage IN ({','.join('?' * len(_NON_TERMINAL))})"
                    " ORDER BY created_at LIMIT 1",
                    (compile_key, *_NON_TERMINAL))
                if rows:
                    self.registry.add_alias(rows[0][0], ref)
                    return dict(zip(("uuid", "ref", "compile_key", "stage",
                                     "attempt", "error_code", "progress"),
                                    rows[0]))
            return self._create(ref, compile_key, duration_s, succeed,
                                fail_state, device_required)

    def _create(self, ref: str, compile_key: str | None,
                duration_s: float, succeed: bool, fail_state: str,
                device_required: bool) -> dict:
        job_uuid = uuid.uuid4().hex
        self.registry.create_job(job_uuid, ref, compile_key, self.boot_token)
        self._backends[job_uuid] = self.backend_factory(
            source_sha256="", compile_key=compile_key or "",
            duration_s=duration_s, succeed=succeed, fail_state=fail_state,
            device_required=device_required,
            device_held_by_worker=device_required)
        return self.registry.get_job(job_uuid)

    def pump(self, job_uuid: str, dt_s: float) -> dict:
        """Advance one job; mirror backend state into the registry."""
        backend = self._backends.get(job_uuid)
        if backend is None:
            job = self.registry.get_job(job_uuid)
            if job is None:
                raise FxdnaError(NOT_READY, "UNKNOWN_JOB",
                                 f"no such job {job_uuid}")
            return job
        stage = backend.poll(dt_s)
        self.registry.set_job(
            job_uuid, stage,
            error_code=stage if stage in TERMINAL_ERROR_STATES else None,
            progress=f"elapsed={backend.elapsed_s:.1f}s")
        return self.registry.get_job(job_uuid)

    def wait(self, job_uuid: str, timeout_s: float,
             step_s: float = 0.05) -> dict:
        """Pump until terminal state or timeout. Never fabricates progress."""
        deadline = time.monotonic() + timeout_s
        while True:
            job = self.pump(job_uuid, step_s)
            if job["stage"] in TERMINAL_STATES:
                return job
            if time.monotonic() >= deadline:
                raise FxdnaError(NOT_READY, "WAIT_TIMEOUT",
                                 f"job {job_uuid} not terminal after "
                                 f"{timeout_s}s (stage {job['stage']})")
            time.sleep(min(step_s, 0.05))
