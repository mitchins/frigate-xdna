"""Preparation job management (Task 02: fake compile backend).

Same-key requests join one job (no duplicates). Terminal failures
carry structured records (retry_policy); automatic retries are
bounded (backoff, persisted attempts) and only for eligible classes —
a verified correction or safe interrupt resumes, everything else waits
for an explicit operator recover. The fake backend stands in for the
audited compiler job (Task 03) behind this exact interface.
"""
from __future__ import annotations

import time
import uuid

from .. import retry_policy as _retry
from ..errors import NOT_READY, FxdnaError
from .fake import TERMINAL_STATES, FakeCompileJob

# FakeCompileJob constructor params; anything else in a submit's extra
# mapping belongs to real backends and is filtered out here.
_FAKE_PARAMS = frozenset({
    "source_sha256", "compile_key", "duration_s", "succeed", "fail_state",
    "device_required", "device_held_by_worker", "job_uuid", "detail"})

_NON_TERMINAL = ("QUEUED", "WAITING_FOR_DEVICE", "COMPILING",
                 "FETCHING", "DOWNLOADED", "INSPECTED")

TERMINAL_ERROR_STATES = frozenset({
    "COMPILE_FAILED", "RESOURCE_EXCEEDED", "VALIDATION_FAILED",
    "UNSUPPORTED_CONTRACT", "QUARANTINED", "INTERRUPTED",
})


class JobManager:
    def __init__(self, registry, backend_factory=None,
                 boot_token: str | None = None, listener=None):
        self.registry = registry
        self.backend_factory = backend_factory or (
            lambda **kw: FakeCompileJob(
                **{k: v for k, v in kw.items() if k in _FAKE_PARAMS}))
        self.boot_token = boot_token
        # listener(event) receives retry/resume transitions for console
        # reporting (supervisor maps them to reporter events).
        self.listener = listener
        self._backends: dict[str, FakeCompileJob] = {}

    def submit(self, ref: str, compile_key: str | None,
               duration_s: float = 0.0, succeed: bool = True,
               fail_state: str = "COMPILE_FAILED",
               device_required: bool = False,
               extra: dict | None = None, **kw) -> dict:
        """Submit or join a preparation job. Returns the job record.

        Lookup and insert run inside one exclusive transaction so
        concurrent submitters cannot create duplicate live jobs for the
        same ref/compile key. Extra backend params (e.g. scripted
        failure detail) flow into new backend objects only.
        """
        with self.registry.transaction():
            existing = self.registry.find_job(ref, compile_key)
            if existing and existing["stage"] not in TERMINAL_STATES:
                # Join the live job; record this ref as an alias so status
                # advances for every alias sharing the compile key (S9).
                self.registry.add_alias(existing["uuid"], ref)
                return existing
            if existing and existing["stage"] in TERMINAL_ERROR_STATES:
                return self._submit_terminal(
                    ref, compile_key, existing,
                    duration_s=duration_s, succeed=succeed,
                    fail_state=fail_state,
                    device_required=device_required,
                    extra={**(extra or {}), **kw})
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
                                fail_state, device_required,
                                **(extra or {}))

    def backend_for(self, job_uuid: str):
        """Backend job object (for result retrieval); None if unknown."""
        return self._backends.get(job_uuid)

    def _submit_terminal(self, ref: str, compile_key: str | None,
                         existing: dict, duration_s: float = 0.0,
                         succeed: bool = True,
                         fail_state: str = "COMPILE_FAILED",
                         device_required: bool = False,
                         extra: dict | None = None) -> dict:
        """Submit against a terminal row: resume when eligible,
        otherwise return the row unchanged."""
        full = self.registry.get_job(existing["uuid"])
        resumed = self._maybe_resume(
            ref, compile_key, full,
            duration_s=duration_s, succeed=succeed,
            fail_state=fail_state, device_required=device_required,
            extra=extra, trigger="submit")
        return resumed if resumed is not None else existing

    def retry_due(self, now: float | None = None, **kw) -> list[dict]:
        """Open bounded attempts for eligible terminal rows whose time
        has come (backoff elapsed, correction verified, safe resume).

        The serve loop calls this every pump so backoff expiry is
        noticed without a new prepare. Returns the new live rows.
        """
        import time as _time
        now = _time.time() if now is None else now
        opened = []
        for row in self.registry.query(
                "SELECT uuid, ref, compile_key FROM jobs WHERE stage IN"
                " ('COMPILE_FAILED','RESOURCE_EXCEEDED',"
                "'VALIDATION_FAILED','UNSUPPORTED_CONTRACT',"
                "'QUARANTINED','INTERRUPTED')"
                " AND failure_json IS NOT NULL"):
            # One transaction per row: the live check, eligibility and
            # creation are atomic, so a concurrent submit (which holds
            # its own transaction) cannot interleave a duplicate
            # attempt between this row's check and creation.
            with self.registry.transaction():
                full = self.registry.get_job(row[0])
                if full is None:
                    continue
                resumed = self._maybe_resume(
                    row[1], row[2], full, trigger="pump", **kw)
                if resumed is not None:
                    opened.append(resumed)
        return opened

    def _maybe_resume(self, ref: str, compile_key: str | None,
                      failed: dict, duration_s: float = 0.0,
                      succeed: bool = True,
                      fail_state: str = "COMPILE_FAILED",
                      device_required: bool = False,
                      extra: dict | None = None,
                      trigger: str = "submit",
                      **kw) -> dict | None:
        """Open a new bounded attempt for an eligible terminal row.

        Returns the new live job, or None when the row stays terminal
        (ineligible class, exhausted budget, backoff pending, live
        duplicate, or any safety inhibition). Retries never duplicate
        a live compile for the key.
        """
        record = failed.get("failure")
        if compile_key and self.registry.live_job_for_key(compile_key):
            return None
        if self.registry.get_state("inhibition"):
            return None
        if record is None:
            return None
        eligible, why = _retry.auto_eligible(record, time.time())
        if not eligible:
            return None
        new = self._new_attempt(
            ref, compile_key, record, why, trigger,
            duration_s=duration_s, succeed=succeed, fail_state=fail_state,
            device_required=device_required,
            extra={**(extra or {}), **kw},
            prev_uuid=failed.get("uuid"),
            source_sha256=(extra or {}).get("source_sha256")
            or failed.get("source_sha256"))
        if new is None:
            return None
        # Consume this terminal row: it must never spawn a second
        # resume (its backoff stays expired; without this every pump
        # would duplicate the attempt).
        consumed = {**record, "resumed_to": new["uuid"]}
        self.registry.set_failure(failed["uuid"], consumed)
        return new

    def _new_attempt(self, ref: str, compile_key: str | None,
                     prev: dict, why: str, trigger: str,
                     duration_s: float = 0.0, succeed: bool = True,
                     fail_state: str = "COMPILE_FAILED",
                     device_required: bool = False,
                     extra: dict | None = None,
                     prev_uuid: str | None = None,
                     source_sha256: str | None = None) -> dict | None:
        """Create attempt N+1 as a new job row (history preserved on
        the old terminal rows and carried on the new one; committed
        artifacts never touched). The attempt binds the failed
        attempt's source bytes, never the ref's current bytes. None
        when backend construction refuses (e.g. unresolvable resume
        source)."""
        job_uuid = uuid.uuid4().hex
        attempt = int(prev.get("attempts", 0)) + 1
        self.registry.create_job(job_uuid, ref, compile_key,
                                 self.boot_token, attempt=attempt,
                                 source_sha256=source_sha256)
        self.registry.set_failure(
            job_uuid, {"resumed_from": prev_uuid,
                       "history": list(prev.get("history", []))})
        self._carry_aliases(job_uuid, ref, prev_uuid)
        kwargs = {"source_sha256": source_sha256 or "",
                  "compile_key": compile_key or "",
                  "duration_s": duration_s, "succeed": succeed,
                  "fail_state": fail_state,
                  "device_required": device_required,
                  "device_held_by_worker": device_required,
                  "job_uuid": job_uuid}
        kwargs.update(extra or {})
        if not self._spawn_backend(job_uuid, kwargs, prev_uuid):
            return None
        self.registry.set_ref_state(ref, "QUEUED")
        if self.listener is not None:
            self.listener({"kind": "resumed" if why in (
                "safe resume", "blocking condition corrected",
                "operator recover") else "retry_scheduled",
                "ref": ref, "attempt": attempt, "reason": why,
                "trigger": trigger})
        job = self.registry.get_job(job_uuid)
        assert job is not None
        return job

    def _carry_aliases(self, job_uuid: str, ref: str,
                       prev_uuid: str | None) -> None:
        """Joined aliases follow the retry: without them, an alias
        ref would report PREPARED with no reachable key after the new
        attempt succeeds."""
        if prev_uuid is None:
            return
        for alias in self.registry.refs_for_job(prev_uuid):
            if alias != ref:
                self.registry.add_alias(job_uuid, alias)

    def _spawn_backend(self, job_uuid: str, kwargs: dict,
                       prev_uuid: str | None) -> bool:
        """Construct the backend object. False when refused: the row
        is rolled back so the key stays recompilable, and the source
        row is stamped so automatic retries stop burning
        constructions against the same refusal."""
        try:
            self._backends[job_uuid] = self.backend_factory(**kwargs)
        except Exception as e:
            self.registry.execute("DELETE FROM jobs WHERE uuid=?",
                                  (job_uuid,))
            if prev_uuid is not None:
                prior = self.registry.get_job(prev_uuid)
                if prior is not None:
                    self.registry.set_failure(
                        prev_uuid,
                        {**(prior.get("failure") or {}),
                         "resume_refused": str(e)[:200]})
            return False
        return True

    def _requeue_prev(self, full: dict) -> dict | None:
        """Previous-record baseline for an operator cycle: None when
        the class refuses. Attempts reset; history carries over."""
        record = full.get("failure")
        kind = (record or {}).get("kind", "unknown")
        if kind in ("safety", "permanent"):
            return None
        if kind == "unknown" and record is None:
            # Legacy row without evidence: acknowledged operator risk,
            # one bounded cycle, attempts restart.
            return {"attempts": 0, "history": []}
        if not (record or {}).get("retryable", False):
            return None
        prev = {**(record or {})}
        prev["attempts"] = 0
        return prev

    def requeue(self, ref: str, compile_key: str | None) -> tuple:
        """Explicit acknowledged operator retry (recover path).

        Returns (new_row, None) with attempts reset for the new
        cycle, or (None, reason) spelling out the refusal (safety /
        permanent class, active inhibition, live duplicate,
        unresolvable resume source). Callers surface the reason;
        refusals are never silent.
        """
        with self.registry.transaction():
            existing = self.registry.find_job(ref, compile_key)
            if existing is None or existing["stage"] not in \
                    TERMINAL_ERROR_STATES:
                return None, "no retryable terminal job for this ref"
            inh = self.registry.get_state("inhibition")
            if inh is not None:
                # Safety is device-global: recover clears a same-ref
                # inhibition before reaching here, so anything still
                # present (any ref, any class) blocks new compiles.
                return None, (f"inhibition active for {inh.get('ref')}"
                              f" ({inh.get('reason')}); resolve it first")
            full = self.registry.get_job(existing["uuid"])
            assert full is not None
            prev = self._requeue_prev(full)
            if prev is None:
                record = full.get("failure") or {}
                return None, (f"{record.get('kind', 'unknown')} failure"
                               f" ({record.get('code', '?')}) never"
                               " retries; see"
                               f" `fxdna status {ref}` for guidance")
            if compile_key and self.registry.live_job_for_key(
                    compile_key):
                return None, "a live attempt already exists"
            new = self._new_attempt(
                ref, compile_key, prev, "operator recover", "recover",
                prev_uuid=existing["uuid"],
                source_sha256=full.get("source_sha256"))
            if new is None:
                return None, ("resume source unresolvable; re-submit"
                               " source explicitly")
            # Consume the source row like automatic resumes do: after
            # the new cycle goes terminal, the old row must not open a
            # second chain around the attempt bound.
            self.registry.set_failure(
                existing["uuid"],
                {**(full.get("failure") or {}),
                 "resumed_to": new["uuid"]})
            return new, None

    def _create(self, ref: str, compile_key: str | None,
                duration_s: float, succeed: bool, fail_state: str,
                device_required: bool, **extra) -> dict:
        job_uuid = uuid.uuid4().hex
        self.registry.create_job(job_uuid, ref, compile_key,
                                 self.boot_token,
                                 source_sha256=extra.get("source_sha256"))
        kwargs = {"source_sha256": "", "compile_key": compile_key or "",
                  "duration_s": duration_s, "succeed": succeed,
                  "fail_state": fail_state,
                  "device_required": device_required,
                  "device_held_by_worker": device_required,
                  "job_uuid": job_uuid}
        kwargs.update(extra or {})
        self._backends[job_uuid] = self.backend_factory(**kwargs)
        return self.registry.get_job(job_uuid)

    def record_terminal(self, job_uuid: str, stage: str,
                        error_code: str | None, detail: str = "",
                        error_text: str = "") -> dict:
        """Write the structured failure record for a terminal job,
        preserving attempt history across retries."""
        job = self.registry.get_job(job_uuid)
        assert job is not None
        prev = (job.get("failure") or {}).get("history", [])
        self.registry.set_failure(
            job_uuid,
            _retry.new_record(stage, error_code, detail,
                              job.get("attempt", 1),
                              history_base=prev,
                              error_text=error_text))
        row = self.registry.get_job(job_uuid)
        assert row is not None
        return row

    def pump(self, job_uuid: str, dt_s: float) -> dict:
        """Advance one job; mirror backend state into the registry.

        Terminal stages get structured failure records; eligible
        rows open a new bounded attempt (backoff/correction/resume),
        otherwise the row stays terminal for operator review."""
        backend = self._backends.get(job_uuid)
        if backend is None:
            job = self.registry.get_job(job_uuid)
            if job is None:
                raise FxdnaError(NOT_READY, "UNKNOWN_JOB",
                                 f"no such job {job_uuid}")
            return job
        prev = self.registry.get_job(job_uuid)
        prev_stage = (prev or {}).get("stage")
        stage = backend.poll(dt_s)
        phase = getattr(backend, "phase", None)
        progress = f"elapsed={backend.elapsed_s:.1f}s"
        if phase and phase != stage:
            progress += f" phase={phase}"
        self.registry.set_job(
            job_uuid, stage,
            error_code=stage if stage in TERMINAL_ERROR_STATES else None,
            progress=progress)
        job = self.registry.get_job(job_uuid)
        assert job is not None
        if stage in TERMINAL_ERROR_STATES and not (
                prev_stage == stage and job.get("failure")):
            return self._finish_terminal(job_uuid, backend, job)
        return job

    def _finish_terminal(self, job_uuid: str, backend, job: dict) -> dict:
        """Record a fresh terminal outcome, then open a bounded retry
        when eligible (otherwise the row stays for operator review)."""
        detail, error_text = "", ""
        result = getattr(backend, "result", None)
        if result is not None:
            detail = getattr(result, "detail", "") or ""
            error_text = getattr(result, "error", "") or ""
        elif hasattr(backend, "detail"):
            detail = getattr(backend, "detail") or ""
        job = self.record_terminal(job_uuid, job["stage"],
                                   job.get("error_code"), detail,
                                   error_text=error_text)
        resumed = self._maybe_resume(
            job["ref"], job["compile_key"], job, trigger="pump")
        return resumed if resumed is not None else job

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
