"""Live state projection (v0.1.1 checkpoint 4).

One account of current state shared by daemon status, read-only CLI
status, waits and console reporting. Stored ref states
(NEW/QUEUED/.../PREPARED/terminals) are overlaid with:

* ACTIVE — a live worker serves this ref's prepared artifact. Never
  stored: a persisted `active` record is history, not proof, so after
  a crash/restart nothing claims readiness without a live worker.
* VERIFIED — the ref's prepared artifact has a recorded native-check
  pass (in-compile probe or activation LOAD) in the validations
  table. Sticky across restarts; failed checks never clear it.
* phase/elapsed_s — from the newest job row (sub-phase when the
  backend reports one, else the stage).
* error_code — the terminal code when the newest job failed.

Rank order PREPARED(1) < VERIFIED(2) < ACTIVE(3) lets `wait --state`
treat a better state as satisfying: an ACTIVE model is verified and
prepared by construction (activation records verification).
"""
from __future__ import annotations

import re

_ELAPSED_RE = re.compile(r"elapsed=(\d+(?:\.\d+)?)s")
_PHASE_RE = re.compile(r"phase=([A-Z_]+)")

STATE_RANK = {"PREPARED": 1, "VERIFIED": 2, "ACTIVE": 3}


def parse_progress(progress: str | None) -> tuple[float | None, str | None]:
    """(elapsed_s, phase) from a job progress string; Nones when absent."""
    if not progress:
        return None, None
    elapsed, phase = None, None
    m = _ELAPSED_RE.search(progress)
    if m:
        try:
            elapsed = float(m.group(1))
        except ValueError:
            elapsed = None
    m = _PHASE_RE.search(progress)
    if m:
        phase = m.group(1)
    return elapsed, phase


def _ranked_state(stored: str, verified: bool,
                  live_worker_key: str | None, key: str | None) -> str:
    if live_worker_key and key == live_worker_key:
        return "ACTIVE"
    if stored == "PREPARED" and verified:
        return "VERIFIED"
    return stored


_LEGACY_NO_RETRY = frozenset({
    "QUARANTINED", "VALIDATION_FAILED", "UNSUPPORTED_CONTRACT"})


def _failure_view(failure, job) -> tuple[dict | None, bool]:
    """(display record, retryable) for a job failure value. Lineage
    markers and refusal stamps on rows (no phase) display nothing and
    are never retryable — mirroring _requeue_prev, which refuses
    anything without retryable=True. Legacy terminal rows with NO
    record at all mirror the requeue rule: retryable unless a
    safety/permanent stage; live rows never are."""
    from .compiler.jobs import TERMINAL_ERROR_STATES
    if not isinstance(failure, dict) or "phase" not in failure:
        if failure is not None:
            return None, False
        stage = (job or {}).get("stage")
        return None, bool(stage and stage in TERMINAL_ERROR_STATES
                          and stage not in _LEGACY_NO_RETRY)
    return {"phase": failure.get("phase"), "code": failure.get("code"),
            "reason": failure.get("reason"),
            "kind": failure.get("kind"),
            "guidance": failure.get("guidance"),
            "attempts": failure.get("attempts", 0)}, \
        bool(failure.get("retryable"))


def project_ref(registry, ref: str,
                live_worker_key: str | None) -> dict:
    """Projected view for one ref: {state, phase, elapsed_s, verified,
    error_code, compile_key}. ACTIVE only with a live worker on this
    ref's prepared key."""
    rec = registry.get_ref(ref)
    stored = (rec or {}).get("state") or "NEW"
    source = (rec or {}).get("source_sha256") or ""
    job = registry.latest_job_for_ref(ref)
    key = registry.prepared_key_for_ref(ref)
    verified = bool(key) and registry.is_verified(key)
    error = (job or {}).get("error_code")
    elapsed, sub = parse_progress((job or {}).get("progress"))
    phase = sub or ((job or {}).get("stage")
                    if job and job["stage"] not in ("PREPARED",) else None)
    failure, retryable = _failure_view((job or {}).get("failure"),
                                         job)
    return {"ref": ref, "source_sha256": source,
            "state": _ranked_state(stored, verified, live_worker_key,
                                   key),
            "phase": phase, "elapsed_s": elapsed, "verified": verified,
            "error_code": error, "compile_key": key,
            "attempts": (job or {}).get("attempt", 0) or 0,
            "retryable": retryable, "failure": failure}


def satisfies(actual: str, want: str) -> bool:
    """Wait predicate: a better state satisfies (ACTIVE implies
    VERIFIED implies PREPARED)."""
    if actual == want:
        return True
    return STATE_RANK.get(actual, 0) >= STATE_RANK.get(want, 0) > 0
