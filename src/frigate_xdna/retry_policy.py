"""Failure classification and bounded retry (v0.1.1 checkpoint 5).

Every terminal job carries a structured failure record (phase, stable
code, reason, retry eligibility, attempt accounting, corrective
guidance, vendor evidence). Classification decides:

* transient — recognized temporary resource problem: bounded automatic
  retry with backoff (≤3 attempts per cycle, persisted).
* config_blocked — known inadequate configuration (e.g. memlock):
  no retries while unchanged; a verified correction starts a new
  bounded cycle automatically.
* interrupted_safe — restart-interrupted idempotent work with a clean
  safety state: safe to resume automatically (fresh workdir, atomic
  publish, never overwrite).
* permanent — unsupported input, auth, corrupt artifact, validation
  failure: never automatic; most never operator-retried either.
* safety — quarantine, device fault, suspect host reset, unclean
  operation: never automatic, never operator-bypassed via recover.
* unknown — legacy rows or unattributable failures: never automatic;
  an explicit acknowledged operator retry may start one bounded cycle.

An errno alone never establishes safety: attribution requires the
record (phase evidence + current limits + safety state), not a
substring. In particular a memlock diagnosis requires BOTH the
locked-memory vendor pattern AND an actually inadequate
RLIMIT_MEMLOCK — never EAGAIN alone.
"""
from __future__ import annotations

import re
import resource as _resource
import time

from .deploy_checks import MEMLOCK_NEED_BYTES

MAX_ATTEMPTS = 3
BACKOFF_BASE_S = 30.0
BACKOFF_CAP_S = 600.0

# Narrow locked-memory vendor pattern (the observed field failure):
# an 8209-flagged mmap failing err=-11, or mmap + "Resource
# temporarily unavailable" on the same tail. Anything broader is an
# unknown resource failure, never a memlock diagnosis.
_MEMLOCK_RES = (
    re.compile(r"flags=8209"),
    re.compile(r"mmap.*err=-11"),
)

SAFETY_REASONS = (
    "QUARANTINED",
    "SAFETY_INHIBITED",
    "HOST_RESET",
    "DEVICE_FAULT",
    "unclean_operation",
    "interrupted_unsafe_operation",
)

# Terminal stages that are never retried automatically. Most are also
# never operator-retried; VALIDATION_FAILED additionally preserves
# evidence and blocks publish/activation elsewhere.
PERMANENT_STAGES = frozenset({
    "UNSUPPORTED_CONTRACT",
    "VALIDATION_FAILED",
    "QUARANTINED",
    "SOURCE_CHANGED",
    "CACHE_CORRUPT",
    "INVALID_MODEL",
})

# Error codes that are never retried automatically (auth/contract/
# conflict/operator errors rather than resource failures).
PERMANENT_CODES = frozenset({
    "AUTH_FAILED",
    "UNSUPPORTED_CONTRACT",
    "INVALID_ARGS",
    "INVALID_MODEL",
    "INVALID_CONFIG",
    "SOURCE_CHANGED",
    "CACHE_CORRUPT",
    "NOT_PREPARED",
    "OWNERSHIP_CONFLICT",
    "MODEL_IN_USE",
    "QUARANTINED",
    "SAFETY_INHIBITED",
    "DEVICE_FAULT",
    "DEVICE_BUSY",
    "VALIDATION_FAILED",
    "ACTIVATION_FAILED",
})

# Inhibition reasons recover must never bypass (quarantine / safety).
SAFETY_PREFIXES = (
    "QUARANTINED",
    "SAFETY_INHIBITED",
)


def is_safety_inhibition(reason: str) -> bool:
    """True when an inhibition is safety-class (recover must refuse)."""
    reason = reason or ""
    if reason.startswith(SAFETY_PREFIXES):
        return True
    return any(token in reason for token in SAFETY_REASONS)


def backoff_s(attempt: int) -> float:
    """Backoff before attempt N (1-based): 30s, 60s, 120s, capped."""
    return min(BACKOFF_CAP_S, BACKOFF_BASE_S * (2 ** max(0, attempt - 1)))


def _memlock_pattern(detail: str) -> bool:
    if not detail:
        return False
    if _MEMLOCK_RES[0].search(detail):
        return True
    return bool(_MEMLOCK_RES[1].search(detail)
                and "Resource temporarily unavailable" in detail)


def memlock_adequate() -> bool:
    """Current locked-memory allowance covers the 64 MiB mapping."""
    try:
        soft, _hard = _resource.getrlimit(_resource.RLIMIT_MEMLOCK)
    except (OSError, ValueError):
        return False
    return soft == _resource.RLIM_INFINITY or soft >= MEMLOCK_NEED_BYTES


def classify(stage: str, error_code: str | None,
             detail: str = "") -> dict:
    """Classify a terminal job outcome.

    Returns {kind, retryable, auto, guidance} where kind is one of
    transient/config_blocked/permanent/safety/unknown. `retryable`
    admits an explicit acknowledged operator retry; `auto` admits an
    automatic one (bounded, backoff). Unknown vendor crashes,
    device faults and safety stages are never automatic.
    """
    code = error_code or stage
    if stage == "INTERRUPTED":
        return {"kind": "interrupted_safe", "retryable": True,
                "auto": True,
                "guidance": "Restart-interrupted idempotent work;"
                            " resumes automatically with a clean safety"
                            " state, else waits for operator review."}
    if stage in ("QUARANTINED",) or code in ("QUARANTINED",
                                             "SAFETY_INHIBITED",
                                             "DEVICE_FAULT"):
        return {"kind": "safety", "retryable": False, "auto": False,
                "guidance": "Safety inhibition: explicit operator review"
                            " required; recover refuses safety-class"
                            " inhibitions."}
    if stage == "RESOURCE_EXCEEDED" or code == "RESOURCE_EXCEEDED":
        return {"kind": "transient", "retryable": True, "auto": True,
                "guidance": "Resource exhausted: free the constrained"
                            " resource, then recover explicitly or wait"
                            " for the bounded automatic retry."}
    if stage in PERMANENT_STAGES or code in PERMANENT_CODES:
        if stage == "VALIDATION_FAILED" or code == "VALIDATION_FAILED":
            guidance = ("Validation failed: evidence preserved; artifact"
                        " neither published nor activated. New source"
                        " bytes start a new job; this row never retries.")
        else:
            guidance = ("Permanent failure: fix the input/configuration"
                        " and submit new source; this row never retries.")
        return {"kind": "permanent", "retryable": False, "auto": False,
                "guidance": guidance}
    if _memlock_pattern(detail):
        if memlock_adequate():
            return {"kind": "unknown", "retryable": False, "auto": False,
                    "guidance": "Locked-memory vendor pattern with an"
                                " adequate memlock allowance: attribution"
                                " insufficient; explicit operator review"
                                " required (see status/recover)."}
        return {"kind": "config_blocked", "retryable": True,
                "auto": False,
                "guidance": "Inadequate locked-memory allowance: set"
                            " ulimits memlock soft+hard to -1"
                            " (examples/compose.yaml) and recreate the"
                            " container on the same /data; preparation"
                            " resumes automatically."}
    if stage == "COMPILE_FAILED" and "timeout" in (detail or "").lower():
        return {"kind": "transient", "retryable": True, "auto": True,
                "guidance": "Compile timeout: bounded automatic retry"
                            " with backoff."}
    if stage in ("DOWNLOAD_FAILED", "ACQUISITION_FAILED"):
        return {"kind": "transient", "retryable": True, "auto": True,
                "guidance": "Temporary acquisition failure: bounded"
                            " automatic retry with backoff."}
    # A bare compile failure with no attributable evidence is not a
    # recognized temporary problem: never automatic, but an explicit
    # acknowledged operator retry may open one bounded cycle.
    return {"kind": "unknown", "retryable": True, "auto": False,
            "guidance": "Unrecognized failure: explicit operator review"
                        " required; recover reports what remains unknown"
                        " and may open one bounded cycle."}


def new_record(stage: str, error_code: str | None, detail: str,
               attempt: int, now: float | None = None,
               history_base: tuple | list = ()) -> dict:
    """Build the structured failure record for a terminal job."""
    now = time.time() if now is None else now
    cls = classify(stage, error_code, detail)
    history = list(history_base) + [{"attempt": attempt, "stage": stage,
                                     "code": error_code or stage,
                                     "at": now}]
    record = {"phase": stage, "code": error_code or stage,
              "reason": (detail or "")[:500],
              "kind": cls["kind"], "retryable": cls["retryable"],
              "guidance": cls["guidance"], "attempts": attempt,
              "not_before": now, "history": history}
    if cls["auto"] and cls["kind"] == "transient":
        record["not_before"] = now + backoff_s(attempt)
    return record


def auto_eligible(record: dict | None, now: float) -> tuple[bool, str]:
    """Whether the daemon may automatically open a new attempt now.

    Returns (eligible, why). Config-blocked rows become eligible only
    with a verified correction; transient rows after backoff; safe
    interrupts immediately; everything else never automatically.
    """
    if not record:
        return False, "legacy failure without evidence"
    if record.get("resumed_to"):
        return False, "already resumed"
    if "phase" not in record:
        return False, "not a terminal failure"
    kind = record.get("kind", "unknown")
    attempts = record.get("attempts", 0)
    if attempts >= MAX_ATTEMPTS:
        return False, "attempt budget exhausted"
    if kind == "transient":
        if now >= record.get("not_before", 0):
            return True, "backoff elapsed"
        return False, "backing off"
    if kind == "config_blocked":
        if memlock_adequate():
            return True, "blocking condition corrected"
        return False, "blocking condition unchanged"
    if kind == "interrupted_safe":
        return True, "safe resume"
    return False, f"{kind} never retries automatically"
