"""Stability-diagnostic journal, report and acknowledgement
(docs/COMPATIBILITY.md, Compatibility Observations & Stability
Diagnostics v1).

A stability run owns `/data/stability/<RUN_ID>/`:

```text
metadata.json    run identity: model/compile key, profile, host-info
current.json     latest durable checkpoint (rewritten, fsynced)
timeline.jsonl   append-only STARTED/COMPLETED breadcrumbs (fsynced)
result.json      only present when a final result exists
acknowledge.json only present after an operator acknowledged an
                 interrupted run
```

A host reset may erase the final log line, so `/data` — not console
output — is the diagnostic record. `result.json` present means the
run finished deliberately (PASS or FAIL). Its absence after device-
sensitive STARTED breadcrumbs means the run was interrupted; that
is a fact about the last durable checkpoint, never a causal claim.

The runner (`fxdna stability run`) arrives in a later release; the
journal contract here is final so its records can be read now.
"""
from __future__ import annotations

import json
import os
import re
import time

RUNS_DIRNAME = "stability"
TIMELINE = "timeline.jsonl"
METADATA = "metadata.json"
CURRENT = "current.json"
RESULT = "result.json"
ACKNOWLEDGE = "acknowledge.json"

# Bounded tail carried into diagnose bundles.
DIAGNOSE_TAIL_LINES = 200

# RUN_IDs are operator/CLI-supplied path components: restrict to a
# flat, separator-free token so no run_id can traverse outside
# <data>/stability/ (Sonar S8707).
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def safe_run_id(run_id: str) -> bool:
    """True when run_id is a flat directory name, never a path."""
    return bool(_RUN_ID_RE.fullmatch(run_id or ""))


def runs_root(data_dir: str) -> str:
    return os.path.join(data_dir, RUNS_DIRNAME)


def run_dir(data_dir: str, run_id: str) -> str:
    return os.path.join(runs_root(data_dir), run_id)


def list_runs(data_dir: str) -> list[str]:
    """Run IDs that own a metadata.json, oldest first (RUN_IDs are
    timestamp-sortable by construction)."""
    root = runs_root(data_dir)
    try:
        names = os.listdir(root)
    except OSError:
        return []
    return sorted(n for n in names
                  if safe_run_id(n)
                  and os.path.isfile(os.path.join(root, n, METADATA)))


def latest_run(data_dir: str) -> str | None:
    runs = list_runs(data_dir)
    return runs[-1] if runs else None


def _read_json(path: str):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def read_timeline(data_dir: str, run_id: str) -> list[dict]:
    path = os.path.join(run_dir(data_dir, run_id), TIMELINE)
    try:
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except (OSError, ValueError):
        return []


def detect_interrupted(timeline: list[dict], has_result: bool) -> dict | None:
    """Last STARTED step with no matching COMPLETED, when the run
    has no final result.

    Matching is by (phase, step, cycle): the completion of a
    started operation closes it. Reports the evidence we possess —
    the last durable breadcrumb — and makes no causal claim.
    """
    if has_result:
        return None
    open_ops: dict[tuple, dict] = {}
    for rec in timeline:
        state = rec.get("state")
        key = (rec.get("phase"), rec.get("step"), rec.get("cycle"))
        if state == "STARTED":
            open_ops[key] = rec
        elif state == "COMPLETED" and key in open_ops:
            del open_ops[key]
    if not open_ops:
        return None
    last = max(open_ops.values(), key=lambda r: r.get("sequence", 0))
    return {
        "phase": last.get("phase"),
        "step": last.get("step"),
        "cycle": last.get("cycle"),
        "sequence": last.get("sequence"),
        "submitted": last.get("submitted"),
        "completed": last.get("completed"),
        "wall_time": last.get("wall_time"),
    }


def load_report(data_dir: str, run_id: str | None = None) -> dict | None:
    """Full report record for a run (latest when run_id is None)."""
    if run_id is not None and not safe_run_id(run_id):
        return None
    rid = run_id or latest_run(data_dir)
    if rid is None or not os.path.isfile(
            os.path.join(run_dir(data_dir, rid), METADATA)):
        return None
    timeline = read_timeline(data_dir, rid)
    result = _read_json(os.path.join(run_dir(data_dir, rid), RESULT))
    current = _read_json(os.path.join(run_dir(data_dir, rid), CURRENT))
    ack = _read_json(os.path.join(run_dir(data_dir, rid), ACKNOWLEDGE))
    interrupted = detect_interrupted(timeline, result is not None)
    return {
        "schema_version": 1,
        "run_id": rid,
        "metadata": _read_json(
            os.path.join(run_dir(data_dir, rid), METADATA)),
        "result": result,
        "interrupted": interrupted,
        "acknowledged": bool(ack),
        "acknowledge": ack,
        "current": current,
        "timeline_records": len(timeline),
    }


def _ack_refusal(rid: str | None, report: dict | None) -> dict | None:
    """Refusal record for acknowledge, or None when allowed."""
    if rid is None:
        return {"acknowledged": False, "error_code": "NOT_FOUND",
                "message": "no stability runs recorded"}
    if report is None:
        return {"acknowledged": False, "error_code": "NOT_FOUND",
                "message": f"run {rid} has no metadata"}
    if report["acknowledged"]:
        return {"acknowledged": True, "error_code": "ALREADY",
                "message": f"run {rid} already acknowledged",
                "run_id": rid}
    if report["interrupted"] is None:
        return {"acknowledged": False, "error_code": "NOT_INTERRUPTED",
                "message": f"run {rid} has a final result; nothing to "
                           "acknowledge", "run_id": rid}
    return None


def acknowledge_run(data_dir: str, reason: str,
                    run_id: str | None = None) -> dict:
    """Record explicit operator acknowledgement of an interrupted
    run. Clears the diagnostic-run latch only: ordinary production
    device-fault inhibition, quarantine and cooldown state are not
    touched (they live in the registry and keep their own recovery
    route)."""
    if run_id is not None and not safe_run_id(run_id):
        return {"acknowledged": False, "error_code": "INVALID_ARGS",
                "message": "run id must be a flat directory name"}
    rid = run_id or latest_run(data_dir)
    report = (load_report(data_dir, rid) if rid is not None else None)
    refusal = _ack_refusal(rid, report)
    if refusal is not None:
        return refusal
    assert rid is not None and report is not None
    rec = {"schema_version": 1, "run_id": rid, "reason": reason,
           "acknowledged_at": time.time()}
    path = os.path.join(run_dir(data_dir, rid), ACKNOWLEDGE)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp, path)
    return {"acknowledged": True, "run_id": rid, "reason": reason}


def format_report(report: dict) -> str:
    """Human-readable rendering for `fxdna stability report`."""
    lines: list[str] = []
    res = report.get("result")
    interrupted = report.get("interrupted")
    if res is not None:
        lines.append(f"frigate-xdna stability run: {res.get('outcome')}")
    elif interrupted is not None:
        lines.append("frigate-xdna stability run: INTERRUPTED")
    else:
        lines.append("frigate-xdna stability run: (no final result)")
    meta = report.get("metadata") or {}
    model = meta.get("model") or {}
    if model:
        lines += ["", "Model",
                  f"  {model.get('family', '?')} / "
                  f"{model.get('resolution', '?')}",
                  f"  prepared compile key: {model.get('compile_key', '?')}"]
    phases = meta.get("phases") or []
    if phases or res or interrupted:
        lines += ["", "Phases"]
        for ph in phases:
            lines.append(f"  {ph}")
    if res is not None:
        lines += ["", "Result",
                  f"  {res.get('outcome')} through "
                  f"{res.get('last_phase', '?')}.",
                  "  This is an observation, not certification of "
                  "host stability."]
    if interrupted is not None:
        lines += ["", "Interrupted",
                  f"  {interrupted.get('phase')} cycle="
                  f"{interrupted.get('cycle')}",
                  f"  last step: {interrupted.get('step')} STARTED",
                  f"  last completed request: "
                  f"{interrupted.get('completed')}",
                  "",
                  "  The host/process disappeared before a completion "
                  "record was persisted.",
                  "  No causal conclusion is made from this alone."]
        if report.get("acknowledged"):
            ack = report.get("acknowledge") or {}
            lines += ["", f"  Acknowledged: {ack.get('reason', '')}"]
        else:
            lines += ["", "  Unacknowledged. Next stability runs must "
                          "refuse to start until:",
                      "  fxdna stability acknowledge --last --reason \"...\""]
    return "\n".join(lines) + "\n"
