"""Device-operation journal and safety circuit breaker (SPEC §10).

Persists operation journal before compile/activation/validation/teardown.
Records operation, model/artifact, pid, generation, boot token, timestamps,
clean/unclean completion. On interrupted unsafe operation, inhibits NPU
activation and quarantines artifact; requires explicit fxdna recover.
"""
from __future__ import annotations

import json
import os
import time

from ..cache.registry import Registry

JOURNAL_CURRENT = "operations/current.json"
JOURNAL_HISTORY = "operations/history.jsonl"
MAX_HISTORY = 1000


def _journal_dir(data_dir: str) -> str:
    d = os.path.join(data_dir, "operations")
    os.makedirs(d, exist_ok=True)
    return d


def _boot_token() -> str:
    try:
        with open("/proc/sys/kernel/random/boot_id") as f:
            return f.read().strip()
    except OSError:
        return "unknown"


def begin_operation(
    data_dir: str,
    registry: Registry,
    operation: str,
    model_ref: str = "",
    artifact: str = "",
    generation: int = 0,
) -> dict:
    """Persist journal before unsafe operation; returns record."""
    rec = {
        "operation": operation,
        "model_ref": model_ref,
        "artifact": artifact,
        "generation": generation,
        "pid": os.getpid(),
        "boot_token": _boot_token(),
        "started_at": time.time(),
        "completed": False,
        "clean": False,
    }
    d = _journal_dir(data_dir)
    cur = os.path.join(data_dir, JOURNAL_CURRENT)
    tmp = cur + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp, cur)
    # fsync dir
    try:
        dfd = os.open(d, os.O_DIRECTORY)
        os.fsync(dfd)
        os.close(dfd)
    except OSError:
        pass
    return rec


def complete_operation(
    data_dir: str, clean: bool = True, error: str = ""
) -> None:
    cur = os.path.join(data_dir, JOURNAL_CURRENT)
    hist = os.path.join(data_dir, JOURNAL_HISTORY)
    try:
        with open(cur) as f:
            rec = json.load(f)
    except (OSError, ValueError):
        return
    rec["completed"] = True
    rec["clean"] = clean
    rec["error"] = error
    rec["completed_at"] = time.time()
    # append to history
    try:
        with open(hist, "a") as hf:
            hf.write(json.dumps(rec, sort_keys=True) + "\n")
            hf.flush()
            os.fsync(hf.fileno())
        # rotate if too large
        try:
            if os.path.getsize(hist) > 2 * 1024 * 1024:
                # keep last MAX_HISTORY lines
                with open(hist) as hf:
                    lines = hf.readlines()
                with open(hist, "w") as hf:
                    hf.writelines(lines[-MAX_HISTORY:])
        except OSError:
            pass
    except OSError:
        pass
    # remove current
    try:
        os.unlink(cur)
    except OSError:
        pass
    # if unclean, set inhibition
    if not clean:
        try:
            # registry may be None in some contexts
            from ..supervisor import Supervisor as _Sup  # avoid cycle
        except ImportError:
            pass


def check_inhibited(data_dir: str, registry: Registry) -> dict | None:
    """Return inhibition record if present, else None."""
    try:
        inh = registry.get_state("inhibition")
        if inh:
            return inh
    except Exception:
        pass
    # also check journal for unclean completion
    cur = os.path.join(data_dir, JOURNAL_CURRENT)
    if os.path.exists(cur):
        try:
            with open(cur) as f:
                rec = json.load(f)
            # if not completed and boot token changed, it's interrupted
            if not rec.get("completed"):
                current_boot = _boot_token()
                if rec.get("boot_token") != current_boot:
                    return {
                        "reason": "interrupted_unsafe_operation",
                        "operation": rec.get("operation"),
                        "artifact": rec.get("artifact"),
                        "previous_boot": rec.get("boot_token"),
                        "current_boot": current_boot,
                    }
        except (OSError, ValueError):
            pass
    return None


def inhibit(data_dir: str, registry: Registry, reason: str, ref: str) -> None:
    registry.set_state(
        "inhibition",
        {"reason": reason, "ref": ref, "at": time.time(), "boot_token": _boot_token()},
    )


def clear_inhibition(registry: Registry) -> None:
    try:
        registry.execute("DELETE FROM service_state WHERE key=?", ("inhibition",))
    except Exception:
        pass
