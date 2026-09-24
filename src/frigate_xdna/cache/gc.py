"""Pins and pruning (docs/CACHE.md §6).

Default: no time-based eviction of configured, active, in-progress,
quarantined-evidence or rollback-pinned records. `prune` is dry-run unless
`--apply`; locks held during apply; active/read leases excluded.
"""
from __future__ import annotations

import os
import shutil

PIN_KINDS = ("configured", "active", "manual", "rollback", "running-job",
             "quarantine-evidence")


def pin_ref(registry, ref: str, kind: str = "manual") -> None:
    if kind not in PIN_KINDS:
        raise ValueError(f"unknown pin kind {kind!r}")
    registry.add_pin(f"ref:{ref}", kind, ref)


def plan_prune(data_dir: str, registry, max_bytes: int | None = None) -> dict:
    """Compute deletion candidates without deleting anything.

    Protected: anything reachable from pins, the active ref, running jobs,
    configured refs, or quarantine evidence. Two aliases sharing bytes are
    counted once.
    """
    pins = registry.list_pins()
    protected_targets = {p["target"] for p in pins}
    active = registry.get_state("active")
    if active and active.get("ref"):
        protected_targets.add(active["ref"])
    protected_digests: set[str] = set()
    for target in protected_targets:
        if target.startswith(("plus://", "local://")) or "/" in target \
                or target.endswith((".onnx", ".rai")):
            ref = registry.get_ref(target)
            if ref and ref.get("source_sha256"):
                protected_digests.add(ref["source_sha256"])
        elif len(target) == 64 and all(
                c in "0123456789abcdef" for c in target):
            protected_digests.add(target)
    candidates: list[dict] = []
    total = 0
    sources = os.path.join(data_dir, "sources")
    if os.path.isdir(sources):
        for digest in sorted(os.listdir(sources)):
            if digest in protected_digests:
                continue
            d = os.path.join(sources, digest)
            size = 0
            for root, _, files in os.walk(d):
                for fn in files:
                    try:
                        size += os.path.getsize(os.path.join(root, fn))
                    except OSError:
                        pass
            candidates.append({"kind": "source", "digest": digest,
                               "bytes": size})
            total += size
    plan = {"candidates": candidates, "reclaimable_bytes": total,
            "protected": sorted(protected_targets)}
    if max_bytes is not None:
        plan["max_bytes"] = max_bytes
    return plan


def apply_prune(data_dir: str, registry, plan: dict,
                max_bytes: int | None = None) -> dict:
    """Delete unreferenced content per a dry-run plan, registry first."""
    removed = []
    freed = 0
    for cand in plan["candidates"]:
        if cand["kind"] != "source":
            continue
        digest = cand["digest"]
        # recheck protection under lock before deleting
        if registry.get_source(digest) is None:
            continue
        still_pinned = any(
            p["target"] == digest or (
                registry.get_ref(p["target"]) or {}).get("source_sha256")
            == digest for p in registry.list_pins())
        if still_pinned:
            continue
        path = os.path.join(data_dir, "sources", digest)
        try:
            size = cand["bytes"]
            shutil.rmtree(path)
            freed += size
            removed.append(digest)
        except FileNotFoundError:
            pass
        # Registry cleanup AFTER filesystem deletion, per CACHE.md §6:
        # drop the sources row and clear dangling ref pointers so status
        # can never report a missing source as present.
        registry.execute("DELETE FROM sources WHERE sha256=?", (digest,))
        for row in registry.query(
                "SELECT ref FROM model_refs WHERE source_sha256=?",
                (digest,)):
            registry.execute(
                "UPDATE model_refs SET source_sha256=NULL WHERE ref=?",
                (row[0],))
        if max_bytes is not None and freed >= max_bytes:
            break
    return {"removed": removed, "freed_bytes": freed}
