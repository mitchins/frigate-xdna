#!/usr/bin/env python3
"""Re-verify public YOLO contract evidence against the manifest.

Reads tests/fixtures/yolo-public-contracts-0.1.2.manifest.json, inspects
the ONNX binaries in the evidence area, and fails on any mismatch of
bytes, graph contract or verdict.

No network, no hardware, no secrets. Binaries are never committed;
this script is how a later checkout re-proves the manifest against
re-acquired or archived evidence files.

Paths are module constants, not CLI options: this is a fixed-purpose
evidence check, and file access must never depend on caller-supplied
arguments.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

MANIFEST = os.path.join(
    REPO, "tests", "fixtures",
    "yolo-public-contracts-0.1.2.manifest.json")
MODELS_DIR = "/mnt/downloads/fxdna-012-local"


# Case B's banked location (outside the evidence area). The only
# absolute path this script ever opens; everything else is confined
# to --models (see _confined).
_BANKED_B = "/mnt/downloads/xdna-task03-work/yolov8n.onnx"


_EVIDENCE_NAME_RE = re.compile(r"[A-Za-z0-9_.-]+\.onnx\Z")


def _confined(models_dir: str, filename: str) -> str:
    """Resolve a manifest-controlled filename inside the models dir.

    The name must be a bare ONNX filename (no directories, no
    escapes); the resolved path must stay inside the directory.
    Defense in depth: names come from the committed manifest, but a
    corrupted manifest must never become a directory traversal.
    """
    if not _EVIDENCE_NAME_RE.fullmatch(filename):
        raise ValueError(f"refusing non-file evidence name: {filename!r}")
    base = os.path.realpath(models_dir)
    candidate = os.path.realpath(os.path.join(base, filename))
    if os.path.commonpath([base, candidate]) != base:
        raise ValueError(f"refusing path outside models dir: {filename!r}")
    return candidate


def read_evidence(path: str) -> bytes:
    """Read one evidence file whole (binaries here are ~10 MB)."""
    with open(path, "rb") as f:
        return f.read()


class _Unreadable(Exception):
    """Evidence bytes cannot be produced for a case."""


def load_case_bytes(cid: str, want: dict, models_dir: str) -> bytes:
    """Exact manifest bytes for a case, or raise _Unreadable."""
    try:
        path = _confined(models_dir, want["filename"])
        try:
            return read_evidence(path)
        except OSError:
            # Case B lives in its banked location, not the
            # evidence area.
            if cid != "B":
                raise
            return read_evidence(_BANKED_B)
    except (ValueError, OSError) as e:
        raise _Unreadable(f"cannot read evidence"
                          f" {want['filename']}: {e}") from e


def check_bytes(cid: str, want: dict, raw: bytes) -> str | None:
    """Mismatch description, or None when the bytes match."""
    digest = hashlib.sha256(raw).hexdigest()
    if digest != want["sha256"] or len(raw) != want["bytes"]:
        return (f"{cid}: bytes differ (got {digest[:12]}... {len(raw)}B,"
                f" want {want['sha256'][:12]}... {want['bytes']}B)")
    return None


def check_contract(case: dict, contract, cls: dict,
                   summary_line: str | None) -> list[str]:
    """Graph/verdict mismatches against the recorded case."""
    cid = case["id"]
    found: list[str] = []
    got = {"input": {"name": contract["input_name"],
                     "shape": contract["input_shape"],
                     "dtype": contract["input_dtype"]},
           "outputs": contract["outputs"],
           "opset": contract["opset"],
           "ir_version": contract["ir_version"],
           "node_count": contract["node_count"],
           "class_count": (cls.get("channels") or 0) - 4
           if cls.get("profile") else None}
    exp = case["graph"]
    for key in ("input", "outputs", "opset", "ir_version",
                "node_count", "class_count"):
        if got[key] != exp[key]:
            found.append(f"{cid}: {key} differs (got {got[key]!r},"
                         f" want {exp[key]!r})")
    verdict = case["verdict"]
    ok = cls.get("profile") == verdict.get("profile")
    if verdict.get("compatible") and not ok:
        found.append(f"{cid}: expected compatible,"
                     f" classifier says {cls!r}")
    if not verdict.get("compatible") and ok:
        found.append(f"{cid}: expected refusal,"
                     f" classifier says {cls!r}")
    if summary_line != verdict.get("serving_summary"):
        found.append(f"{cid}: serving summary differs"
                     f" (got {summary_line!r},"
                     f" want {verdict.get('serving_summary')!r})")
    return found


def check_case(case: dict, models_dir: str, inspect) -> list[str]:
    """All mismatches for one manifest case (empty means match)."""
    cid = case["id"]
    want = case["onnx"]
    try:
        raw = load_case_bytes(cid, want, models_dir)
    except _Unreadable as e:
        return [f"{cid}: {e}"]
    mismatch = check_bytes(cid, want, raw)
    if mismatch is not None:
        return [mismatch]
    model, _ = inspect.load_graph_bytes(raw)
    try:
        contract = inspect.inspect_model(model)
    except Exception as e:  # noqa: BLE001 - recorded as a mismatch
        return [f"{cid}: inspect refused: {e}"]
    cls = inspect.classify_output(contract["outputs"])
    summary = inspect.summarize_inspection(contract, cls, None)
    return check_contract(case, contract, cls, summary.get("output"))


def main() -> int:
    from frigate_xdna.models import inspect as _inspect
    models_dir = os.path.realpath(MODELS_DIR)
    with open(os.path.realpath(MANIFEST), encoding="utf-8") as f:
        manifest = json.load(f)
    failures = []
    for case in manifest["cases"]:
        found = check_case(case, models_dir, _inspect)
        failures.extend(found)
        if not found:
            print(f"{case['id']}: {case['onnx']['filename']} OK"
                  f" ({case['verdict'].get('serving_summary')})")
    if failures:
        print("MISMATCHES:")
        for f in failures:
            print(" -", f)
        return 1
    print("all cases match the manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
