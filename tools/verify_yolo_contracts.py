#!/usr/bin/env python3
"""Re-verify public YOLO contract evidence against the manifest.

Reads tests/fixtures/yolo-public-contracts-0.1.2.manifest.json, inspects
the ONNX binaries found under --models (default: the v0.1.2 evidence
area), and fails on any mismatch of bytes, graph contract or verdict.

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


def main() -> int:
    from frigate_xdna.models import inspect as _inspect
    models_dir = os.path.realpath(MODELS_DIR)
    with open(os.path.realpath(MANIFEST), encoding="utf-8") as f:
        manifest = json.load(f)
    failures = []
    for case in manifest["cases"]:
        cid = case["id"]
        want = case["onnx"]
        try:
            path = _confined(models_dir, want["filename"])
            try:
                raw = read_evidence(path)
            except OSError:
                # Case B lives in its banked location, not the
                # evidence area.
                if cid != "B":
                    raise
                raw = read_evidence(_BANKED_B)
        except (ValueError, OSError) as e:
            failures.append(f"{cid}: cannot read evidence"
                            f" {want['filename']}: {e}")
            continue
        digest = hashlib.sha256(raw).hexdigest()
        if digest != want["sha256"] or len(raw) != want["bytes"]:
            failures.append(
                f"{cid}: bytes differ (got {digest[:12]}... {len(raw)}B,"
                f" want {want['sha256'][:12]}... {want['bytes']}B)")
            continue
        model, _ = _inspect.load_graph_bytes(raw)
        try:
            contract = _inspect.inspect_model(model)
        except Exception as e:  # noqa: BLE001 - verdict comparison below
            failures.append(f"{cid}: inspect refused: {e}")
            continue
        cls = _inspect.classify_output(contract["outputs"])
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
                failures.append(
                    f"{cid}: {key} differs (got {got[key]!r},"
                    f" want {exp[key]!r})")
        verdict = case["verdict"]
        ok = cls.get("profile") == verdict.get("profile")
        if verdict.get("compatible") and not ok:
            failures.append(f"{cid}: expected compatible,"
                            f" classifier says {cls!r}")
        if not verdict.get("compatible") and ok:
            failures.append(f"{cid}: expected refusal,"
                            f" classifier says {cls!r}")
        if not failures or not any(f.startswith(cid + ":")
                                   for f in failures):
            print(f"{cid}: {want['filename']} OK"
                  f" ({verdict.get('serving_summary')})")
    if failures:
        print("MISMATCHES:")
        for f in failures:
            print(" -", f)
        return 1
    print("all cases match the manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
