#!/usr/bin/env python3
"""Re-verify public YOLO contract evidence against the manifest.

Reads tests/fixtures/yolo-public-contracts-0.1.2.manifest.json, inspects
the ONNX binaries found under --models (default: the v0.1.2 evidence
area), and fails on any mismatch of bytes, graph contract or verdict.

No network, no hardware, no secrets. Binaries are never committed;
this script is how a later checkout re-proves the manifest against
re-acquired or archived evidence files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

MANIFEST = os.path.join(
    REPO, "tests", "fixtures",
    "yolo-public-contracts-0.1.2.manifest.json")


# Case B's banked location (outside the evidence area). The only
# absolute path this script ever opens; everything else is confined
# to --models (see _confined).
_BANKED_B = "/mnt/downloads/xdna-task03-work/yolov8n.onnx"


def _confined(models_dir: str, filename: str) -> str:
    """Resolve a manifest-controlled filename inside the models dir.

    Refuses anything escaping the directory (defense in depth: the
    names come from the committed manifest, not the operator).
    """
    base = os.path.realpath(models_dir)
    candidate = os.path.realpath(os.path.join(base, filename))
    if os.path.commonpath([base, candidate]) != base:
        raise ValueError(f"refusing path outside models dir: {filename!r}")
    return candidate


def sha_of(path: str) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="/mnt/downloads/fxdna-012-local",
                    help="directory holding the evidence ONNX files")
    ap.add_argument("--manifest", default=MANIFEST)
    args = ap.parse_args()
    from frigate_xdna.models import inspect as _inspect
    models_dir = os.path.realpath(args.models)
    with open(os.path.realpath(args.manifest), encoding="utf-8") as f:
        manifest = json.load(f)
    failures = []
    for case in manifest["cases"]:
        cid = case["id"]
        want = case["onnx"]
        try:
            path = _confined(models_dir, want["filename"])
        except ValueError as e:
            failures.append(f"{cid}: {e}")
            continue
        # Case B lives in its banked location, not the evidence area.
        if not os.path.isfile(path) and cid == "B" and os.path.isfile(
                _BANKED_B):
            path = _BANKED_B
        if not os.path.isfile(path):
            failures.append(f"{cid}: missing file for {want['filename']}")
            continue
        digest, size = sha_of(path)
        if digest != want["sha256"] or size != want["bytes"]:
            failures.append(
                f"{cid}: bytes differ (got {digest[:12]}... {size}B,"
                f" want {want['sha256'][:12]}... {want['bytes']}B)")
            continue
        with open(path, "rb") as f:
            model, _ = _inspect.load_graph_bytes(f.read())
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
