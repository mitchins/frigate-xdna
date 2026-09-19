#!/usr/bin/env python3
"""Import banked proof manifests into a local (uncommitted) evidence record.

Reads the research evidence directories named in SPEC §2.1 and writes
`evidence/proofs.manifest.json` with FULL hashes, recipe flags and licence
origins. Never invents values: any unreadable input is recorded under
"missing" and fails the run.

Usage: python3 tools/import_proofs.py [--evidence-dir DIR] [--out FILE]

No hardware, network, or vendor imports. Do not commit evidence/ (gitignored:
it contains absolute local paths and references private model bytes).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

EVIDENCE_DIRS = {
    "phase5": "/mnt/downloads/xdna-phase5/20260916",
    "phase5_license": "/mnt/downloads/xdna-phase5-license/20260916",
    "phase7": "/mnt/downloads/xdna-phase7/20260916",
    "phase7_soak": "/mnt/downloads/xdna-phase7-soak/20260917",
    "compiler_audit": "/mnt/downloads/xdna-compiler-audit/20260918",
    "research_root": "/root/xdna",
}

# Exact files the product reuses or cites. Missing entries are reported,
# never synthesized.
REQUIRED_FILES = {
    "native_worker_cc": "/root/xdna/phase7/xdna-zmq-worker/worker.cc",
    "native_runner_cc": "/root/xdna/phase5/runner/yolo_flexml.cc",
    "soak_worker_cc": "/root/xdna/phase7-soak/worker_v9s320.cc",
    "soak_worker_bin": "/root/xdna/phase7-soak/xdna-zmq-worker-v9s320",
    "soak_report": "/mnt/downloads/xdna-phase7-soak/20260917/REPORT.md",
    "soak_rai": "/mnt/downloads/xdna-model-sweep/20260916/rai/yolov9s-320.rai",
    "compiler_audit": "/root/xdna/compiler-audit/COMPILER-AUDIT.md",
    "compiler_recipe": "/root/xdna/compiler-audit/A1_recipe.md",
    "b1_manifest": (
        "/mnt/downloads/xdna-compiler-audit/20260918/b1-trace/used_manifest.json"
    ),
    "clean_rai": "/mnt/downloads/xdna-compiler-audit/20260918/clean_compile.rai",
    "vaiml_config": "/root/xdna/vaiml_config.json",
    "unwind": "/root/xdna/UNWIND.md",
}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence-dir", default="evidence")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.evidence_dir, "proofs.manifest.json")

    manifest: dict = {"schema_version": 1, "files": {}, "missing": [],
                      "evidence_dirs": {}}
    for name, path in EVIDENCE_DIRS.items():
        manifest["evidence_dirs"][name] = {
            "path": path, "present": os.path.isdir(path)}
    failed = False
    for name, path in REQUIRED_FILES.items():
        if not os.path.isfile(path):
            manifest["missing"].append({"name": name, "path": path,
                                        "reason": "not found"})
            failed = True
            continue
        try:
            manifest["files"][name] = {
                "path": path,
                "sha256": sha256_file(path),
                "size_bytes": os.path.getsize(path),
            }
        except OSError as e:
            manifest["missing"].append({"name": name, "path": path,
                                        "reason": f"{type(e).__name__}: {e}"})
            failed = True
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote {out}: {len(manifest['files'])} files, "
          f"{len(manifest['missing'])} missing")
    for m in manifest["missing"]:
        print(f"  MISSING {m['name']}: {m['path']}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
