#!/usr/bin/env python3
"""Verify a candidate vendor payload dir against vendor-files.manifest.json.

Usage: verify_payload.py --manifest <manifest> --payload <site-packages>
       --xrt <xrt-prefix>

Checks every manifest entry exists with matching sha256/size, and reports
any extra files not in the manifest. Exit 0 iff exact match.

Scope notes: soname symlinks are recreated at install and are not hashed;
__pycache__/pip scaffolding is excluded at manifest build time and ignored
here as well.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--payload", required=True)
    ap.add_argument("--xrt", required=True)
    args = ap.parse_args()
    manifest = json.load(open(args.manifest))
    errors = 0
    seen = set()
    for entry in manifest["files"]:
        rel = entry["path"]
        base = args.xrt if rel.startswith("xrt/") else args.payload
        local = rel[4:] if rel.startswith("xrt/") else rel
        full = os.path.join(base, local)
        seen.add(os.path.normpath(full))
        if not os.path.isfile(full) or os.path.islink(full):
            print(f"MISSING {rel}")
            errors += 1
            continue
        if os.path.getsize(full) != entry["size_bytes"]:
            print(f"SIZE {rel}")
            errors += 1
            continue
        if sha256_file(full) != entry["sha256"]:
            print(f"HASH {rel}")
            errors += 1
    for root, label in ((args.payload, ""), (args.xrt, "xrt/")):
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            top = os.path.relpath(dirpath, root).split(os.sep)[0]
            if label == "" and top in ("pip", "pip-24.0.dist-info"):
                dirnames[:] = []
                continue
            for fn in filenames:
                if fn.endswith((".pyc", ".pyo")):
                    continue
                full = os.path.join(dirpath, fn)
                if os.path.islink(full):
                    continue
                if os.path.normpath(full) not in seen:
                    print(f"EXTRA {label}{os.path.relpath(full, root)}")
                    errors += 1
    print(f"checked={len(manifest['files'])} errors={errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
