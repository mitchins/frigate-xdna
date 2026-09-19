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


ALLOWLISTED_SYMLINKS = {
    "lib/libxrt_core.so.2": "libxrt_core.so.2.25.37",
    "lib/libxrt_coreutil.so.2": "libxrt_coreutil.so.2.25.37",
    "lib/libxrt_driver_xdna.so.2": "libxrt_driver_xdna.so.2.25.260102.56.release",
}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--payload", required=True)
    ap.add_argument("--xrt", required=True)
    ap.add_argument("--flexmlrt", required=False)
    ap.add_argument("--calib", required=False)
    ap.add_argument("--legal", required=False)
    args = ap.parse_args()
    manifest = json.load(open(args.manifest))
    errors = 0
    seen = set()
    # Map manifest prefix -> (cli arg, strip)
    prefix_map = {
        "xrt/": (args.xrt, 4),
        "flexmlrt/": (args.flexmlrt, 9),
        "calib/": (args.calib, 6),
        "legal/": (args.legal, 6),
    }
    def base_for(rel: str):
        for pfx, (base, strip) in prefix_map.items():
            if rel.startswith(pfx):
                return base, rel[strip:]
        return args.payload, rel
    for entry in manifest["files"]:
        rel = entry["path"]
        base, local = base_for(rel)
        if base is None:
            print(f"MISSING {rel} (no base dir supplied)")
            errors += 1
            continue
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
    # Check for extra files and unexpected symlinks in each supplied tree
    roots = [(args.payload, ""), (args.xrt, "xrt/")]
    if args.flexmlrt:
        roots.append((args.flexmlrt, "flexmlrt/"))
    if args.calib:
        roots.append((args.calib, "calib/"))
    if args.legal:
        roots.append((args.legal, "legal/"))
    for root, label in roots:
        if root is None or not os.path.isdir(root):
            continue
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
                    rel = os.path.relpath(full, root)
                    # Allowlisted XRT soname links are not in the manifest
                    # (recreated at install); any other symlink is an error.
                    check_key = rel if label == "xrt/" else rel
                    if label == "xrt/" and check_key in ALLOWLISTED_SYMLINKS:
                        target = os.readlink(full)
                        if os.path.basename(target) != ALLOWLISTED_SYMLINKS[check_key]:
                            print(f"SYMLINK_TARGET_MISMATCH {label}{rel} -> {target}")
                            errors += 1
                        continue
                    print(f"UNEXPECTED_SYMLINK {label}{rel} -> {os.readlink(full)}")
                    errors += 1
                    continue
                if os.path.normpath(full) not in seen:
                    print(f"EXTRA {label}{os.path.relpath(full, root)}")
                    errors += 1
    print(f"checked={len(manifest['files'])} errors={errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
