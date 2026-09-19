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


def safe_join(base: str, rel: str) -> str | None:
    """Join a manifest-relative path under base.

    Returns None when the manifest entry is absolute or escapes base via
    ``..``. The manifest is machine-generated, but this verifier is
    supply-chain code: never let a manifest entry redirect reads outside
    the payload root it claims to describe.
    """
    if os.path.isabs(rel):
        return None
    norm_base = os.path.normpath(base)
    full = os.path.normpath(os.path.join(norm_base, rel))
    if full != norm_base and not full.startswith(norm_base + os.sep):
        return None
    return full

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--payload", required=True)
    ap.add_argument("--xrt", required=True)
    ap.add_argument("--flexmlrt", required=False)
    ap.add_argument("--calib", required=False)
    ap.add_argument("--legal", required=False)
    args = ap.parse_args()
    with open(args.manifest) as mf:
        manifest = json.load(mf)
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
        full = safe_join(base, local)
        if full is None:
            print(f"TRAVERSAL {rel}")
            errors += 1
            continue
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
    def _check_symlink(root: str, label: str, rel: str, target: str) -> bool:
        # Only the 3 XRT file symlinks are allowlisted; directory symlinks
        # are never allowlisted.
        if label != "xrt/" or rel not in ALLOWLISTED_SYMLINKS:
            print(f"UNEXPECTED_SYMLINK {label}{rel} -> {target}")
            return False
        if os.path.isabs(target):
            print(f"SYMLINK_ABSOLUTE_TARGET {label}{rel} -> {target}")
            return False
        if ".." in target.split(os.sep):
            print(f"SYMLINK_TRAVERSAL {label}{rel} -> {target}")
            return False
        expected = ALLOWLISTED_SYMLINKS[rel]
        if target != expected and os.path.basename(target) != expected:
            print(f"SYMLINK_TARGET_MISMATCH {label}{rel} -> {target} "
                  f"expected {expected}")
            return False
        # Validate resolved target is inside XRT root and not dangling
        link_path = os.path.join(root, rel)
        try:
            resolved = os.path.realpath(link_path)
            real_root = os.path.realpath(root)
            if not resolved.startswith(real_root + os.sep):
                print(f"SYMLINK_OUTSIDE_ROOT {label}{rel} -> {resolved}")
                return False
            if not os.path.exists(resolved):
                print(f"SYMLINK_DANGLING {label}{rel} -> {target}")
                return False
        except OSError as e:
            print(f"SYMLINK_ERROR {label}{rel}: {e}")
            return False
        return True

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
            # Directory symlinks: always reject (no allowlisted dir links)
            for d in list(dirnames):
                full_d = os.path.join(dirpath, d)
                if os.path.islink(full_d):
                    rel_d = os.path.relpath(full_d, root)
                    print(f"UNEXPECTED_DIR_SYMLINK {label}{rel_d} -> "
                          f"{os.readlink(full_d)}")
                    errors += 1
            dirnames[:] = [d for d in dirnames
                           if d != "__pycache__" and not os.path.islink(
                               os.path.join(dirpath, d))]
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
                    target = os.readlink(full)
                    if _check_symlink(root, label, rel, target):
                        continue
                    errors += 1
                    continue
                if os.path.normpath(full) not in seen:
                    print(f"EXTRA {label}{os.path.relpath(full, root)}")
                    errors += 1
    print(f"checked={len(manifest['files'])} errors={errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
