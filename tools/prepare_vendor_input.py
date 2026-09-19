#!/usr/bin/env python3
"""Stage the private vendor build input (maintainer-side, never committed).

Copies the audited payload subtrees into packaging/vendor-input/ and
verifies them byte-for-byte against packaging/vendor-files.manifest.json:

  vendor-input/
    compile-site-packages/   audited compile files (voe/flexml/peano/...)
    xrt/                     minimal XRT userspace (lib/, share/)
    flexmlrt/                standalone FlexMLRT lib/
    legal/                   AMD EULA + TPN texts (image flow-down notices)

Usage: prepare_vendor_input.py --payload-src DIR --xrt-src DIR
       --flexmlrt-lib DIR --legal-src DIR [--out packaging/vendor-input]

The output directory is gitignored. The Docker build bind-mounts it;
nothing proprietary enters the repository or the build context listing.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "..", "packaging", "vendor-input")


def copy_tree(src: str, dest: str, ignore=()) -> None:
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    # symlinks=True: soname links (libxrt_*.so.2) travel as links; the
    # Dockerfile recreates the same links idempotently at install.
    shutil.copytree(src, dest, symlinks=True,
                    ignore=shutil.ignore_patterns(*ignore))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload-src", required=True)
    ap.add_argument("--xrt-src", required=True)
    ap.add_argument("--flexmlrt-lib", required=True)
    ap.add_argument("--legal-src", required=True)
    ap.add_argument("--calib-src", required=True)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    copy_tree(args.payload_src,
              os.path.join(args.out, "compile-site-packages"),
              ignore=("__pycache__", "pip", "pip-24.0.dist-info"))
    copy_tree(args.xrt_src, os.path.join(args.out, "xrt"))
    os.makedirs(os.path.join(args.out, "flexmlrt"), exist_ok=True)
    for fn in ("libflexmlrt.so",):
        shutil.copy2(os.path.join(args.flexmlrt_lib, fn),
                     os.path.join(args.out, "flexmlrt", fn))
    copy_tree(args.legal_src, os.path.join(args.out, "legal"))
    # Calibration images: the exact 32 COCO images from the audited proof
    # (public ultralytics COCO128 assets, 1.9 MB). Same bytes => same BF16.
    copy_tree(args.calib_src, os.path.join(args.out, "calib"))

    rc = subprocess.run(
        [sys.executable,
         os.path.join(HERE, "..", "packaging", "verify_payload.py"),
         "--manifest", os.path.join(HERE, "..", "packaging",
                                    "vendor-files.manifest.json"),
         "--payload", os.path.join(args.out, "compile-site-packages"),
         "--xrt", os.path.join(args.out, "xrt")]).returncode
    if rc != 0:
        print("vendor input FAILED manifest verification")
        return rc
    for fn in ("libflexmlrt.so",):
        if not os.path.isfile(os.path.join(args.out, "flexmlrt", fn)):
            print(f"missing flexmlrt/{fn}")
            return 1
    print(f"vendor input staged at {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
