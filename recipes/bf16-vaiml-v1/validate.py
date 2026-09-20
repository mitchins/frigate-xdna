#!/usr/bin/env python3
"""bf16-vaiml-v1 step 3: artifact validation (audited recipe).

Checks the produced .rai: exists, size bounds, sha256 recorded. Deeper
native validation (activation smoke) belongs to the serving path (Task 04);
this gate ensures only well-formed compiler output reaches atomic
publication. A compile exit code plus an existing file is NOT correctness.

Usage: validate.py --rai MODEL.rai --min-bytes N --max-bytes M
Prints: VALIDATE_OK <sha256> <bytes> as the last line.
"""
import argparse
import hashlib
import os
import resource
import sys

# No RLIMIT_AS (see prepare.py note); the file-size cap stays: it
# bounds artifact writes, not address space.
try:
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024 * 1024, 8 * 1024 * 1024))
except (ValueError, OSError):
    pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rai", required=True)
    ap.add_argument("--min-bytes", type=int, default=1 << 20)
    ap.add_argument("--max-bytes", type=int, default=256 * 1024 * 1024)
    args = ap.parse_args()
    try:
        size = os.path.getsize(args.rai)
    except OSError:
        print("VALIDATE_FAILED: unreadable artifact", flush=True)
        return 1
    if not args.min_bytes <= size <= args.max_bytes:
        print(f"VALIDATE_FAILED: size {size} out of bounds", flush=True)
        return 1
    h = hashlib.sha256()
    with open(args.rai, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    print(f"VALIDATE_OK {h.hexdigest()} {size}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
