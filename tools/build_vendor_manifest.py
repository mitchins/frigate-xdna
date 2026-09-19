#!/usr/bin/env python3
"""Build the product vendor manifests from the audited Phase-7.7 payload.

Reads the trace-accurate payload (as proven in the clean-room chroot) plus
the pinned recipe/licence evidence, and writes:
  packaging/vendor-files.manifest.json  (per-file path/sha256/size/origin)
  packaging/vendor.lock.json            (per-component pins + digests)
  packaging/legal/component-map.json    (per-component governing terms)

Usage: build_vendor_manifest.py --payload <dir> --xrt <dir> --out <repo>

The payload dir layout mirrors the image prefixes:
  <payload>/site-packages/...   (compile venv site-packages content)
  <xrt>/lib/... <xrt>/share/... (minimal XRT userspace)

No binaries are copied or committed — only hashes and metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

# path prefix -> (source package, version, licence id, role)
ORIGINS = [
    ("voe/lib/", ("voe", "1.8.0", "amd-eula",
                   "xcompiler/aiecompiler/dyn-dispatch/flexmlrt backend")),
    ("flexml/flexml_extras/lib/", ("flexml", "1.8.0", "amd-eula",
                                     "VAIML partitioner + compile VFS")),
    ("lnx64.o/tools/peano/lib/", ("llvm-aie", "1.8.0", "amd-eula",
                                   "peano AIE runtime lib (LLVM-sourceable)")),
    ("include/", ("vitis-aie-essentials", "1.8.0", "amd-eula",
                   "ADF/AIE-API compile headers")),
    ("vitis_mllib/", ("vitis-mllib", "1.8.0", "amd-eula",
                       "AIE kernel compile includes/metadata")),
    ("data/", ("vitis-aie-essentials", "1.8.0", "amd-eula",
                "device schema data")),
    ("onnxruntime/", ("onnxruntime-vitisai", "1.27.0", "amd-eula",
                       "VitisAI EP driver + opened subset")),
    ("onnx/", ("onnx", "1.23.0", "onnx-mit",
                "ONNX graph inspection (public PyPI)")),
    ("onnx-1.23.0.dist-info/", ("onnx", "1.23.0", "onnx-mit",
                                "ONNX dist-info")),
    ("google/protobuf/", ("protobuf", "7.36.2", "protobuf-bsd",
                          "ONNX protobuf runtime")),
    ("protobuf-7.36.2.dist-info/", ("protobuf", "7.36.2", "protobuf-bsd",
                                    "ONNX protobuf runtime")),
    ("google/", ("protobuf", "7.36.2", "protobuf-bsd",
                 "ONNX protobuf runtime")),
    ("typing_extensions", ("typing_extensions", "4.16.0", "typing-extensions-mit",
                           "ONNX typing support")),
    ("typing_extensions-", ("typing_extensions", "4.16.0", "typing-extensions-mit",
                            "ONNX typing support")),
    ("ml_dtypes", ("ml_dtypes", "0.6.0", "ml_dtypes-apache",
                   "ONNX ML dtypes")),
    ("numpy", ("numpy", "2.5.3", "numpy-bsd",
                "compile-harness array support")),
    ("numpy.libs", ("numpy", "2.5.3", "numpy-bsd",
                     "bundled openblas/gfortran")),
    ("vaiml_config.json", ("ryzenai-sw-example", "1.8", "ryzenai-sw-mit",
                             "VAIML partition recipe")),
]

XRT_ORIGINS = [
    ("lib/libxrt_core", ("xrt-base", "2.25.37", "apache-2.0",
                          "XRT core runtime")),
    ("lib/libxrt_coreutil", ("xrt-base", "2.25.37", "apache-2.0",
                              "XRT core utilities")),
    ("lib/libxrt_driver_xdna", ("xdna-driver-plugin", "2.25.260102.56",
                                 "amdnpu-binary",
                                 "XDNA userspace shim (Apache-2.0 sources)")),
    ("share/amdxdna/version.json", ("xdna-driver-plugin", "2.25.260102.56",
                                     "amdnpu-binary", "shim version record")),
]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def origin_for(rel: str, table) -> tuple:
    for prefix, origin in table:
        if rel == prefix.rstrip("/") or rel.startswith(prefix):
            return origin
    return ("unknown", "unknown", "UNMAPPED", "unmapped file")


ALLOWLISTED_SYMLINKS = {
    "lib/libxrt_core.so.2": "libxrt_core.so.2.25.37",
    "lib/libxrt_coreutil.so.2": "libxrt_coreutil.so.2.25.37",
    "lib/libxrt_driver_xdna.so.2": "libxrt_driver_xdna.so.2.25.260102.56.release",
}

def _validate_allowlisted_symlink(root: str, rel: str, target: str) -> None:
    """Validate an allowlisted XRT soname symlink's target chain."""
    if rel not in ALLOWLISTED_SYMLINKS:
        raise SystemExit(f"unexpected symlink {rel!r} -> {target!r}")
    # Reject absolute targets and parent traversal
    if os.path.isabs(target):
        raise SystemExit(f"symlink {rel!r} has absolute target {target!r}")
    if ".." in target.split(os.sep):
        raise SystemExit(f"symlink {rel!r} target escapes via '..': {target!r}")
    expected = ALLOWLISTED_SYMLINKS[rel]
    if target != expected and os.path.basename(target) != expected:
        # Require exact basename match at minimum; full relative check above
        raise SystemExit(f"symlink {rel!r} target mismatch: {target!r} "
                         f"expected {expected!r}")
    # Must not be dangling and must resolve inside the XRT root
    link_path = os.path.join(root, rel)
    try:
        resolved = os.path.realpath(link_path)
        # realpath resolves the symlink; must be inside root and exist
        if not resolved.startswith(os.path.realpath(root) + os.sep):
            raise SystemExit(f"symlink {rel!r} resolves outside XRT root: "
                             f"{resolved!r}")
        if not os.path.isfile(resolved):
            raise SystemExit(f"symlink {rel!r} dangles (target missing): "
                             f"{target!r} -> {resolved!r}")
    except OSError as e:
        raise SystemExit(f"symlink {rel!r} validation failed: {e}") from e

def walk(root: str, table, strip: str) -> list[dict]:
    # venv scaffolding (pip, dist-info) and bytecode caches are recreated
    # at build time, never part of the audited set.
    skip_dirs = {"__pycache__"}
    skip_top = {"pip", "pip-24.0.dist-info"}
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Reject directory symlinks outright (no allowlisted dir links)
        for d in list(dirnames):
            full_d = os.path.join(dirpath, d)
            if os.path.islink(full_d):
                rel_d = os.path.relpath(full_d, root)
                target_d = os.readlink(full_d)
                raise SystemExit(f"unexpected directory symlink {rel_d!r} "
                                 f"-> {target_d!r}")
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        rel_dir = os.path.relpath(dirpath, root)
        if rel_dir.split(os.sep)[0] in skip_top:
            continue
        for fn in sorted(filenames):
            if fn.endswith((".pyc", ".pyo")):
                continue
            full = os.path.join(dirpath, fn)
            if os.path.islink(full):
                rel = os.path.relpath(full, root)
                target = os.readlink(full)
                _validate_allowlisted_symlink(root, rel, target)
                continue
            rel = os.path.relpath(full, root)
            pkg, ver, lic, role = origin_for(rel, table)
            out.append({"path": rel, "sha256": sha256_file(full),
                        "size_bytes": os.path.getsize(full),
                        "package": pkg, "version": ver,
                        "licence": lic, "role": role})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload", required=True)
    ap.add_argument("--xrt", required=True)
    ap.add_argument("--flexmlrt", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--legal", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sp = os.path.join(args.payload, "site-packages")
    files = walk(sp, ORIGINS, sp)
    files += [{"path": "xrt/" + f["path"], **{k: v for k, v in f.items()
                                               if k != "path"}}
              for f in walk(args.xrt, XRT_ORIGINS, args.xrt)]
    if args.flexmlrt:
        files += [{"path": "flexmlrt/" + os.path.basename(f["path"]),
                   "sha256": f["sha256"], "size_bytes": f["size_bytes"],
                   "package": "flexmlrt", "version": "1.8.0",
                   "licence": "amd-eula", "role": "standalone FlexMLRT runtime"}
                  for f in walk(args.flexmlrt, [], args.flexmlrt)
                  if f["path"].endswith("libflexmlrt.so")]
    if args.calib:
        files += [{"path": "calib/" + f["path"], "sha256": f["sha256"],
                   "size_bytes": f["size_bytes"], "package": "coco-calib",
                   "version": "coco128", "licence": "cc-by-4.0",
                   "role": "BF16 calibration images (public COCO)"}
                  for f in walk(args.calib, [], args.calib)]
    if args.legal:
        files += [{"path": "legal/" + f["path"], "sha256": f["sha256"],
                   "size_bytes": f["size_bytes"], "package": "amd-legal",
                   "version": "1.8", "licence": "amd-eula",
                   "role": "EULA/TPN flow-down notices"}
                  for f in walk(args.legal, [], args.legal)
                  if f["path"].endswith(".txt") or f["path"].endswith(".pdf")]
    unmapped = [f for f in files if f["licence"] == "UNMAPPED"]
    if unmapped:
        print(f"UNMAPPED FILES: {[f['path'] for f in unmapped][:10]}")
        return 1

    files_manifest = {"schema_version": 1, "file_count": len(files),
                      "total_bytes": sum(f["size_bytes"] for f in files),
                      "files": files}
    with open(os.path.join(args.out, "vendor-files.manifest.json"),
              "w") as f:
        json.dump(files_manifest, f, indent=1, sort_keys=True)

    components: dict[tuple, dict] = {}
    for entry in files:
        key = (entry["package"], entry["version"], entry["licence"])
        comp = components.setdefault(key, {
            "package": entry["package"], "version": entry["version"],
            "licence": entry["licence"], "roles": sorted(set()),
            "file_count": 0, "total_bytes": 0, "files_sha256": None})
        comp["file_count"] += 1
        comp["total_bytes"] += entry["size_bytes"]
        comp["roles"] = sorted(set(comp["roles"]) | {entry["role"]})
    for comp in components.values():
        subset = sorted(f["sha256"] for f in files
                        if (f["package"], f["version"], f["licence"]) == (
                            comp["package"], comp["version"],
                            comp["licence"]))
        comp["files_sha256"] = hashlib.sha256(
            "\n".join(subset).encode()).hexdigest()
        comp["roles"] = "; ".join(comp["roles"])
    vendor_lock = {
        "schema_version": 1,
        "status": "pinned: exact audited Phase-7.7 payload (B1/B2 evidence)",
        "payload_manifest_sha256": "8a8d28b751974205ffc4e2b87b34cd3e93584b70d8f4f3de557965aee975f4bd",
        "components": sorted(components.values(),
                             key=lambda c: c["package"]),
    }
    with open(os.path.join(args.out, "vendor.lock.json"), "w") as f:
        json.dump(vendor_lock, f, indent=2, sort_keys=True)

    # Synchronize legal/component-map.json from the same locked collection
    # (keeps both outputs consistent; NumPy and all components included).
    # Read the committed source, not args.out (which may be a clean temp
    # during generation); fail loudly if it is missing or malformed.
    committed_map = os.path.join(
        os.path.dirname(__file__), "..", "packaging", "legal",
        "component-map.json")
    try:
        with open(committed_map) as f:
            existing = json.load(f)
    except (OSError, ValueError) as e:
        print(f"cannot read committed component-map {committed_map!r}: {e}",
              file=sys.stderr)
        return 1
    governing = existing.get("governing_terms")
    rules = existing.get("rules")
    if not isinstance(governing, list) or not isinstance(rules, list):
        print("committed component-map missing governing_terms/rules",
              file=sys.stderr)
        return 1
    comp_map = {
        "schema_version": 1,
        "status": vendor_lock["status"] + "; per-file manifest in "
                  "packaging/vendor-files.manifest.json",
        "components": [
            {
                "package": c["package"], "version": c["version"],
                "licence": c["licence"], "file_count": c["file_count"],
                "files_sha256": c["files_sha256"],
                "total_bytes": c["total_bytes"],
                "purpose": c["roles"],
            } for c in vendor_lock["components"]
        ],
        "governing_terms": governing,
        "rules": rules,
    }
    os.makedirs(os.path.join(args.out, "legal"), exist_ok=True)
    with open(os.path.join(args.out, "legal", "component-map.json"), "w") as f:
        json.dump(comp_map, f, indent=2, sort_keys=True)

    print(f"files={len(files)} bytes={files_manifest['total_bytes']} "
          f"components={len(components)}")
    for comp in sorted(components.values(), key=lambda c: -c["total_bytes"]):
        print(f"  {comp['package']:24s} {comp['total_bytes'] // 1024 // 1024:5d} MB"
              f"  {comp['file_count']:5d} files  {comp['licence']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
