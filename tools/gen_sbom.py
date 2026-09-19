#!/usr/bin/env python3
"""Generate the release SBOM (CycloneDX 1.5 JSON) from pinned inputs.

Inputs (all committed, audited):
  packaging/vendor.lock.json  audited binary payload components
  requirements.lock            pinned Python deps (runtime section)
  packaging/legal/component-map.json  file-level licence mapping

No network, no image pull, no secrets. Refuses to run if any input is
missing so a partial SBOM can never be mistaken for a complete one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_json(rel: str):
    path = os.path.join(ROOT, rel)
    if not os.path.isfile(path):
        raise SystemExit(f"refusing partial SBOM: missing {rel}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_requirements(rel: str) -> list[tuple[str, str]]:
    path = os.path.join(ROOT, rel)
    if not os.path.isfile(path):
        raise SystemExit(f"refusing partial SBOM: missing {rel}")
    pins: list[tuple[str, str]] = []
    runtime = True  # image installs the runtime section only; record all,
    for line in open(path, encoding="utf-8"):  # ...flagging section below
        line = line.strip()
        if line.startswith("#"):
            # The dev/test section header is the single boundary;
            # earlier headers mention dev tooling in passing.
            if "development/test-only" in line.lower():
                runtime = False
            continue
        if "==" in line and not line.startswith(("-", " ")):
            name, ver = line.split("==", 1)
            pins.append((name.strip(), ver.strip(),
                         "runtime" if runtime else "dev-test"))
    return pins


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="SBOM JSON path")
    ap.add_argument("--image", required=True,
                    help="Image reference this SBOM describes")
    ap.add_argument("--image-id", default="",
                    help="Local image ID (config hash unknown until push)")
    args = ap.parse_args()

    vendor = load_json("packaging/vendor.lock.json")
    # Existence-checked AND bound: the file-level licence map digest
    # pins exactly which mapping this SBOM was generated against.
    load_json("packaging/legal/component-map.json")
    with open(os.path.join(ROOT, "packaging/legal/component-map.json"),
              "rb") as f:
        legal_sha = hashlib.sha256(f.read()).hexdigest()
    pins = parse_requirements("requirements.lock")

    components: list[dict] = []
    for comp in vendor.get("components", []):
        components.append({
            "type": "library",
            "bom-ref": f"vendor:{comp['package']}@{comp['version']}",
            "name": comp["package"],
            "version": str(comp["version"]),
            "scope": "required",
            "hashes": [{"alg": "SHA-256",
                        "content": comp["files_sha256"]}],
            "licenses": [{"license": {"id": comp["licence"]}}],
            "properties": [
                {"name": "fxdna:roles", "value": comp.get("roles", "")},
                {"name": "fxdna:total_bytes",
                 "value": str(comp.get("total_bytes", 0))},
            ],
        })
    for name, ver, scope in pins:
        components.append({
            "type": "library",
            "bom-ref": f"pypi:{name}@{ver}",
            "name": name,
            "version": ver,
            "scope": scope,
            "purl": f"pkg:pypi/{name}@{ver}",
        })
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {
            "component": {
                "type": "container",
                "bom-ref": args.image,
                "name": args.image,
                "version": args.image_id or "unpushed",
            },
            "tools": [{"name": "fxdna-gen-sbom",
                       "version": "8.4"}],
            "properties": [
                {"name": "fxdna:component-map-sha256", "value": legal_sha},
            ],
        },
        "components": sorted(components, key=lambda c: c["bom-ref"]),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(sbom, f, indent=2, sort_keys=True)
        f.write("\n")
    digest = hashlib.sha256(
        open(args.out, "rb").read()).hexdigest()
    print(f"components={len(components)} sha256={digest} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
