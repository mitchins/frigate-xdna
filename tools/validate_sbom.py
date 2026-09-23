#!/usr/bin/env python3
"""Fail-closed SBOM gate for the release path (Task 8.5.2).

Verifies the generated document is structurally the CycloneDX 1.5 the
GitHub attestation path accepts — before any image is pushed, so an
obviously unusable SBOM can never survive until `docker push`.

Checks: JSON parses; bomFormat/specVersion/$schema pinned;
serialNumber present and urn:uuid:<RFC-4122 UUID>; components is an
array; every emitted licence choice carries a non-empty `name` and
never puts our internal vendor labels through `license.id` (which
CycloneDX reserves for SPDX identifiers).

No network, no third-party SBOM package. Exit 0 on accept, 1 with a
useful error on reject. Directly unit-tested.
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid

SCHEMA_URI = "http://cyclonedx.org/schema/bom-1.5.schema.json"


def fail(reason: str) -> int:
    print(f"validate_sbom: REJECT: {reason}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sbom", help="SBOM JSON path to validate")
    args = ap.parse_args(argv)

    try:
        with open(args.sbom, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError) as e:
        return fail(f"unreadable JSON: {e}")
    if not isinstance(doc, dict):
        return fail("top-level document is not an object")
    if doc.get("bomFormat") != "CycloneDX":
        return fail(f"bomFormat is {doc.get('bomFormat')!r}, want CycloneDX")
    if doc.get("specVersion") != "1.5":
        return fail(f"specVersion is {doc.get('specVersion')!r}, want 1.5")
    if doc.get("$schema") != SCHEMA_URI:
        return fail(f"$schema is {doc.get('$schema')!r}")
    serial = doc.get("serialNumber")
    if not serial:
        return fail("missing serialNumber")
    if not serial.startswith("urn:uuid:"):
        return fail(f"serialNumber is not a urn:uuid: {serial!r}")
    try:
        parsed = uuid.UUID(serial[len("urn:uuid:"):])
    except ValueError:
        return fail(f"serialNumber is not a UUID: {serial!r}")
    if parsed.variant != uuid.RFC_4122:
        return fail(f"serialNumber is not RFC-4122: {serial!r}")
    components = doc.get("components")
    if not isinstance(components, list):
        return fail("components is not an array")
    for comp in components:
        for choice in comp.get("licenses", []):
            lic = choice.get("license", {})
            if "id" in lic:
                return fail(
                    f"component {comp.get('bom-ref')!r} puts {lic['id']!r}"
                    " through license.id (SPDX-only; use name)")
            if not lic.get("name"):
                return fail(
                    f"component {comp.get('bom-ref')!r} licence choice"
                    " has no non-empty name")
    print(f"validate_sbom: ACCEPT components={len(components)}"
          f" serial={serial}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
