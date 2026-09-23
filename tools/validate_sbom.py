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


def load_document(path: str):
    """Read and parse the SBOM; raises SystemExit(1) failing closed."""
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError) as e:
        print(f"validate_sbom: REJECT: unreadable JSON: {e}",
              file=sys.stderr)
        raise SystemExit(1)
    if not isinstance(doc, dict):
        print("validate_sbom: REJECT: top-level document is not an object",
              file=sys.stderr)
        raise SystemExit(1)
    return doc


def check_identity(doc: dict) -> str:
    """Verify CycloneDX 1.5 document identity; returns the serial."""
    for key, want in (("bomFormat", "CycloneDX"), ("specVersion", "1.5"),
                      ("$schema", SCHEMA_URI)):
        if doc.get(key) != want:
            print(f"validate_sbom: REJECT: {key} is {doc.get(key)!r},"
                  f" want {want!r}", file=sys.stderr)
            raise SystemExit(1)
    serial = doc.get("serialNumber")
    if not serial or not serial.startswith("urn:uuid:"):
        print(f"validate_sbom: REJECT: bad serialNumber: {serial!r}",
              file=sys.stderr)
        raise SystemExit(1)
    try:
        parsed = uuid.UUID(serial[len("urn:uuid:"):])
    except ValueError:
        print(f"validate_sbom: REJECT: serialNumber is not a UUID:"
              f" {serial!r}", file=sys.stderr)
        raise SystemExit(1)
    if parsed.variant != uuid.RFC_4122:
        print(f"validate_sbom: REJECT: serialNumber is not RFC-4122:"
              f" {serial!r}", file=sys.stderr)
        raise SystemExit(1)
    return serial


def check_licences(doc: dict) -> int:
    """Verify licence representation; returns the component count."""
    components = doc.get("components")
    if not isinstance(components, list):
        print("validate_sbom: REJECT: components is not an array",
              file=sys.stderr)
        raise SystemExit(1)
    for comp in components:
        for choice in comp.get("licenses", []):
            lic = choice.get("license", {})
            if "id" in lic:
                print(f"validate_sbom: REJECT: component"
                      f" {comp.get('bom-ref')!r} puts {lic['id']!r}"
                      " through license.id (SPDX-only; use name)",
                      file=sys.stderr)
                raise SystemExit(1)
            if not lic.get("name"):
                print(f"validate_sbom: REJECT: component"
                      f" {comp.get('bom-ref')!r} licence choice"
                      " has no non-empty name", file=sys.stderr)
                raise SystemExit(1)
    return len(components)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sbom", help="SBOM JSON path to validate")
    args = ap.parse_args(argv)

    try:
        doc = load_document(args.sbom)
        serial = check_identity(doc)
        count = check_licences(doc)
    except SystemExit as e:
        return int(e.code or 1)
    print(f"validate_sbom: ACCEPT components={count} serial={serial}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
