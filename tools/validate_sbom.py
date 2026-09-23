#!/usr/bin/env python3
"""Fail-closed SBOM gate for the release path (Task 8.5.2).

Verifies the generated document is structurally the CycloneDX 1.5 the
GitHub attestation path accepts — before any image is pushed, so an
obviously unusable SBOM can never survive until `docker push`.

Checks: JSON parses to an object; bomFormat/specVersion/$schema pinned;
serialNumber present, a string, and urn:uuid:<canonical RFC-4122 UUID>;
components is an array of objects with unique bom-refs (including
metadata.component); every emitted licence choice carries a non-empty
`name` and never puts our internal vendor labels through `license.id`
(which CycloneDX reserves for SPDX identifiers).

Wrongly typed fields are rejected with REJECT, never a traceback.
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


def load_document(path: str) -> tuple[dict | None, str | None]:
    """Read and parse the SBOM: (doc, None), or (None, reason)."""
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError) as e:
        return None, f"unreadable JSON: {e}"
    if not isinstance(doc, dict):
        return None, "top-level document is not an object"
    return doc, None


def check_identity(doc: dict) -> tuple[str | None, str | None]:
    """Verify CycloneDX 1.5 document identity: (serial, None) or reject."""
    for key, want in (("bomFormat", "CycloneDX"), ("specVersion", "1.5"),
                      ("$schema", SCHEMA_URI)):
        if doc.get(key) != want:
            return None, f"{key} is {doc.get(key)!r}, want {want!r}"
    serial = doc.get("serialNumber")
    if not isinstance(serial, str) or not serial:
        return None, f"bad serialNumber: {serial!r}"
    if not serial.startswith("urn:uuid:"):
        return None, f"serialNumber is not a urn:uuid: {serial!r}"
    body = serial[len("urn:uuid:"):]
    try:
        parsed = uuid.UUID(body)
    except ValueError:
        return None, f"serialNumber is not a UUID: {serial!r}"
    if str(parsed) != body:
        return None, f"serialNumber is not canonical: {serial!r}"
    if parsed.variant != uuid.RFC_4122:
        return None, f"serialNumber is not RFC-4122: {serial!r}"
    return serial, None


def _check_licenses(comp: dict) -> str | None:
    """Verify one component's licence choices: None, or the reason."""
    licenses = comp.get("licenses", [])
    if not isinstance(licenses, list):
        return (f"component {comp.get('bom-ref')!r}"
                 " licenses is not an array")
    for choice in licenses:
        lic = choice.get("license") if isinstance(choice, dict) else None
        if not isinstance(lic, dict):
            return (f"component {comp.get('bom-ref')!r}"
                     " licence choice has no license object")
        if "id" in lic:
            return (f"component {comp.get('bom-ref')!r}"
                     f" puts {lic['id']!r} through license.id"
                     " (SPDX-only; use name)")
        if not lic.get("name"):
            return (f"component {comp.get('bom-ref')!r}"
                     " licence choice has no non-empty name")
    return None


def _seed_refs(doc: dict) -> set[str]:
    """bom-refs already claimed by metadata.component (usually one)."""
    metadata = doc.get("metadata")
    if not isinstance(metadata, dict):
        return set()
    component = metadata.get("component")
    if not isinstance(component, dict):
        return set()
    ref = component.get("bom-ref")
    return {ref} if isinstance(ref, str) else set()


def check_components(doc: dict) -> tuple[int | None, str | None]:
    """Verify components array, bom-ref uniqueness, licences: (n, None)."""
    components = doc.get("components")
    if not isinstance(components, list):
        return None, "components is not an array"
    seen = _seed_refs(doc)
    for index, comp in enumerate(components):
        if not isinstance(comp, dict):
            return None, f"components[{index}] is not an object"
        ref = comp.get("bom-ref")
        if isinstance(ref, str):
            if ref in seen:
                return None, f"duplicate bom-ref {ref!r}"
            seen.add(ref)
        error = _check_licenses(comp)
        if error is not None:
            return None, error
    return len(components), None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sbom", help="SBOM JSON path to validate")
    args = ap.parse_args(argv)

    doc, error = load_document(args.sbom)
    if error is not None:
        return fail(error)
    assert doc is not None
    serial, error = check_identity(doc)
    if error is not None:
        return fail(error)
    assert serial is not None
    count, error = check_components(doc)
    if error is not None:
        return fail(error)
    print(f"validate_sbom: ACCEPT components={count} serial={serial}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
