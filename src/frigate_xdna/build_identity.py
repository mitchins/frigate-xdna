"""Authoritative build identity (v0.1.1 checkpoint 2).

One identity — release version + source revision — reported
consistently by startup output, status output, `fxdna --version` and
the release sanity gate. Baked into the image at build time
(`packaging/Dockerfile` writes /opt/fxdna/build-identity.json from
the release workflow's build args); never reconstructed from a
mutable image tag at runtime.

Without a baked file (source checkout, dev venv) the build reports
itself as development explicitly. A present-but-malformed file also
falls back to development rather than crashing startup; the release
sanity gate (which compares reported vs expected identity) fails
such an image loudly instead.

`FXDNA_BUILD_IDENTITY_FILE` overrides the path for tests only.
"""
from __future__ import annotations

import json
import os
import re

IDENTITY_PATH = "/opt/fxdna/build-identity.json"
IDENTITY_ENV = "FXDNA_BUILD_IDENTITY_FILE"

STABLE_RE = re.compile(r"^\d+\.\d+\.\d+$")
RC_RE = re.compile(r"^\d+\.\d+\.\d+-rc\.\d+$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$|^[0-9a-f]{64}$")

CHANNEL_DEVELOPMENT = "development"
CHANNEL_RC = "release-candidate"
CHANNEL_RELEASE = "release"


def _development(version: str) -> dict:
    return {"schema_version": 1, "version": version,
            "revision": "unknown", "channel": CHANNEL_DEVELOPMENT}


def channel_for(version: str) -> str:
    """Release channel derived from the version string alone."""
    if RC_RE.match(version or ""):
        return CHANNEL_RC
    if STABLE_RE.match(version or ""):
        return CHANNEL_RELEASE
    return CHANNEL_DEVELOPMENT


def get_build_identity() -> dict:
    """Read the baked identity, or report a development build."""
    from . import __version__
    path = os.environ.get(IDENTITY_ENV, IDENTITY_PATH)
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return _development(__version__)
    if not isinstance(raw, dict):
        return _development(__version__)
    version = raw.get("version")
    revision = raw.get("revision")
    if not isinstance(version, str) or not isinstance(revision, str):
        return _development(__version__)
    if not (STABLE_RE.match(version) or RC_RE.match(version)):
        return _development(__version__)
    if not REVISION_RE.match(revision):
        return _development(__version__)
    return {"schema_version": 1, "version": version,
            "revision": revision, "channel": channel_for(version)}


def format_identity(ident: dict) -> str:
    """Canonical one-line identity for console output."""
    return (f"frigate-xdna {ident['version']} "
            f"revision={ident['revision']}")


def version_string() -> str:
    """`fxdna --version` output: version, revision and channel."""
    ident = get_build_identity()
    return (f"fxdna {ident['version']} revision={ident['revision']} "
            f"channel={ident['channel']}")
