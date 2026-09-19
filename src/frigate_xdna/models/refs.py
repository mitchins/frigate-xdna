"""Model reference parsing (SPEC §5.4, INTERFACES.md §1).

REF forms: `plus://ID`, a local ONNX path, or a local RAI path with a
descriptor. Content keys identify models; basenames and wire names are
aliases only.
"""
from __future__ import annotations

import os
import re
import urllib.parse

from ..errors import INVALID_ARGS, FxdnaError

PLUS_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")


def parse_ref(ref: str) -> dict:
    """Parse a user-supplied model ref into kind + identity.

    Raises FxdnaError(INVALID_ARGS) on malformed refs. Local paths are NOT
    resolved to content here; ingestion hashes bytes once (Task: store).
    """
    ref = (ref or "").strip()
    if not ref:
        raise FxdnaError(INVALID_ARGS, "INVALID_REF", "empty model ref")
    if "," in ref or "\n" in ref:
        raise FxdnaError(INVALID_ARGS, "INVALID_REF",
                         f"model ref must be a single value: {ref!r}")
    if ref.startswith("plus://"):
        model_id = ref[len("plus://"):]
        if not model_id or not PLUS_ID_RE.fullmatch(model_id):
            raise FxdnaError(INVALID_ARGS, "INVALID_REF",
                             f"invalid Plus model ID: {ref!r}")
        return {"kind": "plus", "id": model_id,
                "ref": f"plus://{model_id}"}
    if "://" in ref:
        scheme = ref.split("://", 1)[0].lower()
        raise FxdnaError(INVALID_ARGS, "INVALID_REF",
                         f"unsupported ref scheme {scheme!r}: use plus://ID "
                         f"or a local ONNX/RAI path")
    # Local path: no shell expansion, no URI tricks; confined to the import
    # root at ingestion time (store.py enforces containment).
    path = os.path.expanduser(ref)
    if not path or path != ref.strip():
        raise FxdnaError(INVALID_ARGS, "INVALID_REF",
                         f"invalid local path: {ref!r}")
    lower = path.lower()
    if lower.endswith(".onnx"):
        return {"kind": "onnx", "path": path, "ref": path}
    if lower.endswith(".rai"):
        return {"kind": "rai", "path": path, "ref": path}
    if lower.endswith((".pt", ".pth", ".pkl", ".pickle", ".bin", ".dll",
                       ".so", ".dylib", ".py")):
        raise FxdnaError(INVALID_ARGS, "INVALID_REF",
                         f"refused executable/pickle/custom-op format: {ref!r}")
    raise FxdnaError(INVALID_ARGS, "INVALID_REF",
                     f"local ref must be .onnx or .rai (+descriptor): {ref!r}")


def wire_alias(ref: dict, wire_name: str | None) -> str:
    """A wire basename is an alias hint only, never an identity."""
    if wire_name:
        name = os.path.basename(wire_name.strip())
        if not name:
            raise FxdnaError(INVALID_ARGS, "INVALID_REF",
                             "empty wire name alias")
        return name
    if ref["kind"] == "plus":
        return ref["id"]
    return os.path.basename(ref["path"])


def quote_path_component(name: str) -> str:
    """Sanitize a ref-derived string for filesystem use (never trusted)."""
    safe = urllib.parse.quote(name, safe="")
    if not safe or safe in (".", ".."):
        raise FxdnaError(INVALID_ARGS, "INVALID_REF",
                         "unusable ref-derived name")
    return safe
