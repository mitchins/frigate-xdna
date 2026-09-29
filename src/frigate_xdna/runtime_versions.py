"""Pinned embedded runtime versions (appliance, not host).

The appliance carries its own XRT/XDNA-shim/FlexMLRT payload and
never uses arbitrary host XRT libraries (docs/COMPATIBILITY.md:
"Appliance runtime vs host tooling"). These constants are the
single source the running build reports via `fxdna host-info`;
unit tests pin them against the vendor lockfile and the Dockerfile
sonames so a payload bump cannot silently desync the report.
"""
from __future__ import annotations

EMBEDDED_XRT = "2.25.37"
EMBEDDED_XDNA_SHIM = "2.25.260102.56"
EMBEDDED_FLEXMLRT = "1.8.0"
# Compile recipe reported alongside (supervisor.RECIPE_ID).
RECIPE_ID = "bf16-vaiml-v1"


def embedded_runtime() -> dict:
    """Embedded (appliance) runtime versions as a JSON-ready dict."""
    return {"schema_version": 1,
            "xrt": EMBEDDED_XRT,
            "xdna_shim": EMBEDDED_XDNA_SHIM,
            "flexmlrt": EMBEDDED_FLEXMLRT,
            "recipe": RECIPE_ID}
