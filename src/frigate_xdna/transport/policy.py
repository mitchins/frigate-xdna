"""Transfer-gate policy (INTERFACES.md §2, §5.4).

Decides whether an inbound ZMQ model transfer is accepted. The default
product mode accepts ordinary ONNX bytes over the trusted Docker network:
stock Frigate rc2 only becomes ready after its startup transfer succeeds,
and every new routing identity must transfer source bytes so the binding
is to an exact content hash, never a basename.

`allow_uploads=false` is a hardening mode for deployments that
intentionally disable Frigate model transfer. The Task 04 ROUTER frontend
calls `check_transfer` before reading transfer bytes; rejection uses the
stock `model_saved=false` + `INVALID_MODEL` reply (INTERFACES.md §3).
"""
from __future__ import annotations

from ..errors import INVALID_ARGS, FxdnaError

MAX_MODEL_BYTES = 256 * 1024 * 1024
MAX_HEADER_BYTES = 16 * 1024


def check_transfer(allow_uploads: bool, nbytes: int) -> None:
    """Raise FxdnaError(INVALID_MODEL) when a transfer must be refused."""
    if not allow_uploads:
        raise FxdnaError(INVALID_ARGS, "INVALID_MODEL",
                         "model transfers disabled by configuration"
                         " (FXDNA_ALLOW_UPLOADS=false)")
    if nbytes <= 0 or nbytes > MAX_MODEL_BYTES:
        raise FxdnaError(INVALID_ARGS, "INVALID_MODEL",
                         f"transfer size out of bounds: {nbytes} bytes")
