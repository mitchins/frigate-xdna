"""ONNX source inspection (SPEC §5.4, §7).

Validates ordinary compatible ONNX with the pinned `onnx` package:
- loadable graph, opset/IR recorded;
- single static-shape batch-one 3-channel float input;
- output tensor contract extracted (names/shapes/dtypes);
- external-data references refused unless inside an explicit bundle
  manifest confined to the import root (bundles deferred: Task 02 rejects
  external data with a clear error — see WORKQUEUE).

No pickle, no custom-op DLLs, no network fetch. Unknown/ambiguous output
contracts are UNSUPPORTED_CONTRACT, never a guess.
"""
from __future__ import annotations

import hashlib
import os

import onnx
from onnx import TensorProto

from ..errors import INVALID_ARGS, UNSUPPORTED_CONTRACT, FxdnaError

MAX_ONNX_BYTES = 256 * 1024 * 1024
ONNX_MAGIC = b"\x08"  # protobuf field 1 (varint) — first byte of ModelProto


def _fail(code: int, error_code: str, message: str) -> FxdnaError:
    return FxdnaError(code, error_code, message)


def load_graph_bytes(data: bytes, max_bytes: int = MAX_ONNX_BYTES):
    """Validate ONNX bytes already bounded by the caller. Returns (model, digest)."""
    if not data or len(data) > max_bytes or data[:1] != ONNX_MAGIC:
        raise _fail(INVALID_ARGS, "INVALID_MODEL",
                    "not an ONNX protobuf (bad magic/size)")
    digest = hashlib.sha256(data).hexdigest()
    try:
        model = onnx.load_from_string(data)
    except Exception as e:
        raise _fail(INVALID_ARGS, "INVALID_MODEL",
                    f"ONNX parse failed: {type(e).__name__}")
    try:
        onnx.checker.check_model(model, full_check=False)
    except Exception as e:
        raise _fail(INVALID_ARGS, "INVALID_MODEL",
                    f"ONNX checker rejected model: {e}")
    return model, digest


def load_graph(path: str, max_bytes: int = MAX_ONNX_BYTES):
    """Load and minimally validate an ONNX file. Returns (model, digest)."""
    try:
        size = os.path.getsize(path)
    except OSError:
        raise _fail(INVALID_ARGS, "INVALID_MODEL",
                    f"cannot stat model file: {path!r}")
    if size <= 0 or size > max_bytes:
        raise _fail(INVALID_ARGS, "INVALID_MODEL",
                    f"model size out of bounds: {size} bytes")
    with open(path, "rb") as f:
        raw = f.read(max_bytes + 1)
    return load_graph_bytes(raw, max_bytes)


def _dim_value(dim) -> int | None:
    if dim.HasField("dim_value"):
        return int(dim.dim_value)
    return None  # dim_param (symbolic) or unknown


def inspect_model(model) -> dict:
    """Extract the serving-relevant contract from a checked ONNX graph."""
    graph = model.graph
    if len(graph.input) != 1:
        raise _fail(UNSUPPORTED_CONTRACT, "UNSUPPORTED_CONTRACT",
                    f"expected 1 graph input, found {len(graph.input)}")
    inp = graph.input[0]
    ttype = inp.type.tensor_type
    if ttype.elem_type != TensorProto.FLOAT:
        raise _fail(UNSUPPORTED_CONTRACT, "UNSUPPORTED_CONTRACT",
                    "only float32 graph inputs are supported")
    shape = [_dim_value(d) for d in ttype.shape.dim]
    if len(shape) != 4:
        raise _fail(UNSUPPORTED_CONTRACT, "UNSUPPORTED_CONTRACT",
                    f"input rank {len(shape)} is not supported;"
                    " export a static batch-one NCHW model")
    if any(v is None for v in shape):
        raise _fail(UNSUPPORTED_CONTRACT, "UNSUPPORTED_CONTRACT",
                    "dynamic input dimensions are not supported;"
                    " export a static batch-one NCHW model")
    if (shape[0] != 1 or shape[1] != 3
            or any(v <= 0 or v > 4096 for v in shape)):
        raise _fail(UNSUPPORTED_CONTRACT, "UNSUPPORTED_CONTRACT",
                    f"unsupported static input shape: {shape!r}")
    outputs = []
    for out in graph.output:
        otype = out.type.tensor_type
        if not otype.HasField("shape"):
            raise _fail(UNSUPPORTED_CONTRACT, "UNSUPPORTED_CONTRACT",
                        f"output {out.name!r} has no static shape")
        odims = [_dim_value(d) for d in otype.shape.dim]
        if any(v is None for v in odims):
            raise _fail(UNSUPPORTED_CONTRACT, "UNSUPPORTED_CONTRACT",
                        f"output {out.name!r} has dynamic dims")
        elem = TensorProto.DataType.Name(otype.elem_type).lower()
        outputs.append({"name": out.name, "shape": odims, "dtype": elem})
    if not outputs:
        raise _fail(UNSUPPORTED_CONTRACT, "UNSUPPORTED_CONTRACT",
                    "graph has no outputs")
    externals = [i.name for i in graph.initializer
                 if i.data_location == TensorProto.EXTERNAL]
    if externals:
        raise _fail(UNSUPPORTED_CONTRACT, "UNSUPPORTED_CONTRACT",
                    f"external-data weights require an explicit bundle "
                    f"({len(externals)} tensors, e.g. {externals[0]!r})")
    opset = max((o.version for o in model.opset_import), default=0)
    return {
        "input_name": inp.name,
        "input_shape": shape,
        "input_dtype": "float32",
        "outputs": outputs,
        "opset": opset,
        "ir_version": int(model.ir_version),
        "node_count": len(graph.node),
    }


def classify_output(outputs: list[dict]) -> dict:
    """Map output tensors to a supported decoder profile or fail.

    Supported: single raw YOLO tensor [1, 4+C, N] (profile yolo-raw).
    Everything else is UNSUPPORTED_CONTRACT.
    """
    if len(outputs) != 1:
        return {"profile": None,
                "error": f"{len(outputs)} outputs: only a single raw-YOLO"
                         f" output tensor is supported in this release"
                         f" (embedded-NMS exports are not supported;"
                         f" export raw predictions instead)"}
    o = outputs[0]
    shape = o["shape"]
    if (len(shape) == 3 and shape[0] == 1 and shape[1] >= 6
            and shape[2] >= 100 and o["dtype"] == "float"):
        return {"profile": "yolo-raw",
                "tensor": o["name"],
                "channels": shape[1],
                "anchors": shape[2]}
    return {"profile": None,
            "error": f"output {o['name']} shape {shape} dtype {o['dtype']}:"
                     f" not a supported raw-YOLO contract"
                     f" ([1,4+C,N] float32 required)"}


def summarize_inspection(contract: dict | None, cls: dict | None,
                         error: FxdnaError | None = None,
                         source_sha256: str | None = None) -> dict:
    """Stable appliance view of one graph inspection.

    The same dict drives preparation, console logs and status: a single
    result, never a second validator with its own rules. Numeric class
    count only (channels-4 for the raw-YOLO profile); label names are
    never inferred — Frigate owns the label map. Compatibility is never
    claimed from a filename or model family, only from this graph.
    `source_sha256` binds the summary to the exact bytes described, so
    a refusal stored for pending bytes is never mistaken for the
    verdict on the ref's current source.
    """
    profile = (cls or {}).get("profile")
    compatible = error is None and profile is not None
    shape = (contract or {}).get("input_shape")
    outputs = (contract or {}).get("outputs") or []
    class_count = None
    output_line = None
    if profile == "yolo-raw" and contract is not None:
        channels = int((cls or {}).get("channels", 0))
        class_count = channels - 4 if channels >= 5 else None
        anchors = (cls or {}).get("anchors", "?")
        output_line = f"raw YOLO [1,{channels},{anchors}]"
    return {
        "compatible": compatible,
        "source_sha256": source_sha256,
        "reason": None if compatible else
        (error.message if error is not None else
         (cls or {}).get("error", "incompatible contract")),
        "input": "x".join(str(v) for v in shape)
        if isinstance(shape, list) else None,
        "input_dtype": (contract or {}).get("input_dtype"),
        "layout": "NCHW" if shape is not None else None,
        "profile": profile,
        "output": output_line,
        "outputs": outputs,
        "class_count": class_count,
        "opset": (contract or {}).get("opset"),
        "ir_version": (contract or {}).get("ir_version"),
    }


def _strict_int_list(values, what: str) -> list[int]:
    """Validate a four-element geometry: ints only (bool is not an int
    here), no nulls, no floats/strings. Coercion is a correctness bug:
    reject instead."""
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise _fail(UNSUPPORTED_CONTRACT, "UNSUPPORTED_CONTRACT",
                    f"invalid {what} geometry: {values!r}")
    out = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, int):
            raise _fail(UNSUPPORTED_CONTRACT, "UNSUPPORTED_CONTRACT",
                        f"invalid {what} geometry value {v!r}: integers only")
        out.append(v)
    return out


def _check_layout_tag(key: str, claimed: str) -> None:
    """Validate a string layout claim (real Frigate+ inputShape "nchw").

    Numeric geometry comes from width/height in that case. Accept NCHW
    (our only serving layout), reject anything else explicitly. Every
    present key is checked: a contradicting pair cannot hide behind
    key order.
    """
    if claimed.lower() != "nchw":
        raise _fail(
            UNSUPPORTED_CONTRACT, "UNSUPPORTED_CONTRACT",
            f"unsupported metadata {key} layout {claimed!r}:"
            " only NCHW is served")


def compare_plus_metadata(info: dict, inspected: dict) -> None:
    """Fail on metadata-vs-graph conflicts (SPEC §5.3).

    Compares input geometry only where the Plus metadata actually provides
    it (common keys width/height, input_shape, ...); absent fields are
    recorded, never guessed. Raises UNSUPPORTED_CONTRACT on conflict.
    """
    want = _want_from_lists(info)
    if want is None:
        want = _want_from_wh(info)
    if want is None:
        return  # metadata carries no geometry claim; nothing to conflict
    got = inspected["input_shape"]
    if want != got:
        raise _fail(UNSUPPORTED_CONTRACT, "UNSUPPORTED_CONTRACT",
                    f"Plus metadata geometry {want} conflicts with ONNX "
                    f"graph input {got}")


def _want_from_lists(info: dict) -> list[int] | None:
    """First int-list geometry claim (string layouts checked, not used)."""
    want = None
    for key in ("input_shape", "inputShape", "dimensions", "input_dims"):
        if key not in info:
            continue
        claimed = info[key]
        if isinstance(claimed, str):
            _check_layout_tag(key, claimed)
            continue
        if want is None:
            want = _strict_int_list(claimed, f"metadata {key}")
    return want


def _want_from_wh(info: dict) -> list[int] | None:
    """[1,3,h,w] from width/height; present-but-invalid is malformed."""
    if "width" not in info and "height" not in info:
        return None
    w, h = info.get("width"), info.get("height")
    for v in (w, h):
        if isinstance(v, bool) or not isinstance(v, int):
            raise _fail(UNSUPPORTED_CONTRACT, "UNSUPPORTED_CONTRACT",
                        f"invalid metadata geometry value {v!r}:"
                        " integers only")
    return [1, 3, h, w]
