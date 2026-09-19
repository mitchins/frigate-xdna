"""ONNX test builders (integration tests only).

Builds tiny valid ONNX graphs in-memory with the pinned `onnx` package —
no model binaries in the repo. Declared shapes mirror the product families
(320/640, raw-YOLO single output); weights are small constants, so files
stay under 3 MB and tests stay OOM-safe.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper


def make_raw_yolo(path: str, res: int = 320, classes: int = 80,
                  seed: int = 3) -> bytes:
    """Minimal valid raw-YOLO-shaped ONNX.

    Declared contract: input images [1,3,res,res] float32, output output0
    [1,4+classes,N] float32. The body is a small Constant (seed-varied so
    different seeds give different bytes); no dense weight matrices.
    """
    n = (res // 32) ** 2  # 100 @320, 400 @640
    rng = np.random.default_rng(seed)
    inp = helper.make_tensor_value_info("images", TensorProto.FLOAT,
                                        [1, 3, res, res])
    out = helper.make_tensor_value_info("output0", TensorProto.FLOAT,
                                        [1, 4 + classes, n])
    filler = rng.random((1, 4 + classes, n), dtype=np.float32)
    const = helper.make_node(
        "Constant", [], ["output0"], name="const",
        value=helper.make_tensor("c", TensorProto.FLOAT,
                                 [1, 4 + classes, n],
                                 filler.flatten().tolist()))
    graph = helper.make_graph([const], "tiny-yolo", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid(
        "", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=False)
    data = model.SerializeToString()
    assert len(data) < 4 * 1024 * 1024, len(data)
    with open(path, "wb") as f:
        f.write(data)
    return data


def make_external_data_model(path: str) -> bytes:
    """ONNX with an external-data initializer pointing outside the file."""
    inp = helper.make_tensor_value_info("images", TensorProto.FLOAT,
                                        [1, 3, 8, 8])
    out = helper.make_tensor_value_info("output0", TensorProto.FLOAT,
                                        [1, 6, 4])
    ext = TensorProto()
    ext.name = "w"
    ext.data_type = TensorProto.FLOAT
    ext.dims.extend([8, 8])
    ext.data_location = TensorProto.EXTERNAL
    ext.external_data.add(key="location", value="../escape.bin")
    node = helper.make_node("Relu", ["images"], ["output0"], name="relu")
    graph = helper.make_graph([node], "ext", [inp], [out],
                              initializer=[ext])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid(
        "", 17)])
    model.ir_version = 8
    data = model.SerializeToString()
    with open(path, "wb") as f:
        f.write(data)
    return data
