#!/usr/bin/env python3
"""bf16-vaiml-v1 step 1: Quark BF16 preparation (audited recipe).

Reads an ordinary ONNX, writes a BF16-quantized ONNX using the pinned
Quark config + BF16QDQToCast over 32 calibration images resized to the
model's own input geometry. No network, no device, no secrets.

Usage: prepare.py --onnx IN.onnx --calib DIR --out OUT.onnx
Prints: BF16_PREPARE_OK <seconds> <out_sha256> as the last line.
"""
import argparse
import copy
import hashlib
import os
import resource
import sys
import time

import numpy as np
import onnx
from onnxruntime.quantization import CalibrationDataReader
from PIL import Image
from quark.onnx import ModelQuantizer
from quark.onnx.quantization.config.config import Config
from quark.onnx.quantization.config.custom_config import get_default_config

# Safe resource limits (replaces unsafe preexec_fn in launcher): 6 GiB AS.
# File-size capping is handled by parent log truncation, not RLIMIT_FSIZE,
# so artifact writes (BF16 ONNX) are not capped at 8 MiB.
try:
    resource.setrlimit(resource.RLIMIT_AS, (6 * 1024 ** 3, 6 * 1024 ** 3))
except (ValueError, OSError):
    pass


class PILDataReader(CalibrationDataReader):
    def __init__(self, folder, input_name, geometry, limit=32):
        all_names = sorted(
            f for f in os.listdir(folder) if f.endswith(".jpg"))
        if len(all_names) != limit:
            raise SystemExit(
                f"calibration images: expected {limit}, found "
                f"{len(all_names)} in {folder!r}")
        names = all_names[:limit]
        self.data = []
        for n in names:
            img = Image.open(os.path.join(folder, n)).convert("RGB").resize(
                (geometry, geometry), Image.BILINEAR)
            x = np.asarray(img, dtype=np.float32) / 255.0
            self.data.append({input_name: np.ascontiguousarray(
                x.transpose(2, 0, 1)[None])})
        self.enum_data = None

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(self.data)
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None


def geometry_of(onnx_path: str) -> tuple[str, int]:
    model = onnx.load(onnx_path)
    dims = model.graph.input[0].type.tensor_type.shape.dim
    shape = [d.dim_value for d in dims]
    if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
        raise SystemExit(f"unsupported input shape {shape}")
    if shape[2] != shape[3]:
        raise SystemExit(f"non-square input {shape}")
    return model.graph.input[0].name, shape[2]


def restore_float_inputs(path: str) -> int:
    """Collapse input quantization chains back to float graph inputs.

    Quark's BF16QDQToCast conversion can leave Cast nodes with BF16
    inputs (e.g. on the images edge) that onnxruntime-vitisai refuses
    to load. Walk each graph input through single-consumer
    QuantizeLinear/DequantizeLinear/Cast nodes, rewire the first real
    consumer back to the graph input, and drop the orphaned chain.
    Numerically exact: restores the original float32 input edge (the
    compile probe feeds float32). Model-agnostic: graphs without such
    chains are byte-unchanged apart from a checker pass. Returns the
    number of collapsed chains.
    """
    model = onnx.load(path)
    graph_inputs = {i.name for i in model.graph.input}
    edge_ops = {"QuantizeLinear", "DequantizeLinear", "Cast"}

    def consumers():
        cons = {}
        for n in model.graph.node:
            for i in n.input:
                cons.setdefault(i, []).append(n)
        return cons

    collapsed = 0
    for gi in sorted(graph_inputs):
        chain = []
        cur = gi
        while True:
            users = consumers().get(cur, [])
            if len(users) != 1:
                break
            nxt = users[0]
            if nxt.op_type not in edge_ops or len(nxt.output) != 1:
                break
            chain.append(nxt)
            cur = nxt.output[0]
        if not chain:
            continue
        tail_users = consumers().get(cur, [])
        if not tail_users:
            continue
        for n in model.graph.node:
            for idx, inp in enumerate(n.input):
                if inp == cur:
                    n.input[idx] = gi
        collapsed += 1

    # Drop edge-op nodes no kept node consumes anymore (cascading),
    # then prune orphaned initializers/value_info. Non-edge nodes and
    # graph outputs are never touched.
    used = set()
    for n in model.graph.node:
        used.update(n.input)
        used.update(n.output)
    used.update(o.name for o in model.graph.output)
    graph_outputs = set(o.name for o in model.graph.output)
    # Index-based liveness (protobuf node wrappers have unstable id()).
    nodes = list(model.graph.node)
    keep = set(range(len(nodes)))
    changed = True
    while changed:
        changed = False
        live_inputs = set()
        for i in keep:
            live_inputs.update(nodes[i].input)
        for i in list(keep):
            n = nodes[i]
            if n.op_type in edge_ops and not any(
                    o in live_inputs or o in graph_outputs
                    for o in n.output):
                keep.discard(i)
                changed = True
    kept = [nodes[i] for i in range(len(nodes)) if i in keep]
    del model.graph.node[:]
    model.graph.node.extend(kept)
    refd = set()
    for n in model.graph.node:
        refd.update(n.input)
    refd.update(o.name for o in model.graph.output)
    kept_init = [t for t in model.graph.initializer if t.name in refd]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_init)
    kept_vi = [v for v in model.graph.value_info if v.name in refd]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_vi)
    onnx.checker.check_model(model, full_check=False)
    with open(path, "wb") as f:
        f.write(model.SerializeToString())
    return collapsed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    input_name, geom = geometry_of(args.onnx)
    qc = get_default_config("BF16")
    qc.extra_options["BF16QDQToCast"] = True
    # Quark's own VAIML-targeted repair: removes BF16 Cast couples and
    # converts stray BF16 weights to float32 (see remove_bf16_cast).
    qc.extra_options["EnableVaimlBF16"] = True
    t0 = time.time()
    ModelQuantizer(Config(global_quant_config=copy.deepcopy(qc))).quantize_model(
        args.onnx, args.out,
        PILDataReader(args.calib, input_name, geom, 32))
    collapsed = restore_float_inputs(args.out)
    digest = hashlib.sha256(open(args.out, "rb").read()).hexdigest()
    print(f"BF16_PREPARE_OK {time.time() - t0:.1f}s {digest}"
          f" collapsed_inputs={collapsed}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
