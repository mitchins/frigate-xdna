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
import sys
import time

import numpy as np
import onnx
from onnxruntime.quantization import CalibrationDataReader
from PIL import Image
from quark.onnx import ModelQuantizer
from quark.onnx.quantization.config.config import Config
from quark.onnx.quantization.config.custom_config import get_default_config


class PILDataReader(CalibrationDataReader):
    def __init__(self, folder, input_name, geometry, limit=32):
        names = sorted(f for f in os.listdir(folder)
                       if f.endswith(".jpg"))[:limit]
        if not names:
            raise SystemExit("no calibration images found")
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    input_name, geom = geometry_of(args.onnx)
    qc = get_default_config("BF16")
    qc.extra_options["BF16QDQToCast"] = True
    t0 = time.time()
    ModelQuantizer(Config(global_quant_config=copy.deepcopy(qc))).quantize_model(
        args.onnx, args.out,
        PILDataReader(args.calib, input_name, geom, 32))
    digest = hashlib.sha256(open(args.out, "rb").read()).hexdigest()
    print(f"BF16_PREPARE_OK {time.time() - t0:.1f}s {digest}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
