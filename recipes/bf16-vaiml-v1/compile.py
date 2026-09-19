#!/usr/bin/env python3
"""bf16-vaiml-v1 step 2: VAIML compile (audited recipe).

Creates the VitisAI EP session over the BF16 ONNX (compile happens inside
session creation), runs one zeros probe, and leaves the .rai + context.json
in the cache dir. No reusable cache: the cache dir must be empty/fresh.

Usage: compile.py --onnx BF16.onnx --config vaiml_config.json
                  --cache-dir DIR --cache-key KEY
Prints: COMPILE_OK <seconds> <rai_sha256> <rai_bytes> as the last line.
"""
import argparse
import hashlib
import os
import sys
import time

import resource

import numpy as np
import onnx
import onnxruntime as ort

try:
    resource.setrlimit(resource.RLIMIT_AS, (6 * 1024 ** 3, 6 * 1024 ** 3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024 * 1024, 8 * 1024 * 1024))
except (ValueError, OSError):
    pass


def geometry_of(onnx_path: str) -> tuple[str, int]:
    model = onnx.load(onnx_path)
    dims = model.graph.input[0].type.tensor_type.shape.dim
    shape = [d.dim_value for d in dims]
    return model.graph.input[0].name, shape[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--cache-key", required=True)
    args = ap.parse_args()
    if os.path.isdir(args.cache_dir) and os.listdir(args.cache_dir):
        raise SystemExit(
            f"cache dir not empty: {args.cache_dir!r} — refusing to reuse; "
            f"provide a fresh empty directory")
    input_name, geom = geometry_of(args.onnx)
    os.makedirs(args.cache_dir, exist_ok=True)
    so = ort.SessionOptions()
    so.log_severity_level = 1
    po = {"config_file": args.config, "cache_dir": args.cache_dir,
          "cache_key": args.cache_key, "enable_cache_file_io_in_mem": "0"}
    t0 = time.perf_counter()
    sess = ort.InferenceSession(args.onnx, sess_options=so,
                                providers=["VitisAIExecutionProvider"],
                                provider_options=[po])
    compile_s = time.perf_counter() - t0
    if sess.get_providers()[0] != "VitisAIExecutionProvider":
        print("COMPILE_FAILED: VitisAI EP unavailable (silent CPU fallback "
              "is never accepted)", flush=True)
        return 1
    x = np.zeros((1, 3, geom, geom), dtype=np.float32)
    out = sess.run(None, {input_name: x})[0]
    if not np.isfinite(out).all():
        print("COMPILE_FAILED: non-finite probe output", flush=True)
        return 1
    rai_path = os.path.join(args.cache_dir, args.cache_key,
                            f"{args.cache_key}.rai")
    if not os.path.isfile(rai_path):
        print("COMPILE_FAILED: no .rai produced", flush=True)
        return 1
    digest = hashlib.sha256(open(rai_path, "rb").read()).hexdigest()
    print(f"COMPILE_OK {compile_s:.1f}s {digest} "
          f"{os.path.getsize(rai_path)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
