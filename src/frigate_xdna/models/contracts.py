"""Model tensor validation and raw-YOLO decoder (SPEC §7).

The sidecar MUST NOT normalize again or resize: Frigate delivers float32
tensors that are already normalized. Validation is explicit; the only
permitted transformation is C-order float32 passthrough. Decoder parameters
come from the serving contract, never from globals.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RawYoloProfile:
    """Tested output contract for raw decoded YOLO boxes/classes."""
    layout: str  # "cxcywh" or "xyxy"
    coordinate_units: str  # "pixels" or "normalized"
    class_count: int
    objectness: bool = False
    score_threshold: float = 0.25
    nms_iou: float = 0.45
    max_detections: int = 20


def validate_tensor(payload: bytes, shape: tuple[int, ...],
                    dtype_name: str) -> np.ndarray:
    """Validate an inference tensor with overflow-safe byte accounting.

    Returns a C-order float32 view/copy. Raises ValueError on any contract
    violation. Never rescales values.
    """
    if dtype_name != "float32":
        raise ValueError(f"unsupported dtype {dtype_name!r}: only the "
                         f"serving contract's float32 input is accepted")
    if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
        raise ValueError(f"unsupported tensor shape {shape!r}: batch-one "
                         f"three-channel required")
    n = 1
    for dim in shape:
        if not isinstance(dim, int) or dim <= 0 or dim > 4096:
            raise ValueError(f"invalid tensor dimension {dim!r}")
        n *= dim
        if n * 4 > 16 * 1024 * 1024:
            raise ValueError("tensor exceeds 16 MiB bound")
    want = n * 4
    if len(payload) != want:
        raise ValueError(f"byte count {len(payload)} != shape product {want}")
    arr = np.frombuffer(payload, dtype=np.float32).reshape(shape)
    if not np.isfinite(arr).all():
        raise ValueError("tensor contains NaN/Inf")
    return np.ascontiguousarray(arr)


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    xx1 = np.maximum(a[0], b[:, 0])
    yy1 = np.maximum(a[1], b[:, 1])
    xx2 = np.minimum(a[2], b[:, 2])
    yy2 = np.minimum(a[3], b[:, 3])
    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a + area_b - inter + 1e-9)


def decode_yolo_raw(raw: np.ndarray, profile: RawYoloProfile,
                    image_wh: tuple[float, float]) -> np.ndarray:
    """Decode raw (84, N) network output to canonical [K,6] detections.

    Rows: [class_id, score, ymin, xmin, ymax, xmax], normalized coords,
    descending score, zero-padded to max_detections. Class-aware NMS keeps
    different classes. Raises ValueError on unknown/ambiguous contracts.
    """
    if profile.layout != "cxcywh" or profile.coordinate_units != "pixels":
        raise ValueError(f"unsupported raw contract {profile.layout}/"
                         f"{profile.coordinate_units}")
    if profile.objectness:
        raise ValueError("objectness profiles need an explicit contract")
    if raw.ndim != 2 or raw.shape[0] != 4 + profile.class_count:
        raise ValueError(f"raw shape {raw.shape} incompatible with "
                         f"class_count={profile.class_count}")
    if not np.isfinite(raw).all():
        raise ValueError("raw output contains NaN/Inf")
    w, h = image_wh
    scores = raw[4:, :].max(axis=0)
    classes = raw[4:, :].argmax(axis=0).astype(np.int64)
    sel = np.where(scores > profile.score_threshold)[0]
    cx, cy, bw, bh = (raw[k, sel].astype(np.float64) for k in range(4))
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2],
                     axis=1)
    order = np.argsort(-scores[sel], kind="stable")
    kept: list[int] = []
    # class-aware NMS: suppress only same-class overlap
    while order.size and len(kept) < profile.max_detections:
        i = int(order[0])
        kept.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        same = classes[sel[rest]] == classes[sel[i]]
        ious = _iou_xyxy(boxes[i], boxes[rest])
        order = rest[~(same & (ious > profile.nms_iou))]
    out = np.zeros((profile.max_detections, 6), dtype=np.float32)
    for r, j in enumerate(kept):
        i = int(sel[j])
        x1, y1, x2, y2 = (float(v) for v in boxes[j])
        out[r] = (float(classes[i]), float(scores[i]),
                  y1 / h, x1 / w, y2 / h, x2 / w)
    # canonical descending-score order (stable for ties)
    det = out[:len(kept)]
    det = det[np.argsort(-det[:, 1], kind="stable")]
    out[:len(kept)] = det
    return out
