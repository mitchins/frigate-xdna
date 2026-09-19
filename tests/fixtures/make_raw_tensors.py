#!/usr/bin/env python3
"""Deterministic generator for golden raw-YOLO tensor fixtures.

Writes tests/fixtures/raw-yolo-{320,640}.npz with:
  raw:      float32 (1, 84, N) raw network output, rows 0-3 cxcywh pixels,
            rows 4-83 class scores (84 = 4 + 80).
  expected: float32 (20, 6) canonical [class, score, ymin, xmin, ymax, xmax]
            NMS result for score_threshold=0.25, iou=0.45, normalized coords.

Seed pinned (RNG 7); regenerate with: python3 tests/fixtures/make_raw_tensors.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESOLUTIONS = {320: 2100, 640: 8400}
CLASSES = 80
CONF = 0.25
IOU = 0.45


def nms(boxes_xyxy, scores, iou_thr, top_k=20):
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size and len(keep) < top_k:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(boxes_xyxy[i, 0], boxes_xyxy[order[1:], 0])
        yy1 = np.maximum(boxes_xyxy[i, 1], boxes_xyxy[order[1:], 1])
        xx2 = np.minimum(boxes_xyxy[i, 2], boxes_xyxy[order[1:], 2])
        yy2 = np.minimum(boxes_xyxy[i, 3], boxes_xyxy[order[1:], 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_i = (boxes_xyxy[i, 2] - boxes_xyxy[i, 0]) * (boxes_xyxy[i, 3] - boxes_xyxy[i, 1])
        area_j = ((boxes_xyxy[order[1:], 2] - boxes_xyxy[order[1:], 0])
                  * (boxes_xyxy[order[1:], 3] - boxes_xyxy[order[1:], 1]))
        union = area_i + area_j - inter + 1e-9
        order = order[1:][(inter / union) <= iou_thr]
    return keep


def build(n, res, seed=7):
    rng = np.random.default_rng(seed)
    raw = np.zeros((1, 4 + CLASSES, n), dtype=np.float32)
    # background: low uniform noise
    raw[0, 4:, :] = (rng.random((CLASSES, n), dtype=np.float32) * 0.05).astype(np.float32)
    # planted detections: (class, score, cx, cy, w, h) in pixels
    plants = [
        (5, 0.93, res * 0.50, res * 0.45, res * 0.95, res * 0.46),
        (5, 0.88, res * 0.51, res * 0.46, res * 0.93, res * 0.44),  # dup of above (NMS kills)
        (0, 0.81, res * 0.18, res * 0.60, res * 0.24, res * 0.44),
        (0, 0.62, res * 0.83, res * 0.58, res * 0.17, res * 0.44),
        (2, 0.41, res * 0.35, res * 0.70, res * 0.12, res * 0.10),
        (7, 0.20, res * 0.60, res * 0.20, res * 0.10, res * 0.10),  # below threshold
    ]
    for k, (c, s, cx, cy, w, h) in enumerate(plants):
        i = 10 + k * 7
        raw[0, 0, i] = cx
        raw[0, 1, i] = cy
        raw[0, 2, i] = w
        raw[0, 3, i] = h
        raw[0, 4 + c, i] = s
    # decode like the product decoder
    scores = raw[0, 4:, :].max(axis=0)
    cls = raw[0, 4:, :].argmax(axis=0)
    sel = np.where(scores > CONF)[0]
    cx, cy, w, h = (raw[0, k, sel] for k in range(4))
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    keep = nms(xyxy, scores[sel], IOU)
    out = np.zeros((20, 6), dtype=np.float32)
    for r, j in enumerate(keep):
        i = sel[j]
        x1, y1, x2, y2 = (float(v) / res for v in xyxy[j])
        out[r] = (float(cls[i]), float(scores[i]), y1, x1, y2, x2)
    return raw, out


def main() -> int:
    for res, n in RESOLUTIONS.items():
        raw, expected = build(n, res)
        path = os.path.join(HERE, f"raw-yolo-{res}.npz")
        np.savez(path, raw=raw, expected=expected)
        print(f"wrote {path} raw={raw.shape} expected detections="
              f"{int((expected[:, 1] > 0).sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
