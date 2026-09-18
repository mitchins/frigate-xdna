"""Unit tests: tensor validation (no renormalisation) + raw-YOLO decode."""
import os
import unittest

import numpy as np

from frigate_xdna.models.contracts import (
    RawYoloProfile,
    decode_yolo_raw,
    validate_tensor,
)

FIX = os.path.join(os.path.dirname(__file__), "..", "fixtures")
PROFILE = RawYoloProfile(layout="cxcywh", coordinate_units="pixels",
                         class_count=80, score_threshold=0.25, nms_iou=0.45)


class TestTensorValidation(unittest.TestCase):
    def test_float32_passthrough_unscaled(self):
        payload = (np.arange(1 * 3 * 4 * 4, dtype=np.float32) + 0.5).tobytes()
        arr = validate_tensor(payload, (1, 3, 4, 4), "float32")
        # exact passthrough: values must not be divided by 255 or otherwise
        # transformed (Frigate already normalized before the plugin).
        self.assertTrue(np.array_equal(
            arr, (np.arange(48, dtype=np.float32) + 0.5).reshape(1, 3, 4, 4)))

    def test_rejects_wrong_dtype(self):
        with self.assertRaises(ValueError):
            validate_tensor(b"\x00" * 48, (1, 3, 4, 4), "uint8")

    def test_rejects_byte_count_mismatch(self):
        with self.assertRaises(ValueError):
            validate_tensor(b"\x00" * 47, (1, 3, 4, 4), "float32")

    def test_rejects_nonfinite(self):
        bad = np.full((1, 3, 2, 2), np.inf, dtype=np.float32).tobytes()
        with self.assertRaises(ValueError):
            validate_tensor(bad, (1, 3, 2, 2), "float32")

    def test_rejects_bad_rank_and_batch(self):
        with self.assertRaises(ValueError):
            validate_tensor(b"\x00" * 16, (2, 2), "float32")


class TestDecodeYoloRaw(unittest.TestCase):
    def _case(self, res):
        d = np.load(os.path.join(FIX, f"raw-yolo-{res}.npz"))
        got = decode_yolo_raw(d["raw"][0], PROFILE, (float(res), float(res)))
        want = d["expected"]
        self.assertEqual(got.shape, (20, 6))
        self.assertTrue(np.allclose(got, want, atol=1e-5),
                        f"res={res}\n{got[got[:,1]>0]}\n{want[want[:,1]>0]}")

    def test_golden_320(self):
        self._case(320)

    def test_golden_640(self):
        self._case(640)

    def test_expected_top_detection(self):
        d = np.load(os.path.join(FIX, "raw-yolo-320.npz"))
        got = decode_yolo_raw(d["raw"][0], PROFILE, (320.0, 320.0))
        self.assertEqual(int(got[0, 0]), 5)
        self.assertAlmostEqual(float(got[0, 1]), 0.93, places=5)

    def test_rejects_unknown_contract(self):
        d = np.load(os.path.join(FIX, "raw-yolo-320.npz"))
        bad = RawYoloProfile(layout="xyxy", coordinate_units="pixels",
                             class_count=80)
        with self.assertRaises(ValueError):
            decode_yolo_raw(d["raw"][0], bad, (320.0, 320.0))

    def test_rejects_wrong_class_count(self):
        d = np.load(os.path.join(FIX, "raw-yolo-320.npz"))
        bad = RawYoloProfile(layout="cxcywh", coordinate_units="pixels",
                             class_count=7)
        with self.assertRaises(ValueError):
            decode_yolo_raw(d["raw"][0], bad, (320.0, 320.0))

    def test_private_label_ids_preserved(self):
        # sparse non-COCO ids must pass through untouched
        raw = np.zeros((4 + 7, 50), dtype=np.float32)
        raw[0, 3] = 100.0
        raw[1, 3] = 100.0
        raw[2, 3] = 40.0
        raw[3, 3] = 40.0
        raw[4 + 6, 3] = 0.9  # class id 6, not COCO
        prof = RawYoloProfile(layout="cxcywh", coordinate_units="pixels",
                              class_count=7)
        got = decode_yolo_raw(raw, prof, (320.0, 320.0))
        self.assertEqual(int(got[0, 0]), 6)

    def test_hand_computed_single_box(self):
        # Independent anchor: one box, hand-computed arithmetic, no fixture
        # generator involved. Box cx=160 cy=160 w=80 h=40 in a 320 image,
        # class 2 score 0.5 -> normalized [y1=0.4375, x1=0.375, y2=0.5625,
        # x2=0.625].
        raw = np.zeros((4 + 3, 4), dtype=np.float32)
        raw[0, 1] = 160.0
        raw[1, 1] = 160.0
        raw[2, 1] = 80.0
        raw[3, 1] = 40.0
        raw[4 + 2, 1] = 0.5
        prof = RawYoloProfile(layout="cxcywh", coordinate_units="pixels",
                              class_count=3)
        got = decode_yolo_raw(raw, prof, (320.0, 320.0))
        self.assertEqual(int((got[:, 1] > 0).sum()), 1)
        self.assertEqual(int(got[0, 0]), 2)
        self.assertAlmostEqual(float(got[0, 1]), 0.5, places=6)
        for v, want in zip(got[0, 2:6], (0.4375, 0.375, 0.5625, 0.625)):
            self.assertAlmostEqual(float(v), want, places=6)
        self.assertTrue((got[1:] == 0).all())


if __name__ == "__main__":
    unittest.main()
