"""Contract tests: Frigate normalizes float input before the plugin.

Source assertions on the pinned `base.py.src` (blob bc7910e4...) pin the
upstream behaviour; behavioural tests assert our tensor gate passes float32
through unscaled. The sidecar must never divide by 255 again.
"""
import os
import unittest

import numpy as np

from frigate_xdna.models.contracts import validate_tensor

UPSTREAM = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "upstream")


class TestNormalizationBoundary(unittest.TestCase):
    def test_pinned_base_normalizes_float(self):
        src = open(os.path.join(UPSTREAM, "base.py.src")).read()
        # BaseLocalDetector._transform_input: transpose, float32 cast, /255
        self.assertIn("tensor_input /= 255", src)
        self.assertIn("tensor_input.astype(np.float32)", src)
        self.assertIn("np.transpose(tensor_input", src)

    def test_sidecar_does_not_renormalize(self):
        # A Frigate-normalized 0..1 tensor must survive validation bit-exact.
        rng = np.random.default_rng(11)
        tensor = rng.random((1, 3, 32, 32), dtype=np.float32)
        out = validate_tensor(tensor.tobytes(), (1, 3, 32, 32), "float32")
        self.assertTrue(np.array_equal(out, np.ascontiguousarray(tensor)))

    def test_sidecar_rejects_unconverted_uint8(self):
        raw = bytes(range(1 * 3 * 4 * 4))
        with self.assertRaises(ValueError):
            validate_tensor(raw, (1, 3, 4, 4), "uint8")


if __name__ == "__main__":
    unittest.main()
