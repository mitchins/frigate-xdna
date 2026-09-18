"""Unit tests: Plus metadata comparison strictness."""
import unittest

from frigate_xdna.errors import FxdnaError
from frigate_xdna.models.inspect import compare_plus_metadata

INSPECTED = {"input_shape": [1, 3, 320, 320]}


class TestComparePlusMetadata(unittest.TestCase):
    def test_matching_geometry_passes(self):
        compare_plus_metadata({"width": 320, "height": 320}, INSPECTED)
        compare_plus_metadata({"input_shape": [1, 3, 320, 320]}, INSPECTED)

    def test_absent_geometry_skipped(self):
        compare_plus_metadata({"id": "X"}, INSPECTED)  # must not raise

    def test_conflict_fails(self):
        with self.assertRaises(FxdnaError) as ctx:
            compare_plus_metadata({"width": 640, "height": 640}, INSPECTED)
        self.assertEqual(ctx.exception.error_code, "UNSUPPORTED_CONTRACT")

    def test_present_but_invalid_fails(self):
        for bad in ({"width": True, "height": 320},
                    {"width": 320, "height": None},
                    {"width": 320.0, "height": 320},
                    {"width": "320", "height": 320},
                    {"width": 320},
                    {"input_shape": [1, 3, 320, "320"]},
                    {"input_shape": [1, 3, 320]},
                    {"input_shape": None}):
            with self.subTest(bad=bad):
                with self.assertRaises(FxdnaError) as ctx:
                    compare_plus_metadata(bad, INSPECTED)
                self.assertEqual(ctx.exception.error_code,
                                 "UNSUPPORTED_CONTRACT")


if __name__ == "__main__":
    unittest.main()
