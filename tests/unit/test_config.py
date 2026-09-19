"""Unit tests: configuration parsing/validation (SPEC §5.1)."""
import unittest

from frigate_xdna.config import default_endpoint, load_config
from frigate_xdna.errors import FxdnaError


class TestConfig(unittest.TestCase):
    def test_defaults(self):
        c = load_config({})
        self.assertEqual(c.models, ())
        self.assertEqual(c.data_dir, "/data")
        self.assertEqual(c.endpoint, "tcp://0.0.0.0:5555")
        self.assertEqual(c.device, "/dev/accel/accel0")
        self.assertEqual(c.log_level, "info")
        self.assertFalse(c.offline)
        self.assertTrue(c.allow_uploads)

    def test_models_split_commas_and_newlines(self):
        c = load_config({"FXDNA_MODELS": "plus://A, plus://B\nplus://C ,, "})
        self.assertEqual(c.models, ("plus://A", "plus://B", "plus://C"))

    def test_secret_sources_mutually_exclusive(self):
        with self.assertRaises(FxdnaError) as ctx:
            load_config({"PLUS_API_KEY": "x", "PLUS_API_KEY_FILE": "/run/k"})
        self.assertEqual(ctx.exception.exit_code, 2)

    def test_secret_never_in_redacted_view(self):
        c = load_config({"PLUS_API_KEY": "supersecret"})
        view = c.redacted()
        self.assertNotIn("supersecret", repr(view))
        self.assertTrue(view["plus_api_key_set"])

    def test_invalid_log_level(self):
        with self.assertRaises(FxdnaError):
            load_config({"FXDNA_LOG_LEVEL": "verbose"})

    def test_invalid_endpoint(self):
        with self.assertRaises(FxdnaError):
            load_config({"FXDNA_ENDPOINT": "not-a-url"})

    def test_bool_parsing(self):
        self.assertTrue(load_config({"FXDNA_OFFLINE": "true"}).offline)
        self.assertFalse(load_config({"FXDNA_ALLOW_UPLOADS": "0"}).allow_uploads)
        with self.assertRaises(FxdnaError):
            load_config({"FXDNA_OFFLINE": "maybe"})

    def test_default_endpoint_context(self):
        self.assertEqual(default_endpoint(True), "tcp://0.0.0.0:5555")
        self.assertEqual(default_endpoint(False), "tcp://127.0.0.1:5555")


if __name__ == "__main__":
    unittest.main()
