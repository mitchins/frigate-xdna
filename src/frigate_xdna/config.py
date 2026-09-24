"""Configuration: environment parsing and validation (SPEC §5.1-5.2).

Stdlib only. No hardware, network, or vendor side effects on import.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from .errors import INVALID_ARGS, FxdnaError

DEFAULT_DATA_DIR_IMAGE = "/data"
DEFAULT_ENDPOINT_IMAGE = "tcp://0.0.0.0:5555"
DEFAULT_ENDPOINT_NATIVE = "tcp://127.0.0.1:5555"
DEFAULT_DEVICE = "/dev/accel/accel0"
DEFAULT_LOG_LEVEL = "info"

VALID_LOG_LEVELS = ("debug", "info", "warning", "error")


DEFAULT_MODEL_DIR_IMAGE = "/models"


@dataclass(frozen=True)
class Config:
    models: tuple[str, ...] = ()
    # The sole Plus credential input (container environment). Never
    # logged, never baked into images, never passed to child envs.
    plus_api_key: str | None = None
    # Local model directory: `local://ID` resolves to
    # `<model_dir>/ID.onnx`. Absolute; the appliance default is /models.
    model_dir: str = DEFAULT_MODEL_DIR_IMAGE
    data_dir: str = DEFAULT_DATA_DIR_IMAGE
    endpoint: str = DEFAULT_ENDPOINT_IMAGE
    device: str = DEFAULT_DEVICE
    log_level: str = DEFAULT_LOG_LEVEL
    offline: bool = False
    allow_uploads: bool = True

    def redacted(self) -> dict:
        """Log-safe view: secrets are never included."""
        return {
            "models": list(self.models),
            "plus_api_key_set": self.plus_api_key is not None,
            "model_dir": self.model_dir,
            "data_dir": self.data_dir,
            "endpoint": self.endpoint,
            "device": self.device,
            "log_level": self.log_level,
            "offline": self.offline,
            "allow_uploads": self.allow_uploads,
        }


def default_endpoint(in_docker: bool = True) -> str:
    """Default ZMQ endpoint: bind-all inside the image, loopback natively.

    The Task 02 `serve` implementation selects via its runtime context;
    `load_config` keeps the documented image default for explicitness.
    """
    return DEFAULT_ENDPOINT_IMAGE if in_docker else DEFAULT_ENDPOINT_NATIVE


def _parse_bool(value: str, name: str) -> bool:
    v = value.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise FxdnaError(INVALID_ARGS, "INVALID_CONFIG",
                     f"{name} must be a boolean, got {value!r}")


def _split_models(raw: str) -> tuple[str, ...]:
    parts: list[str] = []
    for chunk in raw.replace("\n", ",").split(","):
        ref = chunk.strip()
        if ref:
            parts.append(ref)
    return tuple(parts)


def load_config(env: dict[str, str] | None = None) -> Config:
    """Build validated Config from environment mapping (os.environ default)."""
    env = os.environ if env is None else env
    key = env.get("PLUS_API_KEY")
    log_level = env.get("FXDNA_LOG_LEVEL", DEFAULT_LOG_LEVEL).strip().lower()
    if log_level not in VALID_LOG_LEVELS:
        raise FxdnaError(INVALID_ARGS, "INVALID_CONFIG",
                         f"FXDNA_LOG_LEVEL must be one of {VALID_LOG_LEVELS}")
    endpoint = env.get("FXDNA_ENDPOINT", DEFAULT_ENDPOINT_IMAGE).strip()
    if "://" not in endpoint:
        raise FxdnaError(INVALID_ARGS, "INVALID_CONFIG",
                         f"FXDNA_ENDPOINT must be a URL, got {endpoint!r}")
    model_dir = env.get("FXDNA_MODEL_DIR",
                        DEFAULT_MODEL_DIR_IMAGE).strip()
    if not model_dir or not os.path.isabs(model_dir):
        raise FxdnaError(INVALID_ARGS, "INVALID_CONFIG",
                         "FXDNA_MODEL_DIR must be an absolute path, got"
                         f" {model_dir!r}")
    return Config(
        models=_split_models(env.get("FXDNA_MODELS", "")),
        plus_api_key=key,
        model_dir=model_dir,
        data_dir=env.get("FXDNA_DATA_DIR", DEFAULT_DATA_DIR_IMAGE).strip(),
        endpoint=endpoint,
        device=env.get("FXDNA_DEVICE", DEFAULT_DEVICE).strip(),
        log_level=log_level,
        offline=_parse_bool(env.get("FXDNA_OFFLINE", "false"), "FXDNA_OFFLINE"),
        allow_uploads=_parse_bool(env.get("FXDNA_ALLOW_UPLOADS", "true"),
                                  "FXDNA_ALLOW_UPLOADS"),
    )
