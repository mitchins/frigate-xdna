"""TEST SHIM — not upstream Frigate code.

Minimal stand-ins for Frigate-internal modules that the pinned
`detector_config.py` imports but that are NOT part of the pinned rc2
compatibility surface. Values mirror the real defaults only where the
contract tests depend on them; they must never be cited as upstream
behaviour. See tests/upstream/NOTICES.md.
"""

#: Mirrors frigate.const.DEFAULT_ATTRIBUTE_LABEL_MAP default (empty mapping).
DEFAULT_ATTRIBUTE_LABEL_MAP: dict = {}

#: Directory Frigate uses for downloaded Plus models (real default).
MODEL_CACHE_DIR = "/config/model_cache"
