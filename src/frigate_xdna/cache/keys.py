"""Content identity keys (docs/CACHE.md §2, normative).

All digests are SHA-256 over canonical JSON (UTF-8, sorted keys, stable
scalar encoding). Never substitute basenames, timestamps, host BDF, boot
IDs, or image versions for content keys.
"""
from __future__ import annotations

import hashlib
import json


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(obj: dict) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def compile_key(source_sha256: str, compiler_payload_sha256: str,
                recipe_id: str, recipe_config_sha256: str,
                target_profile: str, artifact_compatibility_id: str,
                compile_input: dict) -> str:
    return sha256_bytes(canonical({
        "schema": 1,
        "source_sha256": source_sha256,
        "compiler_payload_sha256": compiler_payload_sha256,
        "recipe_id": recipe_id,
        "recipe_config_sha256": recipe_config_sha256,
        "target_profile": target_profile,
        "artifact_compatibility_id": artifact_compatibility_id,
        "compile_input": compile_input,
    }))


def serving_digest(compile_key_hex: str, contract: dict) -> str:
    """Serving identity: compile key + input semantics + decoder + labels."""
    return sha256_bytes(canonical({
        "schema": 1,
        "compile_key": compile_key_hex,
        "contract": contract,
    }))


def validation_key(artifact_sha256: str, serving_hex: str,
                   runtime_profile: dict, suite_version: str) -> str:
    return sha256_bytes(canonical({
        "schema": 1,
        "artifact_sha256": artifact_sha256,
        "serving_digest": serving_hex,
        "runtime_profile": runtime_profile,
        "suite_version": suite_version,
    }))
