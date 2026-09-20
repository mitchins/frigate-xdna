"""Model binding per ROUTER routing identity + generation (SPEC §6).

Each ROUTER identity is bound to source_sha256 + serving_digest +
worker_generation. No cross-generation inference.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

QUIESCENCE_S = 5.0


@dataclass
class Binding:
    identity: bytes
    source_sha256: str
    serving_digest: str
    compile_key: str
    generation: int
    created_at: float = field(default_factory=time.monotonic)
    last_used_at: float = field(default_factory=time.monotonic)


class SessionTable:
    def __init__(self):
        self._by_identity: dict[bytes, Binding] = {}
        # track superseded models ineligible for auto-reactivation
        self.superseded: set[str] = set()
        self.active_generation: int = 0
        self.active_compile_key: str | None = None
        self.active_source_sha256: str | None = None
        self.active_serving_digest: str | None = None

    def bind(
        self,
        identity: bytes,
        source_sha256: str,
        serving_digest: str,
        compile_key: str,
        generation: int,
    ) -> Binding:
        b = Binding(
            identity=identity,
            source_sha256=source_sha256,
            serving_digest=serving_digest,
            compile_key=compile_key,
            generation=generation,
        )
        self._by_identity[identity] = b
        return b

    def get(self, identity: bytes) -> Binding | None:
        return self._by_identity.get(identity)

    def count(self) -> int:
        """Number of bound identities (observability)."""
        return len(self._by_identity)

    def touch(self, identity: bytes) -> None:
        b = self._by_identity.get(identity)
        if b:
            b.last_used_at = time.monotonic()

    def invalidate_generation(self, generation: int) -> None:
        """Invalidate all bindings of an old generation (A->B switch)."""
        to_del = [k for k, v in self._by_identity.items() if v.generation == generation]
        for k in to_del:
            del self._by_identity[k]

    def is_quiescent(self, generation: int) -> bool:
        """Old generation quiescent: no binding of that generation used
        within QUIESCENCE_S."""
        now = time.monotonic()
        for b in self._by_identity.values():
            if b.generation == generation and (now - b.last_used_at) < QUIESCENCE_S:
                return False
        return True

    def has_active_traffic(self, generation: int) -> bool:
        """Any binding of that generation still active (inverse of quiescent)."""
        return not self.is_quiescent(generation)

    def mark_superseded(self, compile_key: str) -> None:
        self.superseded.add(compile_key)

    def is_superseded(self, compile_key: str) -> bool:
        return compile_key in self.superseded

    def clear_superseded(self, compile_key: str) -> None:
        self.superseded.discard(compile_key)

    def set_active(
        self,
        compile_key: str,
        source_sha256: str,
        serving_digest: str,
        generation: int,
    ) -> None:
        self.active_compile_key = compile_key
        self.active_source_sha256 = source_sha256
        self.active_serving_digest = serving_digest
        self.active_generation = generation
