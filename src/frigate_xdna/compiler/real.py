"""Real audited compiler backend (bf16-vaiml-v1) for the job manager.

Same lifecycle interface as the fake backend: the JobManager drives it
with poll(); terminal stages are PREPARED / COMPILE_FAILED (and friends).
The multi-minute subprocess runs on a worker thread so pumping never
blocks; one real compile at a time (the launcher lock enforces it, extra
jobs stay QUEUED).

Production preparation uses this backend. The fake backend remains for
hardware-free tests. Backend identity flows into artifact manifests and
the compile-key trust check (supervisor COMPILER_BACKEND).
"""
from __future__ import annotations

import threading
import time

from .launcher import CompileResult, CompilerPrefixes, run_compile

BACKEND_ID = "bf16-vaiml-v1"


class RealCompileJob:
    TERMINAL_OK = "PREPARED"

    def __init__(self, source_sha256: str, compile_key: str,
                 source_path: str, workdir: str, prefixes: CompilerPrefixes,
                 timeout_s: float = 2700.0, data_dir: str | None = None,
                 worker_factory=None, **ignored):
        self.source_sha256 = source_sha256
        self.compile_key = compile_key
        self._source_path = source_path
        self._workdir = workdir
        self._prefixes = prefixes
        self._timeout_s = timeout_s
        self.state = "QUEUED"
        self.elapsed_s = 0.0
        self.result: CompileResult | None = None
        self.log: list[str] = []
        self._thread: threading.Thread | None = None
        self._t0 = 0.0
        self._data_dir = data_dir
        self._worker_factory = worker_factory

    def _run(self):
        self.result = run_compile(
            self._prefixes, self._source_path, self._workdir,
            self.compile_key, timeout_s=self._timeout_s,
            data_dir=self._data_dir,
            worker_factory=self._worker_factory)

    def poll(self, dt_s: float) -> str:
        if self.state in ("PREPARED", "COMPILE_FAILED", "RESOURCE_EXCEEDED",
                          "VALIDATION_FAILED", "INTERRUPTED"):
            return self.state
        if self._thread is None:
            self._t0 = time.monotonic()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            self.state = "COMPILING"
            self.log.append("compiling")
            return self.state
        self.elapsed_s = time.monotonic() - self._t0
        if self._thread.is_alive():
            self.state = "COMPILING"
            return self.state
        result = self.result
        if result is None or result.returncode == 124:
            self.state = "COMPILE_FAILED"
            self.log.append("failed:timeout-or-crash")
        elif result.returncode == 98:
            # launcher busy (should not happen: one real job at a time,
            # extra jobs stay QUEUED at the manager). Back off honestly.
            self.state = "QUEUED"
            self._thread = None
            self.log.append("launcher-busy-retry")
        elif result.returncode != 0:
            self.state = "COMPILE_FAILED"
            self.log.append(f"failed:rc={result.returncode}:{result.error}")
        else:
            self.state = "PREPARED"
            self.log.append("prepared")
        return self.state
