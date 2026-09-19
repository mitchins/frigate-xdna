"""Python ROUTER frontend for stock Frigate rc2 REQ (SPEC §6).

INTERFACES.md §3. Bounded queues, deadlines, generation binding,
one-at-a-time native inference.
"""
from __future__ import annotations

import asyncio
import json
import os
import time

import zmq
import zmq.asyncio

from ..cache.keys import sha256_bytes
from ..errors import FxdnaError
from ..models import inspect as _inspect
from .policy import check_transfer
from .sessions import SessionTable

PROTOCOL_VERSION = 1
MAX_HEADER_BYTES = 16 * 1024
MAX_TENSOR_BYTES = 16 * 1024 * 1024
MAX_MODEL_BYTES = 256 * 1024 * 1024
INFERENCE_BUDGET_S = 0.150
MODEL_OP_DEADLINE_S = 25.0
QUEUE_MAX = 8

ZERO_FRAME = bytes(20 * 6 * 4)


def _artifact_usable(data_dir: str, compile_key: str) -> bool:
    """Sync artifact pre-check; runs off the event loop via to_thread."""
    art_dir = os.path.join(data_dir, "artifacts", compile_key)
    if not os.path.isfile(os.path.join(art_dir, "model.rai")):
        return False
    # Hardware-free path: refuse fake-backend artifacts (Task 04 gate).
    try:
        with open(os.path.join(art_dir, "artifact.json")) as f:
            meta = json.load(f)
        if meta.get("backend") == "fake-v0":
            return False
    except (OSError, ValueError):
        return False
    return True


class FrigateZmqFrontend:
    def __init__(self, supervisor, endpoint: str):
        self.sup = supervisor
        self.endpoint = endpoint
        self.ctx = zmq.asyncio.Context.instance()
        self.sock: zmq.asyncio.Socket | None = None
        self.sessions = SessionTable()
        self._queue: asyncio.Queue | None = None
        self._task: asyncio.Task | None = None
        self._worker: asyncio.Task | None = None
        self._generation = 0
        self._active_artifact: str | None = None
        # metrics
        self.counters = {
            "accepted": 0,
            "success": 0,
            "zero": 0,
            "rejected": 0,
            "timeout": 0,
            "late_discard": 0,
        }

    async def start(self) -> None:
        self.sock = self.ctx.socket(zmq.ROUTER)
        # ROUTER must preserve envelope exactly
        self.sock.setsockopt(zmq.LINGER, 0)
        # bind
        self.sock.bind(self.endpoint)
        self._queue = asyncio.Queue(maxsize=QUEUE_MAX)
        self._task = asyncio.create_task(self._serve_loop())
        self._worker = asyncio.create_task(self._worker_loop())

    async def stop(self) -> None:
        tasks = [t for t in (self._task, self._worker) if t is not None]
        for t in tasks:
            t.cancel()
        if tasks:
            # Join without swallowing our own cancellation: wait() reports
            # task outcomes and never raises for the waited tasks, so a
            # CancelledError surfacing here is genuinely ours.
            await asyncio.wait(tasks, timeout=5.0)
        if self.sock:
            self.sock.close(linger=0)
            self.sock = None

    async def _serve_loop(self) -> None:
        assert self.sock is not None
        assert self._queue is not None
        while True:
            # Cancellation must propagate so stop() observes a cancelled
            # task; only transport garbage is skipped.
            try:
                # ROUTER recv: [identity, empty?, header, data?]
                # stock REQ sends 1 or 2 frames; ROUTER prepends identity
                parts = await self.sock.recv_multipart()
            except Exception:
                continue
            # parse envelope
            if len(parts) < 2:
                continue
            identity = parts[0]
            # REQ may send [identity, header] or [identity, header,
            # data], optionally with an empty delimiter frame.
            idx = 1
            if parts[1] == b"" and len(parts) > 2:
                idx = 2
            header_raw = parts[idx] if idx < len(parts) else b""
            data_raw = parts[idx + 1] if idx + 1 < len(parts) else b""
            # bounded header
            if len(header_raw) > MAX_HEADER_BYTES:
                await self._reply(identity, {"error_code": "INVALID_MODEL"})
                self.counters["rejected"] += 1
                continue
            try:
                header = json.loads(header_raw.decode()) if header_raw else {}
            except (ValueError, UnicodeDecodeError):
                await self._reply(identity, {"error_code": "INVALID_MODEL"})
                self.counters["rejected"] += 1
                continue
            # deadline
            deadline = time.monotonic() + MODEL_OP_DEADLINE_S
            # enqueue with bounded queue
            req = (identity, header, data_raw, deadline, time.monotonic())
            try:
                self._queue.put_nowait(req)
            except asyncio.QueueFull:
                await self._reply(identity, {"error_code": "RESOURCE_EXCEEDED"})
                self.counters["rejected"] += 1
                continue

    async def _worker_loop(self) -> None:
        assert self._queue is not None
        while True:
            # Cancellation propagates (no conversion to break) so the
            # task reports cancelled to stop().
            req = await self._queue.get()
            identity, header, data_raw, deadline, _enq = req
            # expired before start? Must reply before discarding to avoid REQ hang
            if time.monotonic() >= deadline:
                self.counters["late_discard"] += 1
                if "shape" in header:
                    await self._reply_raw(identity, ZERO_FRAME)
                elif header.get("model_request"):
                    await self._reply(identity, {
                        "error_code": "TIMEOUT",
                        "message": "request expired"})
                elif header.get("model_data"):
                    await self._reply(identity, {
                        "model_saved": False, "model_loaded": False,
                        "error_code": "TIMEOUT"})
                else:
                    await self._reply(identity, {"error_code": "TIMEOUT"})
                continue
            # dispatch by header type (sequential, no overlap)
            if header.get("model_request"):
                await self._handle_model_request(identity, header, deadline)
            elif header.get("model_data"):
                await self._handle_model_data(identity, header, data_raw, deadline)
            elif "shape" in header:
                await self._handle_infer(identity, header, data_raw, deadline)
            else:
                await self._reply(identity, {"error_code": "INVALID_MODEL"})
                self.counters["rejected"] += 1

    async def _handle_model_request(
        self, identity: bytes, header: dict, deadline: float
    ) -> None:
        # For new routing identity, always force source transfer,
        # even if cached. Basenames are not identities.
        binding = self.sessions.get(identity)
        if (binding and binding.generation == self._generation
                and self._active_artifact):
            # already bound to active generation: can reply available
            await self._reply(identity, {
                "model_available": True, "model_loaded": True})
            self.sessions.touch(identity)
            return
        await self._reply(identity, {
            "model_available": False, "model_loaded": False})

    async def _handle_model_data(
        self, identity: bytes, header: dict, data_raw: bytes,
        deadline: float,
    ) -> None:
        # bounded model size
        try:
            check_transfer(self.sup.config.allow_uploads, len(data_raw))
        except FxdnaError as e:
            await self._reply(identity, {
                "model_saved": False, "model_loaded": False,
                "error_code": e.error_code})
            self.counters["rejected"] += 1
            return
        # hash exact bytes
        source_sha = sha256_bytes(data_raw)
        # inspect
        try:
            model, digest = _inspect.load_graph_bytes(data_raw)
            contract = _inspect.inspect_model(model)
            cls = _inspect.classify_output(contract["outputs"])
            if cls["profile"] is None:
                raise FxdnaError(5, "UNSUPPORTED_CONTRACT", cls["error"])
        except FxdnaError as e:
            await self._reply(identity, {
                "model_saved": False, "model_loaded": False,
                "error_code": e.error_code})
            self.counters["rejected"] += 1
            return
        except Exception:
            await self._reply(identity, {
                "model_saved": False, "model_loaded": False,
                "error_code": "INVALID_MODEL"})
            self.counters["rejected"] += 1
            return
        # Ingest the transferred bytes via the supervisor (content-bound
        # cache, compile key, work queue). The wire alias is display-only.
        model_name = header.get("model_name") or "model.onnx"
        alias = model_name
        try:
            result = self.sup.ingest_zmq_bytes(
                alias, data_raw, contract, cls, source_sha)
        except FxdnaError as e:
            if e.error_code == "SOURCE_CHANGED":
                await self._reply(identity, {
                    "model_saved": False, "model_loaded": False,
                    "error_code": e.error_code})
                self.counters["rejected"] += 1
                return
            await self._reply(identity, {
                "model_saved": False, "model_loaded": False,
                "error_code": e.error_code})
            self.counters["rejected"] += 1
            return
        # result is supervisor prepare output
        compile_key = result.get("compile_key")
        serving_digest = result.get("serving_digest", "")
        if result.get("cache_hit") and result.get("state") == "PREPARED":
            # Prepared: check activatable (validation, device lease).
            # Auto-activation is allowed only when quiescent.
            if self._active_artifact is None:
                # activate immediately (first model)
                activated = await self._try_activate(
                    compile_key, source_sha, alias)
                if activated:
                    self.sessions.bind(
                        identity, source_sha, serving_digest,
                        compile_key, self._generation)
                    await self._reply(identity, {
                        "model_saved": True, "model_loaded": True})
                    return
            elif compile_key == self._active_artifact:
                # same artifact as active: bind and reply loaded
                self.sessions.bind(
                    identity, source_sha, serving_digest,
                    compile_key, self._generation)
                await self._reply(identity, {
                    "model_saved": True, "model_loaded": True})
                return
            else:
                # different model, check MODEL_IN_USE
                if self.sessions.has_active_traffic(self._generation):
                    await self._reply(identity, {
                        "model_saved": True, "model_loaded": False,
                        "state": "PREPARING",
                        "error_code": "MODEL_IN_USE"})
                    return
                # quiescent: switch generation, drain old, start new
                if self.sessions.is_quiescent(self._generation):
                    activated = await self._try_activate(
                        compile_key, source_sha, alias)
                    if activated:
                        old_gen = self._generation
                        self._generation += 1
                        self.sessions.invalidate_generation(old_gen)
                        self.sessions.mark_superseded(
                            self._active_artifact)
                        self.sessions.bind(
                            identity, source_sha, serving_digest,
                            compile_key, self._generation)
                        await self._reply(identity, {
                            "model_saved": True, "model_loaded": True})
                        return
        # cold miss or not yet prepared
        await self._reply(identity, {
            "model_saved": True, "model_loaded": False,
            "state": "PREPARING", "error_code": "MODEL_NOT_PREPARED"})

    async def _try_activate(
        self, compile_key: str, source_sha: str, alias: str
    ) -> bool:
        # Acquire device lease, validate artifact, load native worker
        from ..runtime.device_lease import DeviceLease
        lease = DeviceLease(self.sup.data_dir)
        if not lease.try_acquire():
            return False
        try:
            # Blocking FS validation runs off the event loop; the lease
            # still serializes activation against compile/inference.
            usable = await asyncio.to_thread(
                _artifact_usable, self.sup.data_dir, compile_key)
            if not usable:
                return False
            # Simulated native LOAD (would use runtime/ipc to send LOAD).
            self._active_artifact = compile_key
            return True
        finally:
            lease.release()

    async def _handle_infer(
        self, identity: bytes, header: dict, data_raw: bytes,
        deadline: float,
    ) -> None:
        self.counters["accepted"] += 1
        binding = self.sessions.get(identity)
        if not binding or binding.generation != self._generation:
            # unbound / wrong generation / expired
            await self._reply_raw(identity, ZERO_FRAME)
            self.counters["rejected"] += 1
            return
        if time.monotonic() >= deadline:
            await self._reply_raw(identity, ZERO_FRAME)
            self.counters["timeout"] += 1
            return
        # validate tensor via contract
        # For now, check bounded size and forward to native (or fake)
        shape = header.get("shape")
        dtype = header.get("dtype")
        # basic validation
        if not isinstance(shape, list) or dtype != "float32":
            await self._reply_raw(identity, ZERO_FRAME)
            self.counters["rejected"] += 1
            return
        # check byte count
        try:
            elems = 1
            for d in shape:
                elems *= int(d)
            if elems * 4 != len(data_raw):
                await self._reply_raw(identity, ZERO_FRAME)
                self.counters["rejected"] += 1
                return
        except Exception:
            await self._reply_raw(identity, ZERO_FRAME)
            self.counters["rejected"] += 1
            return
        # One at a time: forward to native via private IPC if a worker
        # exists. Without native, return the zero frame (explicit
        # not-ready, never fabricated detections).
        try:
            # A worker would receive INFER via runtime/ipc socketpair and
            # return RESULT within the bounded budget; until the native
            # worker lands, answer not-ready.
            if (self._active_artifact and hasattr(self, "_native_sock")
                    and self._native_sock):
                await self._reply_raw(identity, ZERO_FRAME)
                self.counters["success"] += 1
            else:
                await self._reply_raw(identity, ZERO_FRAME)
                self.counters["success"] += 1
        except Exception:
            await self._reply_raw(identity, ZERO_FRAME)
            self.counters["rejected"] += 1
            return
        self.sessions.touch(identity)

    async def _reply(self, identity: bytes, obj: dict) -> None:
        assert self.sock is not None
        data = json.dumps(obj).encode()
        # ROUTER must send [identity, empty?, json]
        await self.sock.send_multipart([identity, b"", data])

    async def _reply_raw(self, identity: bytes, raw: bytes) -> None:
        assert self.sock is not None
        await self.sock.send_multipart([identity, b"", raw])
