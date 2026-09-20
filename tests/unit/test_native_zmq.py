"""Hardware-free tests for the ZMQ ROUTER frontend (SPEC §6).

Loopback only. Exercises envelope parsing, deadlines, generation
binding and activation gating with a stub supervisor — no native
worker, no NPU, no Frigate.
"""
import asyncio
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import zmq
import zmq.asyncio

from frigate_xdna.transport import frigate_zmq as fz
from frigate_xdna.transport.frigate_zmq import ZERO_FRAME, FrigateZmqFrontend


def _sup(tmp, worker="no_worker", activate=True):
    sup = SimpleNamespace(
        data_dir=tmp,
        config=SimpleNamespace(allow_uploads=True),
    )
    # Worker seam: ("ok", 480B) | ("no_worker", b"") | ("failed", b"").
    payload = bytes(480) if worker == "ok" else b""
    sup.worker_infer = (  # type: ignore[method-assign]
        lambda data, shape, timeout: (worker, payload))
    sup.activate_worker = (  # type: ignore[method-assign]
        lambda ck: activate)
    return sup


class FrontendCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fe = FrigateZmqFrontend(_sup(self.tmp.name),
                                     "tcp://127.0.0.1:*")
        await self.fe.start()
        ep = self.fe.sock.getsockopt(zmq.LAST_ENDPOINT).decode()
        self.ctx = zmq.asyncio.Context()
        self.cli = self.ctx.socket(zmq.REQ)
        self.cli.setsockopt(zmq.RCVTIMEO, 3000)
        self.cli.setsockopt(zmq.LINGER, 0)
        self.cli.connect(ep)

    async def asyncTearDown(self):
        self.cli.close()
        self.ctx.term()
        await self.fe.stop()
        self.tmp.cleanup()

    async def _req(self, header, data=b""):
        frames = [json.dumps(header).encode()]
        if data:
            frames.append(data)
        await self.cli.send_multipart(frames)
        return await self.cli.recv_multipart()

    def test_endpoint_bound(self):
        self.assertIsNotNone(self.fe.sock)
        self.assertIsNotNone(self.fe._task)
        self.assertIsNotNone(self.fe._worker)

    async def test_infer_unbound_returns_zero(self):
        shape = [1, 3, 2, 2]
        payload = b"\x00" * (1 * 3 * 2 * 2 * 4)
        parts = await self._req(
            {"shape": shape, "dtype": "float32"}, payload)
        self.assertEqual(parts[-1], ZERO_FRAME)
        self.assertGreaterEqual(self.fe.counters["rejected"], 1)

    async def test_model_request_unbound_forces_transfer(self):
        parts = await self._req({"model_request": {"name": "m.onnx"}})
        obj = json.loads(parts[-1].decode())
        self.assertFalse(obj["model_available"])
        self.assertFalse(obj["model_loaded"])

    async def test_oversize_header_rejected(self):
        big = "x" * (fz.MAX_HEADER_BYTES + 1)
        await self.cli.send_multipart([big.encode()])
        parts = await self.cli.recv_multipart()
        obj = json.loads(parts[-1].decode())
        self.assertEqual(obj["error_code"], "INVALID_MODEL")

    async def test_expired_request_replies_before_discard(self):
        old = fz.MODEL_OP_DEADLINE_S
        fz.MODEL_OP_DEADLINE_S = -1.0
        try:
            parts = await self._req({"model_request": {"name": "m.onnx"}})
        finally:
            fz.MODEL_OP_DEADLINE_S = old
        obj = json.loads(parts[-1].decode())
        self.assertEqual(obj["error_code"], "TIMEOUT")
        self.assertGreaterEqual(self.fe.counters["late_discard"], 1)


class DispatchCase(unittest.IsolatedAsyncioTestCase):
    """Direct dispatch tests with stub identities (ROUTER drops sends
    to unknown peers, so only counters/state are asserted)."""

    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fe = FrigateZmqFrontend(_sup(self.tmp.name),
                                     "tcp://127.0.0.1:*")
        await self.fe.start()
        self.ingest_result: dict | None = None
        sup = self.fe.sup
        sup.ingest_zmq_bytes = (  # type: ignore[method-assign]
            lambda alias, data, contract, cls, sha: self.ingest_result)

    async def asyncTearDown(self):
        await self.fe.stop()
        self.tmp.cleanup()

    async def test_generation_binding(self):
        ident = b"ghost"
        shape = [1, 3, 2, 2]
        payload = b"\x00" * (1 * 3 * 2 * 2 * 4)
        dl = asyncio.get_event_loop().time() + 25.0
        # unbound -> rejected
        await self.fe._handle_infer(
            ident, {"shape": shape, "dtype": "float32"}, payload, dl)
        self.assertEqual(self.fe.counters["rejected"], 1)
        # bound to current generation, active artifact unset -> the
        # not-ready path still answers (zero frame) and touches the
        # binding, counted as zero (not success, not rejected)
        self.fe.sessions.bind(ident, "sha", "srv", "ck", 0)
        before = self.fe.sessions.get(ident).last_used_at
        await self.fe._handle_infer(
            ident, {"shape": shape, "dtype": "float32"}, payload, dl)
        self.assertEqual(self.fe.counters["zero"], 1)
        self.assertEqual(self.fe.counters["success"], 0)
        self.assertGreaterEqual(
            self.fe.sessions.get(ident).last_used_at, before)
        # stale generation -> rejected, no touch
        self.fe._generation = 1
        await self.fe._handle_infer(
            ident, {"shape": shape, "dtype": "float32"}, payload, dl)
        self.assertEqual(self.fe.counters["rejected"], 2)

    async def test_infer_validation(self):
        ident = b"ghost2"
        self.fe.sessions.bind(ident, "sha", "srv", "ck", 0)
        dl = asyncio.get_event_loop().time() + 25.0
        await self.fe._handle_infer(
            ident, {"shape": [1, 3], "dtype": "float32"}, b"zz", dl)
        self.assertEqual(self.fe.counters["rejected"], 1)
        await self.fe._handle_infer(
            ident, {"shape": [1, 3, 2, 2], "dtype": "float32"}, b"short",
            dl)
        self.assertEqual(self.fe.counters["rejected"], 2)

    async def test_try_activate_gating(self):
        art = os.path.join(self.tmp.name, "artifacts")
        # missing artifact -> False
        self.assertFalse(
            await self.fe._try_activate("ck-missing"))
        # fake backend -> False
        os.makedirs(os.path.join(art, "ck-fake"))
        with open(os.path.join(art, "ck-fake", "model.rai"), "wb") as f:
            f.write(b"FAKE")
        with open(os.path.join(art, "ck-fake", "artifact.json"),
                  "w") as f:
            json.dump({"backend": "fake-v0"}, f)
        self.assertFalse(
            await self.fe._try_activate("ck-fake"))
        self.assertIsNone(self.fe._active_artifact)
        # non-fake artifact -> True and recorded
        os.makedirs(os.path.join(art, "ck-real"))
        with open(os.path.join(art, "ck-real", "model.rai"), "wb") as f:
            f.write(b"REAL")
        with open(os.path.join(art, "ck-real", "artifact.json"),
                  "w") as f:
            json.dump({"backend": "bf16-vaiml-v1"}, f)
        self.assertTrue(
            await self.fe._try_activate("ck-real"))
        self.assertEqual(self.fe._active_artifact, "ck-real")

    async def test_stop_joins_tasks(self):
        await self.fe.stop()
        self.assertTrue(self.fe._task.done())
        self.assertTrue(self.fe._worker.done())
        self.assertIsNone(self.fe.sock)

    async def test_infer_forwards_to_worker(self):
        self.fe.sup.worker_infer = lambda d, s, t: ("ok", bytes(480))
        ident = b"w-ok"
        shape = [1, 3, 2, 2]
        payload = bytes(1 * 3 * 2 * 2 * 4)
        self.fe.sessions.bind(ident, "sha", "srv", "ck", 0)
        dl = asyncio.get_event_loop().time() + 25.0
        await self.fe._handle_infer(
            ident, {"shape": shape, "dtype": "float32"}, payload, dl)
        self.assertEqual(self.fe.counters["success"], 1)
        self.assertEqual(self.fe.counters["zero"], 0)

    async def test_infer_worker_failure_rejects(self):
        self.fe.sup.worker_infer = lambda d, s, t: ("failed", b"")
        ident = b"w-bad"
        shape = [1, 3, 2, 2]
        payload = bytes(1 * 3 * 2 * 2 * 4)
        self.fe.sessions.bind(ident, "sha", "srv", "ck", 0)
        dl = asyncio.get_event_loop().time() + 25.0
        await self.fe._handle_infer(
            ident, {"shape": shape, "dtype": "float32"}, payload, dl)
        self.assertEqual(self.fe.counters["rejected"], 1)
        self.assertEqual(self.fe.counters["success"], 0)

    async def test_refused_artifact_activates_nothing(self):
        art = os.path.join(self.tmp.name, "artifacts")
        os.makedirs(os.path.join(art, "ck-ref"), exist_ok=True)
        with open(os.path.join(art, "ck-ref", "model.rai"), "wb") as f:
            f.write(b"REAL")
        with open(os.path.join(art, "ck-ref", "artifact.json"),
                  "w") as f:
            json.dump({"backend": "bf16-vaiml-v1"}, f)
        self.fe.sup.activate_worker = lambda ck: False
        self.assertFalse(await self.fe._try_activate("ck-ref"))
        self.assertIsNone(self.fe._active_artifact)


class ActivationMatrixCase(unittest.IsolatedAsyncioTestCase):
    """_handle_model_data activation matrix with a stub supervisor.

    Replies go to unknown ROUTER peers (dropped), so binding state,
    generation and the active artifact are asserted instead.
    """

    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fe = FrigateZmqFrontend(_sup(self.tmp.name),
                                     "tcp://127.0.0.1:*")
        await self.fe.start()
        self.ingest_result: dict = {}
        sup = self.fe.sup
        sup.ingest_zmq_bytes = (  # type: ignore[method-assign]
            lambda alias, data, contract, cls, sha: dict(
                self.ingest_result))
        from tests.integration.onnx_builders import make_raw_yolo
        with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
            self.onnx = make_raw_yolo(f.name, res=320, seed=11)
        self.art = os.path.join(self.tmp.name, "artifacts")
        os.makedirs(self.art, exist_ok=True)

    async def asyncTearDown(self):
        await self.fe.stop()
        self.tmp.cleanup()

    def _real_artifact(self, ck):
        os.makedirs(os.path.join(self.art, ck), exist_ok=True)
        with open(os.path.join(self.art, ck, "model.rai"), "wb") as f:
            f.write(b"REAL")
        with open(os.path.join(self.art, ck, "artifact.json"), "w") as f:
            json.dump({"backend": "bf16-vaiml-v1"}, f)

    def _prepared(self, ck):
        self.ingest_result = {"compile_key": ck, "cache_hit": True,
                              "state": "PREPARED",
                              "serving_digest": "srv-" + ck}

    async def test_first_model_activates(self):
        self._real_artifact("ck-a")
        self._prepared("ck-a")
        await self.fe._handle_model_data(
            b"id-a", {"model_name": "a.onnx"}, self.onnx)
        self.assertEqual(self.fe._active_artifact, "ck-a")
        bound = self.fe.sessions.get(b"id-a")
        self.assertIsNotNone(bound)
        self.assertEqual(bound.compile_key, "ck-a")
        self.assertEqual(bound.generation, 0)

    async def test_same_artifact_binds(self):
        self._real_artifact("ck-a")
        self._prepared("ck-a")
        await self.fe._handle_model_data(
            b"id-a", {"model_name": "a.onnx"}, self.onnx)
        await self.fe._handle_model_data(
            b"id-b", {"model_name": "a.onnx"}, self.onnx)
        self.assertEqual(self.fe.sessions.get(b"id-b").generation, 0)

    async def test_cold_miss_binds_nothing(self):
        self.ingest_result = {"compile_key": "ck-x", "cache_hit": False,
                              "state": "QUEUED", "serving_digest": ""}
        await self.fe._handle_model_data(
            b"id-x", {"model_name": "x.onnx"}, self.onnx)
        self.assertIsNone(self.fe._active_artifact)
        self.assertIsNone(self.fe.sessions.get(b"id-x"))

    async def test_fake_artifact_falls_through(self):
        os.makedirs(os.path.join(self.art, "ck-f"), exist_ok=True)
        with open(os.path.join(self.art, "ck-f", "model.rai"),
                  "wb") as f:
            f.write(b"FAKE")
        with open(os.path.join(self.art, "ck-f", "artifact.json"),
                  "w") as f:
            json.dump({"backend": "fake-v0"}, f)
        self._prepared("ck-f")
        await self.fe._handle_model_data(
            b"id-f", {"model_name": "f.onnx"}, self.onnx)
        self.assertIsNone(self.fe._active_artifact)
        self.assertIsNone(self.fe.sessions.get(b"id-f"))

    async def test_busy_model_gets_model_in_use(self):
        self._real_artifact("ck-a")
        self._real_artifact("ck-b")
        self._prepared("ck-a")
        await self.fe._handle_model_data(
            b"id-a", {"model_name": "a.onnx"}, self.onnx)
        self._prepared("ck-b")
        await self.fe._handle_model_data(
            b"id-b", {"model_name": "b.onnx"}, self.onnx)
        # traffic on generation 0: no switch, old stays active
        self.assertEqual(self.fe._active_artifact, "ck-a")
        self.assertEqual(self.fe._generation, 0)
        self.assertIsNone(self.fe.sessions.get(b"id-b"))

    async def test_quiescent_switches_generation(self):
        import time as _t
        self._real_artifact("ck-a")
        self._real_artifact("ck-b")
        self._prepared("ck-a")
        await self.fe._handle_model_data(
            b"id-a", {"model_name": "a.onnx"}, self.onnx)
        # drain: backdate past the quiescence window
        old = self.fe.sessions.get(b"id-a")
        old.last_used_at = _t.monotonic() - 60.0
        self._prepared("ck-b")
        await self.fe._handle_model_data(
            b"id-b", {"model_name": "b.onnx"}, self.onnx)
        self.assertEqual(self.fe._active_artifact, "ck-b")
        self.assertEqual(self.fe._generation, 1)
        self.assertIsNone(self.fe.sessions.get(b"id-a"))
        self.assertTrue(self.fe.sessions.is_superseded("ck-a"))
        self.assertEqual(
            self.fe.sessions.get(b"id-b").generation, 1)


if __name__ == "__main__":
    unittest.main()
