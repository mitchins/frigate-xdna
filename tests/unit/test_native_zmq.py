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


def _sup(tmp):
    return SimpleNamespace(
        data_dir=tmp,
        config=SimpleNamespace(allow_uploads=True),
    )


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
        # bound to current generation, active artifact unset -> success
        # path still answers (zero frame) and touches the binding
        self.fe.sessions.bind(ident, "sha", "srv", "ck", 0)
        before = self.fe.sessions.get(ident).last_used_at
        await self.fe._handle_infer(
            ident, {"shape": shape, "dtype": "float32"}, payload, dl)
        self.assertEqual(self.fe.counters["success"], 1)
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
            await self.fe._try_activate("ck-missing", "sha", "a"))
        # fake backend -> False
        os.makedirs(os.path.join(art, "ck-fake"))
        with open(os.path.join(art, "ck-fake", "model.rai"), "wb") as f:
            f.write(b"FAKE")
        with open(os.path.join(art, "ck-fake", "artifact.json"),
                  "w") as f:
            json.dump({"backend": "fake-v0"}, f)
        self.assertFalse(
            await self.fe._try_activate("ck-fake", "sha", "a"))
        self.assertIsNone(self.fe._active_artifact)
        # non-fake artifact -> True and recorded
        os.makedirs(os.path.join(art, "ck-real"))
        with open(os.path.join(art, "ck-real", "model.rai"), "wb") as f:
            f.write(b"REAL")
        with open(os.path.join(art, "ck-real", "artifact.json"),
                  "w") as f:
            json.dump({"backend": "bf16-vaiml-v1"}, f)
        self.assertTrue(
            await self.fe._try_activate("ck-real", "sha", "a"))
        self.assertEqual(self.fe._active_artifact, "ck-real")

    async def test_stop_joins_tasks(self):
        await self.fe.stop()
        self.assertTrue(self.fe._task.done())
        self.assertTrue(self.fe._worker.done())
        self.assertIsNone(self.fe.sock)


if __name__ == "__main__":
    unittest.main()
