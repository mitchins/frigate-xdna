"""Unit tests: ZMQ frontend transport guards (no NPU, no bound socket).

Drives the serve/worker loops and handlers with a scripted socket:
malformed envelopes, oversized headers, full queues, expired requests
and failing supervisors must degrade to explicit error replies and
counters — never hang a REQ client or kill the serve task.
"""
import asyncio
import json
import os
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import zmq

from frigate_xdna.errors import FxdnaError
from frigate_xdna.runtime.device_lease import DeviceLease
from frigate_xdna.transport.frigate_zmq import (
    MAX_HEADER_BYTES,
    ZERO_FRAME,
    FrigateZmqFrontend,
    _artifact_usable,
)
from tests.integration.onnx_builders import make_raw_yolo


def eterm():
    e = zmq.ZMQError()
    e.errno = zmq.ETERM
    return e


class FakeSock:
    """Scripted ROUTER socket: recv items or raised errors in order."""

    def __init__(self, script, fail_sends=0):
        self.script = list(script)
        self.sent = []
        self.fail_sends = fail_sends

    async def recv_multipart(self):
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def send_multipart(self, parts):
        if self.fail_sends > 0:
            self.fail_sends -= 1
            raise RuntimeError("send exploded")
        self.sent.append(parts)


def make_frontend(tmp, **sup_kw):
    sup = SimpleNamespace(data_dir=tmp,
                          config=SimpleNamespace(allow_uploads=True),
                          **sup_kw)
    fe = FrigateZmqFrontend(sup, "tcp://127.0.0.1:5559")
    return fe, sup


def run_loop(fe, items):
    """Feed worker-loop items, let them drain, then stop the loop."""
    async def drive():
        for it in items:
            fe._queue.put_nowait(it)
        task = asyncio.ensure_future(fe._worker_loop())
        await asyncio.sleep(0.2)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    fe._queue = asyncio.Queue()
    asyncio.run(drive())


def expired_req(header):
    return (b"id", header, b"data", time.monotonic() - 1.0,
            time.monotonic() - 1.0)


class TestArtifactUsable(unittest.TestCase):
    def test_corrupt_metadata_refuses(self):
        with tempfile.TemporaryDirectory() as d:
            art = os.path.join(d, "artifacts", "ck")
            os.makedirs(art)
            with open(os.path.join(art, "model.rai"), "wb") as f:
                f.write(b"RAI")
            with open(os.path.join(art, "artifact.json"), "w") as f:
                f.write("{not json")
            self.assertFalse(_artifact_usable(d, "ck"))

    def test_fake_backend_refuses(self):
        with tempfile.TemporaryDirectory() as d:
            art = os.path.join(d, "artifacts", "ck")
            os.makedirs(art)
            with open(os.path.join(art, "model.rai"), "wb") as f:
                f.write(b"RAI")
            with open(os.path.join(art, "artifact.json"), "w") as f:
                json.dump({"backend": "fake-v0"}, f)
            self.assertFalse(_artifact_usable(d, "ck"))


class TestServeLoop(unittest.TestCase):
    def drive_serve(self, fe, script):
        async def go():
            fe.sock = FakeSock(script)
            fe._queue = asyncio.Queue()
            await fe._serve_loop()
            return fe.sock
        return asyncio.run(go())

    def test_oversized_header_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            fe, _sup = make_frontend(d)
            sock = self.drive_serve(
                fe, [[b"id", b"x" * (MAX_HEADER_BYTES + 1)], eterm()])
            self.assertEqual(fe.counters["rejected"], 1)
            self.assertEqual(
                json.loads(sock.sent[0][2])["error_code"], "INVALID_MODEL")

    def test_bad_json_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            fe, _sup = make_frontend(d)
            sock = self.drive_serve(fe, [[b"id", b"{nope"], eterm()])
            self.assertEqual(fe.counters["rejected"], 1)
            self.assertEqual(
                json.loads(sock.sent[0][2])["error_code"], "INVALID_MODEL")

    def test_short_envelope_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            fe, _sup = make_frontend(d)
            sock = self.drive_serve(fe, [[b"only-identity"], eterm()])
            self.assertEqual(sock.sent, [])
            self.assertEqual(fe.counters["rejected"], 0)

    def test_again_then_message_queued(self):
        with tempfile.TemporaryDirectory() as d:
            fe, _sup = make_frontend(d)
            header = json.dumps({"model_request": True}).encode()
            self.drive_serve(
                fe, [zmq.Again(), [b"id", header], eterm()])
            self.assertEqual(fe._queue.qsize(), 1)

    def test_transport_garbage_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            fe, _sup = make_frontend(d)
            header = json.dumps({"model_request": True}).encode()
            self.drive_serve(
                fe, [RuntimeError("garbage"), [b"id", header], eterm()])
            self.assertEqual(fe._queue.qsize(), 1)

    def test_nonterminal_transport_error_propagates(self):
        with tempfile.TemporaryDirectory() as d:
            fe, _sup = make_frontend(d)
            err = zmq.ZMQError()
            err.errno = zmq.EAGAIN

            async def go():
                fe.sock = FakeSock([err])
                fe._queue = asyncio.Queue()
                await fe._serve_loop()
            with self.assertRaises(zmq.ZMQError):
                asyncio.run(go())

    def test_full_queue_replies_resource_exceeded(self):
        with tempfile.TemporaryDirectory() as d:
            fe, _sup = make_frontend(d)
            header = json.dumps({"model_request": True}).encode()

            async def go():
                fe.sock = FakeSock([[b"id", header], eterm()])
                fe._queue = asyncio.Queue(maxsize=1)
                fe._queue.put_nowait(("stub", {}, b"", 0.0, 0.0))
                await fe._serve_loop()
                return fe.sock
            sock = asyncio.run(go())
            self.assertEqual(fe.counters["rejected"], 1)
            self.assertEqual(
                json.loads(sock.sent[0][2])["error_code"],
                "RESOURCE_EXCEEDED")


class TestWorkerLoop(unittest.TestCase):
    def test_expired_requests_get_explicit_replies(self):
        with tempfile.TemporaryDirectory() as d:
            fe, _sup = make_frontend(d)
            fe.sock = FakeSock([])
            run_loop(fe, [
                expired_req({"shape": [1, 3, 4, 4]}),
                expired_req({"model_request": True}),
                expired_req({"model_data": True}),
                expired_req({"other": True}),
            ])
            self.assertEqual(fe.counters["late_discard"], 4)
            kinds = [json.loads(p[2]) if len(p[2]) < 480 else "raw"
                     for p in fe.sock.sent]
            self.assertEqual(kinds[0], "raw")
            self.assertEqual(kinds[1]["error_code"], "TIMEOUT")
            self.assertFalse(kinds[2]["model_saved"])
            self.assertEqual(kinds[3]["error_code"], "TIMEOUT")

    def test_unknown_live_request_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            fe, _sup = make_frontend(d)
            fe.sock = FakeSock([])
            future = time.monotonic() + 60.0
            run_loop(fe, [(b"id", {"other": True}, b"", future, future)])
            self.assertEqual(fe.counters["rejected"], 1)
            self.assertEqual(
                json.loads(fe.sock.sent[0][2])["error_code"],
                "INVALID_MODEL")

    def test_live_model_data_dispatch_reaches_handler(self):
        with tempfile.TemporaryDirectory() as d:
            fe, _sup = make_frontend(d)
            fe.sock = FakeSock([])
            future = time.monotonic() + 60.0
            run_loop(fe, [(b"id", {"model_data": True}, b"garbage",
                           future, future)])
            body = json.loads(fe.sock.sent[0][2])
            self.assertFalse(body["model_saved"])
            self.assertEqual(body["error_code"], "INVALID_MODEL")

    def test_live_shape_dispatch_reaches_infer(self):
        with tempfile.TemporaryDirectory() as d:
            fe, _sup = make_frontend(d)
            fe.sock = FakeSock([])
            future = time.monotonic() + 60.0
            run_loop(fe, [(b"id", {"shape": [1, 3, 2, 2]}, b"\x00" * 48,
                           future, future)])
            self.assertEqual(fe.counters["rejected"], 1)
            self.assertEqual(fe.sock.sent[0][2], ZERO_FRAME)


class TestModelHandlers(unittest.TestCase):
    def test_bound_session_reports_available(self):
        with tempfile.TemporaryDirectory() as d:
            fe, _sup = make_frontend(d)
            fe.sock = FakeSock([])
            fe._active_artifact = "ck"
            fe.sessions.bind(b"id", "sha", "digest", "ck",
                             fe._generation)
            asyncio.run(fe._handle_model_request(b"id"))
            body = json.loads(fe.sock.sent[0][2])
            self.assertTrue(body["model_available"])
            self.assertTrue(body["model_loaded"])

    def test_disabled_uploads_refused(self):
        with tempfile.TemporaryDirectory() as d:
            fe, sup = make_frontend(d)
            sup.config.allow_uploads = False
            fe.sock = FakeSock([])
            asyncio.run(fe._handle_model_data(b"id", {}, b"bytes"))
            body = json.loads(fe.sock.sent[0][2])
            self.assertFalse(body["model_saved"])
            self.assertEqual(body["error_code"], "INVALID_MODEL")
            self.assertEqual(fe.counters["rejected"], 1)

    def test_garbage_bytes_refused(self):
        with tempfile.TemporaryDirectory() as d:
            fe, _sup = make_frontend(d)
            fe.sock = FakeSock([])
            asyncio.run(fe._handle_model_data(b"id", {}, b"garbage"))
            body = json.loads(fe.sock.sent[0][2])
            self.assertFalse(body["model_saved"])
            self.assertEqual(body["error_code"], "INVALID_MODEL")

    def test_ingest_source_changed_refused(self):
        with tempfile.TemporaryDirectory() as d:
            def boom(*a):
                raise FxdnaError(2, "SOURCE_CHANGED", "changed")
            fe, _sup = make_frontend(d, ingest_zmq_bytes=boom)
            fe.sock = FakeSock([])
            raw = make_raw_yolo(os.path.join(d, "m.onnx"), res=320,
                                classes=2, seed=9)
            asyncio.run(fe._handle_model_data(
                b"id", {}, raw))
            body = json.loads(fe.sock.sent[0][2])
            self.assertEqual(body["error_code"], "SOURCE_CHANGED")
            self.assertEqual(fe.counters["rejected"], 1)

    def test_ingest_unexpected_error_refused(self):
        with tempfile.TemporaryDirectory() as d:
            def boom(*a):
                raise FxdnaError(6, "COMPILE_FAILED", "busy")
            fe, _sup = make_frontend(d, ingest_zmq_bytes=boom)
            fe.sock = FakeSock([])
            raw = make_raw_yolo(os.path.join(d, "m.onnx"), res=320,
                                classes=2, seed=9)
            asyncio.run(fe._handle_model_data(
                b"id", {}, raw))
            body = json.loads(fe.sock.sent[0][2])
            self.assertEqual(body["error_code"], "COMPILE_FAILED")

    def test_non_yolo_contract_refused(self):
        with tempfile.TemporaryDirectory() as d:
            fe, _sup = make_frontend(d)
            fe.sock = FakeSock([])
            asyncio.run(fe._handle_model_data(
                b"id", {}, make_two_output(os.path.join(d, "t.onnx"))))
            body = json.loads(fe.sock.sent[0][2])
            self.assertFalse(body["model_saved"])
            self.assertEqual(body["error_code"], "UNSUPPORTED_CONTRACT")

    def test_inspect_crash_refused(self):
        with tempfile.TemporaryDirectory() as d:
            from frigate_xdna.transport import frigate_zmq as zmq_mod
            fe, _sup = make_frontend(d)
            fe.sock = FakeSock([])
            raw = make_raw_yolo(os.path.join(d, "m.onnx"), res=320,
                                classes=2, seed=9)
            with mock.patch.object(zmq_mod._inspect, "load_graph_bytes",
                                   side_effect=RuntimeError("odd")):
                asyncio.run(fe._handle_model_data(
                    b"id", {}, raw))
            body = json.loads(fe.sock.sent[0][2])
            self.assertFalse(body["model_saved"])
            self.assertEqual(body["error_code"], "INVALID_MODEL")


def make_two_output(path):
    """Valid ONNX whose two outputs match no decoder profile."""
    from onnx import TensorProto, helper
    inp = helper.make_tensor_value_info("images", TensorProto.FLOAT,
                                        [1, 3, 32, 32])
    outs = [helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, 2])
            for n in ("a", "b")]
    nodes = [helper.make_node(
        "Constant", [], [n], name=n,
        value=helper.make_tensor("c" + n, TensorProto.FLOAT, [1, 2],
                                 [1.0, 2.0])) for n in ("a", "b")]
    graph = helper.make_graph(nodes, "two", [inp], outs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid(
        "", 17)])
    model.ir_version = 8
    data = model.SerializeToString()
    with open(path, "wb") as f:
        f.write(data)
    return data


class TestActivationGuards(unittest.TestCase):
    def test_failed_activation_reports_not_prepared(self):
        with tempfile.TemporaryDirectory() as d:
            fe, _sup = make_frontend(d)
            fe.sock = FakeSock([])
            fe._active_artifact = "old"
            result = asyncio.run(
                fe._handle_prepared(b"id", "missing", "digest", "sha"))
            self.assertFalse(result)

    def test_busy_lease_refuses_activation(self):
        with tempfile.TemporaryDirectory() as d:
            fe, _sup = make_frontend(d)
            held = DeviceLease(d)
            self.assertTrue(held.try_acquire())
            try:
                self.assertFalse(
                    asyncio.run(fe._try_activate("ck")))
            finally:
                held.release()


class TestInferGuards(unittest.TestCase):
    def bind(self, fe):
        fe.sessions.bind(b"id", "sha", "digest", "ck", fe._generation)

    def test_past_deadline_returns_zero_and_counts_timeout(self):
        with tempfile.TemporaryDirectory() as d:
            fe, _sup = make_frontend(d)
            fe.sock = FakeSock([])
            self.bind(fe)
            past = time.monotonic() - 1.0
            header = {"shape": [1, 3, 2, 2], "dtype": "float32"}
            asyncio.run(fe._handle_infer(b"id", header, b"\x00" * 64,
                                         past))
            self.assertEqual(fe.counters["timeout"], 1)
            self.assertEqual(fe.sock.sent[0][2], ZERO_FRAME)

    def test_bad_shape_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            fe, _sup = make_frontend(d)
            fe.sock = FakeSock([])
            self.bind(fe)
            future = time.monotonic() + 60.0
            asyncio.run(fe._handle_infer(
                b"id", {"shape": "nope", "dtype": "float32"},
                b"\x00" * 64, future))
            self.assertEqual(fe.counters["rejected"], 1)
            self.assertEqual(fe.sock.sent[0][2], ZERO_FRAME)

    def test_byte_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            fe, _sup = make_frontend(d)
            fe.sock = FakeSock([])
            self.bind(fe)
            future = time.monotonic() + 60.0
            asyncio.run(fe._handle_infer(
                b"id", {"shape": [1, 3, 2, 2], "dtype": "float32"},
                b"\x00" * 10, future))
            self.assertEqual(fe.counters["rejected"], 1)

    def test_unparsable_dims_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            fe, _sup = make_frontend(d)
            fe.sock = FakeSock([])
            self.bind(fe)
            future = time.monotonic() + 60.0
            asyncio.run(fe._handle_infer(
                b"id", {"shape": ["x"], "dtype": "float32"},
                b"\x00" * 10, future))
            self.assertEqual(fe.counters["rejected"], 1)

    def test_worker_exception_counts_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            def boom(*a):
                raise RuntimeError("worker gone")
            fe, _sup = make_frontend(d, worker_infer=boom)
            fe.sock = FakeSock([])
            self.bind(fe)
            future = time.monotonic() + 60.0
            asyncio.run(fe._handle_infer(
                b"id", {"shape": [1, 3, 2, 2], "dtype": "float32"},
                b"\x00" * 48, future))
            self.assertEqual(fe.counters["rejected"], 1)
            self.assertEqual(fe.sock.sent[0][2], ZERO_FRAME)

    def test_reply_failure_still_counts_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            fe, _sup = make_frontend(d, worker_infer=lambda *a: ("ok",
                                                                 b"R" * 480))
            fe.sock = FakeSock([], fail_sends=1)
            self.bind(fe)
            future = time.monotonic() + 60.0
            asyncio.run(fe._handle_infer(
                b"id", {"shape": [1, 3, 2, 2], "dtype": "float32"},
                b"\x00" * 48, future))
            self.assertEqual(fe.counters["rejected"], 1)
            self.assertEqual(fe.sock.fail_sends, 0)


if __name__ == "__main__":
    unittest.main()
