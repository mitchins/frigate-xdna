"""Contract tests: unmodified stock rc2 ZmqIpcDetector against a scripted server.

Imports the byte-identical pinned plugin (see test_pinned.py) with minimal
test shims for Frigate-internal transitive imports. A scripted in-process
REP server plays the sidecar role under test control.

Hardware/vendor-free: loopback IPC only, canned 480-byte frames.
"""
import json
import os
import sys
import tempfile
import threading
import time
import unittest

import numpy as np
import zmq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "upstream"))

from frigate.detectors.detector_config import (  # noqa: E402  (pinned file)
    ModelConfig,
    ModelTypeEnum,
)
from frigate.detectors.plugins.zmq_ipc import (  # noqa: E402  (pinned file)
    ZmqDetectorConfig,
    ZmqIpcDetector,
)

CANNED = np.zeros((20, 6), dtype=np.float32)
CANNED[0] = (5.0, 0.93, 0.22, 0.025, 0.68, 0.975)
CANNED[1] = (0.0, 0.81, 0.38, 0.06, 0.82, 0.30)


class ScriptedServer(threading.Thread):
    """REP server with a programmed availability/transfer/inference script."""

    def __init__(self, endpoint, *, available_first=True, transfer_ok=True,
                 infer_delay_s=0.0, infer_frame=CANNED.tobytes()):
        super().__init__(daemon=True)
        self.endpoint = endpoint
        self.available_first = available_first
        self.transfer_ok = transfer_ok
        self.infer_delay_s = infer_delay_s
        self.infer_frame = infer_frame
        self.requests = []  # (kind, header, nbytes)
        self.received_model_bytes = None
        self._stop_event = threading.Event()

    def run(self):
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REP)
        sock.setsockopt(zmq.RCVTIMEO, 500)
        sock.bind(self.endpoint)
        try:
            while not self._stop_event.is_set():
                try:
                    frames = sock.recv_multipart()
                except zmq.Again:
                    continue
                header = json.loads(frames[0].decode())
                if header.get("model_request"):
                    self.requests.append(("model_request", header, 0))
                    if self.available_first:
                        sock.send_json({"model_available": True,
                                        "model_loaded": True})
                    else:
                        sock.send_json({"model_available": False,
                                        "model_loaded": False})
                elif header.get("model_data"):
                    self.requests.append(
                        ("model_data", header, len(frames[1])))
                    self.received_model_bytes = bytes(frames[1])
                    sock.send_json({"model_saved": self.transfer_ok,
                                    "model_loaded": self.transfer_ok})
                else:
                    nbytes = len(frames[1]) if len(frames) > 1 else 0
                    self.requests.append(("infer", header, nbytes))
                    if self.infer_delay_s:
                        time.sleep(self.infer_delay_s)
                    sock.send(self.infer_frame)
        finally:
            sock.close(linger=0)
            ctx.term()

    def stop(self):
        self._stop_event.set()


def make_detector(endpoint, model_path, timeout_ms=200):
    cfg = ZmqDetectorConfig(
        type="zmq",
        model=ModelConfig(path=model_path,
                          model_type=ModelTypeEnum.yologeneric),
        endpoint=endpoint,
        request_timeout_ms=timeout_ms,
        linger_ms=0,
    )
    return ZmqIpcDetector(cfg)


class TestStockPlugin(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(self.tmp.name, "yolov9s-320.onnx")
        with open(self.model_path, "wb") as f:
            f.write(b"FAKE-ONNX-BYTES-" * 64)
        self.endpoint = (f"ipc:///tmp/fxdna-test-{os.getpid()}-"
                         f"{time.monotonic_ns()}")
        self.servers = []

    def tearDown(self):
        for s in self.servers:
            s.stop()
        for s in self.servers:
            s.join(timeout=5)
        self.tmp.cleanup()

    def serve(self, **kw):
        s = ScriptedServer(self.endpoint, **kw)
        s.start()
        time.sleep(0.2)
        self.servers.append(s)
        return s

    def test_basename_handshake_and_forced_transfer(self):
        srv = self.serve(available_first=False, transfer_ok=True)
        det = make_detector(self.endpoint, self.model_path)
        self.assertTrue(det._model_ready)
        kinds = [k for k, _, _ in srv.requests]
        self.assertEqual(kinds[0], "model_request")
        # basename only, never a path
        self.assertEqual(srv.requests[0][1]["model_name"], "yolov9s-320.onnx")
        self.assertIn("model_data", kinds)
        # exact transferred bytes hashed by the server side
        self.assertEqual(srv.received_model_bytes, b"FAKE-ONNX-BYTES-" * 64)

    def test_inference_header_and_480byte_decode(self):
        srv = self.serve(available_first=True)
        det = make_detector(self.endpoint, self.model_path)
        tensor = (np.arange(1 * 3 * 320 * 320, dtype=np.float32)
                  .reshape(1, 3, 320, 320))
        out = det.detect_raw(tensor)
        self.assertEqual(out.shape, (20, 6))
        self.assertEqual(out.dtype, np.float32)
        infers = [r for r in srv.requests if r[0] == "infer"]
        self.assertEqual(len(infers), 1)
        header = infers[0][1]
        # enum NAME on the wire, not the config value
        self.assertEqual(header, {"shape": [1, 3, 320, 320],
                                 "dtype": "float32",
                                 "model_type": "yologeneric"})
        # no per-frame model identity on the wire
        self.assertNotIn("model_name", header)
        self.assertNotIn("model_id", header)
        self.assertNotIn("sha256", header)
        # canned frame decoded exactly
        self.assertAlmostEqual(float(out[0, 1]), 0.93, places=5)
        self.assertEqual(int(out[0, 0]), 5)

    def test_not_ready_returns_zeros_without_traffic(self):
        srv = self.serve(available_first=False, transfer_ok=False)
        det = make_detector(self.endpoint, self.model_path)
        self.assertFalse(det._model_ready)
        n_before = len(srv.requests)
        out = det.detect_raw(np.zeros((1, 3, 320, 320), np.float32))
        self.assertTrue((out == 0).all())
        self.assertEqual(len(srv.requests), n_before)  # no retry, no poll

    def test_inference_timeout_returns_zeros(self):
        self.serve(available_first=True, infer_delay_s=2.0)
        det = make_detector(self.endpoint, self.model_path, timeout_ms=150)
        out = det.detect_raw(np.zeros((1, 3, 320, 320), np.float32))
        self.assertTrue((out == 0).all())

    def test_malformed_response_size_returns_zeros(self):
        self.serve(available_first=True, infer_frame=b"\x00" * 100)
        det = make_detector(self.endpoint, self.model_path)
        out = det.detect_raw(np.zeros((1, 3, 320, 320), np.float32))
        self.assertTrue((out == 0).all())

    def test_model_operation_timeout_is_30s(self):
        import inspect
        import frigate.detectors.plugins.zmq_ipc as plugin_mod
        src = inspect.getsource(plugin_mod)
        # both model paths raise the socket RCVTIMEO to 30000 ms
        self.assertEqual(src.count("RCVTIMEO, 30000"), 2)


if __name__ == "__main__":
    unittest.main()
