"""Hardware-free tests for supervisor<->native IPC framing (SPEC §3.2).

Length-prefixed JSON header + bounded payload over a socketpair. No
native worker is spawned; both ends are the Python framing helpers.
"""
import json
import struct
import unittest

from frigate_xdna.runtime.ipc import (
    MAX_HEADER_BYTES,
    MAX_MODEL_BYTES,
    MAX_TENSOR_BYTES,
    create_socketpair,
    recv_message,
    send_message,
    validate_tensor_request,
)


def _hdr(mtype, **kw):
    h = {"message_type": mtype, "request_id": "r1", "generation": 0}
    h.update(kw)
    return h


class TestFraming(unittest.TestCase):
    def test_roundtrip_all_types(self):
        for mtype in ("LOAD", "INFER", "RESULT", "STATUS", "SHUTDOWN"):
            a, b = create_socketpair()
            try:
                send_message(a, _hdr(mtype), b"pay")
                h, p = recv_message(b, timeout=5.0)
                self.assertEqual(h["message_type"], mtype)
                self.assertEqual(h["protocol_version"], 1)
                self.assertEqual(h["payload_length"], 3)
                self.assertEqual(p, b"pay")
            finally:
                a.close()
                b.close()

    def test_empty_payload(self):
        a, b = create_socketpair()
        try:
            send_message(a, _hdr("STATUS"))
            h, p = recv_message(b, timeout=5.0)
            self.assertEqual(p, b"")
        finally:
            a.close()
            b.close()

    def test_bad_message_type_rejected(self):
        a, b = create_socketpair()
        try:
            with self.assertRaises(ValueError):
                send_message(a, {"message_type": "NOPE"})
        finally:
            a.close()
            b.close()

    def test_version_mismatch_rejected(self):
        a, b = create_socketpair()
        self.addCleanup(a.close)
        self.addCleanup(b.close)
        bad = _hdr("STATUS", protocol_version=999)
        with self.assertRaises(ValueError):
            send_message(a, bad)

    def test_tensor_bound_enforced(self):
        a, b = create_socketpair()
        self.addCleanup(a.close)
        self.addCleanup(b.close)
        hdr = _hdr("INFER")
        with self.assertRaises(ValueError):
            send_message(a, hdr, b"x" * (MAX_TENSOR_BYTES + 1))

    def test_model_bound_enforced(self):
        a, b = create_socketpair()
        self.addCleanup(a.close)
        self.addCleanup(b.close)
        hdr = _hdr("LOAD")
        with self.assertRaises(ValueError):
            send_message(a, hdr, b"x" * (MAX_MODEL_BYTES + 1))

    def test_garbage_header_length_rejected(self):
        a, b = create_socketpair()
        try:
            a.sendall(struct.pack("!I", MAX_HEADER_BYTES + 1))
            with self.assertRaises(ValueError):
                recv_message(b, timeout=5.0)
        finally:
            a.close()
            b.close()

    def test_non_json_header_rejected(self):
        a, b = create_socketpair()
        try:
            bad = b"\xff\xfe not json"
            a.sendall(struct.pack("!I", len(bad)))
            a.sendall(bad)
            with self.assertRaises(ValueError):
                recv_message(b, timeout=5.0)
        finally:
            a.close()
            b.close()

    def test_wrong_version_on_wire_rejected(self):
        a, b = create_socketpair()
        try:
            h = _hdr("STATUS")
            h["protocol_version"] = 2
            js = json.dumps(h).encode()
            a.sendall(struct.pack("!I", len(js)))
            a.sendall(js)
            with self.assertRaises(ValueError):
                recv_message(b, timeout=5.0)
        finally:
            a.close()
            b.close()

    def test_peer_close_raises(self):
        a, b = create_socketpair()
        a.close()
        with self.assertRaises(OSError):
            recv_message(b, timeout=5.0)
        b.close()


class TestValidateTensorRequest(unittest.TestCase):
    def _payload(self, shape, fill=0.5):
        n = 1
        for d in shape:
            n *= d
        return struct.pack(f"<{n}f", *([fill] * n))

    def test_ok(self):
        shape = [1, 3, 4, 4]
        self.assertIsNone(validate_tensor_request(
            {"shape": shape, "dtype": "float32"},
            self._payload(shape)))

    def test_bad_rank(self):
        self.assertEqual(
            validate_tensor_request({"shape": [1, 3, 4], "dtype": "float32"},
                                    b"x" * 48), "bad rank")

    def test_bad_batch_channels(self):
        self.assertEqual(
            validate_tensor_request(
                {"shape": [2, 3, 4, 4], "dtype": "float32"}, b""),
            "bad batch/channels")

    def test_non_square(self):
        self.assertEqual(
            validate_tensor_request(
                {"shape": [1, 3, 4, 8], "dtype": "float32"}, b""),
            "non-square")

    def test_bad_dtype(self):
        self.assertEqual(
            validate_tensor_request(
                {"shape": [1, 3, 4, 4], "dtype": "float16"}, b""),
            "bad dtype")

    def test_bad_dim(self):
        self.assertEqual(
            validate_tensor_request(
                {"shape": [1, 3, 0, 0], "dtype": "float32"}, b""),
            "bad dim")
        self.assertEqual(
            validate_tensor_request(
                {"shape": [1, 3, 5000, 5000], "dtype": "float32"}, b""),
            "bad dim")

    def test_byte_mismatch(self):
        self.assertTrue(
            validate_tensor_request(
                {"shape": [1, 3, 4, 4], "dtype": "float32"}, b"short"
            ).startswith("byte count mismatch"))

    def test_non_finite_rejected(self):
        import math
        shape = [1, 3, 4, 4]
        n = 48
        bad = struct.pack("<f", math.nan) + b"\x00" * ((n - 1) * 4)
        self.assertEqual(
            validate_tensor_request(
                {"shape": shape, "dtype": "float32"}, bad), "non-finite")
        bad = struct.pack("<f", math.inf) + b"\x00" * ((n - 1) * 4)
        self.assertEqual(
            validate_tensor_request(
                {"shape": shape, "dtype": "float32"}, bad), "non-finite")


if __name__ == "__main__":
    unittest.main()
