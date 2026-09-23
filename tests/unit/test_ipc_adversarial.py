"""Unit tests: IPC framing adversarial bounds (no NPU, socketpairs only).

Oversized headers, unknown message types and out-of-range lengths
must be refused before any byte is trusted — in both directions.
"""
import json
import socket
import struct
import unittest

from frigate_xdna.runtime import ipc


def pair():
    return socket.socketpair()


class TestIpcAdversarial(unittest.TestCase):
    def test_oversized_header_refused_on_send(self):
        a, b = pair()
        try:
            with self.assertRaises(ValueError) as ctx:
                ipc.send_message(a, {"message_type": "INFER",
                                     "pad": "x" * (32 * 1024)}, b"")
            self.assertIn("header too large", str(ctx.exception))
        finally:
            a.close()
            b.close()

    def test_unknown_message_type_refused_on_recv(self):
        a, b = pair()
        try:
            raw = b'{"message_type": "NOPE", "payload_length": 0,\
 "protocol_version": 1}'
            b.sendall(struct.pack("!I", len(raw)) + raw)
            with self.assertRaises(ValueError) as ctx:
                ipc.recv_message(a, timeout=5.0)
            self.assertIn("bad message type", str(ctx.exception))
        finally:
            a.close()
            b.close()

    def test_negative_payload_length_refused(self):
        a, b = pair()
        try:
            raw = b'{"message_type": "INFER", "payload_length": -5,\
 "protocol_version": 1}'
            b.sendall(struct.pack("!I", len(raw)) + raw)
            with self.assertRaises(ValueError) as ctx:
                ipc.recv_message(a, timeout=5.0)
            self.assertIn("bad payload_length", str(ctx.exception))
        finally:
            a.close()
            b.close()

    def test_infer_payload_over_tensor_bound_refused_on_recv(self):
        a, b = pair()
        try:
            raw = json.dumps({"message_type": "INFER",
                                "payload_length": ipc.MAX_TENSOR_BYTES + 4,
                                "protocol_version": 1}).encode()
            b.sendall(struct.pack("!I", len(raw)) + raw)
            with self.assertRaises(ValueError) as ctx:
                ipc.recv_message(a, timeout=5.0)
            self.assertIn("infer payload too large", str(ctx.exception))
        finally:
            a.close()
            b.close()


if __name__ == "__main__":
    unittest.main()
