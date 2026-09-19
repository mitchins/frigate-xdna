"""Private supervisor↔native IPC (SPEC §3.2, INTERFACES.md §4).

Length-prefixed JSON header + bounded binary payload over Unix socketpair.
Message types: LOAD, INFER, RESULT, STATUS, SHUTDOWN.
Every request carries protocol version, request ID, worker generation,
tensor contract, bounded payload length. Native independently validates.
"""
from __future__ import annotations

import json
import os
import socket
import struct

PROTOCOL_VERSION = 1
MAX_HEADER_BYTES = 16 * 1024
MAX_TENSOR_BYTES = 16 * 1024 * 1024
MAX_MODEL_BYTES = 256 * 1024 * 1024

MESSAGE_TYPES = ("LOAD", "INFER", "RESULT", "STATUS", "SHUTDOWN")


def create_socketpair() -> tuple[socket.socket, socket.socket]:
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    return a, b


def _send_all(sock: socket.socket, data: bytes) -> None:
    sock.sendall(data)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("ipc peer closed")
        buf.extend(chunk)
    return bytes(buf)


def send_message(
    sock: socket.socket, header: dict, payload: bytes = b""
) -> None:
    header = dict(header)
    header.setdefault("protocol_version", PROTOCOL_VERSION)
    if header.get("message_type") not in MESSAGE_TYPES:
        raise ValueError(f"bad message_type {header.get('message_type')!r}")
    if header["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("bad protocol version")
    # bounded checks
    if len(payload) > MAX_TENSOR_BYTES and header.get("message_type") == "INFER":
        raise ValueError("tensor payload too large")
    if len(payload) > MAX_MODEL_BYTES and header.get("message_type") == "LOAD":
        raise ValueError("model payload too large")
    header["payload_length"] = len(payload)
    js = json.dumps(header, sort_keys=True).encode()
    if len(js) > MAX_HEADER_BYTES:
        raise ValueError("header too large")
    _send_all(sock, struct.pack("!I", len(js)))
    _send_all(sock, js)
    if payload:
        _send_all(sock, payload)


def recv_message(sock: socket.socket, timeout: float | None = None) -> tuple[dict, bytes]:
    if timeout is not None:
        sock.settimeout(timeout)
    hdr_len = struct.unpack("!I", _recv_exact(sock, 4))[0]
    if hdr_len == 0 or hdr_len > MAX_HEADER_BYTES:
        raise ValueError(f"bad header length {hdr_len}")
    js = _recv_exact(sock, hdr_len)
    try:
        header = json.loads(js.decode())
    except (ValueError, UnicodeDecodeError) as e:
        raise ValueError(f"bad header json: {e}") from e
    # validate
    if header.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("bad protocol version")
    if header.get("message_type") not in MESSAGE_TYPES:
        raise ValueError("bad message type")
    payload_len = int(header.get("payload_length", 0))
    if payload_len < 0 or payload_len > MAX_MODEL_BYTES:
        raise ValueError("bad payload_length")
    payload = b""
    if payload_len:
        # also enforce per-type bounds
        mt = header.get("message_type")
        if mt == "INFER" and payload_len > MAX_TENSOR_BYTES:
            raise ValueError("infer payload too large")
        payload = _recv_exact(sock, payload_len)
    return header, payload


def validate_tensor_request(header: dict, payload: bytes) -> str | None:
    """Native-side validation: rank/shape/dtype/byte count/finite/generation.
    Returns error string or None if ok. Payload is not normalized here.
    """
    # generation already checked by caller
    shape = header.get("shape")
    dtype = header.get("dtype")
    # shape must be [1,3,H,W] with H==W and finite
    if not isinstance(shape, list) or len(shape) != 4:
        return "bad rank"
    if shape[0] != 1 or shape[1] != 3:
        return "bad batch/channels"
    if shape[2] != shape[3]:
        return "non-square"
    if dtype != "float32":
        return "bad dtype"
    # overflow-safe byte count
    try:
        elems = 1
        for d in shape:
            if not isinstance(d, int) or d <= 0 or d > 2048:
                return "bad dim"
            elems *= d
        expected = elems * 4
    except (OverflowError, ValueError):
        return "overflow"
    if expected != len(payload):
        return f"byte count mismatch {len(payload)} vs {expected}"
    # finite check (sampled)
    import struct as _st

    # quick finite check: unpack a few floats
    for i in range(0, min(len(payload), 4096), 4):
        v = _st.unpack_from("<f", payload, i)[0]
        if v != v or v == float("inf") or v == float("-inf"):
            return "non-finite"
    return None
