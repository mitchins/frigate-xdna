#!/usr/bin/env python3
"""Fake fxdna-worker binary (hardware-free NativeWorker tests only).

FXDNA-TEST-FIXTURE: scripted vendor boundary. Speaks the private
supervisor<->native IPC framing ([4-byte BE header length][JSON
header][payload] over an inherited socketpair fd passed via --fd N or
FXDNA_NATIVE_FD) with canned STATUS/RESULT replies. Never touches
hardware. Behavior knobs via argv (the child env is minimal by
design, so no configuration passes through environ):

  --fail load|infer   refuse LOAD / INFER with an error
  --fail-code CODE    error_code for refusals (default INVALID_ARGS;
                      LOAD default INVALID_MODEL without it)
  --frame zeros|ones|nan|short:N
                      INFER payload variant (default ones)
  --expect-shape DIMS require INFER tensor_spec.shape == DIMS
                      (comma separated); else INVALID_TENSOR
  --sleep S           sleep S seconds before each INFER reply
  --exit N            exit(N) immediately (death simulation)
  --die-on infer|load
                      os._exit(1) on that message (crash simulation)
  --ignore-shutdown   never reply to / exit on SHUTDOWN (retire path)

Stdlib only.
"""
import json
import os
import struct
import sys
import time

RESULT_BYTES = 20 * 6 * 4


def parse_arg(argv, name, default=None):
    for i, a in enumerate(argv[1:]):
        if a == name:
            # Valueless trailing flags (e.g. --ignore-shutdown) are
            # True, not an IndexError.
            nxt = argv[i + 2] if i + 2 < len(argv) else None
            if nxt is None or nxt.startswith("--"):
                return True
            return nxt
        if a.startswith(name + "="):
            return a[len(name) + 1:]
    return default


def parse_fd(argv):
    fd = parse_arg(argv, "--fd")
    if fd is not None:
        return int(fd)
    env = os.environ.get("FXDNA_NATIVE_FD")
    if env:
        return int(env)
    return 3


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed")
        buf.extend(chunk)
    return bytes(buf)


def send(sock, header, payload=b""):
    js = json.dumps(header).encode()
    sock.sendall(struct.pack("!I", len(js)) + js + payload)


def frame_bytes(kind):
    import struct as _struct
    if kind == "zeros":
        return b"\x00" * RESULT_BYTES
    if kind == "nan":
        return _struct.pack(f"<{RESULT_BYTES // 4}f",
                            *[float("nan")] * (RESULT_BYTES // 4))
    if kind.startswith("short:"):
        return b"\x00" * int(kind.split(":", 1)[1])
    return b"\x01" * RESULT_BYTES


def main():
    import socket as _socket
    fd = parse_fd(sys.argv)
    sock = _socket.socket(fileno=fd)
    generation = 0
    loaded = False
    fail = parse_arg(sys.argv, "--fail", "") or ""
    fail_code = parse_arg(sys.argv, "--fail-code", "") or ""
    frame_kind = parse_arg(sys.argv, "--frame", "ones") or "ones"
    expect_shape = parse_arg(sys.argv, "--expect-shape", "") or ""
    if expect_shape:
        expect_shape = [int(v) for v in expect_shape.split(",")]
    sleep_s = float(parse_arg(sys.argv, "--sleep", "0") or 0)
    die_on = parse_arg(sys.argv, "--die-on", "") or ""
    ignore_shutdown = parse_arg(sys.argv, "--ignore-shutdown",
                                None) is not None
    early_exit = parse_arg(sys.argv, "--exit")
    if early_exit is not None:
        sys.exit(int(early_exit))
    while True:
        try:
            raw_len = recv_exact(sock, 4)
        except ConnectionError:
            break
        (hlen,) = struct.unpack("!I", raw_len)
        header = json.loads(recv_exact(sock, hlen).decode())
        payload_len = int(header.get("payload_length", 0))
        if payload_len:
            recv_exact(sock, payload_len)
        mt = header.get("message_type")
        rid = header.get("request_id", 0)
        if mt == "SHUTDOWN":
            if ignore_shutdown:
                # Wedged child: detach from the socket and never exit
                # on our own; the supervisor must SIGTERM us after its
                # bounded grace period.
                try:
                    sock.close()
                except OSError:
                    pass
                time.sleep(60)
                continue
            send(sock, {"protocol_version": 1, "message_type": "STATUS",
                        "request_id": rid,
                        "worker_generation": generation,
                        "payload_length": 0})
            break
        if die_on == mt.lower():
            os._exit(1)
        if mt == "LOAD":
            if fail == "load" or not header.get("artifact_path"):
                send(sock, {"protocol_version": 1,
                            "message_type": "STATUS", "request_id": rid,
                            "worker_generation": generation,
                            "error_code": fail_code or "INVALID_MODEL",
                            "error_message": "fake refuse",
                            "payload_length": 0})
                continue
            generation = int(header.get("worker_generation", 0))
            loaded = True
            send(sock, {"protocol_version": 1, "message_type": "STATUS",
                        "request_id": rid,
                        "worker_generation": generation,
                        "payload_length": 0})
            continue
        if mt == "STATUS":
            rep = {"protocol_version": 1, "message_type": "STATUS",
                   "request_id": rid, "worker_generation": generation,
                   "payload_length": 0}
            if not loaded:
                rep["error_code"] = "MODEL_NOT_PREPARED"
            send(sock, rep)
            continue
        if mt == "INFER":
            if sleep_s:
                time.sleep(sleep_s)
            spec = header.get("tensor_spec") or {}
            if expect_shape and list(spec.get("shape", [])) \
                    != expect_shape:
                send(sock, {"protocol_version": 1,
                            "message_type": "RESULT", "request_id": rid,
                            "worker_generation": generation,
                            "error_code": "INVALID_TENSOR",
                            "error_message": "shape mismatch",
                            "payload_length": 0})
                continue
            if (fail == "infer" or not loaded
                    or int(header.get("worker_generation", -1))
                    != generation):
                send(sock, {"protocol_version": 1,
                            "message_type": "RESULT", "request_id": rid,
                            "worker_generation": generation,
                            "error_code": fail_code or "INVALID_ARGS",
                            "error_message": "fake refuse",
                            "payload_length": RESULT_BYTES},
                     b"\x00" * RESULT_BYTES)
                continue
            body = frame_bytes(frame_kind)
            send(sock, {"protocol_version": 1, "message_type": "RESULT",
                        "request_id": rid,
                        "worker_generation": generation,
                        "payload_length": len(body)},
                 body)
            continue
        send(sock, {"protocol_version": 1, "message_type": "STATUS",
                    "request_id": rid, "worker_generation": generation,
                    "error_code": "INVALID_ARGS",
                    "error_message": "unknown",
                    "payload_length": 0})


if __name__ == "__main__":
    main()
