#!/usr/bin/env python3
"""Fake fxdna-worker binary (hardware-free NativeWorker tests only).

Speaks the private supervisor<->native IPC framing ([4-byte BE header
length][JSON header][payload] over an inherited socketpair fd passed
via --fd N or FXDNA_NATIVE_FD) with canned STATUS/RESULT replies.
Never touches hardware. Behavior knobs via argv (the child env is
minimal by design, so no configuration passes through environ):

  --fail load|infer   refuse LOAD / INFER with an error
  --sleep S           sleep S seconds before each INFER reply
  --exit N            exit(N) immediately (death simulation)

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
        if a == name and i + 2 < len(argv) + 1:
            return argv[i + 2]
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


def main():
    import socket as _socket
    fd = parse_fd(sys.argv)
    sock = _socket.socket(fileno=fd)
    generation = 0
    loaded = False
    fail = parse_arg(sys.argv, "--fail", "") or ""
    sleep_s = float(parse_arg(sys.argv, "--sleep", "0") or 0)
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
            send(sock, {"protocol_version": 1, "message_type": "STATUS",
                        "request_id": rid,
                        "worker_generation": generation,
                        "payload_length": 0})
            break
        if mt == "LOAD":
            if fail == "load" or not header.get("artifact_path"):
                send(sock, {"protocol_version": 1,
                            "message_type": "STATUS", "request_id": rid,
                            "worker_generation": generation,
                            "error_code": "INVALID_MODEL",
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
            if (fail == "infer" or not loaded
                    or int(header.get("worker_generation", -1))
                    != generation):
                send(sock, {"protocol_version": 1,
                            "message_type": "RESULT", "request_id": rid,
                            "worker_generation": generation,
                            "error_code": "INVALID_ARGS",
                            "error_message": "fake refuse",
                            "payload_length": RESULT_BYTES},
                     b"\x00" * RESULT_BYTES)
                continue
            send(sock, {"protocol_version": 1, "message_type": "RESULT",
                        "request_id": rid,
                        "worker_generation": generation,
                        "payload_length": RESULT_BYTES},
                 b"\x01" * RESULT_BYTES)
            continue
        send(sock, {"protocol_version": 1, "message_type": "STATUS",
                    "request_id": rid, "worker_generation": generation,
                    "error_code": "INVALID_ARGS",
                    "error_message": "unknown",
                    "payload_length": 0})


if __name__ == "__main__":
    main()
