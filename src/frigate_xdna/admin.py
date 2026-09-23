"""Private administrative Unix socket (INTERFACES.md §5.2).

Newline-delimited JSON requests/responses on a filesystem-access-controlled
socket. No public network API. The daemon is the only registry writer;
online CLI commands are thin clients of this socket.
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time

SOCKET_NAME = "control.sock"
MAX_MESSAGE_BYTES = 64 * 1024


def socket_path(data_dir: str) -> str:
    return os.path.join(data_dir, "control.sock")


class AdminServer(threading.Thread):
    def __init__(self, data_dir: str, handler):
        super().__init__(daemon=True)
        self.path = socket_path(data_dir)
        self.handler = handler
        self._stop_event = threading.Event()
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass

    def run(self):
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(self.path)
        os.chmod(self.path, 0o700)
        srv.listen(8)
        srv.settimeout(0.2)
        try:
            while not self._stop_event.is_set():
                try:
                    conn, _ = srv.accept()
                except TimeoutError:
                    continue
                # One thread per connection: a long `wait` must not wedge
                # other admin commands behind it (SF2).
                t = threading.Thread(target=self._serve_one, args=(conn,),
                                     daemon=True)
                t.start()
        finally:
            srv.close()
            self._cleanup_socket()

    def _serve_one(self, conn):
        with conn:
            conn.settimeout(60.0)
            buf = b""
            try:
                while b"\n" not in buf:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    buf += chunk
                    if len(buf) > MAX_MESSAGE_BYTES:
                        break
                line, _, _ = buf.partition(b"\n")
                req = json.loads(line.decode() or "{}")
            except (ValueError, OSError):
                resp = {"ok": False, "error_code": "INVALID_ARGS",
                        "message": "malformed admin request"}
            else:
                try:
                    resp = self.handler(req) or {}
                    resp.setdefault("ok", True)
                except Exception as e:  # never drop the connection
                    resp = {"ok": False,
                            "error_code": getattr(
                                e, "error_code", "INTERNAL"),
                            "message": str(e)[:500]}
            try:
                conn.sendall(json.dumps(resp).encode() + b"\n")
            except OSError:
                pass

    def stop(self):
        self._stop_event.set()

    def _cleanup_socket(self):
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass


def admin_call(data_dir: str, request: dict, timeout_s: float = 10.0) -> dict:
    """Single admin request; raises on transport failure."""
    from .errors import NOT_READY, FxdnaError
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout_s)
    try:
        try:
            deadline = time.monotonic() + 2.0
            while True:
                try:
                    s.connect(socket_path(data_dir))
                    break
                except (ConnectionRefusedError, FileNotFoundError):
                    # Server thread starts asynchronously: the path may
                    # be unbound yet, or bound but not listening
                    # (bind->listen window). Retry briefly, then report
                    # unreachable as before.
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.05)
        except OSError as e:
            raise FxdnaError(NOT_READY, "DAEMON_UNREACHABLE",
                             f"admin socket unreachable: {e.strerror or e}")
        try:
            s.sendall(json.dumps(request).encode() + b"\n")
            buf = b""
            while b"\n" not in buf:
                chunk = s.recv(4096)
                if not chunk:
                    break
                buf += chunk
        except OSError as e:
            raise FxdnaError(NOT_READY, "DAEMON_UNREACHABLE",
                             f"admin call failed: {e.strerror or e}")
        line, _, _ = buf.partition(b"\n")
        return json.loads(line.decode() or "{}")
    finally:
        s.close()
