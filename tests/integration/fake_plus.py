"""Fake Frigate+ API and download servers (integration tests only).

Programmable in-process HTTP servers speaking the narrow read-only flow:
token exchange (basic auth), model metadata, signed URL, model bytes.
Records every Authorization header seen to prove none leaks to download
hosts. Loopback only.
"""
from __future__ import annotations

import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TEST_KEY_ID = "abcd1234-ef56-7890-abcd-ef1234567890"
TEST_KEY_SECRET = "a" * 40
TEST_KEY = f"{TEST_KEY_ID}:{TEST_KEY_SECRET}"


class FakePlusState:
    def __init__(self):
        self.token_calls = 0
        self.token_expires_in_s = 3600
        self.token_mode = "ok"  # ok | 401 | malformed
        self.models: dict[str, dict] = {}
        self.signed_urls: dict[str, str] = {}
        self.model_status: dict[str, int] = {}
        self.flaky_counts: dict[str, int] = {}
        self.auth_headers_seen: list[str] = []
        self.download_auth_headers: list = []


class FakePlusHandler(BaseHTTPRequestHandler):
    server_version = "FakePlus/1"
    protocol_version = "HTTP/1.1"

    def _send_json(self, code, obj, length=True):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        if length:
            self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if length:
            self.wfile.write(body)

    def log_message(self, *a):
        pass

    def do_GET(self):
        st: FakePlusState = self.server.state
        auth = self.headers.get("Authorization", "")
        if self.path == "/v1/auth/token":
            st.token_calls += 1
            want = "Basic " + base64.b64encode(
                f"{TEST_KEY_ID}:{TEST_KEY_SECRET}".encode()).decode()
            if st.token_mode == "401" or auth != want:
                return self._send_json(401, {"error": "bad credentials"})
            if st.token_mode == "malformed":
                return self._send_json(200, {"nope": True})
            return self._send_json(200, {
                "accessToken": "test-token-1",
                "expires": time.time() + st.token_expires_in_s})
        if not auth.startswith("Bearer "):
            return self._send_json(401, {"error": "missing bearer"})
        st.auth_headers_seen.append(auth)
        if self.path.startswith("/v1/model/") and self.path.endswith("/signed_url"):
            model_id = self.path[len("/v1/model/"):-len("/signed_url")]
            code = st.model_status.get(model_id, 200)
            if code != 200:
                return self._send_json(code, {"error": "nope"})
            return self._send_json(200, {"url": st.signed_urls[model_id]})
        if self.path.startswith("/v1/model/"):
            model_id = self.path[len("/v1/model/"):]
            code = st.model_status.get(model_id, 200)
            if code == "FLAKY_ONCE":
                n = st.flaky_counts.get(model_id, 0)
                st.flaky_counts[model_id] = n + 1
                if n == 0:
                    self.send_response(429)
                    self.send_header("Retry-After", "0")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                code = 200
            if code == 429:
                self.send_response(429)
                self.send_header("Retry-After", "1")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if code != 200:
                return self._send_json(code, {"error": "nope"})
            meta = st.models.get(model_id)
            if meta == "MALFORMED":
                body = b"{{{bad"
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            return self._send_json(200, meta)
        return self._send_json(404, {"error": "unknown"})


class FakeDownloadHandler(BaseHTTPRequestHandler):
    server_version = "FakeDL/1"
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_GET(self):
        st = self.server.state
        st.download_auth_headers.append(self.headers.get("Authorization"))
        try:
            self._serve()
        except (OSError, ConnectionResetError, BrokenPipeError):
            pass  # client went away (timeouts/bounds); not a server bug

    def _serve(self):
        st = self.server.state
        mode = st.download_modes.get(self.path, "ok")
        if mode == "slow":
            time.sleep(15)
            return
        if mode == "404":
            body = b"gone"
            self.send_response(404)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if mode == "redirect-http":
            self.send_response(302)
            self.send_header("Location", st.redirect_target)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if mode == "redirect-private":
            self.send_response(302)
            self.send_header("Location", st.redirect_target)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if mode == "huge":
            self.send_response(200)
            self.send_header("Content-Length", str(300 * 1024 * 1024))
            self.end_headers()
            self.wfile.write(b"\x00" * (2 * 1024 * 1024))
            return
        body = st.download_bodies.get(self.path, b"")
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class FakeDownloadState:
    def __init__(self):
        self.download_bodies: dict[str, bytes] = {}
        self.download_modes: dict[str, str] = {}
        self.redirect_target = ""
        self.download_auth_headers: list = []


def start_server(handler, state, host="127.0.0.1", port=0):
    srv = ThreadingHTTPServer((host, port), handler)
    srv.state = state
    srv.daemon_threads = True
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv
