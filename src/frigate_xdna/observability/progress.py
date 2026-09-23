"""Console progress reporting (v0.1.1 checkpoint 4).

The supervisor emits event dicts for real preparation/worker
transitions; a reporter renders them. The serve path uses
ConsoleReporter (plain flushed lines on stderr, suitable for Docker /
Portainer logs); machine-readable CLI JSON on stdout stays clean.
Tests use RecordingReporter. A None reporter is silent.

No invented percentages: phases carry names and measured elapsed
times. Failures carry phase, stable code and reason from the same
structured records status/JSON reports.
"""
from __future__ import annotations

import sys


def _elapsed(event: dict) -> str:
    return f"elapsed={event['elapsed_s']:.0f}s"


def _formatters() -> dict:
    return {
        "preparing_model":
            lambda e: f"Preparing model {e['ref']}...",
        "model_cached":
            lambda e: (f"Model {e['ref']} already prepared; no compilation."),
        "downloading_model":
            lambda e: f"Downloading model {e['ref']}...",
        "inspection_complete":
            lambda e: (f"Model inspection complete: {e['ref']}:"
                       f" {e.get('profile', '?')}"
                       f" {e.get('shape', '')}".rstrip()),
        "bf16_running":
            lambda e: (f"BF16 preparation running: {e['ref']}:"
                       f" {_elapsed(e)}"),
        "compiling":
            lambda e: (f"XDNA compilation running: {e['ref']}:"
                       f" {_elapsed(e)}"),
        "validating":
            lambda e: (f"Artifact validation running: {e['ref']}:"
                       f" {_elapsed(e)}"),
        "waiting_for_device":
            lambda e: (f"Waiting for device: {e['ref']} (compilation needs"
                       f" the NPU while a worker owns it)."),
        "worker_lost":
            lambda e: (f"Worker lost: {e['reason']}."
                       f" Daemon live; see `fxdna status`."),
        "model_prepared":
            lambda e: (f"Model prepared: {e['ref']}; waiting for Frigate at"
                       f" {e['endpoint']}"),
        "preparation_failed":
            lambda e: (f"Preparation failed: {e['ref']}:"
                       f" phase={e['phase']} code={e['code']}"
                       f"{' reason=' + e['reason'] if e.get('reason') else ''}."
                       f" See `fxdna status {e['ref']}`."),
        "worker_active":
            lambda e: (f"Worker active: generation={e['generation']}"
                       f" model={e.get('compile_key', '')[:12]}..."),
        "handshake_complete":
            lambda _e: "Frigate model handshake complete.",
        "heartbeat":
            lambda e: (f"Still preparing {e['ref']}: {e['phase']}:"
                       f" {_elapsed(e)}"),
        "resumed":
            lambda e: (f"Resuming {e['ref']}: attempt {e['attempt']}"
                       f" ({e['reason']})."),
        "retry_scheduled":
            lambda e: (f"Retry scheduled: {e['ref']}: attempt"
                       f" {e['attempt']} ({e['reason']})."),
        "fetch_failed_cached":
            lambda e: (f"Fetch failed for {e['ref']} [{e['code']}:"
                       f" {e.get('reason', '')}]; serving cached"
                       f" artifact."),
    }


def format_event(event: dict) -> str:
    """One plain log line per event. Unknown kinds are never dropped
    silently: they render generically."""
    kind = event.get("kind", "?")
    render = _formatters().get(kind)
    if render is None:
        return f"fxdna: {kind} {event}"
    return render(event)


class ConsoleReporter:
    """Render events as plain flushed stderr lines."""

    def __init__(self, stream=None):
        self.stream = stream if stream is not None else sys.stderr

    def emit(self, event: dict) -> None:
        self.stream.write(format_event(event) + "\n")
        self.stream.flush()


class RecordingReporter:
    """Test helper: collect events without printing."""

    def __init__(self):
        self.events: list[dict] = []

    def emit(self, event: dict) -> None:
        self.events.append(dict(event))

    def kinds(self) -> list[str]:
        return [e.get("kind", "?") for e in self.events]
