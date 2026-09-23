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


def format_event(event: dict) -> str:
    """One plain log line per event. Unknown kinds are never dropped
    silently: they render generically."""
    kind = event.get("kind", "?")
    if kind == "preparing_model":
        return f"Preparing model {event['ref']}..."
    if kind == "model_cached":
        return (f"Model {event['ref']} already prepared"
                f"{' (' + event['note'] + ')' if event.get('note') else ''};"
                f" no compilation.")
    if kind == "downloading_model":
        return f"Downloading model {event['ref']}..."
    if kind == "inspection_complete":
        return (f"Model inspection complete: {event['ref']}:"
                f" {event.get('profile', '?')}"
                f" {event.get('shape', '')}".rstrip())
    if kind == "bf16_running":
        return (f"BF16 preparation running: {event['ref']}:"
                f" elapsed={event['elapsed_s']:.0f}s")
    if kind == "compiling":
        return (f"XDNA compilation running: {event['ref']}:"
                f" elapsed={event['elapsed_s']:.0f}s")
    if kind == "validating":
        return (f"Artifact validation running: {event['ref']}:"
                f" elapsed={event['elapsed_s']:.0f}s")
    if kind == "waiting_for_device":
        return (f"Waiting for device: {event['ref']} (compilation needs"
                f" the NPU while a worker owns it).")
    if kind == "worker_lost":
        return (f"Worker lost: {event['reason']}."
                f" Daemon live; see `fxdna status`.")
    if kind == "model_prepared":
        return (f"Model prepared: {event['ref']}; waiting for Frigate at"
                f" {event['endpoint']}")
    if kind == "preparation_failed":
        return (f"Preparation failed: {event['ref']}:"
                f" phase={event['phase']} code={event['code']}"
                f"{' reason=' + event['reason'] if event.get('reason') else ''}."
                f" See `fxdna status {event['ref']}`.")
    if kind == "worker_active":
        return (f"Worker active: generation={event['generation']}"
                f" model={event.get('compile_key', '')[:12]}...")
    if kind == "handshake_complete":
        return "Frigate model handshake complete."
    if kind == "heartbeat":
        return (f"Still preparing {event['ref']}: {event['phase']}:"
                f" elapsed={event['elapsed_s']:.0f}s")
    return f"fxdna: {kind} {event}"


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
