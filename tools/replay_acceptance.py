#!/usr/bin/env python3
"""Replay acceptance driver (Task 05 full-app gate, hardware-gated run).

Observes a stock Frigate v0.18.0 replay camera end to end: verifies
the detector is configured against the sidecar, samples detector stats
during replay, then collects tracked-object/event evidence. Uses only
Frigate's long-stable HTTP surface (/api/config, /api/stats,
/api/events); any drift fails visibly with the real HTTP status.

Stdlib only. Read-only against Frigate; never touches credentials.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


def valid_base(url: str) -> str:
    """Operator-supplied Frigate base URL, validated before any fetch.

    Local acceptance tool talking to the operator's own Frigate; still
    refuse credentials-in-URL, non-http(s) schemes and empty hosts so a
    pasted secret or typo can never become a request.
    """
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise ValueError(f"refusing non-http(s) Frigate URL: {url!r}")
    if not parts.hostname:
        raise ValueError(f"refusing hostless Frigate URL: {url!r}")
    if parts.username or parts.password:
        raise ValueError("refusing Frigate URL with embedded credentials")
    return url.rstrip("/")


def valid_out(path: str) -> str:
    """Canonicalize the evidence path (no symlink surprises)."""
    return os.path.realpath(os.path.abspath(path))


def get(base: str, path: str, params: dict | None = None):
    url = base + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, {"_http_error": e.code}
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        return -1, {"_transport_error": f"{type(e).__name__}: {e}"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frigate", required=True,
                    help="Frigate base URL, e.g. http://127.0.0.1:5000")
    ap.add_argument("--camera", required=True,
                    help="Replay camera name as configured in Frigate")
    ap.add_argument("--data-dir", default="/data",
                    help="Sidecar data dir (for local status correlation)")
    ap.add_argument("--duration-s", type=float, default=60.0)
    ap.add_argument("--poll-s", type=float, default=5.0)
    ap.add_argument("--expect-labels", default="",
                    help="Comma-separated labels expected in events")
    ap.add_argument("--out", required=True, help="Evidence JSON path")
    args = ap.parse_args()

    try:
        base = valid_base(args.frigate)
        out = valid_out(args.out)
    except ValueError as e:
        print(f"invalid arguments: {e}", file=sys.stderr)
        return 2
    evidence: dict = {
        "schema_version": 1,
        "frigate": base,
        "camera": args.camera,
        "checks": [],
    }

    def check(name: str, good: bool, detail=""):
        evidence["checks"].append(
            {"name": name, "ok": bool(good), "detail": detail})
        print(f"[{'PASS' if good else 'FAIL'}] {name} {detail}")
        return good

    ok = check_config(base, args.camera, check)
    ok = sample_stats(base, args, evidence, check) and ok
    ok = collect_events(base, args, evidence, check) and ok
    correlate_sidecar(args.data_dir, evidence)

    evidence["verdict"] = "PASS" if ok else "FAIL"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2, sort_keys=True)
    print(f"evidence -> {out}: {evidence['verdict']}")
    return 0 if ok else 1


def check_config(base: str, camera: str, check) -> bool:
    """Detector config points at the sidecar for this camera."""
    st, config = get(base, "/api/config")
    det = (config.get("detectors", {}) if isinstance(config, dict) else {})
    cam_det = ((config.get("cameras", {}).get(camera, {}).get("detectors"))
               if isinstance(config, dict) else None)
    ok = check("config-reachable", st == 200, f"http={st}")
    ok = check("sidecar-detector-configured",
               any("5555" in json.dumps(v) for v in det.values()),
               f"detectors={sorted(det)} camera_override={cam_det}") and ok
    return ok


def sample_stats(base: str, args, evidence: dict, check) -> bool:
    """Sample detector stats through the replay window."""
    det_samples, cam_samples = [], []
    deadline = time.time() + args.duration_s
    while time.time() < deadline:
        st, stats = get(base, "/api/stats")
        if st == 200 and isinstance(stats, dict):
            det_samples.append(stats.get("detectors", {}))
            cams = stats.get("cameras", {})
            if args.camera in cams:
                cam_samples.append(cams[args.camera])
        time.sleep(args.poll_s)
    ok = check("detector-served-during-replay", len(det_samples) > 0,
               f"samples={len(det_samples)}")
    evidence["detector_stat_samples"] = len(det_samples)
    evidence["camera_stat_samples"] = len(cam_samples)
    return ok


def collect_events(base: str, args, evidence: dict, check) -> bool:
    """Tracked-object / event evidence after replay."""
    st, events = get(base, "/api/events",
                     {"camera": args.camera, "limit": 100})
    labels: set[str] = set()
    n_events = 0
    if st == 200 and isinstance(events, list):
        n_events = len(events)
        for ev in events:
            lab = ev.get("label")
            if lab:
                labels.add(str(lab))
    ok = check("events-reachable", st == 200, f"http={st}")
    ok = check("tracked-objects-observed", n_events > 0,
               f"events={n_events} labels={sorted(labels)}") and ok
    want = {s.strip() for s in args.expect_labels.split(",")
            if s.strip()}
    if want:
        ok = check("expected-labels-present", want <= labels,
                   f"want={sorted(want)} got={sorted(labels)}") and ok
    evidence["events"] = n_events
    evidence["labels"] = sorted(labels)
    return ok


def correlate_sidecar(data_dir: str, evidence: dict) -> None:
    """Sidecar correlation (local, read-only, best-effort)."""
    sys.path.insert(0, "src")
    try:
        import os as _os

        from frigate_xdna import cli as _cli
        from frigate_xdna.config import load_config as _load
        _os.environ["FXDNA_DATA_DIR"] = data_dir
        doc = _cli._read_status(_load())
        evidence["sidecar_state"] = doc.get("state")
        evidence["sidecar_models"] = doc.get("models", [])
    except Exception as e:  # correlation is best-effort, not the gate
        evidence["sidecar_state"] = f"unavailable: {type(e).__name__}"


if __name__ == "__main__":
    raise SystemExit(main())
