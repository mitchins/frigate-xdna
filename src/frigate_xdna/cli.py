"""`fxdna` command-line interface (docs/INTERFACES.md §1, normative).

Online commands talk to the daemon over the private admin socket; read-only
commands open the registry directly without taking the writer lock.
`prepare --standalone` takes the same exclusive lock as the daemon and
refuses it when a daemon owns the data directory.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time

from . import __version__
from .admin import admin_call, socket_path
from .build_identity import format_identity, get_build_identity
from .cache.registry import Registry
from .config import load_config
from .errors import (
    INVALID_ARGS,
    NOT_IMPLEMENTED,
    NOT_READY,
    SUCCESS,
    FxdnaError,
)
from .supervisor import Supervisor

COMMANDS = (
    "serve", "prepare", "status", "wait", "activate", "cache",
    "doctor", "health", "recover", "diagnose",
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fxdna",
        description="Self-contained XDNA detector sidecar for Frigate "
                    "(stock Frigate unchanged).",
    )
    from .build_identity import version_string
    p.add_argument("--version", action="version", version=version_string())
    sub = p.add_subparsers(dest="command", required=True, metavar="COMMAND")

    sub.add_parser("serve", help="Start manager, ZMQ frontend and supervision.")

    pp = sub.add_parser("prepare", help="Register durable preparation requests.")
    pp.add_argument("refs", nargs="+", help="plus://ID, local ONNX or local RAI.")
    pp.add_argument("--wait", action="store_true")
    pp.add_argument("--refresh", action="store_true")
    pp.add_argument("--maintenance", action="store_true")
    pp.add_argument("--standalone", action="store_true")
    pp.add_argument("--descriptor", default=None)
    pp.add_argument("--wire-name", default=None)

    sp = sub.add_parser("status", help="Show model/job/service state (no NPU).")
    sp.add_argument("ref", nargs="?", default=None)
    sp.add_argument("--json", action="store_true")
    sp.add_argument("--show-identifiers", action="store_true",
                    help="Reveal raw Plus IDs/full digests (owning machine "
                         "only; default is pseudonymized).")

    dp_diag = sub.add_parser(
        "diagnose",
        help="Export a redacted support bundle (no keys/IDs/URLs/bytes).")
    dp_diag.add_argument("--out", required=True,
                         help="Destination directory (created if missing).")
    dp_diag.add_argument("--show-identifiers", action="store_true",
                         help="Reveal raw Plus IDs/full digests (owning "
                              "machine only; default is pseudonymized).")
    dp_diag.add_argument("--history-lines", type=int, default=200,
                         help="Journal history tail lines to include.")

    wp = sub.add_parser("wait", help="Wait for a model state via admin socket.")
    wp.add_argument("ref")
    wp.add_argument("--state", default="prepared",
                    choices=("prepared", "verified", "active"))
    wp.add_argument("--timeout", type=float, default=1800.0)

    ap = sub.add_parser("activate", help="Activate an already prepared model.")
    ap.add_argument("ref")
    ap.add_argument("--maintenance", action="store_true")
    ap.add_argument("--wait", action="store_true",
                    help="Wait until the ref projects ACTIVE (bounded).")
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="Bound for --wait (seconds).")

    cp = sub.add_parser("cache", help="Inspect/prune the content cache.")
    csub = cp.add_subparsers(dest="cache_command", required=True)
    cl = csub.add_parser("list")
    cl.add_argument("--json", action="store_true")
    pr = csub.add_parser("prune")
    pr.add_argument("--apply", action="store_true")
    pr.add_argument("--max-bytes", type=int, default=None)

    dp = sub.add_parser("doctor", help="Passive checks; --hardware needs lease.")
    dp.add_argument("--hardware", action="store_true")
    dp.add_argument("--json", action="store_true")

    hp = sub.add_parser("health", help="Supervisor liveness (no NPU probe).")
    hp.add_argument("--ready", action="store_true")

    rp = sub.add_parser("recover", help="Clear a recoverable inhibition.")
    rp.add_argument("ref")
    rp.add_argument("--acknowledge", action="store_true", required=True)
    return p


def cmd_health(config, ready: bool) -> int:
    """Liveness (default) or readiness (--ready) from the real daemon.

    A dead daemon, stale socket, or failed bounded admin request is
    liveness-false with a nonzero exit — never success with
    alive=false. Readiness requires a loaded, alive, permitted worker
    with no inhibition; it never opens the NPU, compiles, or infers.
    """
    try:
        resp = admin_call(config.data_dir, {"command": "health"})
        if not resp.get("ok", True) or "health" not in resp:
            raise FxdnaError(NOT_READY, "DAEMON_UNREACHABLE",
                             "bounded admin request failed")
        doc = resp["health"]
    except FxdnaError:
        doc = {"alive": False,
               "reason": "daemon unreachable (absent or stale socket)"}
    out = {"schema_version": 1, "alive": bool(doc.get("alive", True)),
           "ready": bool(doc.get("ready", False)),
           "reason": doc.get("reason", ""),
           "worker": doc.get("worker"),
           "inhibition": doc.get("inhibition")}
    print(json.dumps(out, indent=2, sort_keys=True))
    if not out["alive"]:
        return NOT_READY
    if ready and not out["ready"]:
        return NOT_READY
    return SUCCESS


def _daemon_alive(data_dir: str, timeout_s: float = 2.0) -> bool:
    """Liveness is a bounded connection probe, not a socket-file check.

    A stale control.sock after a crash must not read as a live daemon.
    """
    import socket as _socket
    path = socket_path(data_dir)
    s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    s.settimeout(timeout_s)
    try:
        s.connect(path)
        return True
    except OSError:
        return False
    finally:
        s.close()


def _read_status(config, ref=None, show_identifiers: bool = False) -> dict:
    """Read-only status: no lock, no NPU, works with or without a daemon.

    Never creates state: a missing registry means STARTING, not an error
    (the redaction key is only ensured once a registry exists). By
    default Plus IDs are HMAC pseudonyms and digests abbreviated;
    --show-identifiers reveals raw values on the owning machine only.
    """
    db = os.path.join(config.data_dir, "registry.sqlite3")
    build = get_build_identity()
    if not os.path.isfile(db):
        return {"schema_version": 1, "service": "frigate-xdna",
                "version": build["version"], "build": build,
                "state": "STARTING",
                "active": None, "models": [],
                "note": "no registry yet; daemon not started"}
    from .model_view import project_ref
    from .observability.redact import load_or_create_key
    key = None if show_identifiers else load_or_create_key(config.data_dir)
    view = _StatusView(show_identifiers, key)
    # SERVING is reported only when a live manager owns the directory;
    # otherwise STARTING (or INHIBITED) — never inference readiness.
    # Without a live manager no ACTIVE projection is provable: the
    # persisted record stays history, `active` stays null.
    daemon = _daemon_alive(config.data_dir)
    reg = Registry(db, read_only=True)
    try:
        if ref:
            from .models.refs import parse_ref
            parsed = parse_ref(ref)["ref"]
            rec = reg.get_ref(parsed)
            models = [view.project(project_ref(reg, parsed, None))
                      ] if rec else []
        else:
            models = [view.project(project_ref(reg, r[0], None))
                      for r in reg.query("SELECT ref FROM model_refs")]
        inhibition = reg.get_state("inhibition")
        state = "SERVING" if daemon else (
            "INHIBITED" if inhibition else "STARTING")
        ready = False
        reason = ("daemon not running" if not daemon
                  else "no live worker projection offline")
        return {"schema_version": 1, "service": "frigate-xdna",
                "version": build["version"], "build": build,
                "state": state,
                "active": None, "worker": None,
                "ready": ready, "reason": reason,
                "models": models,
                "inhibition": inhibition}
    finally:
        reg.close()


class _StatusView:
    """Raw-or-redacted projection for status output (one policy object)."""

    def __init__(self, show_identifiers: bool, key: bytes | None):
        self.show = show_identifiers
        self.key = key

    def project(self, view: dict) -> dict:
        """Redact identity fields of a projected model view; phase,
        elapsed, verified, error and key pass through (no secrets)."""
        redacted = self.triple(view["ref"], view.get("source_sha256"),
                               view["state"])
        redacted.update({k: view.get(k) for k in
                         ("phase", "elapsed_s", "verified", "error_code",
                          "compile_key", "attempts", "retryable",
                          "failure")})
        return redacted

    def triple(self, raw_ref: str, source_sha: str | None, state: str,
               ) -> dict:
        from .observability.redact import abbreviate_digest, sanitize_ref
        if self.show:
            return {"ref": raw_ref, "source_sha256": source_sha or "",
                    "state": state}
        assert self.key is not None
        return {"ref": sanitize_ref(raw_ref, self.key),
                "source_sha256": abbreviate_digest(source_sha or ""),
                "state": state}

    def active(self, value) -> object:
        from .observability.redact import sanitize_obj, sanitize_ref
        if isinstance(value, dict):
            # Machine struct {compile_key, worker_generation,
            # serving_digest}: content hashes, no Plus IDs to redact.
            return value if self.show else sanitize_obj(value, self.key)
        if value and not self.show:
            assert self.key is not None
            return sanitize_ref(value, self.key)
        return value


def cmd_diagnose(config, args) -> int:
    """Export a redacted support bundle (Task 05 external-privacy gate).

    Collects status/config/journal/registry-inventory through the
    sanitizer. NEVER includes: redaction.key, the raw registry DB, model
    bytes (.rai/.onnx), key/secret material, bearer tokens, signed URLs.
    Raw Plus IDs appear only with --show-identifiers (owning machine).
    """
    from .observability.redact import load_or_create_key, sanitize_obj
    # Canonicalize the operator-chosen directory (no symlink surprises);
    # writing where the local operator points is the command's purpose.
    out = os.path.realpath(os.path.abspath(args.out))
    os.makedirs(out, exist_ok=True)
    show = args.show_identifiers
    key = None if show else load_or_create_key(config.data_dir)
    writer = _BundleWriter(out)

    doc = _read_status(config, show_identifiers=show)
    writer.write("status.json", doc if show else sanitize_obj(doc, key))
    writer.write("config.json", sanitize_obj(config.redacted(), key) if key
                 else config.redacted())
    writer.write("journal-current.json",
                 _diagnose_current(config, show, key))
    writer.write("journal-history.tail.jsonl",
                 _diagnose_history(config, args.history_lines, key))
    refs_out, arts_out = _diagnose_inventory(config, show, key)
    writer.write("registry-refs.json", refs_out)
    writer.write("cache-inventory.json", arts_out)
    writer.write("manifest.json", {
        "schema_version": 1,
        "generator": f"fxdna {__version__} diagnose",
        "identifiers": "raw" if show else "pseudonymized",
        "note": "Redacted bundle: no keys/tokens/signed URLs/model "
                "bytes/raw registry. redaction.key never exported.",
        "files": sorted(writer.files),
    })
    print(json.dumps({"schema_version": 1, "out": out,
                      "identifiers": "raw" if show else "pseudonymized",
                      "files": sorted(
                          writer.files + ["manifest.json"])},
                     indent=2, sort_keys=True))
    return SUCCESS


class _BundleWriter:
    """One JSON/text file writer for the diagnose bundle directory."""

    def __init__(self, out: str):
        self.out = out
        self.files: list[str] = []

    def write(self, name: str, payload) -> None:
        path = os.path.join(self.out, name)
        if isinstance(payload, (dict, list)):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(payload)
        self.files.append(name)


def _diagnose_current(config, show: bool, key: bytes | None):
    """Sanitized journal current.json ({} when absent)."""
    from .observability.redact import sanitize_obj
    cur_path = os.path.join(config.data_dir, "operations", "current.json")
    if not os.path.isfile(cur_path):
        return {}
    with open(cur_path, encoding="utf-8") as f:
        cur = json.load(f)
    return cur if show else sanitize_obj(cur, key)


def _diagnose_history(config, history_lines: int, key: bytes | None) -> str:
    """Bounded, line-sanitized journal history tail."""
    from .observability.redact import sanitize_text
    hist_path = os.path.join(config.data_dir, "operations", "history.jsonl")
    if not os.path.isfile(hist_path):
        return ""
    with open(hist_path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    tail = lines[-max(history_lines, 0):]
    if key:
        tail = [sanitize_text(line, key) for line in tail]
    return "\n".join(tail) + "\n" if tail else ""


def _diagnose_inventory(config, show: bool, key: bytes | None):
    """Sanitized (refs, artifacts) inventory: refs aliased, source
    digests abbreviated, compile keys (content hashes) kept full."""
    from .observability.redact import abbreviate_digest, sanitize_ref
    refs_out: list[dict] = []
    arts_out: list[dict] = []
    db = os.path.join(config.data_dir, "registry.sqlite3")
    if not os.path.isfile(db):
        return refs_out, arts_out
    reg = Registry(db, read_only=True)
    try:
        for r in reg.query("SELECT ref, kind, state FROM model_refs"):
            refs_out.append({
                "ref": r[0] if show else sanitize_ref(r[0], key),
                "kind": r[1], "state": r[2]})
        for r in reg.query(
                "SELECT compile_key, source_sha256, artifact_size,"
                " recipe_id FROM artifacts"):
            arts_out.append({
                "compile_key": r[0],
                "source_sha256": r[1] if show else
                abbreviate_digest(r[1] or ""),
                "bytes": r[2], "recipe": r[3]})
    finally:
        reg.close()
    return refs_out, arts_out


def _install_serve_handlers(handler) -> None:
    """Install SIGTERM/SIGINT serve handlers (PID-1 safe). Split out so
    unit tests can verify registration without sending real signals."""
    signal.signal(signal.SIGTERM, handler)
    try:
        signal.signal(signal.SIGINT, handler)
    except (OSError, ValueError):
        pass


# Admin error_code -> CLI exit mapping (mirrors _TERMINAL_EXIT for the
# daemon path so online/standalone exits agree — SF1).
_ADMIN_EXIT = {
    "ACQUISITION_FAILED": 4, "AUTH_FAILED": 4, "DOWNLOAD_FAILED": 4,
    "UNSUPPORTED_CONTRACT": 5,
    "COMPILE_FAILED": 6, "RESOURCE_EXCEEDED": 6,
    "VALIDATION_FAILED": 7,
    "DEVICE_BUSY": 8, "DEVICE_FAULT": 8, "SAFETY_INHIBITED": 8,
    "QUARANTINED": 8, "ACTIVATION_FAILED": 8,
    "CACHE_CORRUPT": 10,
    "OWNERSHIP_CONFLICT": 9, "MODEL_IN_USE": 9,
    "SOURCE_CHANGED": 2, "INTERRUPTED": 3,
    "NOT_PREPARED": 3, "UNKNOWN_JOB": 3, "WAIT_TIMEOUT": 3,
    "DAEMON_UNREACHABLE": 3,
}


def _admin_or_raise(data_dir, request):
    resp = admin_call(data_dir, request)
    if not resp.get("ok", True):
        code = resp.get("error_code", "INVALID_ARGS")
        raise FxdnaError(_ADMIN_EXIT.get(code, INVALID_ARGS), code,
                         resp.get("message", "daemon request failed"))
    return resp


# INTERFACES.md §1 exit-code mapping for terminal preparation states.
_TERMINAL_EXIT = {
    "AUTH_FAILED": 4, "DOWNLOAD_FAILED": 4,
    "UNSUPPORTED_CONTRACT": 5,
    "COMPILE_FAILED": 6, "RESOURCE_EXCEEDED": 6,
    "VALIDATION_FAILED": 7,
    "DEVICE_BUSY": 8, "DEVICE_FAULT": 8, "SAFETY_INHIBITED": 8,
    "QUARANTINED": 8,
    "CACHE_CORRUPT": 10,
    "SOURCE_CHANGED": 2, "INTERRUPTED": 3,
}


def _terminal_exit(state: str) -> int:
    return _TERMINAL_EXIT.get(state, NOT_READY)


def run_preflight(config, header: bool = True,
                  extra_refs: tuple = ()) -> int | None:
    """Deployment preflight before expensive work. Returns an exit
    code when a blocking check fails, else None (servable)."""
    from .deploy_checks import blocking_failures, run_preflight
    if header:
        print("Checking deployment requirements...", file=sys.stderr)
    checks = run_preflight(config, extra_refs=extra_refs)
    bad = blocking_failures(checks)
    if not bad:
        return None
    for check in bad:
        print(f"fxdna: requirement failed: {check['message']}"
              f" [{check.get('error_code', 'ERROR')}]",
              file=sys.stderr)
    return bad[0]["code"]


def cmd_serve(config) -> int:
    if "FXDNA_ENDPOINT" not in os.environ:
        # SPEC §5.1: loopback by default for native development; the image
        # default (bind-all) applies only when explicitly configured.
        config = config.__class__(**{**config.__dict__,
                                     "endpoint": "tcp://127.0.0.1:5555"})
    failed = run_preflight(config)
    if failed is not None:
        return failed
    from .observability.progress import ConsoleReporter
    sup = Supervisor(config, reporter=ConsoleReporter())
    sup.start_admin()
    # Start ROUTER frontend (Task 04) if endpoint is configured
    zfrontend = None
    zthread = None
    zloop = None
    if config.endpoint:
        import asyncio as _asyncio
        import threading as _thr

        from .transport.frigate_zmq import FrigateZmqFrontend

        zfrontend = FrigateZmqFrontend(sup, config.endpoint)
        sup.frontend = zfrontend
        zloop = _asyncio.new_event_loop()
        ready = _thr.Event()
        exc: list[Exception] = []

        def _run_zmq():
            _asyncio.set_event_loop(zloop)
            try:
                zloop.run_until_complete(zfrontend.start())
                ready.set()
                zloop.run_forever()
            except Exception as e:
                exc.append(e)
                ready.set()

        zthread = _thr.Thread(target=_run_zmq, daemon=True)
        zthread.start()
        ok = ready.wait(timeout=5.0)
        if not ok or exc:
            # Startup did not signal readiness: close frontend, wait for
            # thread, clean up supervisor before propagating
            try:
                if zfrontend.sock:
                    zfrontend.sock.close(linger=0)
            except Exception:
                pass
            # Ensure synchronous startup/bind has completed; wait for thread
            zthread.join(timeout=5.0)
            if exc:
                try:
                    sup.stop()
                finally:
                    raise exc[0]
            try:
                sup.stop()
            finally:
                raise RuntimeError("ZMQ frontend startup timeout or failed to bind")
        if not zfrontend.sock:
            try:
                sup.stop()
            finally:
                raise RuntimeError("ZMQ frontend failed to bind")
    for ref in config.models:
        try:
            sup.prepare(ref)
        except FxdnaError as e:
            print(f"fxdna: startup prepare {ref}: {e.message} "
                  f"[{e.error_code}]", file=sys.stderr)
    print(format_identity(get_build_identity()), file=sys.stderr)
    print(f"fxdna: serving endpoint={config.endpoint} "
          f"data={sup.data_dir}", file=sys.stderr)
    # Explicit handlers: as container PID 1 the default SIGTERM action
    # is ignored by the kernel, so without these the process never
    # stops gracefully (grace timeout -> SIGKILL -> exit 137) and the
    # worker is never retired cleanly. Handlers only set the event;
    # teardown stays in the finally below (bounded, ordered).
    stop_event = threading.Event()

    def _on_signal(signum, _frame):
        print(f"fxdna: signal {signum}, stopping", file=sys.stderr)
        stop_event.set()

    _install_serve_handlers(_on_signal)
    try:
        while not stop_event.is_set():
            sup.pump(0.2)
            stop_event.wait(0.2)
    finally:
        if zfrontend is not None and zloop is not None and zthread is not None:
            try:
                import asyncio as _asyncio
                fut = _asyncio.run_coroutine_threadsafe(zfrontend.stop(), zloop)
                fut.result(timeout=5.0)
                zloop.call_soon_threadsafe(zloop.stop)
                zthread.join(timeout=5.0)
            except Exception:
                pass
        sup.stop()
    return SUCCESS


def cmd_prepare(config, args) -> int:
    results = []
    if _daemon_alive(config.data_dir) and not args.standalone:
        for ref in args.refs:
            resp = _admin_or_raise(
                config.data_dir,
                {"command": "prepare", "ref": ref,
                 "refresh": args.refresh,
                 "maintenance": args.maintenance,
                 "descriptor": args.descriptor,
                 "wire_name": args.wire_name})
            job = resp["job"]
            if args.wait:
                # Poll ref state (same loop as `wait`); the daemon pumps.
                # No status print here: prepare emits one results document.
                rc = _wait_for_state(config, ref, "PREPARED", 1800.0,
                                     print_success=False)
                if rc != SUCCESS:
                    return rc
                job = _admin_or_raise(
                    config.data_dir,
                    {"command": "status", "ref": ref})["status"]
            results.append(job)
    else:
        failed = run_preflight(config, extra_refs=tuple(args.refs))
        if failed is not None:
            return failed
        sup = Supervisor(config)  # takes exclusive lock or raises
        try:
            for ref in args.refs:
                job = sup.prepare(ref, descriptor_path=args.descriptor,
                                  wire_name=args.wire_name,
                                  refresh=args.refresh,
                                  maintenance=args.maintenance)
                if args.wait and job.get("job_uuid"):
                    job = sup.wait_job(job["job_uuid"], 1800.0)
                results.append(job)
        finally:
            sup.stop()
    print(json.dumps({"schema_version": 1, "results": results}, indent=2,
                     sort_keys=True))
    return SUCCESS


def _wait_for_state(config, ref: str, want: str, timeout: float,
                    print_success: bool = True) -> int:
    from .model_view import satisfies
    daemon = _daemon_alive(config.data_dir)
    deadline = time.monotonic() + timeout
    while True:
        if daemon:
            doc = _admin_or_raise(
                config.data_dir,
                {"command": "status", "ref": ref})["status"]
        else:
            doc = _read_status(config, ref)
        states = [m["state"] for m in doc.get("models", [])]
        # Ranked satisfaction: ACTIVE implies VERIFIED implies
        # PREPARED (activation records verification), so waiting for
        # a lesser state succeeds once a better one projects.
        if states and satisfies(states[0], want):
            if print_success:
                print(json.dumps(doc, indent=2, sort_keys=True))
            return SUCCESS
        if states and states[0] in _TERMINAL_EXIT:
            print(json.dumps(doc, indent=2, sort_keys=True))
            return _terminal_exit(states[0])
        if time.monotonic() >= deadline:
            print(f"fxdna: timed out waiting for {ref}={want}",
                  file=sys.stderr)
            return NOT_READY
        time.sleep(0.5)


def _recover_exit(resp: dict) -> int:
    """Recover exit truthfulness: success only when something was
    cleared or requeued. A safety refusal exits 8 (operator scripts
    must not proceed as though the inhibition lifted); any other
    no-op exits 3."""
    if resp.get("cleared") or resp.get("requeued"):
        return SUCCESS
    if resp.get("refused_safety"):
        from .errors import DEVICE_UNAVAILABLE
        return DEVICE_UNAVAILABLE
    return NOT_READY


def _wait_live_active(sup, ref: str, timeout: float) -> int:
    """Bounded ACTIVE wait against a live standalone supervisor."""
    deadline = time.monotonic() + timeout
    while True:
        doc = sup.status(ref)
        states = [m["state"] for m in doc.get("models", [])]
        if states and states[0] == "ACTIVE":
            print(json.dumps(doc, indent=2, sort_keys=True))
            return SUCCESS
        if states and states[0] in _TERMINAL_EXIT:
            print(json.dumps(doc, indent=2, sort_keys=True))
            return _terminal_exit(states[0])
        if time.monotonic() >= deadline:
            print(f"fxdna: timed out waiting for {ref}=ACTIVE",
                  file=sys.stderr)
            return NOT_READY
        time.sleep(0.5)


def cmd_wait(config, args) -> int:
    return _wait_for_state(config, args.ref, args.state.upper(),
                           args.timeout)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config()
    except FxdnaError as e:
        print(f"fxdna: error: {e.message} [{e.error_code}]", file=sys.stderr)
        return e.exit_code
    try:
        if args.command == "serve":
            return cmd_serve(config)
        if args.command == "prepare":
            return cmd_prepare(config, args)
        if args.command == "status":
            doc = _read_status(config, args.ref, args.show_identifiers)
            if args.json:
                print(json.dumps(doc, indent=2, sort_keys=True))
            else:
                for m in doc.get("models", []):
                    print(f"{m['ref']}: {m['state']}")
                if not doc.get("models"):
                    print("(no models registered)")
            return SUCCESS
        if args.command == "diagnose":
            return cmd_diagnose(config, args)
        if args.command == "wait":
            return cmd_wait(config, args)
        if args.command == "activate":
            if _daemon_alive(config.data_dir):
                resp = _admin_or_raise(
                    config.data_dir,
                    {"command": "activate", "ref": args.ref,
                     "maintenance": args.maintenance})
                print(json.dumps(resp, indent=2, sort_keys=True))
                if not args.wait:
                    return SUCCESS
                return _wait_for_state(config, args.ref, "ACTIVE",
                                       args.timeout)
            sup = Supervisor(config)
            try:
                print(json.dumps(sup.activate(args.ref, args.maintenance)))
                if not args.wait:
                    return SUCCESS
                # Standalone wait polls the live supervisor (still
                # owning its worker); the read-only path cannot
                # project ACTIVE after stop().
                return _wait_live_active(sup, args.ref, args.timeout)
            finally:
                sup.stop()
        if args.command == "cache":
            if args.cache_command == "list":
                if _daemon_alive(config.data_dir):
                    resp = _admin_or_raise(config.data_dir,
                                           {"command": "cache_list"})
                    print(json.dumps({"schema_version": 1,
                                      "entries": resp.get("entries", []),
                                      "sources": resp.get("sources", []),
                                      "pins": resp.get("pins", [])},
                                     indent=2, sort_keys=True))
                    return SUCCESS
                db = os.path.join(config.data_dir, "registry.sqlite3")
                entries, sources, pins = [], [], []
                if os.path.isfile(db):
                    reg = Registry(db, read_only=True)
                    try:
                        entries = [
                            dict(zip(("compile_key", "source_sha256",
                                      "bytes", "recipe"), r))
                            for r in reg.query(
                                "SELECT compile_key, source_sha256,"
                                " artifact_size, recipe_id FROM artifacts")]
                        sources = [
                            dict(zip(("sha256", "bytes", "origin"), r))
                            for r in reg.query(
                                "SELECT sha256, size_bytes, origin"
                                " FROM sources")]
                        pins = reg.list_pins()
                    finally:
                        reg.close()
                print(json.dumps({"schema_version": 1, "entries": entries,
                                  "sources": sources, "pins": pins},
                                 indent=2, sort_keys=True))
                return SUCCESS
            if _daemon_alive(config.data_dir):
                # Both dry-run and applied prune go through the daemon's
                # own locking; standalone owns the lock only when no
                # daemon runs.
                resp = _admin_or_raise(
                    config.data_dir,
                    {"command": "prune", "apply": args.apply,
                     "max_bytes": args.max_bytes})
                print(json.dumps(resp, indent=2, sort_keys=True))
                return SUCCESS
            sup = Supervisor(config)
            try:
                print(json.dumps(sup.prune(args.apply, args.max_bytes),
                                 indent=2, sort_keys=True))
                return SUCCESS
            finally:
                sup.stop()
        if args.command == "doctor":
            if args.hardware:
                print("fxdna: hardware probe needs the Task 04 device "
                      "lease; refusing without it [DEVICE_UNAVAILABLE]",
                      file=sys.stderr)
                return 8
            from .deploy_checks import run_preflight
            doc = {"schema_version": 1, "checks": [
                {"name": "registry-readable", "ok": os.path.isfile(
                    os.path.join(config.data_dir, "registry.sqlite3"))},
                {"name": "endpoint-configured",
                 "ok": "://" in config.endpoint}] + [
                {"name": c["name"], "ok": c["ok"],
                 "message": c["message"]}
                for c in run_preflight(config,
                                       create_data_dir=False)],
                "note": "passive checks only; no NPU probed"}
            print(json.dumps(doc, indent=2, sort_keys=True))
            return SUCCESS
        if args.command == "health":
            return cmd_health(config, args.ready)
        if args.command == "recover":
            if _daemon_alive(config.data_dir):
                resp = _admin_or_raise(config.data_dir,
                                       {"command": "recover",
                                        "ref": args.ref})
                print(json.dumps(resp, indent=2, sort_keys=True))
                return _recover_exit(resp)
            sup = Supervisor(config)
            try:
                resp = sup.recover_ref(args.ref)
                print(json.dumps(resp, indent=2, sort_keys=True))
                return _recover_exit(resp)
            finally:
                sup.stop()
        print(f"fxdna: '{args.command}' is not implemented in this build "
              f"[{NOT_IMPLEMENTED}]", file=sys.stderr)
        return NOT_READY
    except FxdnaError as e:
        print(f"fxdna: error: {e.message} [{e.error_code}]", file=sys.stderr)
        return e.exit_code


if __name__ == "__main__":
    sys.exit(main())
