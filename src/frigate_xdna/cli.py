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
import sys
import time

from . import __version__
from .admin import admin_call, socket_path
from .cache.registry import Registry
from .config import load_config
from .errors import (
    FxdnaError,
    INVALID_ARGS,
    NOT_IMPLEMENTED,
    NOT_READY,
    SUCCESS,
)
from .supervisor import Supervisor

COMMANDS = (
    "serve", "prepare", "status", "wait", "activate", "cache",
    "doctor", "health", "recover",
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fxdna",
        description="Self-contained XDNA detector sidecar for Frigate "
                    "(stock Frigate unchanged).",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
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

    wp = sub.add_parser("wait", help="Wait for a model state via admin socket.")
    wp.add_argument("ref")
    wp.add_argument("--state", default="prepared",
                    choices=("prepared", "verified", "active"))
    wp.add_argument("--timeout", type=float, default=1800.0)

    ap = sub.add_parser("activate", help="Activate an already prepared model.")
    ap.add_argument("ref")
    ap.add_argument("--maintenance", action="store_true")
    ap.add_argument("--wait", action="store_true")

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


def _daemon_alive(data_dir: str) -> bool:
    return os.path.exists(socket_path(data_dir))


def _read_status(config, ref=None) -> dict:
    """Read-only status: no lock, no NPU, works with or without a daemon.

    Never creates state: a missing registry means STARTING, not an error.
    """
    db = os.path.join(config.data_dir, "registry.sqlite3")
    if not os.path.isfile(db):
        return {"schema_version": 1, "service": "frigate-xdna",
                "version": __version__, "state": "STARTING",
                "active": None, "models": [],
                "note": "no registry yet; daemon not started"}
    # SERVING is reported only when a live manager owns the directory;
    # otherwise STARTING (or INHIBITED) — never inference readiness (B2).
    daemon = _daemon_alive(config.data_dir)
    reg = Registry(db)
    try:
        if ref:
            from .models.refs import parse_ref
            rec = reg.get_ref(parse_ref(ref)["ref"])
            models = [{"ref": rec["ref"],
                       "source_sha256": rec.get("source_sha256") or "",
                       "state": rec.get("state") or "NEW"}] if rec else []
        else:
            models = [{"ref": r[0], "source_sha256": r[3] or "",
                       "state": r[5]} for r in reg.query(
                "SELECT ref, kind, model_id, source_sha256,"
                " metadata_sha256, state FROM model_refs")]
        inhibition = reg.get_state("inhibition")
        state = "SERVING" if daemon else (
            "INHIBITED" if inhibition and not inhibition.get("cleared")
            else "STARTING")
        return {"schema_version": 1, "service": "frigate-xdna",
                "version": __version__, "state": state,
                "active": reg.get_state("active"), "models": models,
                "inhibition": inhibition}
    finally:
        reg.close()


# Admin error_code -> CLI exit mapping (mirrors _TERMINAL_EXIT for the
# daemon path so online/standalone exits agree — SF1).
_ADMIN_EXIT = {
    "ACQUISITION_FAILED": 4, "AUTH_FAILED": 4, "DOWNLOAD_FAILED": 4,
    "UNSUPPORTED_CONTRACT": 5,
    "COMPILE_FAILED": 6, "RESOURCE_EXCEEDED": 6,
    "VALIDATION_FAILED": 7,
    "DEVICE_BUSY": 8, "DEVICE_FAULT": 8, "SAFETY_INHIBITED": 8,
    "QUARANTINED": 8,
    "CACHE_CORRUPT": 10,
    "OWNERSHIP_CONFLICT": 9,
    "SOURCE_CHANGED": 2, "INTERRUPTED": 3,
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


def cmd_serve(config) -> int:
    if "FXDNA_ENDPOINT" not in os.environ:
        # SPEC §5.1: loopback by default for native development; the image
        # default (bind-all) applies only when explicitly configured.
        config = config.__class__(**{**config.__dict__,
                                     "endpoint": "tcp://127.0.0.1:5555"})
    sup = Supervisor(config)
    sup.start_admin()
    for ref in config.models:
        try:
            sup.prepare(ref)
        except FxdnaError as e:
            print(f"fxdna: startup prepare {ref}: {e.message} "
                  f"[{e.error_code}]", file=sys.stderr)
    print(f"fxdna: serving endpoint={config.endpoint} "
          f"data={sup.data_dir}", file=sys.stderr)
    try:
        while True:
            sup.pump(0.2)
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
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
                 "descriptor": args.descriptor,
                 "wire_name": args.wire_name})
            results.append(resp["job"])
    else:
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


def cmd_wait(config, args) -> int:
    want = args.state.upper()
    deadline = time.monotonic() + args.timeout
    daemon = _daemon_alive(config.data_dir)
    while True:
        # INTERFACES.md: wait goes through the private admin socket when a
        # daemon owns the data dir; standalone polls the local registry.
        if daemon:
            doc = _admin_or_raise(
                config.data_dir,
                {"command": "status", "ref": args.ref})["status"]
        else:
            doc = _read_status(config, args.ref)
        states = [m["state"] for m in doc.get("models", [])]
        if states and states[0] == want:
            print(json.dumps(doc, indent=2, sort_keys=True))
            return SUCCESS
        if states and states[0] in _TERMINAL_EXIT:
            print(json.dumps(doc, indent=2, sort_keys=True))
            return _terminal_exit(states[0])
        if time.monotonic() >= deadline:
            print(f"fxdna: timed out waiting for {args.ref}={want}",
                  file=sys.stderr)
            return NOT_READY
        time.sleep(0.5)


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
            doc = _read_status(config, args.ref)
            if args.json:
                print(json.dumps(doc, indent=2, sort_keys=True))
            else:
                for m in doc.get("models", []):
                    print(f"{m['ref']}: {m['state']}")
                if not doc.get("models"):
                    print("(no models registered)")
            return SUCCESS
        if args.command == "wait":
            return cmd_wait(config, args)
        if args.command == "activate":
            if _daemon_alive(config.data_dir):
                resp = _admin_or_raise(
                    config.data_dir,
                    {"command": "activate", "ref": args.ref,
                     "maintenance": args.maintenance})
                print(json.dumps(resp, indent=2, sort_keys=True))
                return SUCCESS
            sup = Supervisor(config)
            try:
                print(json.dumps(sup.activate(args.ref, args.maintenance)))
                return SUCCESS
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
                    reg = Registry(db)
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
            if _daemon_alive(config.data_dir) and not args.apply:
                # Dry-run prune is read-only: answer without the writer lock.
                resp = _admin_or_raise(
                    config.data_dir,
                    {"command": "prune", "apply": False,
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
            doc = {"schema_version": 1, "checks": [
                {"name": "registry-readable", "ok": os.path.isfile(
                    os.path.join(config.data_dir, "registry.sqlite3"))},
                {"name": "endpoint-configured",
                 "ok": "://" in config.endpoint}],
                "note": "passive checks only; no NPU probed"}
            print(json.dumps(doc, indent=2, sort_keys=True))
            return SUCCESS
        if args.command == "health":
            alive = _daemon_alive(config.data_dir)
            if not args.ready:
                print(json.dumps({"schema_version": 1, "alive": alive}))
                return SUCCESS
            # --ready requires an ACTIVE healthy worker (Task 04). No
            # worker exists in Task 02: never claim readiness.
            print(json.dumps({"schema_version": 1, "alive": alive,
                              "ready": False,
                              "note": "no native worker in this build"}))
            return NOT_READY
        if args.command == "recover":
            if _daemon_alive(config.data_dir):
                resp = _admin_or_raise(config.data_dir,
                                       {"command": "recover",
                                        "ref": args.ref})
                print(json.dumps(resp, indent=2, sort_keys=True))
                return SUCCESS
            sup = Supervisor(config)
            try:
                print(json.dumps(sup.recover_ref(args.ref), indent=2,
                                 sort_keys=True))
                return SUCCESS
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
