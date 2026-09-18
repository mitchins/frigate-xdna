"""`fxdna` command-line interface (docs/INTERFACES.md §1, normative).

Foundation scope (Task 01): full command surface with exact exit codes;
only `--help`, config validation and offline `status` are functional.
Hardware/model operations are explicit stubs returning honest errors —
never fake readiness.
"""
from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .config import load_config
from .errors import (
    FxdnaError,
    NOT_IMPLEMENTED,
    NOT_READY,
    SUCCESS,
)

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


def cmd_status(args: argparse.Namespace) -> dict:
    """Offline status document; schema_version 1. No NPU access."""
    return {
        "schema_version": 1,
        "service": "frigate-xdna",
        "version": __version__,
        "state": "NOT_IMPLEMENTED",
        "active": None,
        "models": [],
        "note": "Task 01 foundation: daemon not implemented; no hardware "
                "probed, no readiness claimed.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        load_config()  # validate environment eagerly on every command
    except FxdnaError as e:
        print(f"fxdna: error: {e.message} [{e.error_code}]", file=sys.stderr)
        return e.exit_code
    try:
        if args.command == "status":
            doc = cmd_status(args)
            if args.json:
                print(json.dumps(doc, indent=2, sort_keys=True))
            else:
                print(f"frigate-xdna {__version__}: {doc['state']} "
                      f"(foundation build; daemon not implemented)")
            return SUCCESS
        if args.command == "health" and not args.ready:
            print(json.dumps({"schema_version": 1, "alive": False,
                              "note": "daemon not implemented"}))
            return NOT_READY
        if args.command == "doctor" and not args.hardware:
            print(json.dumps({"schema_version": 1, "checks": [],
                              "note": "daemon not implemented"}))
            return SUCCESS
        # Honest stubs: every state-changing or hardware operation refuses
        # rather than faking success.
        print(f"fxdna: '{args.command}' is not implemented in this build "
              f"[{NOT_IMPLEMENTED}]", file=sys.stderr)
        return NOT_READY
    except FxdnaError as e:
        print(f"fxdna: error: {e.message} [{e.error_code}]", file=sys.stderr)
        return e.exit_code


if __name__ == "__main__":
    sys.exit(main())
