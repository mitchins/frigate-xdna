"""Deployment preflight (v0.1.1 checkpoint 4).

Inherited limits and file/credential accessibility are checked BEFORE
expensive acquisition/compilation, so an inadequate environment fails
fast with a corrective action instead of after minutes of wasted
work. Used by `serve`, standalone `prepare`, and `doctor`.

Each check returns {name, ok, message, code}: `code` is the CLI exit
for a blocking failure. The `models-configured` check is advisory
(empty selection stays servable for later prepare/ZMQ upload).
"""
from __future__ import annotations

import os
import resource
import shutil

MEMLOCK_NEED_BYTES = 64 * 1024 * 1024
DATA_FREE_MIN_BYTES = 512 * 1024 * 1024


def _uid_can_read(path: str, uid: int, gid: int, groups: tuple) -> bool:
    """Permission-bit readability for one UID (no root bypass).

    os.access answers for the CALLER; in tests running as root that
    would bless a root-owned 0600 file the UID-10001 service user
    cannot read. Evaluating the bits explicitly keeps the documented
    rule ("a root-owned 0600 file is not readable by UID 10001")
    honest everywhere.
    """
    try:
        st = os.stat(path)
    except OSError:
        return False
    mode = st.st_mode
    if st.st_uid == uid:
        return bool(mode & 0o400)
    if st.st_gid == gid or st.st_gid in (groups or ()):
        return bool(mode & 0o040)
    return bool(mode & 0o004)


def check_memlock() -> dict:
    """Locked-memory allowance must cover the 64 MiB FlexMLRT mapping."""
    try:
        soft, _hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    except (OSError, ValueError):
        return {"name": "memlock", "ok": False,
                "message": "cannot query RLIMIT_MEMLOCK",
                "code": 6, "error_code": "RESOURCE_EXCEEDED"}
    need = MEMLOCK_NEED_BYTES
    if soft == resource.RLIM_INFINITY or soft >= need:
        return {"name": "memlock", "ok": True,
                "message": f"RLIMIT_MEMLOCK soft="
                           f"{'unlimited' if soft == resource.RLIM_INFINITY else soft}",
                "code": 0, "error_code": "RESOURCE_EXCEEDED"}
    return {"name": "memlock", "ok": False,
            "message": f"RLIMIT_MEMLOCK soft={soft} below {need} bytes:"
                       " FlexMLRT cannot lock its 64 MiB mapping"
                       " (mmap fails EAGAIN after minutes of compiling)."
                       " Set ulimits memlock soft+hard to -1"
                       " (examples/compose.yaml).",
            "code": 6, "error_code": "RESOURCE_EXCEEDED"}


def check_data_dir(data_dir: str, create: bool = True) -> dict:
    try:
        if create:
            os.makedirs(data_dir, exist_ok=True)
        elif not os.path.isdir(data_dir):
            return {"name": "data-dir", "ok": False,
                    "message": f"{data_dir} does not exist yet.",
                    "code": 6, "error_code": "RESOURCE_EXCEEDED"}
        probe = os.path.join(data_dir, ".fxdna-write-probe")
        with open(probe, "wb") as f:
            f.write(b"ok")
        os.unlink(probe)
        free = shutil.disk_usage(data_dir).free
    except OSError as e:
        return {"name": "data-dir", "ok": False,
                "message": f"{data_dir} not writable: {e}",
                "code": 6, "error_code": "RESOURCE_EXCEEDED"}
    if free < DATA_FREE_MIN_BYTES:
        return {"name": "data-dir", "ok": False,
                "message": f"{data_dir}: only {free} bytes free,"
                           f" need {DATA_FREE_MIN_BYTES}; refuse new work"
                           " rather than filling the filesystem.",
                "code": 6, "error_code": "RESOURCE_EXCEEDED"}
    return {"name": "data-dir", "ok": True,
            "message": f"{data_dir} writable, {free} bytes free",
            "code": 0, "error_code": "RESOURCE_EXCEEDED"}


def check_device(device: str) -> dict:
    """Presence/accessibility only: never opens or probes the NPU."""
    if not os.path.exists(device):
        return {"name": "device", "ok": False,
                "message": f"{device} missing: pass the accelerator node"
                           " (devices: /dev/accel/accel0) and set"
                           " FXDNA_DEVICE (default /dev/accel/accel0).",
                "code": 8, "error_code": "DEVICE_UNAVAILABLE"}
    if not os.access(device, os.R_OK | os.W_OK):
        return {"name": "device", "ok": False,
                "message": f"{device} not accessible: fix the device"
                           " group (NPU_GID) mapping.",
                "code": 8, "error_code": "DEVICE_UNAVAILABLE"}
    return {"name": "device", "ok": True,
            "message": f"{device} present", "code": 0,
            "error_code": "DEVICE_UNAVAILABLE"}


def check_key_file(key_file: str | None) -> dict:
    if not key_file:
        return {"name": "key-file", "ok": True,
                "message": "no key file configured", "code": 0}
    if not os.path.isfile(key_file):
        return {"name": "key-file", "ok": False,
                "message": f"PLUS_API_KEY_FILE={key_file} not found:"
                           " the bind source must be an absolute path on"
                           " the Docker host.",
                "code": 2, "error_code": "INVALID_CONFIG"}
    if not _uid_can_read(key_file, os.geteuid(), os.getegid(),
                         os.getgroups()):
        return {"name": "key-file", "ok": False,
                "message": f"PLUS_API_KEY_FILE={key_file} not readable"
                           " by this UID: the container runs as UID 10001"
                           " and a root-owned 0600 file is not readable"
                           " by it. Use the secrets overlay or adjust"
                           " ownership/permissions.",
                "code": 2, "error_code": "INVALID_CONFIG"}
    return {"name": "key-file", "ok": True,
            "message": f"{key_file} readable", "code": 0,
            "error_code": "INVALID_CONFIG"}


def check_plus_credential(config) -> dict:
    """Plus refs without any credential fail acquisition; say so now."""
    from .models.refs import parse_ref
    wants_plus = False
    for ref in config.models:
        try:
            if parse_ref(ref)["kind"] == "plus":
                wants_plus = True
                break
        except Exception:
            continue
    if wants_plus and not (config.plus_api_key or
                           config.plus_api_key_file):
        return {"name": "plus-credential", "ok": False,
                "message": "Plus models configured but neither PLUS_API_KEY"
                           " nor PLUS_API_KEY_FILE is set.",
                "code": 4, "error_code": "ACQUISITION_FAILED"}
    return {"name": "plus-credential", "ok": True,
            "message": "Plus credential present or not needed", "code": 0,
            "error_code": "ACQUISITION_FAILED"}


def check_models(config) -> dict:
    if not config.models:
        return {"name": "models-configured", "ok": True,
                "advisory": True,
                "message": "no models selected: sidecar idles until"
                           " prepare or a Frigate upload arrives.",
                "code": 0}
    return {"name": "models-configured", "ok": True,
            "message": f"{len(config.models)} model(s) selected",
            "code": 0}


def run_preflight(config, create_data_dir: bool = True) -> list[dict]:
    """All deployment checks, in a stable order: configuration errors
    (key, credential) before environment limits (memlock, disk,
    device), so a fixable config mistake is never masked by the
    runner's own limits. `doctor` passes create_data_dir=False to
    stay side-effect free."""
    return [check_key_file(config.plus_api_key_file),
            check_plus_credential(config),
            check_memlock(),
            check_data_dir(config.data_dir, create=create_data_dir),
            check_device(config.device),
            check_models(config)]


def blocking_failures(checks: list[dict]) -> list[dict]:
    """All non-advisory failing checks (every one is reported)."""
    return [c for c in checks
            if not c["ok"] and not c.get("advisory")]


def blocking_failure(checks: list[dict]) -> dict | None:
    """First non-advisory failing check, or None when servable."""
    failures = blocking_failures(checks)
    return failures[0] if failures else None
