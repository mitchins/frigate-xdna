"""Content-addressed filesystem store (docs/CACHE.md §1, §5).

Layout under the data dir; same-filesystem staging + fsync + atomic rename;
fcntl single-writer locks; crash recovery reconciling work/ against the
registry; disk preflight with reserve.
"""
from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import time
from contextlib import contextmanager

from ..errors import CACHE_CORRUPT, FxdnaError

RESERVE_BYTES = 2 * 1024 ** 3
RESERVE_FRACTION = 0.10

LAYOUT_DIRS = ("sources", "metadata", "artifacts", "validations", "work",
               "failures", "quarantine", "operations", "locks")


def ensure_layout(data_dir: str) -> None:
    os.makedirs(data_dir, exist_ok=True)
    for sub in LAYOUT_DIRS:
        os.makedirs(os.path.join(data_dir, sub), exist_ok=True)


@contextmanager
def locked(lock_path: str, exclusive: bool = True):
    """Advisory file lock. Same-filesystem semantics required (no NFS)."""
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    with open(lock_path, "a+") as f:
        fcntl.flock(f.fileno(),
                    fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield f
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def try_exclusive(lock_path: str):
    """Non-blocking exclusive lock; returns the file or None."""
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    f = open(lock_path, "a+")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    return f


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write(path: str, data: bytes) -> None:
    """Write via same-directory temp file + fsync + atomic rename."""
    directory = os.path.dirname(path)
    tmp = f"{path}.tmp-{os.getpid()}-{time.monotonic_ns()}"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp, path)
    _fsync_dir(directory)


def ingest_bytes(data_dir: str, data: bytes, subdir: str,
                 filename: str) -> str:
    """Store bytes under sources/<sha256>/; returns the digest.

    Idempotent: identical bytes map to the same path and are never
    rewritten. Basenames never become identities.
    """
    digest = hashlib.sha256(data).hexdigest()
    dest_dir = os.path.join(data_dir, subdir, digest)
    dest = os.path.join(dest_dir, filename)
    os.makedirs(dest_dir, exist_ok=True)
    if os.path.isfile(dest):
        if sha256_file(dest) != digest:
            raise FxdnaError(CACHE_CORRUPT, "CACHE_CORRUPT",
                             f"content mismatch at {dest}")
        return digest
    atomic_write(dest, data)
    return digest


def disk_preflight(data_dir: str, need_bytes: int,
                   statvfs=None) -> dict:
    """Refuse work when scratch + reserve do not fit. No deletion rescues."""
    st = (statvfs or os.statvfs)(data_dir)
    free = st.f_bavail * st.f_frsize
    total = st.f_blocks * st.f_frsize
    reserve = max(RESERVE_BYTES, int(total * RESERVE_FRACTION))
    ok = free >= need_bytes + reserve
    return {"ok": ok, "free_bytes": free, "need_bytes": need_bytes,
            "reserve_bytes": reserve,
            "shortfall_bytes": max(0, need_bytes + reserve - free)}


def publish_artifact(data_dir: str, compile_key: str, staged_dir: str,
                     files: dict[str, bytes]) -> str:
    """Commit staged artifact files atomically into artifacts/<compile_key>/.

    Writes every file with fsync, then renames the staging directory into
    place. Content-hash verification of the artifact happens at activation
    time against the registry row (not here: this function has no digest
    inputs by design). Never overwrites a committed artifact.
    """
    dest = os.path.join(data_dir, "artifacts", compile_key)
    if os.path.isdir(dest):
        raise FxdnaError(CACHE_CORRUPT, "CACHE_CORRUPT",
                         f"refusing to overwrite committed {compile_key}")
    for name, data in files.items():
        if os.path.isabs(name) or ".." in name.split(os.sep):
            raise FxdnaError(CACHE_CORRUPT, "CACHE_CORRUPT",
                             f"unsafe artifact member {name!r}")
        atomic_write(os.path.join(staged_dir, name), data)
    os.rename(staged_dir, dest)
    _fsync_dir(os.path.join(data_dir, "artifacts"))
    return dest


def recover(data_dir: str, registry) -> dict:
    """Reconcile work/ scratch against the registry after a crash.

    - work/<uuid>/ with a non-terminal job row: mark INTERRUPTED, move
      scratch to failures/<uuid>/ (bounded evidence).
    - work/<uuid>/ with no job row: orphan scratch; move to
      failures/orphan-<uuid>/ (evidence preserved, work/ reclaimed).
    - artifacts/<key>/ without a registry row: adopt only after full
      revalidation (artifact.json present with source/artifact hashes and
      model.rai matching); otherwise quarantine the key. Adoption unblocks
      the crash-between-rename-and-commit case instead of deadlocking the
      compile key forever.
    Returns a summary; never deletes committed artifacts.
    """
    import json as _json

    summary = {"interrupted": [], "quarantined": [], "adopted": [],
               "orphans": []}
    work = os.path.join(data_dir, "work")
    for uuid in sorted(os.listdir(work) if os.path.isdir(work) else []):
        wpath = os.path.join(work, uuid)
        if not os.path.isdir(wpath):
            continue
        job = registry.get_job(uuid)
        if job is None:
            dest = os.path.join(data_dir, "failures", f"orphan-{uuid}")
            if os.path.isdir(dest):
                shutil.rmtree(dest)
            os.rename(wpath, dest)
            summary["orphans"].append(uuid)
            continue
        if job["stage"] in ("PREPARED", "COMPILE_FAILED", "RESOURCE_EXCEEDED",
                            "VALIDATION_FAILED", "UNSUPPORTED_CONTRACT",
                            "QUARANTINED", "INTERRUPTED"):
            continue
        registry.set_job(uuid, "INTERRUPTED", error_code="INTERRUPTED")
        ref = (registry.get_job(uuid) or {}).get("ref")
        if ref:
            registry.set_ref_state(ref, "INTERRUPTED")
        dest = os.path.join(data_dir, "failures", uuid)
        if os.path.isdir(dest):
            shutil.rmtree(dest)
        os.rename(wpath, dest)
        summary["interrupted"].append(uuid)
    arts = os.path.join(data_dir, "artifacts")
    for key in sorted(os.listdir(arts) if os.path.isdir(arts) else []):
        if registry.get_artifact(key) is not None:
            continue
        manifest_p = os.path.join(arts, key, "artifact.json")
        rai_p = os.path.join(arts, key, "model.rai")
        adopted = False
        try:
            with open(manifest_p, "rb") as f:
                manifest = _json.loads(f.read().decode())
            if (isinstance(manifest, dict)
                    and manifest.get("compile_key") == key
                    and sha256_file(rai_p) == manifest.get("artifact_sha256")
                    and manifest.get("source_sha256")
                    and manifest.get("recipe_id")):
                registry.add_artifact(
                    key, manifest["source_sha256"],
                    manifest["artifact_sha256"],
                    os.path.getsize(rai_p), manifest["recipe_id"],
                    manifest.get("target_profile", ""))
                adopted = True
        except (OSError, ValueError):
            adopted = False
        if adopted:
            summary["adopted"].append(key)
        else:
            q = os.path.join(data_dir, "quarantine", f"{key}.json")
            atomic_write(q, b'{"reason":"orphan-without-valid-manifest"}')
            summary["quarantined"].append(key)
    return summary
