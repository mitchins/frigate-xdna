"""SQLite job/model registry (docs/CACHE.md §3).

Single-writer discipline is enforced by the supervisor holding the data-dir
lock; this module provides schema, migrations, and transactional helpers.
Readers refuse newer unsupported schemas instead of rebuilding them.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time

SCHEMA_VERSION = 2

# v1 -> v2: structured failure records on jobs (nullable: legacy rows
# read as unknown-evidence failures, never as safe).
_SCHEMA_V2 = """
ALTER TABLE jobs ADD COLUMN failure_json TEXT;
"""

_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY);
CREATE TABLE IF NOT EXISTS model_refs (
  ref TEXT PRIMARY KEY, kind TEXT NOT NULL, model_id TEXT,
  source_sha256 TEXT, metadata_sha256 TEXT, pin TEXT,
  state TEXT NOT NULL DEFAULT 'NEW', updated_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS sources (
  sha256 TEXT PRIMARY KEY, size_bytes INTEGER NOT NULL,
  rel_path TEXT NOT NULL, origin TEXT NOT NULL, checked_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS artifacts (
  compile_key TEXT PRIMARY KEY, source_sha256 TEXT NOT NULL,
  artifact_sha256 TEXT NOT NULL, artifact_size INTEGER NOT NULL,
  recipe_id TEXT NOT NULL, target_profile TEXT NOT NULL,
  created_at REAL NOT NULL, compile_s REAL, peak_rss_kb INTEGER);
CREATE TABLE IF NOT EXISTS serving_contracts (
  digest TEXT PRIMARY KEY, compile_key TEXT NOT NULL,
  contract_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS validations (
  validation_key TEXT PRIMARY KEY, compile_key TEXT NOT NULL,
  serving_digest TEXT NOT NULL, passed INTEGER NOT NULL,
  reason TEXT, fixture_ids TEXT, runtime_json TEXT, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS jobs (
  uuid TEXT PRIMARY KEY, ref TEXT NOT NULL, compile_key TEXT,
  stage TEXT NOT NULL, attempt INTEGER NOT NULL DEFAULT 0,
  created_at REAL NOT NULL, updated_at REAL NOT NULL,
  boot_token TEXT, progress TEXT, error_code TEXT,
  failure_json TEXT);
CREATE TABLE IF NOT EXISTS pins (
  name TEXT PRIMARY KEY, kind TEXT NOT NULL, target TEXT NOT NULL,
  created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS job_aliases (
  job_uuid TEXT NOT NULL, ref TEXT NOT NULL,
  PRIMARY KEY (job_uuid, ref));
CREATE TABLE IF NOT EXISTS service_state (
  key TEXT PRIMARY KEY, value_json TEXT NOT NULL);
"""


def _parse_failure(raw) -> dict | None:
    """Structured failure record, or None for legacy rows without
    evidence (unknown-evidence failures, never auto-safe)."""
    if not raw:
        return None
    try:
        record = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return record if isinstance(record, dict) else None


class Registry:
    def __init__(self, db_path: str, read_only: bool = False):
        """Open the registry.

        read_only=True is for status/cache-list style commands: opens with
        SQLite mode=ro and skips WAL setup, schema creation and version
        writes. Newer schemas are still refused. Write methods raise
        sqlite3.OperationalError in this mode by the engine itself.
        """
        self.db_path = db_path
        self.read_only = read_only
        # The daemon pumps jobs on its main thread while the admin socket
        # serves requests on another; serialize all access. The daemon
        # remains the only writer by discipline; the lock is correctness,
        # not a second-writer licence.
        self._lock = threading.RLock()
        if read_only:
            uri = f"file:{db_path}?mode=ro"
            self.cx = sqlite3.connect(uri, timeout=30.0, uri=True,
                                      check_same_thread=False)
        else:
            self.cx = sqlite3.connect(db_path, timeout=30.0,
                                      isolation_level=None,
                                      check_same_thread=False)
            self._execute("PRAGMA journal_mode=WAL")
            self._execute("PRAGMA synchronous=FULL")
            with self._lock:
                self.cx.executescript(_SCHEMA_V1)
        row = self.query("SELECT version FROM schema_version")
        if not row:
            if read_only:
                raise RuntimeError(
                    f"registry at {db_path} has no schema version;"
                    " refusing read-only open of an uninitialized store")
            self._execute("INSERT INTO schema_version VALUES (?)",
                            (SCHEMA_VERSION,))
        elif row[0][0] > SCHEMA_VERSION:
            raise RuntimeError(
                f"registry schema v{row[0][0]} is newer than supported "
                f"v{SCHEMA_VERSION}; refusing to open")
        elif row[0][0] < SCHEMA_VERSION:
            if read_only:
                raise RuntimeError(
                    f"registry schema v{row[0][0]} needs migration;"
                    " refusing read-only open of a pre-migration store")
            self._migrate(row[0][0])


    def _migrate(self, from_version: int) -> None:
        """Small, idempotent migrations. v1 -> v2 adds the nullable
        failure_json column; legacy rows keep NULL (unknown-evidence
        failures). No explicit transaction: executescript commits
        implicitly, and the column check makes a crash between the
        two statements safely re-runnable."""
        if from_version == 1:
            cols = [r[1] for r in self.query("PRAGMA table_info(jobs)")]
            if "failure_json" not in cols:
                self.cx.executescript(_SCHEMA_V2)
            self._execute("UPDATE schema_version SET version=?",
                          (SCHEMA_VERSION,))

    def _execute(self, sql, params=()):
        with self._lock:
            return self.cx.execute(sql, params)

    def query(self, sql, params=()):
        """Locked SELECT returning all rows (safe from any thread)."""
        with self._lock:
            return self.cx.execute(sql, params).fetchall()

    def execute(self, sql, params=()):
        """Locked write (safe from any thread). Autocommit mode."""
        with self._lock:
            return self.cx.execute(sql, params)

    def transaction(self):
        """Exclusive write transaction spanning multiple statements.

        Takes the registry RLock and issues BEGIN IMMEDIATE so concurrent
        submitters serialize: check-then-insert sequences inside the block
        cannot interleave into duplicate live jobs.
        """
        import contextlib as _cl

        @_cl.contextmanager
        def _tx():
            with self._lock:
                self.cx.execute("BEGIN IMMEDIATE")
                try:
                    yield
                except BaseException:
                    self.cx.execute("ROLLBACK")
                    raise
                else:
                    self.cx.execute("COMMIT")
        return _tx()

    def close(self):
        self.cx.close()

    # -- generic transactional helpers -------------------------------
    def upsert_ref(self, ref: str, kind: str, model_id: str | None,
                   state: str = "NEW"):
        """Register a ref; never clobber an existing row's state/source.

        State transitions are explicit (set_ref_state/set_ref_source); a
        repeated prepare — e.g. a failed offline re-fetch — must not reset
        a previous PREPARED/QUEUED state to NEW.
        """
        now = time.time()
        self._execute(
            "INSERT INTO model_refs(ref, kind, model_id, state, updated_at)"
            " VALUES (?,?,?,?,?) ON CONFLICT(ref) DO UPDATE SET"
            " updated_at=excluded.updated_at",
            (ref, kind, model_id, state, now))

    def get_ref(self, ref: str) -> dict | None:
        rows = self.query(
            "SELECT ref, kind, model_id, source_sha256, metadata_sha256,"
            " pin, state FROM model_refs WHERE ref=?", (ref,))
        if not rows:
            return None
        return dict(zip(("ref", "kind", "model_id", "source_sha256",
                         "metadata_sha256", "pin", "state"), rows[0]))

    def set_ref_source(self, ref: str, source_sha256: str,
                       metadata_sha256: str | None, state: str):
        self._execute(
            "UPDATE model_refs SET source_sha256=?, metadata_sha256=?,"
            " state=?, updated_at=? WHERE ref=?",
            (source_sha256, metadata_sha256, state, time.time(), ref))

    def set_ref_state(self, ref: str, state: str):
        self._execute("UPDATE model_refs SET state=?, updated_at=?"
                        " WHERE ref=?", (state, time.time(), ref))

    def add_source(self, sha256: str, size: int, rel_path: str, origin: str):
        self._execute(
            "INSERT OR IGNORE INTO sources(sha256, size_bytes, rel_path,"
            " origin, checked_at) VALUES (?,?,?,?,?)",
            (sha256, size, rel_path, origin, time.time()))

    def get_source(self, sha256: str) -> dict | None:
        rows = self.query(
            "SELECT sha256, size_bytes, rel_path, origin FROM sources"
            " WHERE sha256=?", (sha256,))
        return dict(zip(("sha256", "size_bytes", "rel_path", "origin"),
                        rows[0])) if rows else None

    def add_artifact(self, compile_key: str, source_sha256: str,
                     artifact_sha256: str, size: int, recipe_id: str,
                     target: str):
        self._execute(
            "INSERT OR IGNORE INTO artifacts(compile_key, source_sha256,"
            " artifact_sha256, artifact_size, recipe_id, target_profile,"
            " created_at) VALUES (?,?,?,?,?,?,?)",
            (compile_key, source_sha256, artifact_sha256, size, recipe_id,
             target, time.time()))

    def get_artifact(self, compile_key: str) -> dict | None:
        rows = self.query(
            "SELECT compile_key, source_sha256, artifact_sha256,"
            " artifact_size, recipe_id, target_profile FROM artifacts"
            " WHERE compile_key=?", (compile_key,))
        return dict(zip(("compile_key", "source_sha256", "artifact_sha256",
                         "artifact_size", "recipe_id", "target_profile"),
                        rows[0])) if rows else None

    def create_job(self, uuid: str, ref: str, compile_key: str | None,
                   boot_token: str | None, attempt: int = 1) -> None:
        now = time.time()
        self._execute(
            "INSERT INTO jobs(uuid, ref, compile_key, stage, attempt,"
            " created_at, updated_at, boot_token)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (uuid, ref, compile_key, "QUEUED", attempt, now, now,
             boot_token))

    def set_job(self, uuid: str, stage: str, error_code: str | None = None,
                progress: str | None = None):
        self._execute(
            "UPDATE jobs SET stage=?, error_code=?, progress=?,"
            " updated_at=? WHERE uuid=?",
            (stage, error_code, progress, time.time(), uuid))

    def get_job(self, uuid: str) -> dict | None:
        rows = self.query(
            "SELECT uuid, ref, compile_key, stage, attempt, error_code,"
            " progress, failure_json FROM jobs WHERE uuid=?", (uuid,))
        if not rows:
            return None
        job = dict(zip(("uuid", "ref", "compile_key", "stage", "attempt",
                        "error_code", "progress", "failure_json"),
                       rows[0]))
        job["failure"] = _parse_failure(job.pop("failure_json"))
        return job

    def set_failure(self, uuid: str, record: dict) -> None:
        self._execute(
            "UPDATE jobs SET failure_json=?, updated_at=? WHERE uuid=?",
            (json.dumps(record, sort_keys=True), time.time(), uuid))

    def live_job_for_key(self, compile_key: str) -> dict | None:
        """A non-terminal job row for a key (duplicate-compile guard)."""
        rows = self.query(
            "SELECT uuid FROM jobs WHERE compile_key=? AND stage NOT IN"
            " ('PREPARED','COMPILE_FAILED','RESOURCE_EXCEEDED',"
            " 'VALIDATION_FAILED','UNSUPPORTED_CONTRACT','QUARANTINED',"
            " 'INTERRUPTED') ORDER BY updated_at DESC LIMIT 1",
            (compile_key,))
        return self.get_job(rows[0][0]) if rows else None

    def find_job(self, ref: str, compile_key: str | None) -> dict | None:
        if compile_key is None:
            rows = self.query(
                "SELECT uuid, ref, compile_key, stage, attempt, error_code,"
                " progress FROM jobs WHERE ref=? AND compile_key IS NULL"
                " ORDER BY created_at DESC LIMIT 1", (ref,))
        else:
            rows = self.query(
                "SELECT uuid, ref, compile_key, stage, attempt, error_code,"
                " progress FROM jobs WHERE ref=? AND compile_key=?"
                " ORDER BY created_at DESC LIMIT 1",
                (ref, compile_key))
        return dict(zip(("uuid", "ref", "compile_key", "stage", "attempt",
                         "error_code", "progress"), rows[0])) if rows else None

    def add_alias(self, job_uuid: str, ref: str):
        self._execute(
            "INSERT OR IGNORE INTO job_aliases(job_uuid, ref) VALUES (?,?)",
            (job_uuid, ref))

    def refs_for_job(self, job_uuid: str) -> list[str]:
        rows = self.query(
            "SELECT ref FROM jobs WHERE uuid=?", (job_uuid,))
        refs = [r[0] for r in rows]
        rows = self.query(
            "SELECT ref FROM job_aliases WHERE job_uuid=?", (job_uuid,))
        refs.extend(r[0] for r in rows)
        return refs

    def add_pin(self, name: str, kind: str, target: str):
        self._execute(
            "INSERT OR REPLACE INTO pins(name, kind, target, created_at)"
            " VALUES (?,?,?,?)", (name, kind, target, time.time()))

    def remove_pin(self, name: str):
        self._execute("DELETE FROM pins WHERE name=?", (name,))

    def list_pins(self) -> list[dict]:
        return [dict(zip(("name", "kind", "target"), r)) for r in
                self.query("SELECT name, kind, target FROM pins")]

    def record_validation(self, validation_key: str, compile_key: str,
                            serving_digest: str, passed: bool, reason: str,
                            fixture_ids: str, runtime_json: str) -> None:
        """Persist native-check evidence (first evidence wins; failures
        never overwrite a pass, passes never need repeating)."""
        self._execute(
            "INSERT OR IGNORE INTO validations(validation_key,"
            " compile_key, serving_digest, passed, reason, fixture_ids,"
            " runtime_json, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (validation_key, compile_key, serving_digest,
             1 if passed else 0, reason, fixture_ids, runtime_json,
             time.time()))

    def is_verified(self, compile_key: str) -> bool:
        rows = self.query(
            "SELECT 1 FROM validations WHERE compile_key=? AND passed=1"
            " LIMIT 1", (compile_key,))
        return bool(rows)

    def verification_for(self, compile_key: str) -> dict | None:
        rows = self.query(
            "SELECT validation_key, serving_digest, reason, fixture_ids,"
            " runtime_json, created_at FROM validations"
            " WHERE compile_key=? AND passed=1"
            " ORDER BY created_at DESC LIMIT 1", (compile_key,))
        if not rows:
            return None
        return dict(zip(("validation_key", "serving_digest", "reason",
                         "fixture_ids", "runtime_json", "created_at"),
                        rows[0]))

    def latest_job_for_ref(self, ref: str) -> dict | None:
        """Newest job row for a ref, including alias rows."""
        rows = self.query(
            "SELECT uuid, ref, compile_key, stage, attempt, error_code,"
            " progress, failure_json FROM jobs WHERE ref=? OR uuid IN"
            " (SELECT job_uuid FROM job_aliases WHERE ref=?)"
            " ORDER BY updated_at DESC LIMIT 1", (ref, ref))
        if not rows:
            return None
        job = dict(zip(("uuid", "ref", "compile_key", "stage", "attempt",
                        "error_code", "progress", "failure_json"),
                       rows[0]))
        job["failure"] = _parse_failure(job.pop("failure_json"))
        return job

    def prepared_key_for_ref(self, ref: str) -> str | None:
        """Newest PREPARED compile key reachable from a ref/alias."""
        rows = self.query(
            "SELECT compile_key FROM jobs WHERE stage='PREPARED' AND"
            " compile_key IS NOT NULL AND (ref=? OR uuid IN"
            " (SELECT job_uuid FROM job_aliases WHERE ref=?))"
            " ORDER BY updated_at DESC LIMIT 1", (ref, ref))
        return rows[0][0] if rows else None

    def set_state(self, key: str, value: dict):
        self._execute(
            "INSERT INTO service_state(key, value_json) VALUES (?,?)"
            " ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json",
            (key, json.dumps(value, sort_keys=True)))

    def get_state(self, key: str) -> dict | None:
        rows = self.query(
            "SELECT value_json FROM service_state WHERE key=?",
            (key,))
        return json.loads(rows[0][0]) if rows else None
