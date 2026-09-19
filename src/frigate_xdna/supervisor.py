"""Supervisor: validated config, registry, preparation queue, admin socket.

The daemon is the only registry writer. No Quark/torch/vendor-ONNX-Runtime/
FlexML imports here or anywhere in the manager process (Task 02 acceptance).
"""
from __future__ import annotations

import hashlib
import json
import os
import time

from . import __version__
from .admin import AdminServer, admin_call
from .cache import gc as _gc
from .cache.keys import compile_key as _compile_key
from .cache.keys import serving_digest as _serving_digest
from .cache.keys import sha256_bytes
from .cache.registry import Registry
from .cache.store import (
    atomic_write,
    disk_preflight,
    ensure_layout,
    ingest_bytes,
    locked,
    publish_artifact,
    recover,
    sha256_file,
    try_exclusive,
)
from .compiler.jobs import TERMINAL_ERROR_STATES, JobManager
from .compiler.real import BACKEND_ID as REAL_BACKEND_ID
from .config import Config
from .errors import (
    CACHE_CORRUPT,
    DEVICE_UNAVAILABLE,
    FxdnaError,
    INVALID_ARGS,
    NOT_READY,
    OWNERSHIP_CONFLICT,
    SUCCESS,
    UNSUPPORTED_CONTRACT,
)
from .models import inspect as _inspect
from .models.refs import parse_ref, wire_alias
from .plus.client import PlusClient

# Compiler backend identity for cache validity (SF3). The fake Task-02
# backend and the audited Task-03 backend produce different bytes for the
# same compile key inputs, so a backend change invalidates cached rows
# instead of trusting them. Task 03 sets this to the audited backend id.
COMPILER_BACKEND = "fake-v0"

# Pinned audited recipe identity (recipes/bf16-vaiml-v1/recipe.json).
RECIPE_ID = "bf16-vaiml-v1"
RECIPE_CONFIG_SHA256 = (
    "4d17acd4393356ff447ba6a91939ef3f0b4fb675bc97a2c4457ed74d894f5782")
COMPILER_PAYLOAD_SHA256 = (
    "8a8d28b751974205ffc4e2b87b34cd3e93584b70d8f4f3de557965aee975f4bd")
TARGET_PROFILE = "xc10AIE2P_ML-die-0x-e-S-es1"
ARTIFACT_COMPAT_ID = "UNASSIGNED-TBD"  # assigned at activation validation
COMPILE_SCRATCH_NEED_BYTES = 4 * 1024 ** 3


def boot_token() -> str:
    try:
        with open("/proc/sys/kernel/random/boot_id") as f:
            return f.read().strip()
    except OSError:
        return "unknown"


def _read_secret(config: Config) -> str | None:
    if config.plus_api_key_file:
        try:
            with open(config.plus_api_key_file) as f:
                return f.read().strip()
        except OSError:
            raise FxdnaError(4, "ACQUISITION_FAILED",
                             "cannot read PLUS_API_KEY_FILE")
    return config.plus_api_key


def _read_bounded(path: str, what: str,
                  max_bytes: int = 256 * 1024 * 1024) -> bytes:
    """Single bounded read; the digest covers exactly these bytes (no TOCTOU)."""
    try:
        with open(path, "rb") as f:
            data = f.read(max_bytes + 1)
    except OSError:
        raise FxdnaError(INVALID_ARGS, "INVALID_MODEL",
                         f"cannot read {what}: {path!r}")
    if not data or len(data) > max_bytes:
        raise FxdnaError(INVALID_ARGS, "INVALID_MODEL",
                         f"{what} size out of bounds")
    return data


class Supervisor:
    def __init__(self, config: Config, data_dir: str | None = None,
                 fake_compile: dict | None = None,
                 plus_client_factory=None,
                 plus_allow_private_hosts: tuple[str, ...] = (),
                 compiler_backend_id: str | None = None,
                 compiler_prefixes=None,
                 compiler_timeout_s: float = 2700.0):
        # Auto-detect the audited appliance prefixes when running inside
        # the image (real backend) vs host dev (fake). Explicit args win;
        # otherwise probe the image layout. Keeps host tests fake without
        # extra configuration.
        if compiler_backend_id is None and compiler_prefixes is None:
            if os.path.isdir("/opt/compile-venv") and os.path.isdir(
                    "/opt/quant-venv") and os.path.isdir("/opt/xilinx-xrt"):
                from .compiler.launcher import CompilerPrefixes
                compiler_prefixes = CompilerPrefixes(
                    quant_python="/opt/quant-venv/bin/python",
                    compile_python="/opt/compile-venv/bin/python",
                    compile_lib="/opt/compile-venv/lib/python3.12/site-packages",
                    xrt_lib="/opt/xilinx-xrt/lib",
                    xrt_root="/opt/xilinx-xrt",
                    recipe_dir="/opt/fxdna/recipes/bf16-vaiml-v1",
                    calib_dir="/opt/fxdna/calib",
                    vaiml_config="/opt/compile-venv/lib/python3.12/"
                                 "site-packages/vaiml_config.json")
                compiler_backend_id = "bf16-vaiml-v1"
            else:
                compiler_backend_id = COMPILER_BACKEND
        elif compiler_backend_id is None:
            compiler_backend_id = COMPILER_BACKEND
        self.config = config
        self.data_dir = data_dir or config.data_dir
        ensure_layout(self.data_dir)
        self.lock_file = try_exclusive(
            os.path.join(self.data_dir, "manager.lock"))
        if self.lock_file is None:
            raise FxdnaError(OWNERSHIP_CONFLICT, "OWNERSHIP_CONFLICT",
                             f"data dir {self.data_dir} is owned by a live "
                             f"daemon; standalone commands refuse its lock")
        self.registry = Registry(os.path.join(self.data_dir,
                                              "registry.sqlite3"))
        self.jobs = JobManager(self.registry,
                               backend_factory=self._make_backend_job,
                               boot_token=boot_token())
        self.fake_compile = fake_compile or {}
        self.compiler_backend_id = compiler_backend_id
        self.compiler_prefixes = compiler_prefixes
        self.compiler_timeout_s = compiler_timeout_s
        self._plus_client_factory = plus_client_factory or PlusClient
        # Test-only affordance for loopback fake Plus servers. Production
        # default is empty: only public https download targets are allowed.
        self._plus_allow_private = tuple(plus_allow_private_hosts)
        self._server: AdminServer | None = None
        recover(self.data_dir, self.registry)

    def _make_backend_job(self, **kw):
        """Job factory: real audited backend when prefixes are configured,
        fake backend otherwise (hardware-free tests + offline development).
        """
        if self.compiler_prefixes is not None and kw.get("compile_key"):
            from .compiler.real import RealCompileJob
            return RealCompileJob(
                source_sha256=kw.get("source_sha256", ""),
                compile_key=kw.get("compile_key", ""),
                source_path=kw.get("source_path", ""),
                workdir=os.path.join(self.data_dir, "work",
                                     kw.get("job_uuid", "nojobs")),
                prefixes=self.compiler_prefixes,
                timeout_s=self.compiler_timeout_s)
        from .compiler.fake import FakeCompileJob
        params = {k: v for k, v in kw.items() if k in (
            "source_sha256", "compile_key", "duration_s", "succeed",
            "fail_state", "device_required", "device_held_by_worker",
            "job_uuid")}
        params.update(self.fake_compile)
        return FakeCompileJob(**params)

    # -- lifecycle ---------------------------------------------------
    def start_admin(self):
        self._server = AdminServer(self.data_dir, self.handle_admin)
        self._server.start()

    def stop(self):
        if self._server is not None:
            self._server.stop()
            self._server.join(timeout=5)
            self._server._cleanup_socket()
        self.registry.close()
        try:
            self.lock_file.close()
        except OSError:
            pass

    def handle_admin(self, req: dict) -> dict:
        cmd = req.get("command")
        if cmd == "status":
            return {"status": self.status(req.get("ref"))}
        if cmd == "prepare":
            job = self.prepare(req.get("ref", ""),
                               descriptor_path=req.get("descriptor"),
                               wire_name=req.get("wire_name"),
                               refresh=bool(req.get("refresh", False)),
                               maintenance=bool(req.get("maintenance", False)))
            return {"job": job}
        if cmd == "wait":
            job = self.wait_job(req.get("job_uuid", ""),
                                float(req.get("timeout", 1800.0)),
                                pump=False)
            return {"job": job}
        if cmd == "activate":
            return {"activation": self.activate(
                req.get("ref", ""),
                maintenance=bool(req.get("maintenance", False)))}
        if cmd == "cache_list":
            return {"entries": self.cache_list(),
                    "sources": self.source_list(),
                    "pins": self.registry.list_pins()}
        if cmd == "prune":
            return self.prune(bool(req.get("apply", False)),
                              req.get("max_bytes"))
        if cmd == "recover":
            return {"recovered": self.recover_ref(req.get("ref", ""))}
        raise FxdnaError(INVALID_ARGS, "INVALID_ARGS",
                         f"unknown admin command {cmd!r}")

    # -- preparation --------------------------------------------------
    def _plus_client(self) -> PlusClient:
        if self.config.offline:
            raise FxdnaError(4, "ACQUISITION_FAILED",
                             "offline mode: Plus acquisition refused")
        secret = _read_secret(self.config)
        if not secret:
            raise FxdnaError(4, "ACQUISITION_FAILED",
                             "no Plus API key configured")
        return self._plus_client_factory(secret)

    def _fetch_plus(self, model_id: str, refresh: bool) -> tuple[bytes, dict]:
        client = self._plus_client()
        info = client.get_model_info(model_id)
        url = client.get_model_download_url(
            model_id, allow_private_hosts=self._plus_allow_private)
        data = client.download_model(
            url, allow_private_hosts=self._plus_allow_private)
        digest = sha256_bytes(data)
        meta_raw = json.dumps(info, sort_keys=True).encode()
        meta_digest = sha256_bytes(meta_raw)
        atomic_write(os.path.join(self.data_dir, "metadata",
                                  f"{meta_digest}.json"), meta_raw)
        return data, {"metadata": info, "source_sha256": digest,
                      "metadata_sha256": meta_digest}

    def prepare(self, ref: str, descriptor_path: str | None = None,
                wire_name: str | None = None, refresh: bool = False,
                maintenance: bool = False) -> dict:
        parsed = parse_ref(ref)
        alias = wire_alias(parsed, wire_name)
        self.registry.upsert_ref(parsed["ref"], parsed["kind"],
                                 parsed.get("id"))
        if parsed["ref"] in self.config.models:
            _gc.pin_ref(self.registry, parsed["ref"], "configured")
        if parsed["kind"] == "plus":
            data, fetched = self._fetch_plus(parsed["id"], refresh)
            # Plus bytes are inspected exactly like local files: metadata
            # conflicts and unsupported contracts fail here, never queue a
            # fake compile to PREPARED. (B3)
            model, digest = _inspect.load_graph_bytes(data)
            contract = _inspect.inspect_model(model)
            _inspect.compare_plus_metadata(fetched["metadata"], contract)
            cls = _inspect.classify_output(contract["outputs"])
            if cls["profile"] is None:
                raise FxdnaError(UNSUPPORTED_CONTRACT, "UNSUPPORTED_CONTRACT",
                                 cls["error"])
            fetched["inspected"] = contract
            fetched["profile"] = cls["profile"]
            return self._ingest_source(parsed["ref"], alias, data, "plus",
                                       fetched, refresh)
        path = parsed["path"]
        if not os.path.isfile(path):
            raise FxdnaError(INVALID_ARGS, "INVALID_MODEL",
                             f"local file not found: {path!r}")
        if parsed["kind"] == "onnx":
            data = _read_bounded(path, "local ONNX")
            model, digest = _inspect.load_graph_bytes(data)
            contract = _inspect.inspect_model(model)
            cls = _inspect.classify_output(contract["outputs"])
            if cls["profile"] is None:
                raise FxdnaError(UNSUPPORTED_CONTRACT, "UNSUPPORTED_CONTRACT",
                                 cls["error"])
            return self._ingest_source(parsed["ref"], alias, data, "local",
                                       {"inspected": contract,
                                        "profile": cls["profile"],
                                        "source_sha256": digest}, refresh)
        # Imported RAI: descriptor mandatory, artifact hash mandatory.
        if descriptor_path is None:
            raise FxdnaError(INVALID_ARGS, "INVALID_MODEL",
                             "local RAI import requires --descriptor FILE")
        try:
            with open(descriptor_path) as f:
                descriptor = json.load(f)
        except OSError as e:
            raise FxdnaError(INVALID_ARGS, "INVALID_MODEL",
                             f"cannot read descriptor: {descriptor_path!r}"
                             ) from e
        except ValueError as e:
            raise FxdnaError(INVALID_ARGS, "INVALID_MODEL",
                             "descriptor is not valid JSON") from e
        if not isinstance(descriptor, dict):
            raise FxdnaError(INVALID_ARGS, "INVALID_MODEL",
                             "descriptor must be a JSON object")
        for need in ("artifact_sha256", "target_profile", "serving"):
            if need not in descriptor:
                raise FxdnaError(INVALID_ARGS, "INVALID_MODEL",
                                 f"RAI descriptor lacks {need!r}")
        data = _read_bounded(path, "local RAI")
        if sha256_bytes(data) != descriptor["artifact_sha256"]:
            raise FxdnaError(CACHE_CORRUPT, "CACHE_CORRUPT",
                             "RAI bytes do not match descriptor hash")
        return self._ingest_source(parsed["ref"], alias, data,
                                   "imported-rai",
                                   {"descriptor": descriptor}, refresh)

    def _publish_result(self, job: dict) -> None:
        """Commit a backend's PREPARED result via the atomic-publish path.

        Fake backend: clearly-marked FAKE bytes; `artifact.json` records
        backend=fake-v0. Activation MUST refuse fake-backend artifacts
        (Task 04 gate — see WORKQUEUE). Never VERIFIED, never ACTIVE.
        Real backend: the compiler child's .rai + manifest carrying the
        audited backend id, recipe, target and compile stats.
        """
        ckey = job["compile_key"]
        if self.registry.get_artifact(ckey):
            return
        # A directory without a row (quarantined/colliding) must not wedge
        # the key: move it aside first (SF4). Committed rows are never
        # overwritten (publish_artifact enforces that).
        dest = os.path.join(self.data_dir, "artifacts", ckey)
        if os.path.isdir(dest):
            aside = os.path.join(
                self.data_dir, "failures",
                f"colliding-{ckey}-{int(time.time())}")
            os.rename(dest, aside)
        staged = os.path.join(self.data_dir, "work", f"stage-{job['uuid']}")
        os.makedirs(staged, exist_ok=True)
        source = (self.registry.get_ref(job["ref"]) or {}).get(
            "source_sha256") or ""
        backend = self.jobs.backend_for(job["uuid"])
        # Backend identity comes from the producer object, never from config,
        # so a fake job can never be published with the audited backend id.
        backend_id = getattr(backend, "BACKEND_ID", None) or getattr(
            backend, "backend_id", None) or self.compiler_backend_id
        # Only the real backend's result carries a .rai path; fake has none.
        result = getattr(backend, "result", None)
        if result is not None and getattr(result, "rai_path", ""):
            with open(result.rai_path, "rb") as f:
                rai_bytes = f.read()
            manifest = {
                "backend": backend_id,
                "compile_key": ckey, "source_sha256": source,
                "artifact_sha256": sha256_bytes(rai_bytes),
                "recipe_id": RECIPE_ID, "target_profile": TARGET_PROFILE,
                "compile_stats": {
                    "wall_s": round(result.wall_s, 1),
                    "peak_rss_kb": result.peak_rss_kb,
                    "bf16_sha256": result.bf16_sha256},
            }
        else:
            rai_bytes = b"FXDNA-FAKE-RAI-v0:" + ckey.encode()
            manifest = {
                "backend": backend_id,
                "compile_key": ckey, "source_sha256": source,
                "artifact_sha256": sha256_bytes(rai_bytes),
                "recipe_id": RECIPE_ID, "target_profile": TARGET_PROFILE,
                "note": "fake compile stand-in; not deployable"}
        publish_artifact(self.data_dir, ckey, staged, {
            "model.rai": rai_bytes,
            "artifact.json": json.dumps(manifest, sort_keys=True).encode(),
        })
        self.registry.add_artifact(
            ckey, source, sha256_bytes(rai_bytes), len(rai_bytes),
            RECIPE_ID, TARGET_PROFILE)

    def _mark_prepared_refs(self, compile_key: str) -> None:
        """All aliases sharing a compile key reach PREPARED together."""
        seen: set[str] = set()
        for row in self.registry.query(
                "SELECT DISTINCT ref FROM jobs WHERE compile_key=?",
                (compile_key,)):
            seen.add(row[0])
        for row in self.registry.query(
                "SELECT DISTINCT ref FROM job_aliases WHERE job_uuid IN"
                " (SELECT uuid FROM jobs WHERE compile_key=?)",
                (compile_key,)):
            seen.add(row[0])
        for ref in seen:
            self.registry.set_ref_state(ref, "PREPARED")

    def _backend_ok(self, compile_key: str) -> bool:
        """A cached row is usable only if its manifest names this backend."""
        try:
            with open(os.path.join(
                    self.data_dir, "artifacts", compile_key,
                    "artifact.json"), "rb") as f:
                manifest = json.loads(f.read().decode())
        except (OSError, ValueError):
            return False
        return isinstance(manifest, dict) and \
            manifest.get("backend") == self.compiler_backend_id

    def _invalidate_artifact(self, compile_key: str) -> None:
        """Move a stale artifact aside and drop its row (never in place)."""
        src = os.path.join(self.data_dir, "artifacts", compile_key)
        if os.path.isdir(src):
            dest = os.path.join(
                self.data_dir, "failures",
                f"stale-{compile_key}-{int(time.time())}")
            os.rename(src, dest)
        self.registry.execute(
            "DELETE FROM artifacts WHERE compile_key=?", (compile_key,))

    def _ingest_source(self, ref: str, alias: str, data: bytes, origin: str,
                       extra: dict, refresh: bool) -> dict:
        digest = sha256_bytes(data)
        prev = self.registry.get_ref(ref)
        old_digest = (prev or {}).get("source_sha256")
        if old_digest and old_digest != digest and not refresh:
            self.registry.set_ref_state(ref, "SOURCE_CHANGED")
            raise FxdnaError(INVALID_ARGS, "SOURCE_CHANGED",
                             f"source bytes changed under {ref}; kept "
                             f"previous artifact (use --refresh to accept)")
        ingest_bytes(self.data_dir, data, "sources", "model.onnx"
                     if origin != "imported-rai" else "model.rai")
        self.registry.add_source(
            digest, len(data), f"sources/{digest}/"
            f"{'model.onnx' if origin != 'imported-rai' else 'model.rai'}",
            origin)
        meta_digest = extra.get("metadata_sha256")
        self.registry.set_ref_source(ref, digest, meta_digest, "DOWNLOADED")
        if origin == "imported-rai":
            return {"ref": ref, "source_sha256": digest,
                    "state": "DOWNLOADED", "note": "imported RAI recorded"}
        # Compile key over the pinned recipe identity. Geometry comes from
        # the inspected graph and is mandatory: a missing shape must fail
        # (UNSUPPORTED_CONTRACT), never fall back to a default that would
        # fabricate a compile identity.
        inspected = extra.get("inspected", {})
        geometry = inspected.get("input_shape")
        if not (isinstance(geometry, list) and len(geometry) == 4 and
                all(isinstance(v, int) for v in geometry)):
            raise FxdnaError(UNSUPPORTED_CONTRACT, "UNSUPPORTED_CONTRACT",
                             "inspected input geometry missing; refusing to"
                             " key compilation on a default shape")
        ckey = _compile_key(
            digest, COMPILER_PAYLOAD_SHA256, RECIPE_ID,
            RECIPE_CONFIG_SHA256, TARGET_PROFILE, ARTIFACT_COMPAT_ID,
            {"shape": geometry, "dtype": "float32"})
        if self.registry.get_artifact(ckey):
            if not self._backend_ok(ckey):
                # Stale backend (e.g. fake row vs real compiler): invalidate
                # and compile fresh instead of trusting foreign bytes (SF3).
                self._invalidate_artifact(ckey)
            else:
                self.registry.set_ref_state(ref, "PREPARED")
                return {"ref": ref, "source_sha256": digest,
                        "compile_key": ckey, "state": "PREPARED",
                        "cache_hit": True}
        pre = disk_preflight(self.data_dir, COMPILE_SCRATCH_NEED_BYTES)
        if not pre["ok"]:
            raise FxdnaError(6, "RESOURCE_EXCEEDED",
                             f"insufficient scratch: need "
                             f"{pre['need_bytes']} + reserve "
                             f"{pre['reserve_bytes']}, free "
                             f"{pre['free_bytes']}")
        job = self.jobs.submit(
            ref, ckey, **self.fake_compile,
            extra={"source_sha256": digest,
                   "source_path": os.path.join(
                       self.data_dir, "sources", digest,
                       "model.onnx" if origin != "imported-rai"
                       else "model.rai")})
        if job["stage"] in TERMINAL_ERROR_STATES:
            # A sticky terminal failure must be visible on the ref, not
            # masked as QUEUED by a job that will never run.
            self.registry.set_ref_state(ref, job["stage"])
        else:
            self.registry.set_ref_state(ref, "QUEUED")
        out = {"ref": ref, "source_sha256": digest, "compile_key": ckey,
               "job_uuid": job["uuid"], "state": job["stage"],
               "cache_hit": False}
        return out

    def pump(self, dt_s: float = 0.05) -> None:
        """Advance all non-terminal jobs (fake backend).

        Per-job isolation: one failing job is marked failed, never kills
        the loop or the daemon (SF4).
        """
        for row in self.registry.query(
                "SELECT uuid FROM jobs WHERE stage NOT IN"
                " ('PREPARED','COMPILE_FAILED','RESOURCE_EXCEEDED',"
                " 'VALIDATION_FAILED','UNSUPPORTED_CONTRACT','QUARANTINED',"
                " 'INTERRUPTED')"):
            try:
                job = self.jobs.pump(row[0], dt_s)
            except FxdnaError as e:
                self.registry.set_job(row[0], "COMPILE_FAILED",
                                      error_code=e.error_code)
                for ref in self.registry.refs_for_job(row[0]):
                    self.registry.set_ref_state(ref, "COMPILE_FAILED")
                continue
            if job["stage"] in TERMINAL_ERROR_STATES:
                for ref in self.registry.refs_for_job(row[0]):
                    self.registry.set_ref_state(ref, job["stage"])
                continue
            if job["stage"] == "PREPARED":
                if job.get("compile_key"):
                    try:
                        self._publish_result(job)
                    except FxdnaError as e:
                        self.registry.set_job(
                            row[0], "COMPILE_FAILED",
                            error_code=e.error_code)
                        for ref in self.registry.refs_for_job(row[0]):
                            self.registry.set_ref_state(ref, "COMPILE_FAILED")
                        continue
                    self._mark_prepared_refs(job["compile_key"])
                else:
                    self.registry.set_ref_state(job["ref"], "PREPARED")

    def wait_job(self, job_uuid: str, timeout_s: float,
                 pump: bool = True) -> dict:
        """Wait for a terminal job state.

        pump=True drives the fake backend (standalone use). The daemon's
        admin path passes pump=False: the serve loop advances jobs, so one
        waiting admin call never blocks the socket behind a compile.
        """
        step = 0.05
        import time as _time
        deadline = _time.monotonic() + timeout_s
        while True:
            job = self.jobs.pump(job_uuid, step) if pump else \
                self.jobs.registry.get_job(job_uuid)
            if job is None:
                raise FxdnaError(NOT_READY, "UNKNOWN_JOB",
                                 f"no such job {job_uuid}")
            if job["stage"] == "PREPARED":
                if pump and job.get("compile_key"):
                    try:
                        self._publish_result(job)
                    except FxdnaError as e:
                        self.registry.set_job(
                            job_uuid, "COMPILE_FAILED",
                            error_code=e.error_code)
                        for ref in self.registry.refs_for_job(job_uuid):
                            self.registry.set_ref_state(ref, "COMPILE_FAILED")
                        return self.registry.get_job(job_uuid)
                if job.get("compile_key"):
                    self._mark_prepared_refs(job["compile_key"])
                else:
                    self.registry.set_ref_state(job["ref"], "PREPARED")
                return job
            if job["stage"] in TERMINAL_ERROR_STATES:
                for ref in self.registry.refs_for_job(job_uuid):
                    self.registry.set_ref_state(ref, job["stage"])
                return job
            if _time.monotonic() >= deadline:
                raise FxdnaError(NOT_READY, "WAIT_TIMEOUT",
                                 f"job {job_uuid} still {job['stage']} "
                                 f"after {timeout_s}s")
            _time.sleep(min(step, 0.05))

    # -- reads ---------------------------------------------------------
    def status(self, ref: str | None = None) -> dict:
        """Live-manager status. State SERVING means this manager process is
        serving the admin/queue plane; it never claims inference readiness
        (no ACTIVE worker exists in Task 02)."""
        if ref:
            parsed = parse_ref(ref)
            rec = self.registry.get_ref(parsed["ref"])
            models = [self._model_view(rec)] if rec else []
        else:
            models = [self._model_view(dict(zip(
                ("ref", "kind", "model_id", "source_sha256",
                 "metadata_sha256", "pin", "state"), r))) for r in
                self.registry.query(
                    "SELECT ref, kind, model_id, source_sha256,"
                    " metadata_sha256, pin, state FROM model_refs")]
        return {"schema_version": 1, "service": "frigate-xdna",
                "version": __version__, "state": "SERVING",
                "active": self.registry.get_state("active"),
                "models": models,
                "inhibition": self.registry.get_state("inhibition")}

    def _model_view(self, rec: dict | None) -> dict:
        if rec is None:
            return {}
        return {"ref": rec["ref"],
                "source_sha256": rec.get("source_sha256") or "",
                "state": rec.get("state") or "NEW"}

    def cache_list(self) -> list[dict]:
        rows = self.registry.query(
            "SELECT compile_key, source_sha256, artifact_size, recipe_id"
            " FROM artifacts")
        return [dict(zip(("compile_key", "source_sha256", "bytes",
                           "recipe"), r)) for r in rows]

    def source_list(self) -> list[dict]:
        rows = self.registry.query(
            "SELECT sha256, size_bytes, origin FROM sources")
        return [dict(zip(("sha256", "bytes", "origin"), r)) for r in rows]

    def activate(self, ref: str, maintenance: bool = False) -> dict:
        raise FxdnaError(NOT_READY, "ACTIVATION_UNAVAILABLE",
                         "native activation arrives with the Task 04 worker; "
                         "preparation state is queryable via status")

    def recover_ref(self, ref: str) -> dict:
        parsed = parse_ref(ref)
        with self.registry.transaction():
            inh = self.registry.get_state("inhibition")
            if not (inh and inh.get("ref") == parsed["ref"]):
                return {"ref": parsed["ref"], "cleared": False,
                        "note": "no inhibition recorded for this ref"}
            # Remove the inhibition; persist the evidence separately so a
            # later recover no longer finds it. One transaction: concurrent
            # recoveries cannot clear the same inhibition twice, and a
            # crash rolls back both halves together.
            self.registry.execute("DELETE FROM service_state WHERE key=?",
                                  ("inhibition",))
            self.registry.set_state("last_inhibition_cleared",
                                    {"ref": parsed["ref"],
                                     "previous": inh,
                                     "at": time.time()})
            return {"ref": parsed["ref"], "cleared": True}

    def prune(self, apply: bool = False,
              max_bytes: int | None = None) -> dict:
        plan = _gc.plan_prune(self.data_dir, self.registry, max_bytes)
        if not apply:
            return {"dry_run": True, **plan}
        with locked(os.path.join(self.data_dir, "locks", "prune.lock")):
            return {"dry_run": False, **_gc.apply_prune(
                self.data_dir, self.registry, plan, max_bytes)}
