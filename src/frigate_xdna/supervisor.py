"""Supervisor: validated config, registry, preparation queue, admin socket.

The daemon is the only registry writer. No Quark/torch/vendor-ONNX-Runtime/
FlexML imports here or anywhere in the manager process (Task 02 acceptance).
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid

from .admin import AdminServer
from .build_identity import get_build_identity
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
    try_exclusive,
)
from .compiler.jobs import TERMINAL_ERROR_STATES, JobManager
from .compiler.real import BACKEND_ID as REAL_BACKEND_ID
from .config import Config
from .errors import (
    CACHE_CORRUPT,
    DEVICE_UNAVAILABLE,
    INVALID_ARGS,
    NOT_READY,
    OWNERSHIP_CONFLICT,
    UNSUPPORTED_CONTRACT,
    VALIDATION_FAILED,
    FxdnaError,
)
from .models import inspect as _inspect
from .models.refs import parse_ref, wire_alias
from .plus.client import PlusClient
from .runtime import native as _native
from .runtime.safety import inhibit as _inhibit

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
SOURCE_ONNX_NAME = "model.onnx"  # sources/<sha>/ content filename


def boot_token() -> str:
    try:
        with open("/proc/sys/kernel/random/boot_id") as f:
            return f.read().strip()
    except OSError:
        return "unknown"


def _read_secret(config: Config) -> str | None:
    # PLUS_API_KEY (container environment) is the sole credential
    # input. It is never logged, never baked into images, and never
    # forwarded to compiler/native child environments.
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
                 compiler_timeout_s: float = 2700.0,
                 worker_factory=None,
                 worker_bin: str | None = None,
                 reporter=None,
                 heartbeat_s: float = 30.0):
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
            if compiler_prefixes is not None:
                compiler_backend_id = REAL_BACKEND_ID
            else:
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
                               boot_token=boot_token(),
                               listener=self._job_event)
        self.fake_compile = fake_compile or {}
        self.compiler_backend_id = compiler_backend_id
        self.compiler_prefixes = compiler_prefixes
        self.compiler_timeout_s = compiler_timeout_s
        self._plus_client_factory = plus_client_factory or PlusClient
        # Test-only affordance for loopback fake Plus servers. Production
        # default is empty: only public https download targets are allowed.
        self._plus_allow_private = tuple(plus_allow_private_hosts)
        # Worker supervision: one resident child per active model.
        # worker_factory is the control-plane seam (tests inject a fake);
        # production default spawns the audited fxdna-worker binary.
        self._worker_factory = worker_factory
        self._worker_bin = worker_bin or _native.worker_binary()
        self._worker = None
        self._worker_generation = 0
        self._worker_compile_key: str | None = None
        self._worker_lock = threading.Lock()
        # Children that failed to retire on a rollback path but may
        # still be live: explicitly tracked for cleanup at stop(),
        # never silently dropped (no untracked live workers).
        self._orphans: list = []
        # Set by serve wiring after frontend creation (None standalone).
        self.frontend = None
        self._server: AdminServer | None = None
        # Console progress: None is silent (tests); serve passes a
        # ConsoleReporter. Events describe real transitions only.
        self.reporter = reporter
        self.heartbeat_s = heartbeat_s
        self._seen_stages: dict[str, str] = {}
        self._seen_ref_states: dict[str, str] = {}
        self._heartbeats: dict[str, float] = {}
        self._last_worker_generation = 0
        self._worker_serving_digest: str | None = None
        recover(self.data_dir, self.registry)
        self._reconcile_jobs()
        self._baseline_progress()

    def emit(self, event: dict) -> None:
        """Report one real transition; silent without a reporter."""
        if self.reporter is not None:
            self.reporter.emit(event)

    def _job_event(self, event: dict) -> None:
        """JobManager retry/resume transitions become console events."""
        self.emit(event)

    def _interrupt_record(self, stage: str, row_boot: str | None,
                          attempt: int) -> dict:
        """Structured record for a restart-interrupted job.

        Resume safety comes from records, never errno alone: a row
        whose boot token differs from this boot may have died with a
        host reboot mid-operation (the suspect-host-reset case), so it
        is unproven safety — never automatic. Same-boot interruption
        with a clean safety state is a safe resume. Any inhibition
        blocks automatic resume either way.
        """
        from . import retry_policy as _retry
        current = boot_token()
        base = _retry.new_record(
            "INTERRUPTED", "INTERRUPTED", "", attempt or 1)
        # A missing row token is unproven too: legacy or partially
        # written rows with NULL boot_token must not bypass the host
        # reboot check just because the current token is available.
        if (self.registry.get_state("inhibition") is not None
                or current == "unknown"
                or not row_boot
                or row_boot != current):
            base["reason"] = (
                "interrupted with an unproven safety state"
                f" (row boot {row_boot or '?'} vs current {current};"
                " a mismatch may mean a host reboot mid-operation);"
                " explicit operator review required")
            base["kind"] = "unknown"
            base["retryable"] = True
            base["auto"] = False
            base["guidance"] = (
                "Interrupted with an unproven safety state (possible"
                " host reboot or active inhibition): no automatic"
                " resume. Review status/diagnose evidence, then recover"
                " explicitly if safe.")
            return base
        base["reason"] = (f"interrupted in {stage} by process restart;"
                          " clean safety state")
        return base

    def _reconcile_jobs(self) -> None:
        """Interrupt stranded rows and record them.

        store.recover handles workdir-backed jobs (marking them
        INTERRUPTED without records); rows without a live backend in
        THIS process are interrupted here. Both shapes get structured
        records via _interrupt_record — including INTERRUPTED rows
        that lack one — so resume decisions never run on bare stages.
        """
        for row in self.registry.query(
                "SELECT uuid, ref, stage, attempt, boot_token FROM jobs"
                " WHERE stage NOT IN ('PREPARED','COMPILE_FAILED',"
                "'RESOURCE_EXCEEDED','VALIDATION_FAILED',"
                "'UNSUPPORTED_CONTRACT','QUARANTINED','INTERRUPTED')"):
            uuid, ref, stage, attempt, row_boot = row
            if self.registry.get_job(uuid) is None:
                continue
            self.registry.set_job(uuid, "INTERRUPTED",
                                  error_code="INTERRUPTED")
            self.registry.set_ref_state(ref, "INTERRUPTED")
            self.registry.set_failure(
                uuid, self._interrupt_record(stage, row_boot,
                                             attempt or 1))
        for row in self.registry.query(
                "SELECT uuid, ref, stage, attempt, boot_token FROM jobs"
                " WHERE stage='INTERRUPTED' AND failure_json IS NULL"):
            uuid, ref, stage, attempt, row_boot = row
            self.registry.set_ref_state(ref, "INTERRUPTED")
            self.registry.set_failure(
                uuid, self._interrupt_record(stage, row_boot,
                                             attempt or 1))

    def _compile_source_path(self, sha256: str) -> str:
        """Bytes path for one source digest (resume binding).

        Raises when the bytes are gone: the key stays recompilable by
        explicit re-submit, but a resume must never start sourceless
        or, worse, on the ref's newer bytes under an old key.
        """
        src = self.registry.get_source(sha256)
        if src is None:
            raise FxdnaError(CACHE_CORRUPT, "CACHE_CORRUPT",
                             f"source bytes missing for resumed compile"
                             f" {sha256[:12]}…; re-submit explicitly")
        path = os.path.join(self.data_dir, "sources",
                            src["sha256"],
                            os.path.basename(src["rel_path"]))
        if not os.path.isfile(path):
            raise FxdnaError(CACHE_CORRUPT, "CACHE_CORRUPT",
                             f"source file missing for resumed compile"
                             f" {sha256[:12]}…; re-submit explicitly")
        return path

    def _make_backend_job(self, **kw):
        """Job factory: real audited backend when prefixes are configured,
        fake backend otherwise (hardware-free tests + offline development).

        A resumed real compile binds the failed attempt's source bytes
        (JobManager carries the sha on the row and in kwargs), never
        the ref's current bytes: compiling newer bytes under an old
        key would corrupt the content-addressed cache.
        """
        if self.compiler_prefixes is not None and kw.get("compile_key"):
            from .compiler.real import RealCompileJob
            source_path = kw.get("source_path", "")
            if not source_path:
                source_path = self._compile_source_path(
                    kw.get("source_sha256") or "")
            return RealCompileJob(
                source_sha256=kw.get("source_sha256", ""),
                compile_key=kw.get("compile_key", ""),
                source_path=source_path,
                workdir=os.path.join(self.data_dir, "work",
                                     kw.get("job_uuid", "nojobs")),
                prefixes=self.compiler_prefixes,
                timeout_s=self.compiler_timeout_s,
                data_dir=self.data_dir,
                worker_factory=self._worker_factory)
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
        with self._worker_lock:
            worker, self._worker = self._worker, None
            self._worker_compile_key = None
            self._worker_serving_digest = None
            if worker is not None:
                try:
                    worker.retire()
                except Exception:
                    pass
            orphans, self._orphans = self._orphans, []
            for orphan in orphans:
                try:
                    orphan.retire()
                except Exception:
                    pass
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
        if cmd == "health":
            return {"health": self.health()}
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
        if cmd == "infer_stats":
            if self.frontend is None:
                return {"infer_stats": None,
                        "note": "no frontend in this process"}
            stats = self.frontend.get_stats()
            stats["worker_generation"] = self._worker_generation
            stats["worker_loaded"] = (
                self._worker is not None
                and getattr(self._worker, "loaded", False))
            stats["inhibition"] = self.registry.get_state("inhibition")
            return {"infer_stats": stats}
        if cmd == "prune":
            return self.prune(bool(req.get("apply", False)),
                              req.get("max_bytes"))
        if cmd == "recover":
            return {"recovered": self.recover_ref(req.get("ref", ""))}
        raise FxdnaError(INVALID_ARGS, "INVALID_ARGS",
                         f"unknown admin command {cmd!r}")

    # -- preparation --------------------------------------------------
    def reconcile_configured(self) -> list[dict]:
        """Startup/desired-state reconciliation for configured refs.

        Each FXDNA_MODELS ref is prepared; failures are reported per
        ref, never raised. Operator-owned local sources (`local://ID`
        and explicit `.onnx` paths) are mutable aliases, so changed
        bytes are accepted here as a new revision: inspected, prepared
        alongside the preserved previous artifact, with no worker
        switch (activation stays byte-bound and explicit). Plus refs
        never auto-refresh (SPEC §4.4: no silent re-acquisition).
        Interactive `prepare` without `--refresh` still reports
        SOURCE_CHANGED for explicit acknowledgement.
        """
        outcomes = []
        for ref in self.config.models:
            try:
                kind = parse_ref(ref, self.config.model_dir)["kind"]
            except FxdnaError:
                kind = None
            try:
                out = self.prepare(
                    ref, refresh=kind in ("local", "onnx"))
                outcomes.append({"ref": ref, "ok": True,
                                 "state": out.get("state"),
                                 "compile_key": out.get("compile_key"),
                                 "cache_hit": out.get("cache_hit")})
            except FxdnaError as e:
                outcomes.append({"ref": ref, "ok": False,
                                 "code": e.error_code,
                                 "reason": e.message})
        return outcomes

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

    def _inspect_checked(self, ref: str, data: bytes,
                           metadata: dict | None = None
                           ) -> tuple[dict, dict, str, dict]:
        """One graph-inspection path for preparation, logs and status.

        Runs the same `inspect_model`/`classify_output` every caller
        uses, records the summary on the ref row (compatible or
        refused), and emits the console event. Failures raise before
        any compile is queued, so an unsupported graph is a stable
        INSPECTION refusal, never a generic COMPILE_FAILED.
        """
        contract = cls = None
        digest = None
        try:
            model, digest = _inspect.load_graph_bytes(data)
            contract = _inspect.inspect_model(model)
            if metadata is not None:
                _inspect.compare_plus_metadata(metadata, contract)
            cls = _inspect.classify_output(contract["outputs"])
            if cls["profile"] is None:
                raise FxdnaError(UNSUPPORTED_CONTRACT,
                                 "UNSUPPORTED_CONTRACT", cls["error"])
        except FxdnaError as e:
            self.registry.set_ref_inspection(
                ref, _inspect.summarize_inspection(contract, cls, e,
                                                   digest))
            self.emit({"kind": "preparation_failed", "ref": ref,
                       "phase": "INSPECTION", "code": e.error_code,
                       "reason": e.message})
            raise
        summary = _inspect.summarize_inspection(contract, cls, None,
                                                digest)
        self.registry.set_ref_inspection(ref, summary)
        shape = contract.get("input_shape")
        self.emit({"kind": "inspection_complete", "ref": ref,
                   "profile": cls["profile"], "shape": shape,
                   "shape_str": "x".join(str(v) for v in shape),
                   "classes": summary["class_count"]})
        return contract, cls, digest, summary

    def prepare(self, ref: str, descriptor_path: str | None = None,
                wire_name: str | None = None, refresh: bool = False,
                maintenance: bool = False) -> dict:
        parsed = parse_ref(ref, self.config.model_dir)
        alias = wire_alias(parsed, wire_name)
        self.registry.upsert_ref(parsed["ref"], parsed["kind"],
                                 parsed.get("id"))
        if parsed["ref"] in self.config.models:
            _gc.pin_ref(self.registry, parsed["ref"], "configured")
        if parsed["kind"] == "plus":
            self.emit({"kind": "downloading_model", "ref": parsed["ref"]})
            try:
                data, fetched = self._fetch_plus(parsed["id"], refresh)
            except FxdnaError as e:
                # SPEC §4.4: never downgrade a working cached model
                # because a refresh/token/DNS lookup fails. New
                # downloads fail visibly and independently; a usable
                # cached artifact keeps serving.
                cached = self._usable_cached_key(parsed["ref"])
                if cached is None:
                    self.emit({"kind": "preparation_failed",
                               "ref": parsed["ref"], "phase": "ACQUISITION",
                               "code": e.error_code, "reason": e.message})
                    raise
                self.emit({"kind": "fetch_failed_cached",
                           "ref": parsed["ref"], "code": e.error_code,
                           "reason": e.message})
                self.registry.set_ref_state(parsed["ref"], "PREPARED")
                return {"ref": parsed["ref"],
                        "compile_key": cached, "state": "PREPARED",
                        "cache_hit": True,
                        "note": f"fetch failed [{e.error_code}];"
                                " serving cached artifact"}
            # Plus bytes are inspected exactly like local files: metadata
            # conflicts and unsupported contracts fail here, never queue a
            # fake compile to PREPARED. (B3)
            contract, cls, digest, _summary = self._inspect_checked(
                parsed["ref"], data, fetched["metadata"])
            fetched["inspected"] = contract
            fetched["profile"] = cls["profile"]
            return self._ingest_source(parsed["ref"], alias, data, "plus",
                                       fetched, refresh)
        if parsed["kind"] == "local":
            # Alias ref: the registry row keeps the stable local://ID
            # while ingestion hashes the resolved file's exact bytes.
            # Changed bytes are a new revision (SOURCE_CHANGED), never
            # a silent mutation — see _ingest_source.
            path = parsed["path"]
        else:
            path = parsed.get("path", "")
        if not path or not os.path.isfile(path):
            raise FxdnaError(INVALID_ARGS, "INVALID_MODEL",
                             f"local file not found: {path!r}")
        if parsed["kind"] in ("onnx", "local"):
            try:
                data = _read_bounded(path, "local ONNX")
            except FxdnaError as e:
                self.emit({"kind": "preparation_failed",
                           "ref": parsed["ref"], "phase": "INSPECTION",
                           "code": e.error_code, "reason": e.message})
                raise
            contract, cls, digest, _summary = self._inspect_checked(
                parsed["ref"], data)
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
        # The artifact was compiled from the job's source bytes, which
        # may no longer be the ref's current source: a revision can
        # advance (reconcile/refresh) while an older attempt is still
        # resumable. Publish with the job-bound digest; the ref digest
        # is only a fallback for legacy rows that predate it.
        source = job.get("source_sha256") or (
            self.registry.get_ref(job["ref"]) or {}).get(
            "source_sha256") or ""
        backend = self.jobs.backend_for(job["uuid"])
        # Backend identity comes from the producer object, never from config.
        if backend is None:
            raise FxdnaError(5, "COMPILE_FAILED",
                             "no backend object for job")
        backend_id = getattr(backend, "BACKEND_ID", None) or getattr(
            backend, "backend_id", None)
        if not backend_id:
            raise FxdnaError(5, "COMPILE_FAILED",
                             "backend identity unavailable")
        # Only the real backend's result carries a .rai path; fake has none.
        # Fake-byte publication is allowed only for FakeCompileJob.
        from .compiler.fake import FakeCompileJob as _Fake
        is_fake = isinstance(backend, _Fake)
        result = getattr(backend, "result", None)
        if result is not None and getattr(result, "rai_path", ""):
            try:
                with open(result.rai_path, "rb") as f:
                    rai_bytes = f.read()
            except OSError as e:
                raise FxdnaError(5, "COMPILE_FAILED",
                                 f"cannot read compiler artifact: {e}") from e
            manifest = {
                "backend": backend_id,
                "compile_key": ckey, "source_sha256": source,
                "artifact_sha256": sha256_bytes(rai_bytes),
                "recipe_id": RECIPE_ID, "target_profile": TARGET_PROFILE,
                "compile_stats": {
                    "wall_s": round(result.wall_s, 1),
                    "peak_rss_kb": result.peak_rss_kb,
                    "vm_peak_kb": result.vm_peak_kb,
                    "bf16_sha256": result.bf16_sha256},
            }
        elif is_fake:
            rai_bytes = b"FXDNA-FAKE-RAI-v0:" + ckey.encode()
            manifest = {
                "backend": backend_id,
                "compile_key": ckey, "source_sha256": source,
                "artifact_sha256": sha256_bytes(rai_bytes),
                "recipe_id": RECIPE_ID, "target_profile": TARGET_PROFILE,
                "note": "fake compile stand-in; not deployable"}
        else:
            raise FxdnaError(5, "COMPILE_FAILED",
                             "real backend produced no artifact")
        publish_artifact(self.data_dir, ckey, staged, {
            "model.rai": rai_bytes,
            "artifact.json": json.dumps(manifest, sort_keys=True).encode(),
        })
        self.registry.add_artifact(
            ckey, source, sha256_bytes(rai_bytes), len(rai_bytes),
            RECIPE_ID, TARGET_PROFILE)
        result = getattr(backend, "result", None)
        if getattr(result, "probe", "") == "ok":
            # Recorded native check for this runtime/target: the
            # artifact ran a finite zeros INFER through the real
            # worker path during compilation. Serving binding is
            # still established at activation.
            self._record_native_check(
                ckey, manifest["artifact_sha256"], "",
                "probe-zeros-finite", "incompile-probe-v1", {})

    def _record_native_check(self, compile_key: str, artifact_sha256: str,
                             serving_digest: str, reason: str, suite: str,
                             extra: dict) -> None:
        """Persist VERIFIED evidence for an artifact (sticky; failures
        never clear it, passes never need repeating)."""
        from .cache.keys import validation_key as _vkey
        runtime = {"backend": self.compiler_backend_id,
                   "target_profile": TARGET_PROFILE}
        runtime.update(extra or {})
        vkey = _vkey(artifact_sha256, serving_digest, runtime, suite)
        self.registry.record_validation(
            vkey, compile_key, serving_digest, True, reason, "",
            json.dumps(runtime, sort_keys=True))

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

    def _usable_cached_key(self, ref: str) -> str | None:
        """Newest prepared key with a committed, backend-matching
        artifact on disk (no download, no compile needed)."""
        key = self.registry.prepared_key_for_ref(ref)
        if not key or not self.registry.get_artifact(key):
            return None
        if not self._backend_ok(key):
            return None
        if not os.path.isfile(os.path.join(
                self.data_dir, "artifacts", key, "model.rai")):
            return None
        return key

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
            self.emit({"kind": "preparation_failed", "ref": ref,
                       "phase": "INGEST", "code": "SOURCE_CHANGED",
                       "reason": "source bytes changed under this ref;"
                                 " previous artifact kept"})
            raise FxdnaError(INVALID_ARGS, "SOURCE_CHANGED",
                             f"source bytes changed under {ref}; kept "
                             f"previous artifact (use --refresh to accept)")
        ingest_bytes(self.data_dir, data, "sources", SOURCE_ONNX_NAME
                     if origin != "imported-rai" else "model.rai")
        self.registry.add_source(
            digest, len(data), f"sources/{digest}/"
            f"{'model.onnx' if origin != 'imported-rai' else 'model.rai'}",
            origin)
        meta_digest = extra.get("metadata_sha256")
        self.registry.set_ref_source(ref, digest, meta_digest, "DOWNLOADED")
        if origin == "imported-rai":
            self.emit({"kind": "preparing_model", "ref": ref})
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
        sdigest = _serving_digest(ckey, inspected)
        if self.registry.get_artifact(ckey):
            if not self._backend_ok(ckey):
                # Stale backend (e.g. fake row vs real compiler): invalidate
                # and compile fresh instead of trusting foreign bytes (SF3).
                self._invalidate_artifact(ckey)
            else:
                # Record the hit as a provenance row (no compile
                # launched, so no job_uuid is reported): every PREPARED
                # ref traces to a job row, which activate-by-ref
                # requires (a ref with no job row could never be
                # activated). Terminal rows are inert to pump/GC.
                hit_uuid = uuid.uuid4().hex
                self.registry.create_job(hit_uuid, ref, ckey,
                                         boot_token(), source_sha256=digest)
                self.registry.set_job(hit_uuid, "PREPARED")
                self.registry.set_ref_state(ref, "PREPARED")
                self.emit({"kind": "model_cached", "ref": ref})
                return {"ref": ref, "source_sha256": digest,
                        "compile_key": ckey, "serving_digest": sdigest,
                        "state": "PREPARED", "cache_hit": True}
        pre = disk_preflight(self.data_dir, COMPILE_SCRATCH_NEED_BYTES)
        if not pre["ok"]:
            self.emit({"kind": "preparation_failed", "ref": ref,
                       "phase": "SCRATCH", "code": "RESOURCE_EXCEEDED",
                       "reason": f"need {pre['need_bytes']} + reserve"
                                 f" {pre['reserve_bytes']}, free"
                                 f" {pre['free_bytes']}"})
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
                       SOURCE_ONNX_NAME if origin != "imported-rai"
                       else "model.rai")})
        self.emit({"kind": "preparing_model", "ref": ref,
                   "job_uuid": job.get("uuid")})
        if job["stage"] in TERMINAL_ERROR_STATES:
            # A sticky terminal failure must be visible on the ref, not
            # masked as QUEUED by a job that will never run.
            self.registry.set_ref_state(ref, job["stage"])
        else:
            self.registry.set_ref_state(ref, "QUEUED")
        out = {"ref": ref, "source_sha256": digest, "compile_key": ckey,
               "serving_digest": sdigest, "job_uuid": job["uuid"],
               "state": job["stage"], "cache_hit": False}
        return out

    def _live_worker_key(self) -> str | None:
        """Compile key of the currently serving worker, or None.

        Passive: poll() never touches the NPU. A dead or unloaded
        worker serves nothing, so it projects nothing."""
        worker = self._worker
        if worker is None or not getattr(worker, "loaded", False):
            return None
        try:
            if not worker.alive():
                return None
        except Exception:
            return None
        return self._worker_compile_key

    def _worker_info(self) -> dict | None:
        worker = self._worker
        if worker is None:
            return None
        try:
            alive = worker.alive()
        except Exception:
            alive = False
        return {"compile_key": self._worker_compile_key,
                "generation": self._worker_generation,
                "serving_digest": self._worker_serving_digest,
                "loaded": bool(getattr(worker, "loaded", False)),
                "alive": alive}

    def health(self) -> dict:
        """Passive liveness + readiness (no NPU context, no compile,
        no test inference)."""
        info = self._worker_info()
        inhibition = self.registry.get_state("inhibition")
        ready, reason = False, "no active worker"
        if info is not None and info["loaded"] and info["alive"]:
            if inhibition:
                reason = f"inhibited: {inhibition.get('reason', '?')}"
            elif not info["compile_key"] or not self.registry.get_artifact(
                    info["compile_key"]):
                reason = "identity inconsistent: artifact missing"
            else:
                ready, reason = True, "serving"
        elif info is not None and not info["alive"]:
            reason = "worker not alive"
        return {"alive": True, "ready": ready, "reason": reason,
                "worker": info, "inhibition": inhibition}

    def _baseline_progress(self) -> None:
        """Snapshot pre-existing row states silently at construction:
        a restart must not announce legacy terminal rows as fresh
        failures. Work submitted after this point reports normally."""
        for row in self.registry.query("SELECT uuid, stage FROM jobs"):
            self._seen_stages[row[0]] = row[1]
        for row in self.registry.query("SELECT ref, state FROM model_refs"):
            self._seen_ref_states[row[0]] = row[1]
        self._last_worker_generation = self._worker_generation

    def report_progress(self) -> None:
        """Emit console events for transitions since the last call.

        Called by the serve loop after pump(). Diff-based: repeated
        calls with no change emit nothing except the bounded
        heartbeat for still-running jobs."""
        import time as _time
        if self.reporter is None:
            return
        now = _time.monotonic()
        if self._worker_generation != self._last_worker_generation:
            self._last_worker_generation = self._worker_generation
            if self._worker is not None:
                self.emit({"kind": "worker_active",
                           "generation": self._worker_generation,
                           "compile_key": self._worker_compile_key or ""})
        for row in self.registry.query(
                "SELECT uuid, ref, compile_key, stage, error_code,"
                " progress FROM jobs"):
            uuid, ref, _ckey, stage, error_code, progress = row
            last = self._seen_stages.get(uuid)
            if last != stage:
                self._seen_stages[uuid] = stage
                self._heartbeats[uuid] = now
                self._emit_stage(ref, uuid, stage, error_code, progress)
            elif stage not in TERMINAL_ERROR_STATES and stage != "PREPARED":
                last_hb = self._heartbeats.get(uuid, 0.0)
                if now - last_hb >= self.heartbeat_s:
                    self._heartbeats[uuid] = now
                    self._emit_heartbeat(ref, stage, progress)

    def _emit_stage(self, ref: str, uuid: str, stage: str,
                    error_code: str | None, progress: str | None) -> None:
        from .model_view import parse_progress
        elapsed, sub = parse_progress(progress)
        elapsed = elapsed or 0.0
        if stage == "WAITING_FOR_DEVICE":
            self.emit({"kind": "waiting_for_device", "ref": ref})
        elif stage == "PREPARED":
            for alias in self.registry.refs_for_job(uuid):
                self.emit({"kind": "model_prepared", "ref": alias,
                           "endpoint": self.config.endpoint})
        elif stage in TERMINAL_ERROR_STATES:
            reason = None
            backend = self.jobs.backend_for(uuid)
            result = getattr(backend, "result", None)
            if result is not None and getattr(result, "detail", ""):
                reason = result.detail
            for alias in self.registry.refs_for_job(uuid):
                self.emit({"kind": "preparation_failed", "ref": alias,
                           "phase": stage,
                           "code": error_code or stage,
                           "reason": reason or ""})
        elif stage == "COMPILING":
            kind = {"PREPARING": "bf16_running",
                    "VALIDATING": "validating"}.get(sub or "", "compiling")
            self.emit({"kind": kind, "ref": ref, "elapsed_s": elapsed})

    def _emit_heartbeat(self, ref: str, stage: str,
                        progress: str | None) -> None:
        from .model_view import parse_progress
        elapsed, sub = parse_progress(progress)
        if stage == "COMPILING":
            phase = {"PREPARING": "BF16 preparation",
                     "VALIDATING": "artifact validation"}.get(
                         sub or "", "XDNA compilation")
        else:
            phase = stage
        self.emit({"kind": "heartbeat", "ref": ref, "phase": phase,
                   "elapsed_s": elapsed or 0.0})

    def pump(self, dt_s: float = 0.05) -> None:
        """Advance all non-terminal jobs (fake backend).

        Per-job isolation: one failing job is marked failed, never kills
        the loop or the daemon (SF4). Eligible terminal rows open new
        bounded attempts here too, so backoff expiry needs no prepare.
        """
        self.jobs.retry_due(**self.fake_compile)
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
                self.jobs.record_terminal(row[0], "COMPILE_FAILED",
                                          e.error_code, e.message)
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
                        self.jobs.record_terminal(
                            row[0], "COMPILE_FAILED", e.error_code,
                            e.message)
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
            if pump and job["uuid"] != job_uuid:
                # The pump opened a bounded retry attempt: follow the
                # live row instead of re-pumping the terminal one.
                job_uuid = job["uuid"]
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
        """Live-manager status. State SERVING means this manager process
        is serving the admin/queue plane. Readiness is explicit
        (ready/reason + live worker binding); the persisted `active`
        record is history, so `active` is the LIVE binding or null —
        a stale record after a crash never claims readiness."""
        from .model_view import project_ref
        live_key = self._live_worker_key()
        if ref:
            parsed = parse_ref(ref, self.config.model_dir)
            rec = self.registry.get_ref(parsed["ref"])
            models = [project_ref(self.registry, parsed["ref"], live_key)
                      ] if rec else []
        else:
            models = [project_ref(self.registry, r[0], live_key)
                      for r in self.registry.query(
                          "SELECT ref FROM model_refs")]
        health = self.health()
        active = None
        if live_key is not None:
            active = {"compile_key": live_key,
                      "worker_generation": self._worker_generation,
                      "serving_digest": self._worker_serving_digest}
        build = get_build_identity()
        return {"schema_version": 1, "service": "frigate-xdna",
                "version": build["version"], "build": build,
                "state": "SERVING",
                "active": active,
                "worker": health["worker"],
                "ready": health["ready"], "reason": health["reason"],
                "models": models,
                "inhibition": self.registry.get_state("inhibition")}

    def _model_view(self, rec: dict | None) -> dict:
        if rec is None:
            return {}
        from .model_view import project_ref
        return project_ref(self.registry, rec["ref"],
                           self._live_worker_key())

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

    def ingest_zmq_bytes(
        self, alias: str, data: bytes, contract: dict, cls: dict, source_sha256: str
    ) -> dict:
        """Ingest bytes received via ZMQ transfer (forced transfer path).

        Hash is already computed (source_sha256), contract and cls already
        inspected. This is the content-bound path; basename is only an alias.
        """
        # Reuse _ingest_source logic but with pre-inspected contract
        # Ref must be content-bound (source_sha256), not alias, so same
        # model_name with different bytes gets distinct refs and never
        # triggers SOURCE_CHANGED on alias collision.
        ref = f"zmq-upload:{source_sha256}"
        # Register synthetic content-bound ref immediately so later
        # _ingest_source / _publish_result find the record
        self.registry.upsert_ref(ref, "onnx", None)
        # Same inspected result status reports: the transport ran the
        # shared inspector, so record its summary here rather than
        # re-validating with a second rule set.
        self.registry.set_ref_inspection(
            ref, _inspect.summarize_inspection(contract, cls, None,
                                               source_sha256))
        # Use the same ingestion as local ONNX but with data already
        # fetched: _ingest_source with origin "local" and inspected extra.
        return self._ingest_source(
            ref,
            alias,
            data,
            "local",
            {
                "inspected": contract,
                "profile": cls["profile"],
                "source_sha256": source_sha256,
            },
            False,
        )

    def activate(self, ref: str, maintenance: bool = False) -> dict:
        """Activate a PREPARED ref: spawn the resident worker on its
        artifact (A→B: the new child LOADs and verifies before the old
        one retires; a failed B never becomes active). Refuses while a
        compile holds the device unless the operator explicitly passes
        maintenance (background-compile-while-serving window)."""
        if not maintenance:
            from .compiler.launcher import compile_in_flight
            if compile_in_flight():
                raise FxdnaError(DEVICE_UNAVAILABLE, "DEVICE_BUSY",
                                 "compile in flight; retry when idle or"
                                 " pass maintenance explicitly")
        parsed = parse_ref(ref, self.config.model_dir)
        # Only terminal-ok rows qualify: a newer failed job must never
        # hide an older valid PREPARED artifact (the old excluded stage
        # names are not even in the job vocabulary).
        row = self.registry.query(
            "SELECT compile_key FROM jobs WHERE ref=? AND compile_key"
            " IS NOT NULL AND stage='PREPARED' ORDER BY updated_at DESC"
            " LIMIT 1", (parsed["ref"],))
        if not row:
            raise FxdnaError(NOT_READY, "NOT_PREPARED",
                             f"no prepared artifact for {parsed['ref']}")
        compile_key = row[0][0]
        if not self.activate_worker(compile_key):
            raise FxdnaError(DEVICE_UNAVAILABLE, "ACTIVATION_FAILED",
                             f"worker refused artifact for {parsed['ref']};"
                             " see inhibition record")
        return {"ref": parsed["ref"], "compile_key": compile_key,
                "worker_generation": self._worker_generation,
                "state": "ACTIVE"}

    def _inspect_activation(self, compile_key: str) -> dict:
        """Content-bound activation inputs for an artifact.

        Re-inspects the stored source ONNX (never trusts a label count
        from metadata alone): contract profile, class count, serving
        digest and artifact path. Raises FxdnaError on any mismatch.
        """
        art = self.registry.get_artifact(compile_key)
        if art is None:
            raise FxdnaError(NOT_READY, "NOT_PREPARED",
                             f"unknown artifact {compile_key[:12]}…")
        source_sha = art["source_sha256"] or ""
        src_path = os.path.join(self.data_dir, "sources", source_sha,
                                SOURCE_ONNX_NAME)
        try:
            with open(src_path, "rb") as f:
                data = f.read(_inspect.MAX_ONNX_BYTES + 1)
        except OSError as e:
            raise FxdnaError(NOT_READY, "CACHE_CORRUPT",
                             f"source bytes missing for {compile_key[:12]}…"
                             f": {e}") from e
        try:
            model, _digest = _inspect.load_graph_bytes(data)
            inspected = _inspect.inspect_model(model)
            cls = _inspect.classify_output(inspected["outputs"])
        except FxdnaError:
            raise
        except Exception as e:
            raise FxdnaError(VALIDATION_FAILED, "VALIDATION_FAILED",
                             f"source re-inspection failed: {e}") from e
        if cls.get("profile") != "yolo-raw":
            raise FxdnaError(UNSUPPORTED_CONTRACT, "UNSUPPORTED_CONTRACT",
                             f"activation supports yolo-raw only:"
                             f" {cls.get('error')}")
        class_count = int(cls["channels"]) - 4
        if class_count <= 0:
            raise FxdnaError(UNSUPPORTED_CONTRACT, "UNSUPPORTED_CONTRACT",
                             "activation needs at least one class")
        rai_path = os.path.join(self.data_dir, "artifacts", compile_key,
                                "model.rai")
        if not os.path.isfile(rai_path):
            raise FxdnaError(NOT_READY, "NOT_PREPARED",
                             f"artifact bytes missing for {compile_key[:12]}…")
        return {"rai_path": rai_path,
                "serving_digest": _serving_digest(compile_key, inspected),
                "class_count": class_count,
                "input_shape": inspected["input_shape"]}

    def _inhibit_quiet(self, reason: str, ref: str) -> None:
        """Best-effort inhibition: failure-path booleans must never turn
        into raises just because the registry is also unhealthy."""
        try:
            _inhibit(self.data_dir, self.registry, reason, ref)
        except Exception:
            pass

    def _retire_confirmed(self, worker) -> bool:
        """Best-effort retire; True only if the child is confirmed dead.

        retire() itself can raise after a wait timeout (e.g. from
        terminate()), leaving the child live — callers must not drop
        the reference in that case (see _orphans)."""
        try:
            worker.retire()
        except Exception:
            pass
        try:
            return not worker.alive()
        except Exception:
            return False

    def _retire_or_track(self, worker) -> bool:
        """Retire a discarded worker; track it if death is unconfirmed.

        True when the child is confirmed dead. False when it may still
        be live — it is appended to _orphans for cleanup at stop() and
        must never be silently dropped."""
        if self._retire_confirmed(worker):
            return True
        self._orphans.append(worker)
        return False

    def _spawn_worker(self):
        if self._worker_factory is not None:
            return self._worker_factory()
        return _native.NativeWorker.spawn(self._worker_bin,
                                          _native.worker_lib_dirs())

    def _ref_for_artifact(self, compile_key: str) -> str:
        row = self.registry.query(
            "SELECT ref FROM jobs WHERE compile_key=? ORDER BY updated_at"
            " DESC LIMIT 1", (compile_key,))
        return row[0][0] if row else compile_key

    def activate_worker(self, compile_key: str) -> bool:
        """Swap the resident worker to an artifact. Serialized; the old
        child retires only after the new one LOADs clean. Any failure
        inhibits (explicit recover) and never respawns: no autoloop."""
        with self._worker_lock:
            try:
                spec = self._inspect_activation(compile_key)
            except FxdnaError as e:
                # Inspection failures inhibit (corrupt source, contract
                # conflict) — except a plain unknown artifact, which is
                # operator error with no safety implication.
                if e.error_code != "NOT_PREPARED":
                    self._inhibit_quiet(
                             f"ACTIVATION_INSPECT:{e.error_code}",
                             self._ref_for_artifact(compile_key))
                return False
            generation = self._worker_generation + 1
            try:
                worker = self._spawn_worker()
            except _native.WorkerError:
                self._inhibit_quiet("WORKER_SPAWN",
                         self._ref_for_artifact(compile_key))
                return False
            try:
                worker.load(spec["rai_path"], generation,
                            spec["serving_digest"], spec["class_count"])
            except _native.WorkerError as e:
                self._retire_or_track(worker)
                self._inhibit_quiet(
                         f"WORKER_LOAD:{e.code}",
                         self._ref_for_artifact(compile_key))
                return False
            old, old_key = self._worker, self._worker_compile_key
            old_generation = self._worker_generation
            old_digest = self._worker_serving_digest
            self._worker = worker
            self._worker_generation = generation
            self._worker_compile_key = compile_key
            self._worker_serving_digest = spec["serving_digest"]
            try:
                self.registry.set_state(
                    "active",
                    {"compile_key": compile_key,
                     "worker_generation": generation,
                     "serving_digest": spec["serving_digest"]})
            except Exception:
                # Publish failed: roll the swap back (retire the new
                # child, restore previous fields) and inhibit — a live
                # worker with no published identity must not exist
                # untracked: if retire did not confirm death, keep an
                # explicit reference for cleanup at stop().
                self._retire_or_track(worker)
                self._worker = old
                self._worker_generation = old_generation
                self._worker_compile_key = old_key
                self._worker_serving_digest = old_digest
                self._inhibit_quiet("ACTIVATION_STATE",
                         self._ref_for_artifact(compile_key))
                return False
            if old is not None:
                self._retire_or_track(old)
            # Recorded native check: this exact artifact LOADed clean
            # in the child that now serves traffic.
            art = self.registry.get_artifact(compile_key)
            if art is not None:
                self._record_native_check(
                    compile_key, art["artifact_sha256"],
                    spec["serving_digest"], "activation-load",
                    "activation-load-v1", {"generation": generation})
            return True

    def _drop_worker(self, reason: str) -> None:
        """Retire + forget the worker after death/fault, then inhibit."""
        worker, self._worker = self._worker, None
        compile_key, self._worker_compile_key = self._worker_compile_key, None
        self._worker_serving_digest = None
        if worker is not None:
            self._retire_or_track(worker)
        ref = (self._ref_for_artifact(compile_key)
               if compile_key else "active")
        self._inhibit_quiet(reason, ref)
        self.emit({"kind": "worker_lost", "reason": reason})

    def worker_infer(self, payload: bytes, shape: list[int],
                     timeout_s: float) -> tuple[str, bytes]:
        """Forward one bounded INFER to the resident worker.

        Returns ("ok", 480B) | ("no_worker", b"") | ("failed", b"").
        Transport death, device faults and timeouts inhibit (explicit
        recover, never respawn); validation refusals fail the request
        only — a bad tensor is client error, not a device fault.
        """
        with self._worker_lock:
            worker = self._worker
            if worker is None or not getattr(worker, "loaded", False):
                return ("no_worker", b"")
            if not worker.alive():
                self._drop_worker("WORKER_DIED")
                return ("failed", b"")
            try:
                out = worker.infer(payload, shape,
                                   worker.generation, timeout_s)
            except _native.WorkerError as e:
                if e.code in ("DEVICE_FAULT", "WORKER_DEAD", "WORKER_IO",
                              "INFER_SHORT", "WORKER_SPAWN"):
                    self._drop_worker(f"WORKER_{e.code}")
                return ("failed", b"")
            return ("ok", out)

    def recover_ref(self, ref: str) -> dict:
        """Documented recovery route through the daemon interface.

        Clears a non-safety inhibition and/or opens one new bounded
        retry cycle for a retryable terminal job. Never bypasses
        quarantine or safety inhibition: safety-class reasons refuse
        with the reason and the explicit next action. Unknown legacy
        failures report what remains unknown; the explicit
        acknowledgement starts one bounded cycle without pretending
        the failure was understood.
        """
        from . import retry_policy as _retry
        parsed = parse_ref(ref, self.config.model_dir)
        with self.registry.transaction():
            inh = self.registry.get_state("inhibition")
            if inh and inh.get("ref") == parsed["ref"]:
                if _retry.is_safety_inhibition(
                        inh.get("reason", "")):
                    return {"ref": parsed["ref"], "cleared": False,
                            "requeued": False, "refused_safety": True,
                            "note": "safety inhibition refused:"
                                    f" {inh.get('reason')}. Explicit"
                                    " operator review required; inspect"
                                    " status/diagnose evidence, resolve"
                                    " the device/fault cause first."}
                # Remove the inhibition; persist the evidence separately
                # so a later recover no longer finds it. One transaction:
                # concurrent recoveries cannot clear the same inhibition
                # twice, and a crash rolls back both halves together.
                # The requeue below runs outside this transaction
                # (requeue manages its own); its live-duplicate guard
                # keeps concurrent recoveries to one new attempt.
                self.registry.execute(
                    "DELETE FROM service_state WHERE key=?",
                    ("inhibition",))
                self.registry.set_state("last_inhibition_cleared",
                                        {"ref": parsed["ref"],
                                         "previous": inh,
                                         "at": time.time()})
                cleared = True
            elif inh is not None:
                return {"ref": parsed["ref"], "cleared": False,
                        "requeued": False,
                        "note": "inhibition active for"
                                f" {inh.get('ref')} ({inh.get('reason')});"
                                " resolve it before recovering"
                                f" {parsed['ref']}."}
            else:
                cleared = False
        row = self._terminal_row_for_ref(parsed["ref"])
        if row is None:
            return {"ref": parsed["ref"], "cleared": cleared,
                    "requeued": False,
                    "note": "no retryable terminal job for this ref"
                            if not cleared else
                            "inhibition cleared; nothing to requeue"}
        new, reason = self.jobs.requeue(parsed["ref"],
                                          row["compile_key"])
        if new is not None:
            return {"ref": parsed["ref"], "cleared": cleared,
                    "requeued": True,
                    "note": "new bounded attempt opened"}
        return {"ref": parsed["ref"], "cleared": cleared,
                "requeued": False, "note": reason or
                "requeue refused"}

    def _terminal_row_for_ref(self, ref: str) -> dict | None:
        """Newest terminal row reachable from a ref (or its aliases)."""
        row = self.registry.query(
            "SELECT uuid FROM jobs WHERE stage IN"
            " ('COMPILE_FAILED','RESOURCE_EXCEEDED','VALIDATION_FAILED',"
            " 'UNSUPPORTED_CONTRACT','QUARANTINED','INTERRUPTED')"
            " AND (ref=? OR uuid IN (SELECT job_uuid FROM job_aliases"
            " WHERE ref=?)) ORDER BY updated_at DESC LIMIT 1",
            (ref, ref))
        if not row:
            return None
        return self.registry.get_job(row[0][0])

    def prune(self, apply: bool = False,
              max_bytes: int | None = None) -> dict:
        plan = _gc.plan_prune(self.data_dir, self.registry, max_bytes)
        if not apply:
            return {"dry_run": True, **plan}
        with locked(os.path.join(self.data_dir, "locks", "prune.lock")):
            return {"dry_run": False, **_gc.apply_prune(
                self.data_dir, self.registry, plan, max_bytes)}
