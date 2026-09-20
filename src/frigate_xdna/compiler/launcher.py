"""Audited compiler launcher (Task 03).

Runs the bf16-vaiml-v1 recipe as short-lived, isolated child processes.
The supervisor never imports the vendor stack; everything vendor happens
inside these children with an explicitly constructed environment.

Isolation contract (tested, not just documented):
- executable, argv, cwd, HOME, TMPDIR, PATH, LD_LIBRARY_PATH, PYTHONPATH,
  thread caps and resource limits are set explicitly per phase;
- the child environment is built from scratch: Plus keys, bearer tokens,
  signed URLs, admin-socket FDs and unrelated manager secrets can never
  leak in (close_fds + env allowlist, asserted by tests);
- one real compile at a time per process (module-level lock);
- wall-clock deadline with process-group kill; bounded logs.
"""
from __future__ import annotations

import os
import resource
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass

COMPILE_TIMEOUT_S = 2700.0  # SPEC §9: 45-minute compiler timeout
COMPILER_THREADS = 4  # SPEC §9: four compiler threads
MEM_LIMIT_BYTES = 6 * 1024 ** 3  # SPEC §9: 6 GiB compiler-child allowance
MAX_LOG_BYTES = 8 * 1024 * 1024

# Secret names that must never appear in a child environment, even if the
# manager process was started with them set.
FORBIDDEN_ENV_KEYS = frozenset({
    "PLUS_API_KEY", "PLUS_API_KEY_FILE", "GH_PAT", "GH_TOKEN",
    "AUTHORIZATION", "BEARER_TOKEN",
})

_compile_lock = threading.Lock()


@dataclass(frozen=True)
class CompilerPrefixes:
    quant_python: str      # /opt/quant-venv/bin/python
    compile_python: str    # /opt/compile-venv/bin/python
    compile_lib: str       # compile site-packages dir
    xrt_lib: str           # minimal XRT lib dir
    xrt_root: str          # minimal XRT prefix (XILINX_XRT)
    recipe_dir: str        # recipes/bf16-vaiml-v1 (prepare/compile/validate)
    calib_dir: str         # calibration images
    vaiml_config: str      # vaiml_config.json path


@dataclass
class CompileResult:
    returncode: int
    wall_s: float
    peak_rss_kb: int
    bf16_sha256: str = ""
    rai_path: str = ""
    rai_sha256: str = ""
    rai_bytes: int = 0
    error: str = ""


def build_compile_env(prefixes: CompilerPrefixes, workdir: str) -> dict[str, str]:
    """Explicit child environment. Anything not listed here does not exist."""
    sp = prefixes.compile_lib
    # XRT first: voe bundles an older libxrt_coreutil that must never win.
    ld = os.pathsep.join([
        prefixes.xrt_lib,
        "/lib/x86_64-linux-gnu",
        f"{sp}/flexml/flexml_extras/lib",
        f"{sp}/onnxruntime/capi",
        f"{sp}/voe/lib",
        f"{sp}/lnx64.o/tools/peano/lib",
        "/usr/lib/x86_64-linux-gnu",
    ])
    env = {
        "PATH": "/usr/bin:/bin",
        "LD_LIBRARY_PATH": ld,
        "RYZEN_AI_INSTALLATION_PATH": os.path.dirname(
            os.path.dirname(prefixes.compile_python)),
        "XILINX_VITIS": sp,
        "XILINX_VITIS_AIETOOLS": sp,
        "XILINX_XRT": prefixes.xrt_root,
        "HOME": os.path.join(workdir, "home"),
        "TMPDIR": os.path.join(workdir, "tmp"),
        "OMP_NUM_THREADS": str(COMPILER_THREADS),
        "MKL_NUM_THREADS": str(COMPILER_THREADS),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for key in FORBIDDEN_ENV_KEYS:
        env.pop(key, None)
    return env


def build_quant_env(workdir: str) -> dict[str, str]:
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": os.path.join(workdir, "home"),
        "TMPDIR": os.path.join(workdir, "tmp"),
        "OMP_NUM_THREADS": str(COMPILER_THREADS),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for key in FORBIDDEN_ENV_KEYS:
        env.pop(key, None)
    return env


def _limit_resources():
    try:
        resource.setrlimit(resource.RLIMIT_AS,
                           (MEM_LIMIT_BYTES, MEM_LIMIT_BYTES))
    except (ValueError, OSError):
        pass


def spawn(argv: list[str], env: dict[str, str], cwd: str,
          timeout_s: float, log_prefix: str) -> tuple[int, float]:
    """Run one child phase; returns (returncode, wall_s). Logs bounded."""
    os.makedirs(cwd, exist_ok=True)
    for key in FORBIDDEN_ENV_KEYS:
        if key in env:
            raise RuntimeError(f"refusing to spawn with secret {key} in env")
    t0 = time.monotonic()
    # Resource limits (RLIMIT_AS / RLIMIT_FSIZE) are enforced inside the
    # audited recipe entry points (prepare/compile/validate) rather than via
    # preexec_fn, which is unsafe in a threaded supervisor (fork without exec
    # must not run Python). Logs are truncated to MAX_LOG_BYTES after wait.
    with open(f"{log_prefix}.stdout.log", "wb") as out, \
            open(f"{log_prefix}.stderr.log", "wb") as err:
        try:
            proc = subprocess.Popen(
                argv, env=env, cwd=cwd, stdin=subprocess.DEVNULL,
                stdout=out, stderr=err, close_fds=True,
                start_new_session=True)
        except OSError:
            return 127, time.monotonic() - t0
        try:
            proc.wait(timeout=timeout_s)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
            proc.wait()
            rc = 124
    # Enforce MAX_LOG_BYTES by truncating any oversize logs (preserves
    # capture of both streams and the timeout behavior; child also sets
    # RLIMIT_FSIZE where applicable).
    for p in (f"{log_prefix}.stdout.log", f"{log_prefix}.stderr.log"):
        try:
            sz = os.path.getsize(p)
            if sz > MAX_LOG_BYTES:
                with open(p, "r+b") as f:
                    f.truncate(MAX_LOG_BYTES)
        except OSError:
            pass
    return rc, time.monotonic() - t0


def _tail_line(path: str, marker: str) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(max(0, os.fstat(f.fileno()).st_size - 4096))
            lines = f.read().decode(errors="replace").splitlines()
    except OSError:
        return ""
    for line in reversed(lines):
        if marker in line:
            return line
    return ""


def probe_artifact(data_dir: str, rai_path: str, class_count: int,
                   shape: list[int], worker_factory=None,
                   timeout_s: float = 180.0) -> tuple[str, str]:
    """Out-of-child zeros probe through the real resident worker path.

    Spawns the audited worker (or the injected factory in tests), LOADs
    the published .rai, runs one zeros INFER and checks for a finite
    480-byte frame. Returns ("ok"|"failed"|"skipped", detail). Skips
    without blocking when the device lease is held (e.g. background
    recompile while serving): activation LOAD validates later, and no
    second NPU client ever polls.
    """
    from ..runtime import native as _native
    from ..runtime.device_lease import DeviceLease
    elems = 1
    for d in shape:
        elems *= int(d)
    lease = DeviceLease(data_dir)
    if not lease.try_acquire():
        return ("skipped", "device busy; activation LOAD validates")
    try:
        worker = (worker_factory() if worker_factory is not None
                  else _native.NativeWorker.spawn(
                      _native.worker_binary(), _native.worker_lib_dirs()))
        try:
            detail = _run_probe_exchange(worker, rai_path, class_count,
                                         shape, elems, timeout_s)
        finally:
            try:
                worker.retire()
            except Exception:
                pass
        return ("ok", "") if detail is None else ("failed", detail)
    finally:
        lease.release()


def _run_probe_exchange(worker, rai_path: str, class_count: int,
                        shape: list[int], elems: int,
                        timeout_s: float) -> str | None:
    """LOAD + zeros INFER + frame validation. None == probe passed."""
    import math as _math
    import struct as _st

    from ..runtime import native as _native
    try:
        worker.load(rai_path, 1, "probe", class_count,
                    timeout_s=min(25.0, timeout_s))
        out = worker.infer(b"\x00" * (elems * 4), list(shape), 1,
                           timeout_s=timeout_s)
    except _native.WorkerError as e:
        return f"{e.code}: {e}"
    if len(out) != _native.RESULT_BYTES:
        return f"short frame {len(out)}"
    vals = _st.unpack(f"<{len(out) // 4}f", out)
    if not all(_math.isfinite(v) for v in vals):
        return "non-finite probe output"
    return None


def compile_in_flight() -> bool:
    """Non-blocking check whether a compile owns the launcher lock."""
    acquired = _compile_lock.acquire(blocking=False)
    if not acquired:
        return True
    _compile_lock.release()
    return False


def _probe_spec(source_onnx: str) -> tuple[int, list[int]]:
    """(class_count, input_shape) re-inspected from the source ONNX."""
    from ..models import inspect as _inspect
    with open(source_onnx, "rb") as f:
        _model, _digest = _inspect.load_graph_bytes(f.read())
    _inspected = _inspect.inspect_model(_model)
    _cls = _inspect.classify_output(_inspected["outputs"])
    if _cls.get("profile") != "yolo-raw":
        raise ValueError(f"unsupported profile: {_cls.get('error')}")
    return int(_cls["channels"]) - 4, _inspected["input_shape"]


def run_compile(prefixes: CompilerPrefixes, source_onnx: str, workdir: str,
                cache_key: str, timeout_s: float = COMPILE_TIMEOUT_S,
                data_dir: str | None = None, worker_factory=None,
                ) -> CompileResult:
    """Execute prepare -> compile -> probe -> validate. One at a time."""
    t_all = time.monotonic()
    acquired = _compile_lock.acquire(blocking=False)
    if not acquired:
        return CompileResult(returncode=98, wall_s=0.0, peak_rss_kb=0,
                             error="another compile owns the launcher lock")
    try:
        return _run_locked(prefixes, source_onnx, workdir, cache_key,
                           timeout_s, t_all, data_dir, worker_factory)
    finally:
        _compile_lock.release()


def _run_vaiml_phase(prefixes, bf16_path: str, workdir: str,
                     cache_key: str, deadline: float, t_all: float):
    """Phase 2 (VAIML compile). Returns (rai_sha, rai_bytes, rai_path)
    or a terminal CompileResult on timeout/failure."""
    def wall() -> float:
        return time.monotonic() - t_all
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return CompileResult(124, wall(), _child_peak_rss(),
                             error="compile timeout")
    rc, _ = spawn(
        [prefixes.compile_python,
         os.path.join(prefixes.recipe_dir, "compile.py"),
         "--onnx", bf16_path, "--config", prefixes.vaiml_config,
         "--cache-dir", os.path.join(workdir, "cache"),
         "--cache-key", cache_key],
        build_compile_env(prefixes, workdir), workdir, remaining,
        os.path.join(workdir, "phase2-compile"))
    if rc != 0:
        return CompileResult(rc, wall(), _child_peak_rss(),
                             error="vaiml-compile failed")
    line = _tail_line(os.path.join(workdir, "phase2-compile.stdout.log"),
                      "COMPILE_OK")
    rai_sha, rai_bytes = "", 0
    if line:
        parts = line.rsplit(" ", 2)
        if len(parts) == 3:
            rai_sha, rai_bytes = parts[1], int(parts[2])
    return (rai_sha, rai_bytes,
            os.path.join(workdir, "cache", cache_key, f"{cache_key}.rai"))


def _run_locked(prefixes, source_onnx, workdir, cache_key, timeout_s,
                t_all, data_dir=None, worker_factory=None) -> CompileResult:
    for sub in ("in", "home", "tmp", "cache"):
        os.makedirs(os.path.join(workdir, sub), exist_ok=True)
    model_in = os.path.join(workdir, "in", "model.onnx")
    shutil.copyfile(source_onnx, model_in)
    bf16_path = os.path.join(workdir, "in", "model-bf16.onnx")
    deadline = t_all + timeout_s

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return CompileResult(124, time.monotonic() - t_all,
                             _child_peak_rss(), error="compile timeout")
    rc, _ = spawn(
        [prefixes.quant_python,
         os.path.join(prefixes.recipe_dir, "prepare.py"),
         "--onnx", model_in, "--calib", prefixes.calib_dir,
         "--out", bf16_path],
        build_quant_env(workdir), workdir, remaining,
        os.path.join(workdir, "phase1-quant"))
    if rc != 0:
        return CompileResult(rc, time.monotonic() - t_all,
                             _child_peak_rss(), error="bf16-prepare failed")
    line = _tail_line(os.path.join(workdir, "phase1-quant.stdout.log"),
                      "BF16_PREPARE_OK")
    bf16_sha = line.rsplit(" ", 1)[-1] if line else ""

    phase2 = _run_vaiml_phase(prefixes, bf16_path, workdir, cache_key,
                              deadline, t_all)
    if isinstance(phase2, CompileResult):
        return phase2
    rai_sha, rai_bytes, rai_path = phase2

    # Out-of-child probe through the real worker path (fresh process;
    # the in-compile probe cannot map device memory under the child's
    # address-space cap). Skipped when the device is busy serving.
    if data_dir is not None and os.path.isfile(rai_path):
        try:
            spec = _probe_spec(source_onnx)
        except Exception as e:
            return CompileResult(7, time.monotonic() - t_all,
                                 _child_peak_rss(),
                                 error=f"probe inspection failed: {e}")
        _status, _detail = probe_artifact(
            data_dir, rai_path, spec[0], spec[1],
            worker_factory=worker_factory, timeout_s=180.0)
        if _status == "failed":
            return CompileResult(7, time.monotonic() - t_all,
                                 _child_peak_rss(),
                                 error=f"probe failed: {_detail}")
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return CompileResult(124, time.monotonic() - t_all,
                             _child_peak_rss(), error="compile timeout")
    rc, _ = spawn(
        [prefixes.compile_python,
         os.path.join(prefixes.recipe_dir, "validate.py"),
         "--rai", rai_path],
        build_quant_env(workdir), workdir, remaining,
        os.path.join(workdir, "phase3-validate"))
    if rc != 0:
        return CompileResult(rc, time.monotonic() - t_all,
                             _child_peak_rss(), error="validate failed")
    return CompileResult(0, time.monotonic() - t_all, _child_peak_rss(),
                         bf16_sha256=bf16_sha, rai_path=rai_path,
                         rai_sha256=rai_sha, rai_bytes=rai_bytes)


def _child_peak_rss() -> int:
    try:
        return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    except OSError:
        return 0
