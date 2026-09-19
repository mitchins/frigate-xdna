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
from dataclasses import dataclass, field

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
    with open(f"{log_prefix}.stdout.log", "wb") as out, \
            open(f"{log_prefix}.stderr.log", "wb") as err:
        try:
            proc = subprocess.Popen(
                argv, env=env, cwd=cwd, stdin=subprocess.DEVNULL,
                stdout=out, stderr=err, close_fds=True,
                start_new_session=True, preexec_fn=_limit_resources)
        except OSError as e:
            return 127, time.monotonic() - t0
        try:
            proc.wait(timeout=timeout_s)
            return proc.returncode, time.monotonic() - t0
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
            proc.wait()
            return 124, time.monotonic() - t0


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


def run_compile(prefixes: CompilerPrefixes, source_onnx: str, workdir: str,
                cache_key: str, timeout_s: float = COMPILE_TIMEOUT_S,
                ) -> CompileResult:
    """Execute prepare -> compile -> validate. One compile at a time."""
    t_all = time.monotonic()
    acquired = _compile_lock.acquire(blocking=False)
    if not acquired:
        return CompileResult(returncode=98, wall_s=0.0, peak_rss_kb=0,
                             error="another compile owns the launcher lock")
    try:
        return _run_locked(prefixes, source_onnx, workdir, cache_key,
                           timeout_s, t_all)
    finally:
        _compile_lock.release()


def _run_locked(prefixes, source_onnx, workdir, cache_key, timeout_s,
                t_all) -> CompileResult:
    for sub in ("in", "home", "tmp", "cache"):
        os.makedirs(os.path.join(workdir, sub), exist_ok=True)
    model_in = os.path.join(workdir, "in", "model.onnx")
    shutil.copyfile(source_onnx, model_in)
    bf16_path = os.path.join(workdir, "in", "model-bf16.onnx")
    deadline = t_all + timeout_s

    rc, _ = spawn(
        [prefixes.quant_python,
         os.path.join(prefixes.recipe_dir, "prepare.py"),
         "--onnx", model_in, "--calib", prefixes.calib_dir,
         "--out", bf16_path],
        build_quant_env(workdir), workdir,
        max(60.0, deadline - time.monotonic()),
        os.path.join(workdir, "phase1-quant"))
    if rc != 0:
        return CompileResult(rc, time.monotonic() - t_all,
                             _child_peak_rss(), error="bf16-prepare failed")
    line = _tail_line(os.path.join(workdir, "phase1-quant.stdout.log"),
                      "BF16_PREPARE_OK")
    bf16_sha = line.rsplit(" ", 1)[-1] if line else ""

    rc, _ = spawn(
        [prefixes.compile_python,
         os.path.join(prefixes.recipe_dir, "compile.py"),
         "--onnx", bf16_path, "--config", prefixes.vaiml_config,
         "--cache-dir", os.path.join(workdir, "cache"),
         "--cache-key", cache_key],
        build_compile_env(prefixes, workdir), workdir,
        max(60.0, deadline - time.monotonic()),
        os.path.join(workdir, "phase2-compile"))
    if rc != 0:
        return CompileResult(rc, time.monotonic() - t_all,
                             _child_peak_rss(), error="vaiml-compile failed")
    line = _tail_line(os.path.join(workdir, "phase2-compile.stdout.log"),
                      "COMPILE_OK")
    rai_sha, rai_bytes = "", 0
    if line:
        parts = line.rsplit(" ", 2)
        if len(parts) == 3:
            rai_sha, rai_bytes = parts[1], int(parts[2])
    rai_path = os.path.join(workdir, "cache", cache_key, f"{cache_key}.rai")

    rc, _ = spawn(
        [prefixes.compile_python,
         os.path.join(prefixes.recipe_dir, "validate.py"),
         "--rai", rai_path],
        build_quant_env(workdir), workdir,
        max(60.0, deadline - time.monotonic()),
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
