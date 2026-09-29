"""Stability runner (`fxdna stability run`, docs/COMPATIBILITY.md §3-§11).

Observes the **production path** — the same audited native worker
child, FlexMLRT/XRT payload, compiled RAI, supervisor device lease
and safety journal that live Frigate traffic uses. There is no
second benchmark client and no alternate direct-XRT path.

v1 scope: `smoke` and `gentle` (E0/H1/H2/H3). The `pm`, `extended`
and `coexistence` profiles stay deferred until the durable
breadcrumbs from smoke/gentle prove they add diagnostic value.

Durable-breadcrumb contract: every phase boundary and device-
sensitive transition is fsynced to `timeline.jsonl` before it
begins and only marked COMPLETED after it ends; `current.json` is
rewritten at most ~1/s during active phases. A host reset
therefore leaves the last durable position, not a causal claim.
"""
from __future__ import annotations

import array
import os
import random
import signal
import time

from .errors import (
    DEVICE_UNAVAILABLE,
    INVALID_ARGS,
    NOT_READY,
    SUCCESS,
)
from .observability import stability
from .observability.stability import StabilityJournal

# Profile -> ordered phases. Deferred profiles are refused with a
# clear message until smoke/gentle evidence says otherwise.
PROFILES: dict[str, tuple[str, ...]] = {
    "smoke": ("E0", "H1", "H2"),
    "gentle": ("E0", "H1", "H2", "H3"),
}
DEFERRED_PROFILES = ("pm", "extended", "coexistence")

PHASE_NAMES = {
    "E0": "ENVIRONMENT",
    "H1": "ACTIVATE",
    "H2": "CALIBRATE",
    "H3": "LOW_STEADY",
}

CAL_WARMUP = 5
CAL_MEASURED = 25
STEADY_FRACTION = 0.10       # of measured sequential capacity
STEADY_DURATION_S = 120.0
CHECKPOINT_INTERVAL_S = 1.0  # current.json rewrite bound (§9.2)
CONSOLE_INTERVAL_S = 5.0
TIMEOUT_FACTOR = 10.0        # tolerance = 10x calibrated p50
TIMEOUT_FLOOR_S = 2.0
TIMEOUT_CEIL_S = 30.0
PROBE_TIMEOUT_S = 30.0       # H1/H2 hang bound (pre-calibration)
DEFAULT_SHAPE = (1, 3, 320, 320)
TENSOR_SEED = 0xF05DA1


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * len(ordered))) - 1))
    return ordered[idx]


class _StopFlag:
    def __init__(self):
        self.stop = False


def _install_stop_handlers(flag: _StopFlag) -> None:
    def _on_signal(_signum, _frame):
        flag.stop = True
    signal.signal(signal.SIGTERM, _on_signal)
    try:
        signal.signal(signal.SIGINT, _on_signal)
    except (OSError, ValueError):
        pass


def _make_tensor(shape: tuple[int, ...]) -> bytes:
    """Deterministic mild-valued float32 tensor (finite, stable)."""
    rng = random.Random(TENSOR_SEED)
    n = 1
    for d in shape:
        n *= d
    vals = array.array("f", (rng.random() * 0.2 for _ in range(n)))
    return vals.tobytes()


def _worker_pid(sup) -> int | None:
    worker = getattr(sup, "_worker", None)
    if worker is None:
        return None
    proc = getattr(worker, "_proc", None)
    pid = getattr(proc, "pid", None)
    if isinstance(pid, int):
        return pid
    return getattr(worker, "pid", None)


def _worker_rss_kb(pid: int | None) -> int | None:
    if not isinstance(pid, int):
        return None
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def _resolve_ref(config, ref: str | None, configured: bool) -> str:
    if configured:
        models = list(config.models or ())
        if len(models) != 1:
            raise ValueError(
                "--configured needs exactly one FXDNA_MODELS entry "
                f"(found {len(models)}); pass an explicit REF instead")
        return models[0]
    if not ref:
        raise ValueError("stability run needs REF or --configured")
    return ref


def _check_startup_latch(data_dir: str, console) -> int | None:
    """§10: refuse to open the NPU while the last run is an
    unacknowledged interrupted run. No automatic continuation."""
    report = stability.load_report(data_dir)
    if not report or not report.get("interrupted"):
        return None
    if report.get("acknowledged"):
        console(f"note: previous run {report['run_id']} was "
                "interrupted and acknowledged")
        return None
    it = report["interrupted"]
    console("PREVIOUS STABILITY RUN INTERRUPTED")
    console("")
    console(f"Run:      {report['run_id']}")
    console(f"Phase:    {it.get('phase')} "
            f"{PHASE_NAMES.get(it.get('phase'), '')}".rstrip())
    console(f"Step:     {it.get('step')} STARTED")
    if it.get("cycle") is not None:
        console(f"Cycle:    {it.get('cycle')}")
    console(f"Last durable completion: request {it.get('completed')}")
    console("")
    console("No automatic continuation will occur. Acknowledge after "
            "reviewing the host:")
    console('  fxdna stability acknowledge --last --reason "..."')
    return DEVICE_UNAVAILABLE


def _exit_for_failure(error_code: str) -> int:
    mapping = {
        "NOT_PREPARED": NOT_READY,
        "SAFETY_INHIBITED": DEVICE_UNAVAILABLE,
        "ACTIVATION_FAILED": DEVICE_UNAVAILABLE,
        "WORKER_FAILED": DEVICE_UNAVAILABLE,
        "DEVICE_FAULT": DEVICE_UNAVAILABLE,
        "TIMEOUT": DEVICE_UNAVAILABLE,
        "INTERRUPTED": NOT_READY,
        "INVALID_ARGS": INVALID_ARGS,
    }
    return mapping.get(error_code, DEVICE_UNAVAILABLE)


class _Run:
    """One stability run: journal + console + counters."""

    def __init__(self, config, sup, journal: StabilityJournal,
                 profile: str, shape: tuple[int, ...], tensor: bytes,
                 console, stop: _StopFlag,
                 steady_duration_s: float, cal_warmup: int,
                 cal_measured: int):
        self.config = config
        self.sup = sup
        self.j = journal
        self.profile = profile
        self.shape = shape
        self.tensor = tensor
        self.console = console
        self.stop = stop
        self.steady_duration_s = steady_duration_s
        self.cal_warmup = cal_warmup
        self.cal_measured = cal_measured
        self.started = time.monotonic()
        self.phases: list[str] = []
        self.last_phase: str | None = None
        self.submitted = 0
        self.completed = 0
        self.errors = 0
        self.timeouts = 0
        self.latencies: list[float] = []
        self.last_ok_monotonic: float | None = None
        self.p50 = 0.0
        self.capacity = 0.0
        self.tolerance_s = PROBE_TIMEOUT_S

    # ---- console -------------------------------------------------
    def say(self, phase: str, state: str, detail: str) -> None:
        elapsed = time.monotonic() - self.started
        self.console(f"[{int(elapsed // 60):02d}:{elapsed % 60:05.2f}] "
                     f"{phase} {PHASE_NAMES.get(phase, ''):<12} "
                     f"{state:<5} {detail}")

    # ---- journal -------------------------------------------------
    def checkpoint(self, phase: str, sub_phase: str,
                   force: bool = False) -> None:
        now = time.monotonic()
        recent = self.latencies[-100:]
        doc = {
            "schema_version": 1,
            "phase": phase,
            "sub_phase": sub_phase,
            "elapsed_s": round(now - self.started, 3),
            "submitted": self.submitted,
            "completed": self.completed,
            "errors": self.errors,
            "timeouts": self.timeouts,
            "last_ok_monotonic": (
                None if self.last_ok_monotonic is None
                else round(self.last_ok_monotonic, 3)),
            "worker_pid": _worker_pid(self.sup),
            "worker_generation": getattr(self.sup, "_worker_generation",
                                         0),
            "rss_kb": _worker_rss_kb(_worker_pid(self.sup)),
            "runtime_status": _device_runtime_status(),
            "latency_ms": {
                "p50": round(_percentile(recent, 0.50) * 1000, 2),
                "p95": round(_percentile(recent, 0.95) * 1000, 2),
            } if recent else None,
            "wall_time": time.time(),
        }
        self.j.current(doc)

    # ---- inference -----------------------------------------------
    def infer_once(self, timeout_s: float) -> str | None:
        """One request through the production worker path.

        Returns None on success (latency recorded), or an error_code
        string ("WORKER_FAILED" | "TIMEOUT"). Any native/device
        failure already travelled the supervisor's inhibition path.
        """
        self.submitted += 1
        t0 = time.monotonic()
        status, _out = self.sup.worker_infer(self.tensor,
                                             list(self.shape),
                                             timeout_s)
        dt = time.monotonic() - t0
        if status != "ok":
            self.errors += 1
            return "WORKER_FAILED"
        self.completed += 1
        self.latencies.append(dt)
        self.last_ok_monotonic = time.monotonic()
        if dt > timeout_s:
            self.timeouts += 1
            return "TIMEOUT"
        return None

    # ---- phases --------------------------------------------------
    def phase_e0(self, status_doc: dict) -> str | None:
        self.last_phase = "E0"
        self.checkpoint("E0", "environment", force=True)
        model = (status_doc.get("models") or [{}])[0]
        inhibition = status_doc.get("inhibition")
        if inhibition:
            self.phases.append("E0 ENVIRONMENT   FAIL (inhibition "
                               "present)")
            return "SAFETY_INHIBITED"
        state = model.get("state")
        from .model_view import satisfies
        if not satisfies(state or "", "PREPARED"):
            self.phases.append("E0 ENVIRONMENT   FAIL (not prepared)")
            return "NOT_PREPARED"
        self.phases.append("E0 ENVIRONMENT   PASS")
        return None

    def phase_h1(self, ref: str) -> str | None:
        self.last_phase = "H1"
        self.j.timeline("H1", "PHASE", "STARTED")
        self.say("H1", "RUN", "activating production worker")
        from .errors import FxdnaError
        try:
            info = self.sup.activate(ref)
        except FxdnaError as e:
            self.j.timeline("H1", "PHASE", "COMPLETED",
                            outcome=e.error_code)
            self.phases.append("H1 ACTIVATE      FAIL "
                               f"({e.error_code})")
            return "ACTIVATION_FAILED"
        except Exception:
            self.j.timeline("H1", "PHASE", "COMPLETED",
                            outcome="ACTIVATION_FAILED")
            self.phases.append("H1 ACTIVATE      FAIL (activation)")
            return "ACTIVATION_FAILED"
        self.checkpoint("H1", "worker-load", force=True)
        err = self.infer_once(PROBE_TIMEOUT_S)
        self.j.timeline("H1", "PHASE", "COMPLETED",
                        outcome="PASS" if err is None else err,
                        generation=info.get("worker_generation"))
        if err is not None:
            self.phases.append("H1 ACTIVATE      FAIL "
                               f"({err.lower()})")
            return err
        self.phases.append("H1 ACTIVATE      PASS")
        self.say("H1", "PASS",
                 f"generation={info.get('worker_generation')}")
        return None

    def phase_h2(self) -> str | None:
        self.last_phase = "H2"
        self.j.timeline("H2", "PHASE", "STARTED")
        for i in range(self.cal_warmup):
            err = self.infer_once(PROBE_TIMEOUT_S)
            if err is not None:
                self.j.timeline("H2", "PHASE", "COMPLETED",
                                outcome=err)
                self.phases.append(f"H2 CALIBRATE     FAIL ({err})")
                return err
        measured: list[float] = []
        for i in range(self.cal_measured):
            err = self.infer_once(PROBE_TIMEOUT_S)
            if err is not None:
                self.j.timeline("H2", "PHASE", "COMPLETED",
                                outcome=err)
                self.phases.append(f"H2 CALIBRATE     FAIL ({err})")
                return err
            measured.append(self.latencies[-1])
            if (i + 1) % 5 == 0:
                self.say("H2", "RUN",
                         f"{i + 1}/{self.cal_measured} "
                         f"p50={_percentile(measured, 0.50) * 1000:.1f}ms")
        self.p50 = _percentile(measured, 0.50)
        if self.p50 <= 0:
            self.phases.append("H2 CALIBRATE     FAIL (zero median)")
            self.j.timeline("H2", "PHASE", "COMPLETED",
                            outcome="ZERO_MEDIAN")
            return "WORKER_FAILED"
        self.capacity = 1.0 / self.p50
        self.tolerance_s = min(TIMEOUT_CEIL_S,
                               max(TIMEOUT_FLOOR_S,
                                   TIMEOUT_FACTOR * self.p50))
        self.j.timeline("H2", "PHASE", "COMPLETED", outcome="PASS",
                        p50_ms=round(self.p50 * 1000, 2),
                        capacity_req_s=round(self.capacity, 2))
        self.phases.append(
            f"H2 CALIBRATE     PASS   p50={self.p50 * 1000:.1f} ms")
        self.say("H2", "PASS",
                 f"capacity={self.capacity:.1f} req/s "
                 f"tolerance={self.tolerance_s:.2f}s")
        return None

    def phase_h3(self) -> str | None:
        self.last_phase = "H3"
        target = max(0.1, STEADY_FRACTION * self.capacity)
        period = 1.0 / target
        self.j.timeline("H3", "PHASE", "STARTED",
                        target_req_s=round(target, 2),
                        duration_s=self.steady_duration_s)
        self.say("H3", "RUN",
                 f"target={target:.1f} req/s "
                 f"duration={self.steady_duration_s:.0f}s")
        t_start = time.monotonic()
        next_req = t_start
        last_ckpt = 0.0
        last_console = 0.0
        while True:
            now = time.monotonic()
            if self.stop.stop:
                self.j.timeline("H3", "PHASE", "COMPLETED",
                                outcome="OPERATOR_STOP")
                self.phases.append("H3 LOW_STEADY    FAIL "
                                   "(operator stop)")
                return "INTERRUPTED"
            if now - t_start >= self.steady_duration_s:
                break
            if now < next_req:
                time.sleep(min(0.05, next_req - now))
                continue
            err = self.infer_once(self.tolerance_s)
            next_req += period
            if err is not None:
                self.j.timeline("H3", "PHASE", "COMPLETED",
                                outcome=err)
                self.phases.append(f"H3 LOW_STEADY    FAIL ({err})")
                return err
            if now - last_ckpt >= CHECKPOINT_INTERVAL_S:
                self.checkpoint("H3", "steady")
                last_ckpt = now
            if now - last_console >= CONSOLE_INTERVAL_S:
                p95 = _percentile(self.latencies[-100:], 0.95) * 1000
                self.say("H3", "RUN",
                         f"ok={self.completed} err={self.errors} "
                         f"p95={p95:.1f}ms")
                last_console = now
        self.j.timeline("H3", "PHASE", "COMPLETED", outcome="PASS",
                        submitted=self.submitted,
                        completed=self.completed)
        self.phases.append(f"H3 LOW_STEADY    PASS   "
                           f"{self.completed}/{self.submitted}")
        self.say("H3", "PASS", f"{self.completed}/{self.submitted}")
        return None


def _device_runtime_status() -> str | None:
    from .observability.host_info import collect_host_info
    return collect_host_info().get("power", {}).get("runtime_status")


def _finish(journal: StabilityJournal, run: _Run | None, outcome: str,
            error_code: str | None = None, extra: dict | None = None):
    doc = {
        "schema_version": 1,
        "outcome": outcome,
        "profile": run.profile if run else None,
        "last_phase": run.last_phase if run else None,
        "phases": run.phases if run else [],
        "submitted": run.submitted if run else 0,
        "completed": run.completed if run else 0,
        "errors": run.errors if run else 0,
        "timeouts": run.timeouts if run else 0,
        "p50_ms": round(run.p50 * 1000, 2) if run and run.p50 else None,
        "capacity_req_s": (round(run.capacity, 2)
                           if run and run.capacity else None),
        "finished_at": time.time(),
    }
    if error_code:
        doc["error_code"] = error_code
    if extra:
        doc.update(extra)
    journal.result(doc)
    journal.timeline("RUN", "RUN", "COMPLETED", outcome=outcome)
    return doc


def run_stability(config, ref: str | None = None,
                  configured: bool = False, profile: str = "gentle",
                  *, steady_duration_s: float = STEADY_DURATION_S,
                  cal_warmup: int = CAL_WARMUP,
                  cal_measured: int = CAL_MEASURED,
                  console=None) -> int:
    """Run one stability diagnostic; returns a CLI exit code.

    The scale kwargs exist for tests only; production profiles use
    the module constants (25 measured, 120 s steady).
    """
    out = console if console is not None else _default_console
    if profile in DEFERRED_PROFILES:
        out(f"fxdna: profile '{profile}' is deferred until smoke and "
            "gentle prove the durable breadcrumbs "
            f"[{NOT_READY}]")
        return NOT_READY
    phases_plan = PROFILES.get(profile)
    if phases_plan is None:
        out(f"fxdna: unknown profile '{profile}' [{INVALID_ARGS}]")
        return INVALID_ARGS
    try:
        ref = _resolve_ref(config, ref, configured)
    except ValueError as e:
        out(f"fxdna: {e} [{INVALID_ARGS}]")
        return INVALID_ARGS

    latch = _check_startup_latch(config.data_dir, out)
    if latch is not None:
        return latch

    from .supervisor import Supervisor
    try:
        sup = Supervisor(config)
    except Exception as e:
        from .errors import FxdnaError
        if isinstance(e, FxdnaError):
            out(f"fxdna: {e.message} [{e.error_code}]")
            return e.exit_code
        raise

    run_id = time.strftime("%Y-%m-%dT%H-%M-%S") + (
        f"-{os.getpid():x}")
    journal = StabilityJournal(config.data_dir, run_id)
    stop = _StopFlag()
    _install_stop_handlers(stop)
    run: _Run | None = None
    try:
        # E0 — environment, persisted before anything device-bound.
        status_doc = sup.status(ref)
        model = (status_doc.get("models") or [{}])[0]
        inspection = model.get("inspection") or {}
        inspected_shape = inspection.get("input_shape")
        shape = tuple(inspected_shape) if (
            isinstance(inspected_shape, list) and len(inspected_shape)
            == 4) else DEFAULT_SHAPE
        from .build_identity import get_build_identity
        from .observability.host_info import collect_host_info
        journal.metadata({
            "schema_version": 1,
            "run_id": run_id,
            "profile": profile,
            "phases_plan": list(phases_plan),
            "model": {
                "ref": model.get("ref", ref),
                "state": model.get("state"),
                "family": (inspection.get("family")
                           or model.get("ref", ref)),
                "resolution": shape[-1] if len(shape) == 4 else None,
                "compile_key": model.get("compile_key"),
                "input_shape": list(shape),
                "shape_source": ("inspection" if inspected_shape
                                 else "fallback"),
            },
            "host_info": collect_host_info(),
            "build": get_build_identity(),
            "inhibition": status_doc.get("inhibition"),
            "started_at": time.time(),
        })
        run = _Run(config, sup, journal, profile, shape,
                   _make_tensor(shape), out, stop,
                   steady_duration_s, cal_warmup, cal_measured)
        journal.timeline("RUN", "RUN", "STARTED", profile=profile,
                         ref=model.get("ref", ref))

        err = run.phase_e0(status_doc)
        if err is None:
            run.say("E0", "PASS", _e0_summary())
        else:
            run.say("E0", "FAIL", err.lower().replace("_", " "))
        if err is None and "H1" in phases_plan:
            err = run.phase_h1(ref)
        if err is None and "H2" in phases_plan:
            err = run.phase_h2()
        if err is None and "H3" in phases_plan:
            err = run.phase_h3()
        if err == "NOT_PREPARED":
            out("MODEL_NOT_PREPARED")
            out("")
            out("Prepare it first:")
            out(f"  fxdna prepare {ref} --wait")
            doc = _finish(journal, run, "FAIL", "NOT_PREPARED")
            _print_result(out, doc)
            return NOT_READY
        if err is not None:
            doc = _finish(journal, run, "FAIL", err)
            _print_result(out, doc)
            return _exit_for_failure(err)
        doc = _finish(journal, run, "PASS")
        _print_result(out, doc)
        return SUCCESS
    finally:
        if run is not None:
            # Final durable checkpoint while the worker still exists
            # (pid/RSS of the retired child are meaningless after
            # stop()); then retire.
            try:
                run.checkpoint(run.last_phase or "E0", "finished",
                               force=True)
            except Exception:
                pass
        try:
            sup.stop()
        except Exception:
            pass


def _e0_summary() -> str:
    from .observability.host_info import collect_host_info
    info = collect_host_info()
    npu = info.get("npu", {})
    dmi = info.get("host", {}).get("dmi", {})
    fw = npu.get("firmware_version") or "?"
    return (f"kernel={info['host']['uname'].split()[1]} "
            f"BIOS={dmi.get('bios_version') or '?'} fw={fw}")


def _print_result(out, doc: dict) -> None:
    out("")
    out(f"frigate-xdna stability run: {doc['outcome']}")
    for ph in doc.get("phases", []):
        out(f"  {ph}")
    if doc.get("outcome") == "PASS":
        out("  This is an observation, not certification of host "
            "stability.")


def _default_console(line: str) -> None:
    import sys
    print(line, file=sys.stderr, flush=True)
