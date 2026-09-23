"""Unit tests: CLI status views, diagnose helpers and serve lifecycle.

Redacted projections, missing-journal passthroughs and the ZMQ
frontend startup/shutdown paths — with stubbed supervisor/frontend
so no socket is ever bound and no signal ever sent.
"""
import hashlib
import os
import tempfile
import threading
import time
import unittest
from unittest import mock

from frigate_xdna import cli as cli_mod
from frigate_xdna.cli import (
    _diagnose_current,
    _diagnose_history,
    _diagnose_inventory,
    _install_serve_handlers,
    _read_status,
    _StatusView,
    cmd_serve,
)
from frigate_xdna.config import Config
from frigate_xdna.errors import SUCCESS
from frigate_xdna.observability.redact import load_or_create_key
from tests.integration.onnx_builders import make_raw_yolo


def make_config(tmp):
    return Config(data_dir=tmp, endpoint="tcp://127.0.0.1:5559")


class TestStatusView(unittest.TestCase):
    def test_triple_redacts_ref_and_abbreviates_digest(self):
        with tempfile.TemporaryDirectory() as d:
            key = load_or_create_key(d)
            view = _StatusView(False, key)
            out = view.triple("plus://model-9", "a" * 64, "PREPARED")
            self.assertTrue(out["ref"].startswith("plus:"))
            self.assertNotIn("model-9", out["ref"])
            self.assertEqual(len(out["source_sha256"]), 17)
            self.assertEqual(out["state"], "PREPARED")

    def test_active_string_aliases_model_ref(self):
        with tempfile.TemporaryDirectory() as d:
            key = load_or_create_key(d)
            view = _StatusView(False, key)
            out = view.active("plus://model-9")
            self.assertTrue(out.startswith("plus:"))
            self.assertNotIn("model-9", out)

    def test_read_status_by_ref_projects_model(self):
        with tempfile.TemporaryDirectory() as d:
            from frigate_xdna.cache.registry import Registry
            data = make_raw_yolo(os.path.join(d, "m.onnx"), res=320,
                                 classes=2, seed=31)
            sha = hashlib.sha256(data).hexdigest()
            reg = Registry(os.path.join(d, "registry.sqlite3"))
            try:
                reg.upsert_ref("plus://model-31", "plus", "model-31")
                reg.add_source(sha, len(data), "m.onnx", "test")
                reg.set_ref_source("plus://model-31", sha, "md",
                                   "PREPARED")
                doc = _read_status(make_config(d),
                                   show_identifiers=True,
                                   ref="plus://model-31")
            finally:
                reg.close()
            self.assertEqual(len(doc["models"]), 1)
            self.assertEqual(doc["models"][0]["ref"], "plus://model-31")
            self.assertEqual(doc["models"][0]["state"], "PREPARED")


class TestDiagnoseHelpers(unittest.TestCase):
    def test_missing_journal_reads_empty(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = make_config(d)
            self.assertEqual(_diagnose_current(cfg, True, None), {})
            self.assertEqual(_diagnose_history(cfg, 50, None), "")
            refs, arts = _diagnose_inventory(cfg, True, None)
            self.assertEqual((refs, arts), ([], []))

    def test_inventory_projects_refs_and_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            from frigate_xdna.cache.registry import Registry
            reg = Registry(os.path.join(d, "registry.sqlite3"))
            try:
                reg.upsert_ref("plus://model-32", "plus", "model-32")
                reg.add_source("b" * 64, 10, "m.onnx", "test")
                reg.set_ref_source("plus://model-32", "b" * 64, "md",
                                   "PREPARED")
                reg.add_artifact("ck32", "b" * 64, "c" * 64, 10,
                                 "bf16-vaiml-v1", "stx-npu")
                refs, arts = _diagnose_inventory(make_config(d), True,
                                                 None)
            finally:
                reg.close()
            self.assertEqual(refs[0]["ref"], "plus://model-32")
            self.assertEqual(arts[0]["compile_key"], "ck32")
            self.assertEqual(arts[0]["source_sha256"], "b" * 64)

    def test_serve_handler_install_tolerates_sigint_failure(self):
        calls = []

        def guarded(signum, handler):
            calls.append(signum)
            if signum == cli_mod.signal.SIGINT:
                raise OSError("no sigint here")
            return None

        with mock.patch.object(cli_mod.signal, "signal",
                               side_effect=guarded):
            _install_serve_handlers(lambda *_a: None)
        self.assertEqual(calls, [cli_mod.signal.SIGTERM,
                                 cli_mod.signal.SIGINT])


class FakeSupervisor:
    def __init__(self, config):
        self.config = config
        self.data_dir = config.data_dir
        self.frontend = None
        self.stopped = False
        self.prepared = []
        self.progress_calls = 0

    def start_admin(self):
        pass

    def pump(self, _timeout):
        pass

    def report_progress(self):
        self.progress_calls += 1

    def prepare(self, ref):
        self.prepared.append(ref)

    def stop(self):
        self.stopped = True


class FakeFrontend:
    instances = []

    def __init__(self, sup, endpoint):
        self.sup = sup
        self.endpoint = endpoint
        self.sock = None
        self.started = False
        self.stopped = False
        FakeFrontend.instances.append(self)

    async def start(self):
        self.sock = object()
        self.started = True

    async def stop(self):
        self.stopped = True


class FailingFrontend(FakeFrontend):
    def __init__(self, sup, endpoint):
        super().__init__(sup, endpoint)
        self.sock = ExplodingSock()

    async def start(self):
        raise RuntimeError("bind boom")


class ExplodingSock:
    def close(self, **_kwargs):
        raise RuntimeError("close boom")


class NoSockFrontend(FakeFrontend):
    async def start(self):
        self.started = True


class HangingFrontend(FakeFrontend):
    async def start(self):
        import asyncio as _asyncio
        self.started = True
        await _asyncio.sleep(30)


class StopFailFrontend(FakeFrontend):
    async def stop(self):
        raise RuntimeError("stop boom")


class TestServeLifecycle(unittest.TestCase):
    def run_serve(self, frontend_cls):
        import frigate_xdna.transport.frigate_zmq as zmq_mod
        FakeFrontend.instances.clear()
        with tempfile.TemporaryDirectory() as d:
            cfg = Config(data_dir=d, endpoint="tcp://127.0.0.1:5559",
                         models=("plus://startup",))
            handlers = []
            sups = []
            real_sup = cli_mod.Supervisor
            real_fe = zmq_mod.FrigateZmqFrontend
            real_handlers = cli_mod._install_serve_handlers
            outcome = {}

            def fake_sup_factory(config, **_kw):
                sup = FakeSupervisor(config)
                sups.append(sup)
                return sup

            def fake_fe_factory(sup, endpoint):
                return frontend_cls(sup, endpoint)

            def target():
                try:
                    outcome["rc"] = cmd_serve(cfg)
                except Exception as e:
                    outcome["exc"] = e

            cli_mod.Supervisor = fake_sup_factory
            zmq_mod.FrigateZmqFrontend = fake_fe_factory
            cli_mod._install_serve_handlers = handlers.append
            real_preflight = cli_mod.run_preflight
            # Lifecycle tests stub the supervisor; preflight has its
            # own dedicated tests (deployment would fail here on
            # stock runners with no NPU).
            cli_mod.run_preflight = lambda _config: None
            try:
                with mock.patch.dict(os.environ, {}, clear=False):
                    os.environ.pop("FXDNA_ENDPOINT", None)
                    thread = threading.Thread(target=target, daemon=True)
                    thread.start()
                    deadline = time.monotonic() + 10.0
                    while (not FakeFrontend.instances
                           and time.monotonic() < deadline):
                        time.sleep(0.02)
                    self.assertTrue(FakeFrontend.instances)
                    fe = FakeFrontend.instances[-1]
                    if frontend_cls in (FailingFrontend, NoSockFrontend,
                                        HangingFrontend):
                        thread.join(timeout=15.0)
                        self.assertFalse(thread.is_alive())
                        raise outcome["exc"]
                    deadline = time.monotonic() + 10.0
                    while (not handlers
                           and time.monotonic() < deadline):
                        time.sleep(0.02)
                    self.assertTrue(fe.started)
                    self.assertTrue(handlers)
                    # Let the serve loop complete at least one
                    # pump+report cycle before stopping; otherwise the
                    # progress-call assertion below races startup.
                    deadline = time.monotonic() + 10.0
                    while (not sups[0].progress_calls
                           and time.monotonic() < deadline):
                        time.sleep(0.02)
                    self.assertGreater(sups[0].progress_calls, 0)
                    handlers[0](15, None)
                    thread.join(timeout=10.0)
                    self.assertFalse(thread.is_alive())
                    return outcome.get("rc"), sups[0], fe
            finally:
                cli_mod.Supervisor = real_sup
                zmq_mod.FrigateZmqFrontend = real_fe
                cli_mod._install_serve_handlers = real_handlers
                cli_mod.run_preflight = real_preflight
                FakeFrontend.instances.clear()

    def test_serve_runs_and_shuts_down_cleanly(self):
        rc, sup, fe = self.run_serve(FakeFrontend)
        self.assertEqual(rc, SUCCESS)
        self.assertTrue(sup.stopped)
        self.assertTrue(fe.stopped)
        self.assertEqual(sup.prepared, ["plus://startup"])
        # The serve loop must drive console reporting every pump;
        # without this the container goes silent during compiles.
        self.assertGreater(sup.progress_calls, 0)

    def test_serve_stop_failure_still_shuts_down(self):
        rc, sup, fe = self.run_serve(StopFailFrontend)
        self.assertEqual(rc, SUCCESS)
        self.assertTrue(sup.stopped)

    def test_serve_startup_failure_propagates(self):
        with self.assertRaises(RuntimeError) as ctx:
            self.run_serve(FailingFrontend)
        self.assertIn("bind boom", str(ctx.exception))

    def test_serve_missing_socket_reports_bind_failure(self):
        with self.assertRaises(RuntimeError) as ctx:
            self.run_serve(NoSockFrontend)
        self.assertIn("failed to bind", str(ctx.exception))

    def test_serve_startup_timeout_reports(self):
        with self.assertRaises(RuntimeError) as ctx:
            self.run_serve(HangingFrontend)
        self.assertIn("timeout or failed to bind", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
