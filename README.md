# frigate-xdna

Self-contained Linux XDNA detector sidecar for **unmodified** Frigate
(`v0.18.0-rc2`): Frigate+ acquisition, local ONNX→RAI compilation, persistent
multi-model cache, resident native inference worker.

Status: **Tasks 01–02 done** (CLI surface, exact contracts, Plus client,
content-addressed cache, serve daemon; fake compile backend). No real
compiler or native worker yet — see `docs/WORKQUEUE.md`.

## Quick start

```sh
make test          # unit + contract + integration, no NPU / SDK / API key
python3 -m frigate_xdna --help        # via PYTHONPATH=src, or:
fxdna status --json                   # after pip install -e .
```

Dev venv (pinned test deps): `/mnt/downloads/frigate-xdna-venvs/dev`
(CI: create a venv and install `requirements.lock` dev pins).

## Layout

* `src/frigate_xdna/` — CLI/config, Plus client, tensor contracts, cache,
  supervisor, fake backends
* `native/reference/` — byte-identical proven FlexML sources + reuse plan
* `recipes/bf16-vaiml-v1/` — audited compile recipe pointer (reference only)
* `schemas/` — model descriptor / artifact / status JSON schemas
* `tests/{unit,contract,integration,fixtures,upstream}/` — hardware-free tests
* `packaging/legal/` — licence component-map skeleton
* `docs/`, `agent-tasks/`, `examples/` — normative spec kit (as issued)
* `tools/import_proofs.py` — local evidence manifest importer (uncommitted output)

## Rules

Read `AGENTS.md` before any task. One task at a time; commit and report.
No hardware side effects from imports, CLI parsing, or test discovery.
