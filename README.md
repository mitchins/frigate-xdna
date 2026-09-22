# frigate-xdna

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=mitchins_frigate-xdna&metric=alert_status)](https://sonarcloud.io/summary/new-code?id=mitchins_frigate-xdna)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=mitchins_frigate-xdna&metric=coverage)](https://sonarcloud.io/summary/new-code?id=mitchins_frigate-xdna)

Self-contained Linux XDNA detector sidecar for **unmodified** Frigate
(`v0.18.0-rc2`): Frigate+ acquisition, local ONNX→RAI compilation, persistent
multi-model cache, resident native inference worker.

Status: **Tasks 01–04 + 8.3.5 done** — CLI surface, exact contracts, Plus
client, content-addressed cache, serve daemon, offline compiler appliance
(`bf16-vaiml-v1`), resident native worker + stock ZMQ ROUTER frontend,
repository hardening (CI, coverage floor 70, SonarCloud gate OK). In
progress: **Task 05 acceptance** (full Frigate rc2 replay, private Plus
model, A→B update, soak, release report) — see `docs/WORKQUEUE.md` and
`docs/RELEASE-8.4.md`.

## Quick start

```sh
make test          # unit + contract + integration, no NPU / SDK / API key
python3 -m frigate_xdna --help        # via PYTHONPATH=src, or:
fxdna status --json                   # after pip install -e .
```

Dev venv (pinned test deps): `/mnt/downloads/frigate-xdna-venvs/dev`
(CI: create a venv and install `requirements.lock` dev pins).

## Install (release image)

```sh
docker pull ghcr.io/mitchins/frigate-xdna:0.1
```

Run with the shipped Compose (see `docs/OPERATIONS.md` for the full
procedure). Replace `NPU_GID` with the numeric group owning
`/dev/accel/accel0` on the Docker host, provide a `/data` volume and
a Plus secret only if you use Frigate+ models:

```sh
NPU_GID=$(stat -c %g /dev/accel/accel0) \
FXDNA_MODELS="plus://<model-id>" \
docker compose -f examples/compose.yaml up -d
# With the sidecar on the xdna-net network, point stock Frigate at it:
# detectors: {xdna: {type: zmq, endpoint: tcp://xdna:5555}}
```

Requirements: `/dev/accel/accel0` passthrough, `NPU_GID` group access,
a persistent `/data` volume, `PLUS_API_KEY_FILE` secret mount for
Frigate+ models (local-only deployments need no secret), and stock
Frigate `v0.18.0-rc2` joining the same network (no published ZMQ port
needed). Certified: Ryzen AI Max 300 / Strix Halo, as recorded in
`docs/RELEASE-8.4.md`; other XDNA2 platforms are not yet certified.
Release automation, SBOM/provenance and tag policy: `docs/RELEASE.md`.

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
