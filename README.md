# frigate-xdna

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=mitchins_frigate-xdna&metric=alert_status)](https://sonarcloud.io/summary/new-code?id=mitchins_frigate-xdna)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=mitchins_frigate-xdna&metric=coverage)](https://sonarcloud.io/summary/new-code?id=mitchins_frigate-xdna)

AMD XDNA2 NPU detector sidecar for stock Frigate.
Downloads compatible Frigate+ / ONNX models, compiles them locally
to Ryzen AI format once, caches them, and serves inference through
Frigate's existing ZMQ detector.

## Status

Release candidate. Validated:

- Ryzen AI Max+ 395 / Strix Halo / XDNA2
- Frigate 0.18.0
- Frigate+ YOLOv9s 320
- 24 h full-service soak
- 1,234,208 inferences
- 0 detector errors / timeouts / late responses

Evidence: `docs/RELEASE-8.4.md`.

## Requirements

- Linux x86_64
- Supported AMD XDNA2 NPU, visible as `/dev/accel/accel0`
- Docker/Podman
- ~8 GiB container memory while compiling
- Persistent `/data` volume
- Frigate+ key only if using Plus models

Certified today: Strix Halo / Ryzen AI Max 300. Other XDNA2 systems
are expected targets but not yet certified.

## Install

```sh
docker pull ghcr.io/mitchins/frigate-xdna:0.1.0-rc.1
```

Images publish from version tags starting at `v0.1.0-rc.1`
(see `docs/RELEASE.md`); `:latest` is only ever published for stable
releases.

Minimal Compose — replace `NPU_GID` with the numeric group owning
`/dev/accel/accel0` on the Docker host:

Create the Plus secret file first (the overlay mounts it read-only):

```sh
mkdir -p examples/secrets
printf '%s\n' 'YOUR_PLUS_KEY' > examples/secrets/PLUS_API_KEY
chmod 600 examples/secrets/PLUS_API_KEY
```

Then start the sidecar:

```sh
NPU_GID=$(stat -c %g /dev/accel/accel0) \
FXDNA_IMAGE=ghcr.io/mitchins/frigate-xdna:0.1.0-rc.1 \
FXDNA_MODELS="plus://<model-id>" \
docker compose -f examples/compose.yaml -f examples/compose.plus.yaml up -d
```

Local-only deployments omit the Plus overlay and need no Plus secret
(see `examples/compose.plus.yaml` and `docs/OPERATIONS.md`). The
sidecar needs no published ZMQ port — stock Frigate joins the same
network.

## Frigate config

Point stock Frigate at the sidecar (no Frigate patches, no new
detector type):

```yaml
detectors:
  xdna:
    type: zmq
    endpoint: tcp://xdna:5555
```

And select a model (prepare the same model in the sidecar first;
Frigate uses its own Plus API key for its model/metadata):

```yaml
model:
  path: plus://MODEL_ID
```

See `examples/frigate-plus.yaml` (Plus) and
`examples/frigate-local.yaml` (local ONNX + label map).

## How it behaves

```text
Frigate+ / local ONNX
        ↓
frigate-xdna
        ↓
compile once → .rai cache
        ↓
resident FlexML/XDNA worker
        ↓
stock Frigate ZMQ
```

- New compatible models compile once (a few minutes on the NPU);
  compilation temporarily needs the device.
- The `.rai` cache survives restarts; cached inference works offline.
- One resident model per NPU initially; several models can stay
  prepared.
- No AMD SDK, account, or activation required.

## Compatibility

| Platform                    | Status         |
| --------------------------- | -------------- |
| Ryzen AI Max 300 / Strix Halo | Certified      |
| Ryzen AI 300 / Strix Point  | Not yet tested |
| Ryzen AI 400                | Not yet tested |
| Older XDNA1                 | Unsupported / not tested |

| Model                          | Status                              |
| ------------------------------ | ----------------------------------- |
| Frigate+ YOLOv9s-320           | Certified                           |
| Compatible YOLO-style fine-tunes | Compile locally; contract dependent |
| Arbitrary ONNX                 | Not promised                        |

## Development

Hardware-free tests, no NPU / SDK / API key:

```sh
make test
```

Pinned dev venv and CI install `requirements.lock`. Read `AGENTS.md`
before contributing (one task at a time; no hardware side effects
from imports, CLI parsing, or test discovery).

## Architecture

- `src/frigate_xdna/` — CLI/config, Plus client, tensor contracts,
  content-addressed cache, supervisor, resident worker supervision
- `native/` — audited FlexML worker sources (`reference/` keeps the
  byte-identical proven sources)
- `recipes/bf16-vaiml-v1/` — audited local compile recipe
- `schemas/` — model descriptor / artifact / status JSON schemas
- `tests/{unit,contract,integration,fixtures,upstream}/` — test suites
- `packaging/` — appliance image, vendor manifest tooling, SBOM inputs

## Evidence

- `docs/RELEASE-8.4.md` — acceptance record and release-boundary inputs
- `docs/ACCEPTANCE.md` — acceptance gates
- `docs/INTERFACES.md` — wire and CLI contracts
- `docs/OPERATIONS.md` — runbook (install, update, offline, faults)
- `docs/RELEASE.md` — release automation, SBOM/provenance, tag policy
- `docs/WORKQUEUE.md` — implementation log

## Licensing

Project source is MIT (`LICENSE`). The release appliance additionally
incorporates third-party and AMD vendor binaries governed by their own
terms (EULA flow-down, third-party notices); see
`THIRD_PARTY_NOTICES.md` and `/opt/fxdna/legal` inside the image.
