# frigate-xdna

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=mitchins_frigate-xdna&metric=alert_status)](https://sonarcloud.io/summary/new-code?id=mitchins_frigate-xdna)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=mitchins_frigate-xdna&metric=coverage)](https://sonarcloud.io/summary/new-code?id=mitchins_frigate-xdna)

## What is this?

Run Frigate object detection on the AMD Ryzen AI/XDNA2 NPU. A Docker sidecar for stock Frigate that prepares compatible models locally, caches them, and serves detections through Frigate's existing ZMQ detector.

## Why use it?

If your machine has an otherwise-idle NPU, this puts it to work on object detection and moves that load away from competing GPU workloads. Power draw has not been measured yet, so no efficiency claim is made here.

## How do I run it?

```text
Configure image, device group, model, and credential
→ deploy the sidecar
→ confirm the model is prepared with `fxdna status` / `fxdna wait`
→ configure the matching Frigate endpoint and model
→ detection starts
```

No SDK installation, manual ONNX conversion, database editing, or compiler-log inspection in the normal path. Details below.

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
- Docker (Compose v2) or Podman
- ~8 GiB container memory while compiling
- Persistent `/data` volume
- Frigate+ key only if using Plus models (local-model users need none)

Certified today: Strix Halo / Ryzen AI Max 300. Other XDNA2 systems
are expected targets but not yet certified.

## Install

```sh
docker pull ghcr.io/mitchins/frigate-xdna:0.1.0-rc.3
```

Images publish from version tags starting at `v0.1.0-rc.2`
(see `docs/RELEASE.md`); `:latest` is only ever published for stable
releases.

Compose files (the files you actually deploy):

- `examples/compose.yaml` — canonical sidecar (required).
- `examples/compose.plus.yaml` — Plus model acquisition overlay.
  Local-only deployments omit it and need no Plus secret.
- `examples/compose.host-port.yaml` — only when Frigate cannot join
  the sidecar's Docker network (see "Frigate config" below).

Deploy with stock Frigate on the same Docker network:

```sh
NPU_GID=$(stat -c %g /dev/accel/accel0) \
FXDNA_MODELS="plus://<model-id>" \
docker compose -f examples/compose.yaml -f examples/compose.plus.yaml up -d
```

`NPU_GID` is the numeric group owning `/dev/accel/accel0` on the
Docker host. `FXDNA_MODELS` is required (the Compose file refuses to
start without it) and uses the `plus://` form for Plus models. The
default image is the latest published release candidate; override
with `FXDNA_IMAGE=...` for a newer or stable tag.

Local-only deployments omit the Plus overlay and need no Plus
secret:

```sh
NPU_GID=$(stat -c %g /dev/accel/accel0) \
FXDNA_MODELS="/models/yolov9s-320.onnx" \
docker compose -f examples/compose.yaml up -d
```

with the local file bind-mounted under the configured import root
(see `examples/compose.yaml` and `docs/OPERATIONS.md`). The sidecar
needs no published ZMQ port when stock Frigate joins the same
network.

### Portainer

Paste the same files as a Portainer stack (or upload them), then set
these stack environment variables:

| Variable | Value | `plus://`? |
|---|---|---|
| `FXDNA_IMAGE` | `ghcr.io/mitchins/frigate-xdna:0.1.0-rc.3` (or newer) | no |
| `FXDNA_MODELS` | `plus://<model-id>` (or a `/models/...` path) | yes, for Plus models |
| `NPU_GID` | numeric group of `/dev/accel/accel0` | no |
| `PLUS_API_KEY` | raw key value only (Plus setups without a key file) | no |
| `FXDNA_BIND_IP` | only for host-networked Frigate (default `127.0.0.1`) | no |

Use **either** `PLUS_API_KEY` **or** `PLUS_API_KEY_FILE` — never
both at once (the sidecar refuses to start with both set).

### Plus key file

The documented overlay (`examples/compose.plus.yaml`) mounts the key
through Docker secrets (container path `PLUS_API_KEY_FILE` =
`/run/secrets/PLUS_API_KEY`):

```sh
mkdir -p examples/secrets
printf '%s\n' 'YOUR_PLUS_KEY' > examples/secrets/PLUS_API_KEY
chmod 600 examples/secrets/PLUS_API_KEY
```

Rules that must hold, whichever method you use:

- The file contains only the key value (one line, no `plus://`).
- The bind source is an absolute path on the Docker host.
- The container runs as UID 10001 — that UID must actually be able
  to read the file. A root-owned `0600` file bind-mounted directly
  is **not** readable by UID 10001; use the secrets overlay above
  (mounted read-only and world-readable inside the container) or
  adjust ownership/permissions so UID 10001 can read it.

## Frigate config

Point stock Frigate at the sidecar (no Frigate patches, no new
detector type). Prepare the same model in the sidecar first; Frigate
uses its own Plus API key for its model/metadata.

**Shared Docker network** (default, no published ZMQ port):

```yaml
detectors:
  xdna:
    type: zmq
    endpoint: tcp://xdna:5555
```

```yaml
model:
  path: plus://MODEL_ID
```

**Host-networked Frigate** (Frigate has `network_mode: host`, or
lives outside the sidecar's Docker network — it cannot resolve the
`xdna` service name, and sharing a Portainer stack does not change
that). Add the host-port overlay so the sidecar listens on host
loopback, then point Frigate at it:

```sh
NPU_GID=$(stat -c %g /dev/accel/accel0) \
FXDNA_MODELS="plus://<model-id>" \
docker compose -f examples/compose.yaml -f examples/compose.plus.yaml \
  -f examples/compose.host-port.yaml up -d
```

```yaml
detectors:
  xdna:
    type: zmq
    endpoint: tcp://127.0.0.1:5555
```

Only set `FXDNA_BIND_IP` to a non-loopback host/LXC address when
Frigate is on another host, and firewall it to the Frigate client.
Never publish unauthenticated ZMQ to all interfaces, and never use a
transient bridge IP.

See `examples/frigate-plus.yaml` (Plus) and
`examples/frigate-local.yaml` (local ONNX + label map).

## Check it is working

```sh
docker compose exec -T xdna fxdna status
docker compose exec -T xdna fxdna wait plus://MODEL_A --timeout 1800
```

`wait` defaults to `PREPARED`: the artifact is built and statically
checked. It is not a claim the model has already run on hardware.
Model activation and its bounded native checks occur before the
successful Frigate handshake.

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
  compilation needs the device, and a cold model cannot finish
  inside Frigate's 30-second model wait — prepare first, then
  reinitialize Frigate's detector.
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

## Troubleshooting and operations

Detailed troubleshooting, architecture, legal notices, and
development material live in the linked documents: `docs/OPERATIONS.md`
(runbook: install, update, offline, faults), `docs/RELEASE-8.4.md`
(acceptance record), `docs/ACCEPTANCE.md` (gates),
`docs/INTERFACES.md` (wire and CLI contracts), `docs/RELEASE.md`
(release automation), `docs/WORKQUEUE.md` (implementation log).

## Licensing

Project source is MIT (`LICENSE`). The release appliance additionally
incorporates third-party and AMD vendor binaries governed by their own
terms (EULA flow-down, third-party notices); see
`THIRD_PARTY_NOTICES.md` and `/opt/fxdna/legal` inside the image.
