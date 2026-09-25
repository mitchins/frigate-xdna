# frigate-xdna

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=mitchins_frigate-xdna&metric=alert_status)](https://sonarcloud.io/summary/new-code?id=mitchins_frigate-xdna)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=mitchins_frigate-xdna&metric=coverage)](https://sonarcloud.io/summary/new-code?id=mitchins_frigate-xdna)

Run Frigate object detection on the AMD Ryzen AI / XDNA2 NPU. It puts
an otherwise-idle NPU to work running the detection model; decoding,
motion and resizing still happen on your CPU/GPU as Frigate configures.

- Works with stock Frigate through its existing ZMQ detector
- Frigate+ models or local YOLO ONNX files
- Compiles each model once and caches it; cached models work offline
- No AMD SDK, account or manual conversion

Tested on Ryzen AI Max 300 (Strix Halo) with Frigate 0.18. Other XDNA2
chips are untested; XDNA1 is unsupported. Power draw has not been measured.

## Requirements

- Linux x86_64 with `/dev/accel/accel0`. If it's missing, fix the host's
  XDNA driver/firmware first; a container can't provide it.
- Docker Compose v2 (or Podman), and ~8 GiB of memory while compiling
- A Frigate+ API key only if you use Frigate+ models

## Run it

Save this as `compose.yaml`:

```yaml
services:
  xdna:
    image: ghcr.io/mitchins/frigate-xdna:0.1.2
    init: true
    restart: unless-stopped
    user: "10001:10001"
    group_add:
      - "${NPU_GID:?Set NPU_GID to the accelerator device group ID}"
    devices:
      - /dev/accel/accel0:/dev/accel/accel0
    environment:
      FXDNA_MODELS: "${FXDNA_MODELS:?Set selected model references}"
      FXDNA_DATA_DIR: /data
      FXDNA_ENDPOINT: tcp://0.0.0.0:5555
      FXDNA_LOG_LEVEL: info
      # Frigate+ only: uncomment and use the same key Frigate uses.
      # PLUS_API_KEY: "${PLUS_API_KEY:?Set PLUS_API_KEY to the Frigate+ API key}"
    volumes:
      - xdna-data:/data
      # Local models only: your directory of .onnx files.
      # - /srv/frigate/models:/models:ro
    networks:
      - xdna-net
    read_only: true
    cap_drop:
      - ALL
    security_opt:
      - no-new-privileges:true
    tmpfs:
      - /run:rw,nosuid,nodev,size=16m,uid=10001,gid=10001,mode=0700
      - /tmp:rw,nosuid,nodev,size=256m,mode=1777
    cpus: "4.0"
    mem_limit: 8g
    memswap_limit: 8g
    pids_limit: 512
    ulimits:
      memlock:
        soft: -1
        hard: -1
    stop_grace_period: 60s
    healthcheck:
      test: ["CMD", "fxdna", "health"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 10s

volumes:
  xdna-data:
    name: frigate-xdna-data

networks:
  xdna-net:
    name: frigate-xdna-net
```

Keep the `memlock` block (the runtime fails without it) and never delete
the `xdna-data` volume to fix a problem; it holds your compiled models.
Start it:

```sh
export NPU_GID=$(stat -c %g /dev/accel/accel0)

# Frigate+ (the key is read without echo and is never logged)
read -rsp 'Frigate+ key: ' PLUS_API_KEY; echo; export PLUS_API_KEY
FXDNA_MODELS=plus://MODEL_ID docker compose up -d

# ...or a local YOLO file: /srv/frigate/models/yolov9-c-320.onnx
FXDNA_MODELS=local://yolov9-c-320 docker compose up -d
```

Export local YOLOv9 files [as Frigate documents](https://docs.frigate.video/configuration/object_detectors/);
the sidecar refuses a model it can't run before it compiles anything.
**Portainer:** paste the same YAML into a Stack and set the same
variables in its environment section.

## Point Frigate at it

Put Frigate on the `frigate-xdna-net` network (as an `external: true`
network in Frigate's Compose file), then:

```yaml
detectors:
  xdna:
    type: zmq
    endpoint: tcp://xdna:5555

# Frigate+
model:
  path: plus://MODEL_ID

# ...or local YOLO
model:
  path: /config/models/yolov9-c-320.onnx
  model_type: yolo-generic
  width: 320
  height: 320
  input_tensor: nchw
  input_dtype: float
  labelmap_path: /labelmap/coco-80.txt
```

Frigate needs the same ONNX file: the path may differ between
containers, but the bytes must match. Custom models need their own
label map instead of `coco-80.txt`.

**Frigate with `network_mode: host`** can't resolve `xdna`. Add
`ports: ["127.0.0.1:5555:5555"]` to the sidecar and use
`tcp://127.0.0.1:5555`. Never expose port 5555 beyond a trusted host.

Start the sidecar first: a first compile takes about 7–25 minutes
(~10 for C-320) and Frigate only waits 30 seconds for a model. If
Frigate got there first, restart it once the model is prepared.

## Check it

```sh
docker compose logs -f xdna
docker compose exec xdna fxdna status
```

On first start the log moves through:

```text
Model inspection complete: …
XDNA compilation running: … elapsed=…
Model prepared: …; waiting for Frigate at …
Frigate model handshake complete. / Worker active: …
```

A `Preparation failed` or requirement error names its cause. For
recovery, upgrades and several models see [docs/OPERATIONS.md](docs/OPERATIONS.md).

## Which model?

YOLOv9, measured on Strix Halo (median detector latency):

| YOLOv9 | 320×320 | 640×640 | |
|---|---:|---:|---|
| T | 7.4 ms | fails validation | smallest |
| S | 9.0 ms | 20.0 ms | fast |
| M | 13.1 ms | 34.4 ms | |
| **C** | **14.1 ms** | **35.2 ms** | **recommended** |
| E | 69.4 ms | 224.8 ms | specialist use |

**YOLOv9-C at 320 gives the best balance of quality and speed.**
C at 640 works if your cameras need no more than about 28 detector
requests per second. E works, but it's much slower. Details:
[320](docs/YOLOV9-320-SCALING.md), [640](docs/YOLOV9-640-SCALING.md).

## More

- [Operations](docs/OPERATIONS.md): updates, several models, failures, offline use
- [Interfaces](docs/INTERFACES.md): CLI and wire contracts
- [Development](docs/DEVELOPMENT.md): `make test` runs without an NPU, SDK or key

## License

Project source is MIT (`LICENSE`). The image also contains
third-party and AMD binaries under their own terms. See
`THIRD_PARTY_NOTICES.md` and `/opt/fxdna/legal` in the image.
