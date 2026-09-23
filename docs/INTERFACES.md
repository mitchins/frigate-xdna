# Interfaces: CLI, environment and stock Frigate wire contract

Normative companion to `SPEC.md`. These are proposed interfaces to implement, not commands already installed on the user's host.

## 1. CLI

`fxdna` is the only user-facing executable. Default command in Docker is `serve`.

| Command | Behaviour |
|---|---|
| `fxdna serve` | Start manager, ZMQ frontend, preparation queue and worker supervision. |
| `fxdna prepare REF... [--wait] [--refresh]` | Register durable preparation requests; deduplicate/cache; do not change the active model. `--refresh` re-fetches Plus metadata/source and detects changed content rather than silently overwriting it. |
| `fxdna prepare REF --descriptor FILE [--wire-name NAME]` | Import an explicitly described local ONNX or RAI. Wire name is an alias only. |
| `fxdna prepare REF --maintenance --wait` | Explicitly permit a device-requiring preparation/validation pause; restore the prior active model afterward when safe. |
| `fxdna prepare REF... --standalone --wait` | Run without daemon, taking the same exclusive cache/device lock. Fail if a daemon already owns the directory. |
| `fxdna status [REF] [--json]` | Show model/job/service state and active worker without opening the NPU. Per-model views carry `phase`/`elapsed_s` (newest job), `verified` (recorded native check), `error_code` and `compile_key`; the top level carries the live `worker` binding (or null), `ready` and `reason`. `ACTIVE` is projected only with a live worker on the ref's key — a persisted record alone never claims it. |
| `fxdna wait REF [--state prepared\|verified\|active] [--timeout SECONDS]` | Wait through private admin socket; default `prepared`, timeout 1800 s. Ranked satisfaction: ACTIVE implies VERIFIED implies PREPARED (activation records verification), so waiting for a lesser state succeeds once a better one projects. Fail immediately on terminal error/quarantine. Never claim prepared means resident. |
| `fxdna activate REF [--maintenance] [--wait] [--timeout SECONDS]` | Activate an already prepared model. Changing a busy active model requires `--maintenance`; new child, no hot swap. No implicit download/compile here. `--wait` waits (default bound 300 s) until the ref projects ACTIVE. |
| `fxdna cache list [--json]` | List source/compiled sizes, pins and last use. |
| `fxdna cache prune [--apply] [--max-bytes N]` | Dry-run by default. Never delete configured/active/job/rollback/evidence pins. |
| `fxdna doctor [--hardware] [--json]` | Passive checks by default; hardware probe requires an idle device lease and explicit flag. |
| `fxdna health [--ready]` | Supervisor liveness by default (bounded admin request; dead daemon, stale socket, or failed request is `alive: false` with a nonzero exit — never success with `alive: false`). `--ready` requires a loaded, alive, permitted worker with no inhibition. No NPU probe, no compile, no test inference. Compose uses liveness (first-time compilation is not daemon failure). |
| `fxdna recover REF --acknowledge` | Clear a specified recoverable inhibition after displaying the reason; does not itself submit inference. Host-reset-suspect artifacts need explicit operator approval and are never retried automatically. |

`REF` is `plus://ID`, a local ONNX path, or a local RAI path with a descriptor. Local paths are ingested once; content keys, not basenames, identify them. The service accepts local files only from its configured import root (Docker `/models`); native CLI may ingest an explicitly supplied path into the cache.

The daemon is the only registry writer. Online commands use `/run/frigate-xdna/control.sock`. Standalone prepare uses the same manager implementation, not a second shell recipe. `serve` ignores no unknown arguments. Log/status JSON has `schema_version: 1`.

Suggested CLI exit codes:

```
0  success / requested state reached
2  invalid arguments/config/descriptor
3  not ready / timed out waiting
4  acquisition/authentication failure
5  unsupported model/target contract
6  compile/resource failure
7  validation failure
8  device unavailable / device fault / safety inhibition
9  ownership lock or active-model conflict
10 cache integrity failure
```

## 2. Environment

| Variable | Default | Purpose |
|---|---|---|
| `FXDNA_MODELS` | empty | Comma/newline separated preparation refs; each is pinned and queued once. |
| `PLUS_API_KEY_FILE` | unset | Read secret from file. |
| `PLUS_API_KEY` | unset | Alternative credential; mutually exclusive with `_FILE`. |
| `FXDNA_DATA_DIR` | `/data` in image | Local persistent registry/cache/work root. |
| `FXDNA_ENDPOINT` | `tcp://0.0.0.0:5555` in image; loopback natively | Stock Frigate detector endpoint; `ipc://` also supported. |
| `FXDNA_DEVICE` | `/dev/accel/accel0` | Device node; does not imply target compatibility. |
| `FXDNA_LOG_LEVEL` | `info` | Structured logging verbosity; debug still redacts secrets. |
| `FXDNA_OFFLINE` | `false` | Prohibit model acquisition; cached/local models still work. |
| `FXDNA_ALLOW_UPLOADS` | `true` | Allow ordinary ONNX model bytes over trusted ZMQ. Disabling does not make ZMQ authenticated. |

Rationale for the `true` default (product decision, 2026-09-18): stock
Frigate rc2 only becomes ready when its startup model transfer returns
`model_saved=true` and `model_loaded=true`, and the sidecar deliberately
requires a source-byte transfer on every new connection — even for a
cached basename — because basenames are not identities and only the
transferred bytes bind a routing identity to an exact source hash.
Defaulting to `false` would break unmodified stock Frigate out of the box
and defeat byte-verified identity. The security boundary is therefore the
narrow network exposure (Compose network only, no published port by
default, loopback/trusted-LAN only when explicitly published) plus
bounded sizes, model validation, no arbitrary URLs/custom ops, and
compiler resource limits. `false` remains a supported hardening mode for
deployments that intentionally disable Frigate model transfer.

Advanced limits should initially be pinned defaults in the versioned config model, with CLI overrides for development where needed—not fifty environment variables. Defaults: compiler concurrency 1, queue 8 inference requests, header 16 KiB, tensor 16 MiB, model 256 MiB, model-operation deadline 25 s (below Frigate's 30 s), inference service budget 150 ms (below default client timeout), compiler timeout 2700 s, four compiler threads, memory allowance 6 GiB. Larger models can fail explicitly; do not silently increase limits.

Docker resource limits remain container settings, not magic application guarantees. A dedicated launcher must enforce compiler child limits or honestly report only aggregate container enforcement.

Never expand environment-variable content through a shell. Commas/newlines delimit model refs, so literal delimiters in file names must be URI-encoded or supplied through CLI. Repeated `prepare` requests are idempotent.

## 3. Frigate 0.18-rc2 wire protocol

Pin `frigate/detectors/plugins/zmq_ipc.py` blob `cc9a538c81160562184901395dafd3988559c4f1` at tags `v0.18.0-rc2` and `v0.18.0` (verified identical). [S1]

The body below excludes ROUTER identity/delimiter frames. ROUTER must retain the exact return envelope. Each request receives exactly one reply; no unsolicited status pushes to a REQ socket.

### Availability

Request: one JSON frame.

```json
{"model_request": true, "model_name": "MODEL_ID_OR_BASENAME"}
```

For a new, content-unverified connection, force transfer:

```json
{"model_available": false, "model_loaded": false}
```

This remains true even when a same-name entry is cached: the server has not established which bytes this client intends. Once the same routing identity is bound to an active generation, it may receive:

```json
{"model_available": true, "model_loaded": true}
```

Do not optimize away source verification by trusting local basenames.

### Transfer

Request: JSON plus one raw model frame.

```json
{"model_data": true, "model_name": "MODEL_ID_OR_BASENAME"}
```

Validate framing/limits; hash bytes; resolve contract; locate compile key.

Cached artifact successfully activated and binding established:

```json
{"model_saved": true, "model_loaded": true}
```

Valid source saved, preparation queued, not yet active:

```json
{"model_saved": true, "model_loaded": false, "state": "PREPARING", "error_code": "MODEL_NOT_PREPARED"}
```

Invalid source or rejected transfer:

```json
{"model_saved": false, "model_loaded": false, "error_code": "INVALID_MODEL"}
```

Extra diagnostic fields are for our logs/custom clients; stock rc2 does not use them to wait/retry. Do not block a transfer until a minutes-long compile finishes. A client that failed initialization needs stock detector reinitialization once prepared.

### Inference

Request: JSON plus raw C-order array bytes.

```json
{"shape": [1, 3, 320, 320], "dtype": "float32", "model_type": "yologeneric"}
```

This is an example, not the contract for every model. The enum name `yologeneric` differs from the configuration value `yolo-generic`. No model identity, label map, pixel-format flag or normalization flag accompanies this frame. The established routing-identity/generation binding is therefore mandatory.

Response: one 480-byte little-endian float32 frame representing `[20,6]`:

```
[class_id, score, ymin, xmin, ymax, xmax]
```

Rows descending by score; unused rows zero; finite values; valid integral class IDs; normalized ordered box coordinates. No COCO relabeling or camera-space conversion in the sidecar.

### Deadline/late-result rules

* Maintain a per-request internal ID, routing identity, generation and deadline.
* Do not start already-expired queued work. Discarded queue entries still
  get an explicit error reply (TIMEOUT JSON, or a zero frame for infer),
  never a silent drop, so stock REQ clients cannot hang.
* Never deliver late results as the answer to a subsequent request.
* A reconnect gets a new identity and must handshake again.
* Individual client timeouts do not unload/reload a healthy native model.
* Errors that return zero frames increment failures, not successful inference. Sidecar health/metrics distinguish no objects from no detector.

## 4. Private native IPC

Define a small versioned framing protocol over an inherited private Unix socket/socketpair. Example message fields:

```
protocol_version
message_type: load | infer | result | status | shutdown
request_id
worker_generation
artifact_path (load only, supervisor-controlled)
serving_contract_path (load only)
tensor_spec + payload_length (infer/result)
error_code + bounded message (failure)
```

Use a fixed-width length prefix and bounded JSON header plus binary payload. The native child may only receive supervisor-validated paths/metadata. C++ must independently validate lengths/dtypes to protect its own boundary. Retain the proven mmap buffer type (`uint8_t*`, not `void*` in `std::any`) and lifetime ordering; exceptions become structured failures, not undefined behaviour.

No runtime compiler API, model hot-swap command, or public administrative endpoint is exposed by the native child.
