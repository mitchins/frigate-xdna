# Frigate-XDNA — v0.1 product and implementation specification

**Status:** implementation baseline, 2026-09-18. This specifies a product to build; it does not claim that its container or service already exists.

**Product:** a self-contained Linux XDNA detector sidecar for unmodified Frigate, with Frigate+ model acquisition, local ONNX-to-RAI compilation, persistent multi-model caching, and a resident native inference worker.

**Primary command:** `fxdna`.

**Normative language:** MUST / MUST NOT are release requirements. SHOULD is the default unless an exception is documented and tested. All behaviour below is a design decision unless marked as existing upstream behaviour or reported experimental evidence.

## 1. The product contract

A user provides a supported AMD NPU, a persistent data directory, and either local model files or Frigate+ credentials plus selected model IDs. The sidecar downloads or receives an ordinary compatible ONNX, compiles it once, and serves detections through Frigate's existing ZMQ detector.

The user MUST NOT need an AMD account, an SDK installation, a licence-server entitlement, manual ONNX conversion, a Frigate fork, or a newly published sidecar image just because a supported model acquired new weights or a new Plus ID.

A new architecture, unsupported operation, incompatible output contract, or different device generation is a compatibility event. A new compatible fine-tune is not. This distinction MUST appear in documentation: zero-day **compatible model/weight support**, not a promise to compile every future ONNX graph.

Public precompiled model catalogues and future Frigate+-side RAI generation are optional optimisations, not dependencies of v0.1.

### 1.1 Release scope

* Linux x86-64; Ubuntu 24.04 userspace inside the appliance; Python 3.12 for orchestration; C++17 for FlexML inference.
* First certified device: the proven Strix Halo XDNA2/AIE2P target. Other Ryzen AI NPUs require explicit compatibility testing; do not infer support from `/dev/accel/accel0` existing.
* Compatibility anchor: unmodified Frigate `v0.18.0`. Pin the actual source and image digests in tests rather than following `dev`.
* Primary model contract: batch-one, static-shape, three-channel YOLO-style object detection, initially the YOLOv9s-320 input/output family. The banked YOLOv8n artifact remains a test fixture.
* Frigate+ YOLOv9 t/s and private fine-tunes are product targets, but each claimed contract must pass on the **actual Plus ONNX**, not merely a similarly named public YOLO export. The earlier v9t NaN result remains a real validation issue until resolved for the relevant graph.
* 320 and 640 are supported geometries where a compiled and validated model supports them. No silent spatial resizing of a model graph.
* Multiple stored models; **one resident model and one NPU execution at a time** in v0.1.
* Detection only. State/object classification, segmentation, pose, general tensor RPC, multiple NPUs, concurrent different-model serving, and an administrative web UI are outside v0.1.

## 2. Banked evidence and its limits

The following are user-reported results to import from the local evidence bundles. They are not newly reproduced by this spec.

| Evidence | Banked result |
|---|---|
| Resident v9s-320 soak | 24 hours, 3,364,689 matched XDNA submissions/completions, no inference errors or host resets, XDNA Err=0 |
| Service observations | Approximately 40 requested inferences/s; p50 around 8.65 ms; 73 FPS maximum in a separate saturation benchmark |
| Availability qualification | 67 client timeouts in one transient host-stall window; immediate recovery. Do not relabel this as zero timeouts or perfect end-to-end availability. |
| Compiler appliance | Clean Ubuntu 24.04 debootstrap chroot, network isolated; fresh ONNX -> Quark BF16 -> VAIML -> RAI -> standalone FlexMLRT inference |
| Compiler measurements | 1.67 GB audited payload, 541 MB compressed; example 2.7 s BF16 preparation + 322 s compile, 1.69 GB peak RSS, 7.72 MB output |
| Deployment runtime | Standalone FlexMLRT + XRT/shim; no ORT/VOE/VAIML/xcompiler loaded into the inference process |
| Binary distribution | Project audit accepts incorporated, unmodified, hash-pinned appliance distribution with the applicable AMD EULA/TPN flow-down and notices |
| Frigate boundary | Actual stock ZMQ detector tested against the native worker; full Frigate application/camera/tracker integration remains a release test |

The compiler size is not the Docker image size. It excludes whatever base OS, manager, ordinary Python dependencies, build/runtime support, and licence material the final image adds. Measure compressed image, unpacked image, persistent cache, and scratch separately.

The soak does **not** establish why earlier host resets happened, prove that a fresh process fixes driver faults, certify model churn, or validate compiler/inference concurrency. Process isolation does not isolate the host kernel from an accelerator fault.

40 requests/s is a useful workload, not the maximum demand of eight cameras at 5 detect FPS: a camera frame may require multiple detection regions. Capacity must be assessed from observed request rate, queue delay, and drops, not cameras multiplied by FPS alone. [S2, S5]

### 2.1 Evidence to import before implementation

Read the actual manifests, successful scripts, outputs and licences at:

```
/mnt/downloads/xdna-phase5/20260916/
/mnt/downloads/xdna-phase5-license/20260916/
/mnt/downloads/xdna-phase7/20260916/
/mnt/downloads/xdna-phase7-soak/20260917/
/mnt/downloads/xdna-compiler-audit/20260918/
/root/xdna/UNWIND.md
```

Import full SHA-256 values, compiler flags and runtime dependencies. Never manufacture complete hashes from the prefixes in a chat summary. Do not modify these evidence directories, CT110 production Frigate, CT113 Home Assistant, the host driver/firmware, or an active research run.

## 3. Architecture

```
Frigate+ authenticated GETs    Local ONNX / RAI + descriptor
             |                            |
             +------------+---------------+
                          v
                 fxdna supervisor
             Python, no accelerator runtime loaded
             |            |              |
       Plus downloader  registry      ZMQ ROUTER frontend
             |          + cache       compatible with stock REQ
             v            |              |
       compile queue -----+              | private local IPC
             |                           v
       compiler child               native inference child
       audited Python/native        C++17 + FlexMLRT + XRT
       short lived                  one model per process
             |                           |
             +-------- device arbiter ---+
                          |
                    /dev/accel/accel0
```

### 3.1 Process responsibilities

**Supervisor:** owns validated configuration, HTTP acquisition, SQLite registry, filesystem cache, compile queue, request routing, worker lifetime, private administrative socket, logs, metrics and health. It MUST NOT import Quark, torch, vendor ONNX Runtime, or FlexML native bindings.

**Compiler child:** executes the audited recipe in a separate prefix/environment, writes only a job workspace, exits after compilation. It receives file paths and explicit configuration, not Plus credentials. Its environment is allowlisted, inherited network FDs are closed, and its operation is offline-capable. A no-network syscall policy for the child SHOULD be added without granting privileged container capabilities; the offline image test is mandatory regardless. Do not claim that an allowlisted environment is a complete filesystem sandbox.

**Native inference child:** loads one artifact once, validates it, retains one FlexML model/context and serves repeated requests. It has no Plus credentials or downloader. Its linked/mapped native closure MUST remain limited to the banked runtime and ordinary system libraries. Exit the process to change the model; never unload A and load B within the same FlexML process.

**Device arbiter:** serializes all service-owned operations that can open or submit to the NPU. One sidecar instance per physical NPU is supported initially. A local lock cannot prevent unrelated software from opening the device; document that boundary.

### 3.2 IPC and request ownership

Use a Python `zmq.asyncio` ROUTER frontend and private, versioned local request/reply IPC to the native child. A ROUTER is wire-compatible with the stock REQ client and exposes routing identities needed to prevent responses from crossing clients or model generations. [S1, S6]

Prefer a private Unix-domain socket with length-prefixed control/tensor frames between supervisor and native child; no shared-memory ring, custom broker, or zero-copy framework in v0.1. Tag every internal request with an ID and worker generation. Never pass pointers across the process boundary.

At most one native inference is in flight. Bound accepted waiting work to eight requests and an explicit deadline; reject expired work before execution. Do not accumulate historical camera frames. ROUTER identities are routing state, **not authentication**.

### 3.3 Preparation is not activation

Adding a model to the preparation list must not switch the active detector. It can fetch, validate metadata, perform CPU-side preparation and, where safe, compile it. Activation happens only through a valid model handshake or explicit administrative action.

The service MUST distinguish:

* `PREPARED`: artifact built and statically checked; activation smoke test still required for this runtime/target if not already recorded.
* `VERIFIED`: artifact has passed the recorded native/reference checks for this runtime/target.
* `ACTIVE`: verified model is resident in the currently serving worker.

Do not report `model_loaded=true` because a file exists.

### 3.4 Compiler/device concurrency policy

The offline-chroot proof establishes lack of network/entitlement dependence; it does not establish lack of NPU dependence.

The packaging task MUST determine whether the exact compile recipe completes with the accelerator device unavailable. Classify recipe stages as `cpu_only` or `requires_device` using evidence.

Until that is established, compilation conservatively requires the device lease. While a model is resident, such a job remains `WAITING_FOR_DEVICE`; downloads and metadata preparation may proceed. The service MUST NOT silently stop live detection to compile an update.

If the compile recipe is proven CPU-only, it may run as one resource-bounded background job while inference continues. Native validation of a new model still requires the exclusive device lease. Validate concurrent compile CPU/memory interference separately before claiming uninterrupted updates.

For device-requiring recipes, `fxdna prepare REF --maintenance` explicitly authorizes a controlled inference pause, compilation/validation, then restoration of the prior active model. It must report that pause. Compilation is still supported on day one; its concurrency/maintenance properties must not be fabricated.

## 4. User-facing workflow

### 4.1 First installation

1. Pull the appliance image, pass the accelerator node, mount `/data`, supply selected model references and optionally a Plus API key.
2. Start the sidecar. It prepares the selected models without requiring Frigate to start first.
3. `fxdna wait REF` waits for `PREPARED` or better and returns an actionable state.
4. Point stock Frigate at the sidecar using `type: zmq` and the same `plus://ID` or local ONNX model.
5. The model transfer identifies the exact bytes, hits the compiled cache, starts the native worker, performs its bounded activation check, and acknowledges only when active.

A cold compilation lasting minutes MUST NOT be held inside Frigate's model-management request. Frigate rc2 uses 30-second model-operation waits and does not keep polling after initial model initialization fails. [S1]

### 4.2 New Plus model / private fine-tune

`fxdna prepare plus://NEW_ID` adds a durable preparation request without restarting the active worker. Editing `FXDNA_MODELS` and restarting the sidecar is also supported, but causes the normal brief worker restart and must not be described as zero downtime.

Once the artifact is prepared, change Frigate to the new model and restart/reinitialize its detector through stock configuration behaviour. The sidecar activates the prepared model in a fresh process, subject to the single-model/client rules below.

No new service release, hardcoded hash catalogue, public upload of private weights, or maintainer-run model compilation is involved.

### 4.3 Unexpected cold model from ZMQ

Receive and hash it, queue one preparation job, and return a prompt truthful not-loaded response. Status/CLI explains `PREPARING` and that rc2 needs detector reinitialization after preparation finishes.

Do not promise transparent eventual activation in an already-failed rc2 client. Do not change Frigate. The recommended no-outage workflow is eager preparation.

### 4.4 Cached and disconnected operation

An already prepared, validated model must continue operating without Plus or Internet connectivity, subject to the applicable model terms. Offline operation must not require re-authentication simply to read locally cached bytes.

Never downgrade a working local model because a metadata refresh, token refresh, or DNS lookup fails. New downloads fail visibly and independently.

## 5. External interfaces

`docs/INTERFACES.md` is normative and supplies exact commands, environment variables, error codes and wire examples.

### 5.1 Minimal configuration

Normal configuration is environment variables plus CLI; no mandatory YAML file or database service:

```
FXDNA_MODELS=plus://MODEL_A,plus://MODEL_B
PLUS_API_KEY=<the key already configured for Frigate>
FXDNA_DATA_DIR=/data
FXDNA_ENDPOINT=tcp://0.0.0.0:5555
FXDNA_DEVICE=/dev/accel/accel0
```

The device and data defaults above are for the Docker image. Native development binds to loopback by default. `PLUS_API_KEY` (container environment) is the sole Plus credential. Secrets never appear in CLI arguments, status output, compiler environment, or cache manifests.

### 5.2 Administrative interface

CLI controls the live daemon over a private Unix-domain socket, with filesystem access control, not another public network API. `prepare`, `status`, `wait`, `activate`, `cache`, `doctor`, and `recover` share the same internal model manager.

`prepare --standalone` works with no daemon and takes an exclusive data/device lock. It fails if the live daemon owns that data directory. There must not be two independent cache writers or compiler supervisors.

No service UI, MQTT control path, external Redis, Docker socket, or Kubernetes operator is required.

### 5.3 Frigate+ acquisition

Implement the authenticated read-only flow reflected in rc2's `PlusApi`: token exchange, model metadata, signed download URL, model bytes. [S3]

* Fetch selected IDs only. Do not train models, spend training credits, upload frames, annotate data, or compile the entire account catalogue.
* Preserve the Plus `labelMap`, attributes, dimensions, input layout/pixel format/dtype and model type. Compare them against the ONNX graph; metadata conflicts are failures, not opportunities to guess. [S4]
* Copy the model bytes before transformations; content hashing uses the original received/downloaded source.
* Use HTTPS with bounded connect/read timeouts and download size; bounded retry/backoff on transient failures. Respect rate limits and retry guidance.
* Never forward API credentials or bearer headers to signed-download hosts. Validate every redirect, deny non-HTTPS and private/local/metadata-service targets, and never log signed URLs or authorization material.
* Keep signed URLs and tokens in memory only. Store model ID, payload digest, sanitized metadata and provenance, not secrets.
* Local/private fine-tuned weights and RAI remain private in `/data`. Do not bake them into release images or public test artifacts.

### 5.4 Model formats

Accepted sources are `plus://ID`, local ONNX paths under an explicit import directory, and local precompiled RAI with a sidecar descriptor.

No `.pt`/pickle loading, untrusted custom Python, arbitrary ONNX custom-op DLLs, or network URL fetch requested by an unauthenticated model name. Frigate's one-file transfer cannot deliver arbitrary external ONNX tensor data: reject that form with a clear error, or import a validated local bundle with all referenced members confined to the bundle. Never follow external-data paths outside the import root.

Bare `.rai` is not enough for a generic client contract. Imported RAI requires full artifact hash, compatible target/runtime profile, tensor semantics and postprocessor/label description. Introspecting shapes cannot recover RGB/BGR, normalization, or semantic label IDs.

## 6. Stock ZMQ contract and identity

### 6.1 Facts from rc2

The plugin sends a basename as `model_name`; model requests/transfers carry that name and optionally bytes. Inference requests carry `shape`, NumPy dtype name, and `model_type` using the enum **name** (for example `yologeneric`), but **no model ID or content hash**. Model waits are 30 seconds; inference timeout defaults to 200 ms. An unsuccessful initial load leaves `_model_ready` false, and `detect_raw` then returns zeros without automatic preparation polling. [S1]

These constraints are not negotiable through a sidecar-only extension. Additional response fields may aid custom clients but rc2 cannot be assumed to use them.

### 6.2 Binding exact bytes

For each new routing identity, the sidecar initially requests model transfer even when a same-name cache entry exists. This prevents an ONNX file overwritten under the same basename from silently selecting an old artifact. A transfer-cache hit is not a compile-cache miss.

After transfer, hash the exact bytes and bind that routing identity to:

```
source_digest + serving_contract_digest + active_generation
```

A basename is only a hint/alias. Never treat it as an authoritative compile key. Plus refs already fetched by authenticated ID provide metadata and expected hashes; compare transferred bytes before success. Unexpected source changes under an already-pinned Plus ID are surfaced, not silently overwritten.

A later request from the same bound identity may receive an immediate available/loaded response if its binding remains valid. No per-frame hashes or model bytes are needed.

### 6.3 Active model and multiple clients

v0.1 supports one logical Frigate deployment per endpoint. Multiple request sockets using the **same** source/serving contract may share the resident worker and bounded queue. Different active models on the same endpoint are not supported concurrently.

A valid request for a different already-prepared model can perform an automatic controlled switch only when no old-generation request is in flight and old-model request bindings are quiescent. Use a 5-second quiescence interval, fit the switch within the model-operation deadline, and invalidate all old-generation bindings before serving the new model. A quiescence interval is an operational guard, not proof a remote process exited.

If old-model traffic is still active, return `MODEL_IN_USE`, keep A running, and require the user to stop/reinitialize the old detector or invoke `fxdna activate B --maintenance` deliberately. Do not ping-pong models because two clients want different artifacts.

After a successful A->B switch, remember superseded A as ineligible for **automatic** reactivation until an explicit administrative activation or a clean configured restart permits it. This prevents old Frigate clients which recreate sockets after a timeout from switching the service back. Preserve A as a manual rollback candidate.

Every inference request must match a current bound identity/generation. Stale requests must never run against the new model, even if their shapes happen to match. This is essential for private Plus models sharing identical tensor geometries but different weights or labels.

### 6.4 Response and error semantics

Successful detection response: one 480-byte frame, little-endian float32, shape `[20,6]`, rows `[class_id, score, ymin, xmin, ymax, xmax]`, descending score, unused rows zero. Coordinates are normalized relative to the submitted detector crop, **not the original camera frame**. [S1, S2]

Malformed tensor or ordinary request rejection: return the protocol's zero detection array, increment an error/rejection counter, log a rate-limited structured diagnostic and expose degraded status as appropriate. A zero response is not counted as successful inference. Transport timeouts remain visible separately. No CPU fallback, fabricated boxes, or cached previous-frame response.

Worker death/device errors: fail requests, trip the safety policy, and never keep advertising loaded status. Do not restart the NPU worker for an individual client timeout when the native execution is still healthy. Discard late results for expired requests or retired generations.

## 7. Model semantics and correctness

Frigate's rc2 input path transposes according to `input_tensor` and, for `input_dtype: float`, casts to float32 and divides by 255 before calling the detector. [S2] Therefore the sidecar MUST NOT normalize again or independently resize/letterbox the incoming detector tensor.

Validate byte length with overflow-safe shape multiplication, batch 1, rank, geometry, C-order, dtype and finite values. Conversion to the FlexML input representation is explicit and versioned. The only permitted transformations are those specified in the serving contract.

### 7.1 Postprocessor profiles

Do not put YOLO family names in an ever-growing ID whitelist. Use tested output contracts:

* Raw decoded YOLO boxes/classes: specified output layout, `cxcywh` or `xyxy`, pixel/normalized coordinate units, class-score representation, optional objectness, and explicit class count.
* End-to-end/NMS-free detections: only when the output contract and tests exist. Do not apply NMS a second time automatically.
* Unknown or ambiguous contracts: `UNSUPPORTED_MODEL_CONTRACT`, not a best guess.

For raw profiles, implement deterministic class-aware NMS; retain different classes where required for Plus attributes; sort by score; cap at 20 only at the Frigate boundary. Record candidate threshold and NMS parameters in the serving contract. Frigate remains responsible for user tracking filters and zones.

Class IDs MUST retain the Plus/model numeric meaning. Never remap to a guessed COCO order, collapse `truck` into `car`, assume 80 classes, or treat the earlier default-labelmap discrepancy as harmless. Attributes and label-map changes are metadata changes even when graph shape is unchanged.

### 7.2 Compilation and activation validation

A compile exit code and an existing `.rai` do not establish correctness. The earlier v9t NaN failure demonstrates the need for a validation gate.

Preparation produces original-ONNX CPU reference probes using the pinned reference runtime and canonical inputs. Probe inputs and output summaries are retained privately. Profiles define versioned numerical and detection-equivalence tolerances derived from known-good experiments; they must be fixed before testing a new candidate and not relaxed per failing model. There is no generic claim that random probes establish real-world accuracy.

On first activation for an artifact/runtime/target combination, run a small bounded native smoke set in the **same child/context** that will serve traffic. Check shapes, dtypes, finiteness, raw-output agreement and meaningful detection agreement where fixtures apply. Keep that child alive after success. Empty scenes alone are not a sufficient correctness fixture.

Record validation separately from compilation. If a prepared artifact fails native checks, quarantine that artifact/recipe combination, retain diagnostic evidence and do not retry it automatically. A previous verified model may be explicitly reactivated; do not silently substitute a different model beneath Frigate's configured labels.

## 8. Smart persistent cache

`docs/CACHE.md` defines keys, manifests and crash recovery. Core rules:

* Source digest = SHA-256 of exact ONNX bytes, or a canonical manifest including every external-data member for an explicitly supported local bundle.
* Compile key = SHA-256 of source digest, audited compiler payload digest, recipe version/options, static input compile contract, target profile and RAI format compatibility ID.
* Serving contract digest separately includes preprocessing semantics, decoder version/options, label map and relevant Plus attributes. Changing a label string or postprocessor does not necessarily require recompilation; it does require a new serving/validation binding.
* Hardware identity used for compilation is architecture/compatibility, not BDF, USB path, physical machine serial, boot ID or arbitrary container path.
* Record runtime/XRT/firmware/driver versions for diagnosis and compatibility validation. Do not recompile for every app rebuild or blindly reuse artifacts across unsupported ABI/target changes.
* Keep original source, successful compiled artifacts and at least the previous active verified artifact. Multiple IDs with the same source/compile key can share compiled bytes while retaining distinct metadata/authorization provenance.
* No compiler execution on a valid cache hit. Do not let a client reconnect retrigger a preparation job.

Use SQLite transactions for the job registry and local filesystem locking for single writers. Use temporary same-filesystem directories, hash verification, fsync and atomic rename for artifacts; publish registry success only after the artifact is durably committed. Recovery reconciles filesystem and registry state without trusting partial files.

Default: no time-based eviction of configured, active, in-progress, quarantined-evidence or rollback-pinned records. Automatic eviction of unpinned entries is optional, off by default. Refuse new work when scratch space is insufficient; do not delete active models or saturate the filesystem. Manual prune is dry-run by default and applies under exclusive locks.

`/data` MUST reside on a local filesystem with reliable locking/atomic rename: ext4, XFS, local Btrfs/ZFS are candidates to validate. Do not put the SQLite/WAL/work queue on CIFS/NFS. Backups may go to network storage separately.

## 9. Compiler integration

Use the audited 1.8 BF16 path, not the unsuccessful INT8/X2-to-FlexML combination. Import the exact clean-room recipe and payload manifest; do not rebuild it from a guessed `pip install` list.

The package has separate immutable runtime and compiler prefixes. Compiler-specific `LD_LIBRARY_PATH`, `PYTHONPATH`, HOME, TMPDIR and caches are set by a dedicated launcher. A compiler environment must never leak into the inference child.

A compiler job:

1. Acquires per-key lock and rechecks the cache.
2. Creates a confined work directory with input copy and immutable job manifest.
3. Validates ONNX and serving metadata; obtains reference probes.
4. Runs BF16 preparation and VAIML with pinned settings, fixed batch/shape and no hidden model cache reuse.
5. Exports only the deployable `.rai` and declared descriptors; validates format/size/hash.
6. Commits the result atomically, records logs/resource metrics, releases scratch and exits.

One compiler job at a time. Initial resource budget: four CPU threads, 6 GiB compiler-child memory allowance, 45-minute timeout, and a measured scratch-space preflight. These are engineering defaults, not claims all models fit. Overall Docker limit starts at four CPUs/8 GiB; measure combined manager/compiler/runtime headroom. Do not raise host limits automatically.

No model-specific graph surgery, arbitrary package upgrades, online downloads or repeated fallback export recipes in a production compile. Fail with the exact unsupported operation/stage and preserve a bounded diagnostic bundle. Changing the recipe creates a new compile key.

## 10. Device safety and lifecycle

The safety objective is resident stability plus controlled infrequent model transitions, not recreating the failed model sweep in a daemon.

* Device-sensitive operations are exclusive; no background `xrt-smi` polling.
* Diagnostics are passive by default; explicit hardware checks run only under the same device lease while idle.
* Worker generation changes on each process start. Do not reuse an old pointer, BO, mmap or context across generations.
* Graceful shutdown: stop accepting inference, drain bounded work, stop the client-facing execution path, terminate the child, wait for cleanup, then mark clean state. Preserve the proven `.rai` mapping/object destruction order.
* A child stuck in uninterruptible kernel sleep is not permission to launch another child, reset the NPU or reboot the host.
* Persist an operation journal before compilation, activation, validation and teardown. Record observed boot token, token scope, process start identity and timestamps. Container-visible boot IDs may be virtualized; never label a change alone as proof of a physical host reset.
* If a device-sensitive operation was interrupted by a changed boot token/unclean restart, quarantine it and pause that operation. Never auto-retry an artifact associated with a suspected host reset.
* A single transient transport timeout does not restart the worker. Unexpected native/device failures trip a circuit breaker; recovery is bounded and explicit, not an infinite Docker restart loop.
* A previously verified resident model may have one guarded recovery after an ordinary unclean process restart. Repeated unclean recovery or any driver/device fault blocks further NPU activation pending `fxdna recover` and operator review.
* Daemon restart may keep the administrative/status process alive while hardware is inhibited. `restart: unless-stopped` must not bypass persistent inhibition.

There is no claim that this policy fixes XRT/firmware bugs. It prevents repeating known-dangerous operations and preserves usable diagnosis.

## 11. Docker appliance and native development

The default release image includes manager, native runner, audited compiler payload, deployment runtime, ordinary dependencies and required legal material. Do not split compilation into a manual SDK install for end users.

Use a multi-stage build. SDK download/authentication, when required for maintainers to obtain legally incorporated inputs, is a **maintainer build concern**, never an end-user runtime step. Do not publish the extracted compiler tar as a standalone asset if the audit only permits incorporation into the appliance. Public source and build recipes are separate from vendor binary rights.

Version and pin all imported payloads. Verify full hashes and file manifests at build time; retain notices; never patch/strip vendor binaries when relying on an unmodified-binary grant. Build ordinary JIT/helper prerequisites into the image or give them a controlled writable cache—no surprise `apt`/`pip` at first compile.

Container requirements:

* Non-root service UID; supplementary access to the host accelerator GID; only the required accelerator character device passed through.
* No `privileged`, no Docker socket, no host networking by default, no `/dev/kfd` or GPU compute exposure, no automatic host module/firmware installation.
* Read-only image root; `/data` persistent; bounded `/tmp` and `/run` tmpfs; compiler temporary data on a capacity-checked `/data/work` rather than a tiny tmpfs.
* Drop capabilities and use `no-new-privileges` where validated. Any exception must be narrow, documented and proven necessary in the acceptance test; never replace a denied operation with blanket privileged mode.
* Use an init/subreaper and explicitly forward signals/reap children.
* Default ZMQ exposure only within the Compose network. Publish to a specified trusted LAN IP only when Frigate is outside that network; raw ZMQ here has no authentication/encryption.
* Liveness healthcheck contacts the supervisor without opening the NPU. Model readiness is separate and checked with `fxdna wait/status`.

Native development uses the same CLI, prefixes, manifests and code paths. It must not rely on the research CT's shell initialization files. Pure unit/contract tests require no NPU/vendor binaries. Hardware tests are explicit, serialized and never run automatically on every agent session.

## 12. Observability

Structured JSON logs by default, with job ID, shortened model alias/hash, stage, elapsed time and error code. Never log credentials, bearer tokens, signed URLs, complete model contents or user frames.

Expose via private admin CLI JSON and optionally a loopback-only read-only HTTP metrics endpoint:

* daemon liveness and safety inhibition;
* each model's source/compile/validation/activation state;
* source hash, compile key, recipe/toolchain/target compatibility;
* active worker PID/generation, current model, uptime and RSS;
* compile queued/running/completed/failed totals and durations;
* request accepted/successful/zero-object/error/rejected/timeout/late-discard counts;
* queue wait, native inference and total sidecar latency histograms;
* cache bytes/hits/misses/pins and disk free;
* restart/quarantine events.

Do not invent compile percentage. Report named stages and elapsed time; only show percentage if the compiler supplies a meaningful denominator.

The native child may report its own execution counts. Hardware counter verification belongs to controlled acceptance tests, not continuous second-client polling. Distinguish submitted/completed from client-delivered responses, particularly during host stalls.

## 13. Security, rights and release boundary

This is a trusted-LAN/local accelerator service, not an Internet-facing multi-tenant model-execution sandbox. Untrusted graphs can reach a complex native compiler and eventually a host kernel driver. Keep the network boundary narrow, apply bounded input/resource checks, and do not claim containers eliminate that risk.

Release relies on the user's exact appliance licence audit. Preserve a per-file manifest linking incorporated unmodified binaries to their governing terms and notices. Do not relabel the complete image MIT, infer all native files are MIT from wheel metadata, or automatically impose AMD terms on unrelated original open-source code. Source code may be published separately under the project's chosen licence; its owner must supply copyright details.

The specification does not independently interpret the EULA PDFs. The release task must carry the audited text/conditions forward, not reopen general licensing research unless the binaries or intended use change. No private model or API key in repository fixtures, image layers, build logs or released reports.

## 14. Acceptance and rollout

`docs/ACCEPTANCE.md` contains traceable gates. Required final demonstration:

```
clean supported host/userspace
+ released candidate image
+ NPU device
+ persistent /data
+ Plus key and a newly selected compatible private model ID

fetch -> fresh compile -> cache -> stock Frigate model handshake
-> correct detections -> actual full Frigate camera/tracker operation
-> restart -> cache hit, no compiler execution
```

The exact private Plus model must not have been baked into the image or already present in its cache. The local-ONNX/no-Plus path must also pass.

Full application testing must include FFmpeg ingest, motion regions, detector scheduling, ZMQ, object tracking, events and recovery. A translated image passed directly to `ZmqIpcDetector` is not a substitute for that release test.

Use a separate Frigate test container and deterministic moving video before production. Keep production CT110/HA/LLM workloads unchanged until the user approves a staged rollout. The installed host driver stays pinned during the first release validation.

Performance gates are relative to the imported controlled baseline, not the invalid OpenVINO-vs-ORT fallback comparisons. Sustain 40 requests/s for v9s-320, record p50/p95/p99 and maximum capacity separately, and account for every timeout/drop. Repeat a 24-hour service soak with no hardware faults, no host reset and no unbounded RSS/latency growth; report availability incidents honestly.

## 15. Repository and execution plan

Use the target structure in `docs/REPOSITORY.md`. `AGENTS.md` applies to every task. Start with `agent-tasks/01-foundation.md`; execute one bounded task, commit its result and report its acceptance evidence before continuing.

The first tasks are foundation/contracts, Plus/cache manager, packaged compiler, resident native/ZMQ integration, and full Docker/Frigate release validation. They reuse the research implementation where proven; they do not restart the compiler archaeology or model sweep.

## References

Source labels S1–S8 resolve in `docs/SOURCES.md`. Experimental claims above are explicitly attributed to the user's supplied Phase 5–7.7 reports; their full local evidence must be imported by Task 01.
