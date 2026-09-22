# Acceptance gates and release evidence

## 1. Non-hardware CI (runs on every change)

### Configuration/CLI

Validate defaults, unknown option rejection, malformed ref handling, environment precedence, mutually exclusive secret sources, redaction, `--json` schema and deterministic exit codes. Importing any CLI module must not open an accelerator or import a vendor runtime.

### Cache/registry

Golden key vectors; same bytes/different basenames share compile key; same name/changed bytes do not. Recipe/target changes invalidate compilation; serving-label/decoder changes invalidate bindings without unnecessary compilation. Simulate interruption before and after fsync/rename/DB commit. Verify concurrent prepare deduplication, two aliases sharing bytes, pins, dry-run prune, low disk and corrupted artifacts.

### Plus API

Use a fake server with token expiry, bad credentials, 401/403/404, 429, timeout, malformed metadata, size overflow, signed URL expiry, redirect and content mismatch cases. Assert no API bearer header reaches the download host, no tokens/signed URLs enter logs/DB, and no write/training endpoint is invoked.

### Stock Frigate protocol

Use the exact unmodified rc2 plugin and versioned dependency fixtures, not an imitation client alone. Verify ordinary handshake, forced byte transfer on new identity, repeated handshake, multi-frame validation, exact 480-byte response, sorting/normalization/zero rows and enum-name handling.

Include two clients with identical shapes but different source hashes/label maps. No frame from A may be processed by B. Test generation invalidation, expired requests, reconnect, superseded-model rejection and client timeout without worker restart.

Use a fake compiler taking over 30 seconds to prove cold misses return truthful not-ready promptly and rc2 does not magically become ready without reinitialization. Then prove eager prepare -> stock reinitialize -> immediate cache-hit activation.

### Tensor/postprocessor

Golden synthetic raw tensors for raw YOLO and each supported end-to-end profile; class count not fixed to 80; sparse/private Plus label IDs; attributes; duplicate boxes; tied confidence; out-of-range/NaN/Inf; byte-count overflow; explicit normalization and RGB/BGR metadata conflicts. Unknown profiles fail; do not infer semantics from shape alone.

## 2. Compiler appliance gate

Use the candidate image, fresh empty data/cache and no mounted SDK/venv/HOME/host runtime libraries. Provision no AMD account/licence file. After supplying a local ONNX, run compilation with network disabled.

Capture the exact toolchain manifest, source hash, compile command, no-network evidence, subprocess/resource stats, artifact hash and deployment proof. Ensure no unexpected pip/apt/compiler-package fetch or JIT build failure. Inspect image layers for hidden SDK copies, stale developer paths, secrets and original 17 GB environment baggage.

Separately test the same compile with no accelerator node. Report **whether it works**; this determines whether background compilation can overlap a resident model. Do not relabel network isolation as device isolation.

A cache-hit test must prove zero compiler-child launches, not merely a fast run.

## 3. Native runtime gate

Only with explicit device ownership and operator approval:

* Reuse the banked v9s-320 artifact and verified standalone library hashes.
* Load once; verify expected metadata, finite raw outputs and profile tolerances on the fixed reference set.
* Capture the actual loaded-library census: no ORT/VOE/VAIML/xcompiler in the inference child.
* Collect hardware proof at controlled boundaries without periodic secondary NPU clients.
* Verify shutdown/mmap/object lifetimes and clean SIGTERM.
* Benchmark service saturation and paced 40 requests/s separately. Initial staging target: no unexplained material regression relative to the imported baseline; define a 15% investigation threshold, not a licence to hide regressions in a different load shape.

Do not reuse quarantined v8l or interrupted suspect artifacts as test fixtures.

## 4. Full Frigate application gate

Run a separate, pinned Frigate 0.18.0 image with a deterministic local moving video and MQTT disabled or isolated. No production cameras/recording volumes/credentials are required for the first test.

Prove:

```
FFmpeg -> decoded frames -> motion regions -> detector scheduler
-> stock ZMQ -> native XDNA -> canonical detections
-> tracker -> observable tracked objects/events
```

Capture actual Frigate model config, tensor geometry, label mapping, request throughput, error logs and API/event evidence. A direct plugin call or translated-image loop is not this test.

Then test a legitimate newly selected private Frigate+ model with the user's explicit credentials. It must not already be in the image/cache. Fetch by ID, compile, preserve its actual label map/attributes, pass the stock Plus model handshake and track expected objects. No training request or model upload is authorised.

## 5. Update and failure gate

1. Serve A resident.
2. Queue B; show downloads/preparation do not select B.
3. Compile B under the verified CPU-only policy or explicit maintenance policy. Do not claim uninterrupted background compilation if NPU exclusivity requires a pause.
4. Reconfigure/reinitialize stock Frigate for B after preparation.
5. Drain/retire A; start B in a new child; verify B; prevent old A identities from receiving B results.
6. Restart sidecar and Frigate; confirm B cache hit, no compilation, offline inference works.
7. Exercise failed B and demonstrate that it is not activated or silently substituted. Preserve an explicit rollback path to A.

Inject daemon/child failures in software with fake native workers before hardware tests. Test disk full, compiler killed, partial download, corrupted RAI, unsupported target, native output NaNs, duplicate jobs, late replies, secret refresh failure and offline Plus outage.

A safety-inhibited daemon must remain inspectable and must not be turned into a reset loop by Docker's restart policy.

## 6. Service soak

After implementation and full-app smoke pass, repeat a 24-hour single-resident-model service soak. Hardware changes/switches are not part of that soak.

Required records: offered/sent/accepted/completed/client-delivered counts, active measurement duration, 1:1 hardware accounting at boundaries, errors, deadline misses, latency histograms by stage, RSS and operation generations. Nominal `40 req/s` is not a substitute for measured offered rate.

Gate: no host reset, device error, unbounded growth, wrong-model inference or unaccounted work. In a controlled quiet-host 30-minute segment, target zero timeouts and p99 end-to-end under 50 ms at 40 req/s. For 24 hours, every timeout window must be attributed and reported; do not rename runtime success as perfect service availability. The prior 67-timeout incident remains in baseline documentation.

Camera-count claims require measured region/request rates in staging. Do not advertise `73 / 5 = 14 cameras` as a capacity guarantee.

## 7. Release gate

* Source/public image rights separated; exact vendor notices and file-level licence mapping included.
* Full, pinned dependency hashes; SBOM; image provenance; no private models/keys or standalone vendor tar release.
* Non-root, unprivileged Docker operation with only NPU passthrough tested.
* Exact supported target/runtime matrix and unmodified Frigate tag/image tested.
* End-user steps require no SDK/AMD account/activation.
* CLI/docs/examples agree and execute against the built image.
* Raw compatibility assumptions and unresolved limitations are explicit.
* User approval before any production rollout or external publication.
