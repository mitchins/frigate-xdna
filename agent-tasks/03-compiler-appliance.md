# Task 8.2 — Integrate the audited compiler into the product appliance

Depends on Tasks 8.0–8.1. Explicitly schedule hardware access; do not overlap a soak or another NPU owner.

## Objective

Turn the existing clean-room compiler proof into the product's deterministic short-lived compile job. This is not a renewed licensing/SDK-minimisation project.

## Work

1. Import the exact Phase-7.7 payload/recipe into immutable, separate compiler/runtime prefixes. Verify every binary hash and attach the audited governing notices/flow-down mapping. Do not reinterpret package MIT metadata or publish the standalone payload tarball.
2. Build a Docker candidate including all audited compile/run requirements. Keep base OS, manager dependencies, compiler and inference footprint separate. Eliminate accidental original-SDK copies and package caches from image layers.
3. Build required ordinary C++/Python helper prerequisites during image creation or ensure the recipe has a controlled writable cache for unavoidable JIT. No first-use apt/pip/AMD fetch.
4. Implement a launcher with explicit executable, environment, cwd, clean HOME/TMPDIR, closed inherited FDs, compiler deadline and resource limits. Never inherit Plus secrets/token state. Keep compiler LD_LIBRARY_PATH out of manager/native runtime.
5. Implement fresh-source validation, BF16 preparation, compile, manifest/hash checks, atomic artifact publication, logs/metrics and failed-job cleanup. One job only; repeated requests join/cache-hit.
6. Reproduce one known-good small model from an empty work/cache directory in the product image **with network disabled** and no SDK/venv/host runtime mounts. Do not rerun the wide sweep.
7. Separately test the same recipe without `/dev/accel/accel0`. This determines the product concurrency policy:
   * works: classify proven CPU-only stages and validate bounded background CPU/resource impact later;
   * requires device: classify them honestly; keep jobs deferred while a model is resident unless explicit maintenance was authorised.
8. After acquiring exclusive device access, validate one newly generated RAI using the proven standalone runner. Capture raw output checks and XDNA accounting. No other model churn or soak.
9. Verify another preparation of the same source/recipe is a cache hit with no compiler process. Change one compile-key input in a mock test and prove it creates a different job, not an overwrite.
10. Add a public-source/private-input build workflow. The public release artifact is the incorporated appliance, not loose vendor tools. Do not actually publish it in this task.

## Acceptance

* Candidate image + local ONNX + network disabled -> fresh RAI.
* Generated RAI passes known runtime checks on XDNA when hardware test authorised.
* No user AMD credentials, activation files, hidden SDK or runtime downloads.
* Exact compile-without-device result determines scheduler behaviour; no invented no-downtime claim.
* Image size, payload size, peak RSS and peak scratch measured separately.
* Failure/timeouts remain inspectable and never publish corrupt cache entries.

## Stop

Commit the compiler integration and image recipe. Do not hook production Frigate to it, start a family sweep, or change host software. Report any implementation-only gap against the banked proof.
