# Task 8.0 — Establish the product repository and exact contracts

Read `AGENTS.md`, `SPEC.md`, `docs/INTERFACES.md` and `docs/SOURCES.md` first.

## Objective

Create a maintainable repository around the proven implementation. Import evidence, pin dependencies/contracts and establish non-hardware tests. Do not execute new accelerator work.

## Inputs

Read the real Phase-5 runner, Phase-7 worker, Phase-7.6 soak and Phase-7.7 compiler audit from the paths in SPEC §2.1. Confirm whether any process/run is active before even considering later hardware tasks. Do not stop it.

## Work

1. Create the code/build/test structure in `docs/REPOSITORY.md` with a minimal runnable CLI whose `--help`, configuration validation and `status` schema can be tested without hardware.
2. Import the proven native runner into a clearly identified reference location first, preserving its original source and hash. Document which code will be reused versus replaced. Do not rewrite the FlexML lifetime/API semantics from memory.
3. Create a complete private/local proof-input manifest: source/RAI/library/worker hashes; native ABI requirements; exact compiler payload manifest and recipe; licence/notice mapping and origins. No truncated hashes. Keep proprietary blobs and private model files outside Git.
4. Pin the exact Frigate tag and source blobs listed in SOURCES. Add MIT notices for reused source. Set up contract-test imports of the **unmodified** `ZmqIpcDetector` and relevant input-transform code; do not vendor a forked plugin and call it stock.
5. Implement/define JSON schemas for model descriptor, compiled artifact and service status. Capture ordinary YOLO raw and private Plus label-map fixtures. No implicit class count 80.
6. Add regression tests documenting the 30-second model timeout, enum-name `yologeneric`, absent per-frame model ID/hash, float normalization before the plugin, and initial-not-ready behaviour.
7. Introduce fake compiler and fake native worker interfaces, so later control-plane tests require no AMD libraries or NPU.
8. Record implementation gaps, not research hypotheses, in a short work queue.

## Acceptance

* Unit/contract tests run on a machine with no `/dev/accel`, SDK or API key.
* `fxdna --help` and config validation have no vendor import/hardware side effects.
* Full provenance values are imported, or precise missing files are reported rather than invented.
* No gated binaries/private models/secrets committed.
* No host, production Frigate or HA modifications.

## Stop

Commit the foundation and report tests/remaining inputs. Do not begin live Plus acquisition, compiler execution, model activation or Docker publication.
