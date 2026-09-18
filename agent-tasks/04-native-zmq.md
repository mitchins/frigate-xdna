# Task 8.3 — Resident native worker and exact stock-ZMQ lifecycle

Depends on Tasks 8.0–8.2. Pure contract tests first; hardware tests explicit and serialized.

## Objective

Implement the sidecar detector service using the proven native FlexML code while preserving identity, model lifecycle, error and cache guarantees.

## Work

1. Wrap the successful C++ runner in the private IPC contract. Load one RAI/model/context per process. Preserve mmap lifetime/destruction order and precise `std::any` buffer types. Do not embed the compiler or vendor Python bindings.
2. Implement model tensor metadata checks and supported output-profile decoders. Preserve Plus class IDs/attributes. No second resize/normalization. Unknown layouts fail explicitly.
3. Implement Python ROUTER frontend compatible with the pinned stock REQ plugin. Bound queues, buffer sizes and timeouts; tag internal work by request ID/worker generation.
4. For each new identity, force a source-byte transfer, calculate SHA-256 and resolve the correct prepared artifact/serving contract. Cache reuse must not trust basenames.
5. Bind each identity to the selected contract/generation. Test two models with identical shapes but different labels/weights: no response may cross the binding.
6. Implement controlled automatic activation of a prepared requested model only when old-model work is quiescent; conflicting live clients receive MODEL_IN_USE. Invalidate old bindings and suppress automatic reactivation of the superseded model. Explicit maintenance activation is the escape hatch, not per-frame model switching.
7. A cold miss queues compilation and replies not-loaded promptly. Test that stock rc2 fails initialization until reinitialized after preparation; do not fake READY or rely on extra response fields for polling.
8. Implement persistent device-operation journal, lease, quarantine/circuit breaker and bounded recovery. Individual client timeouts do not recreate the native context. Boot-token scope is recorded; no assumption that container boot ID always equals physical host boot ID.
9. Keep health/status passive. No xrt-smi polling, automatic device reset or kernel changes.
10. With exclusive access, run one short v9s-320 correctness and paced performance test in one worker generation, followed by clean SIGTERM. Compare with the banked native boundary; report overhead instead of hiding it.

## Acceptance

* Unmodified stock rc2 plugin handshakes and receives correct 480-byte results.
* Cache-hit path performs no compile; cold path tells the truth and remains bounded.
* Native loaded-library census excludes compiler/ORT/VOE/VAIML.
* Old/expired/unbound requests never use the new model or a previous frame's result.
* No fake-success accounting for zero error frames; metrics distinguish no objects and detector failure.
* Worker timeout/crash/quarantine tests pass with fakes; real hardware run is short and one-context.

## Stop

Commit the detector service and report. No production CT110 change or public release. Full Frigate application testing is the next task, not something to infer from a passing detector plugin.
