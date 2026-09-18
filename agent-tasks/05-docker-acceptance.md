# Task 8.4 — Docker-first product acceptance and staged release candidate

Depends on Tasks 8.0–8.3. Do not modify production without explicit user approval.

## Objective

Demonstrate the actual product experience, including a new compatible private Frigate+ model, stock full Frigate and cached restart. Close the gap between a lab plugin and a user-installable appliance.

## Work

1. Build a pinned candidate image with non-root operation, narrow NPU access, no privileged mode/Docker socket/GPU compute, read-only image root, resource limits, secret mount and local `/data` volume. Make example Compose/CLI files execute exactly as documented.
2. Ensure image-init/volume ownership works for a new named volume and document existing bind-mount ownership without recursively chowning arbitrary host paths.
3. Test liveness and separate model readiness; `depends_on: service_healthy` alone must not be described as waiting for model compilation.
4. Run an isolated **full** Frigate 0.18.0-rc2 image against a deterministic local moving video. Capture motion regions, scheduler calls, XDNA accounting and tracked-object/event/API evidence. Do not replace this with direct plugin invocation.
5. With an explicitly supplied Plus key/ID, start from an empty cache and fetch a newly selected compatible private model. Compile it, preserve its metadata/labels, activate through stock Frigate and verify real detections. No private content enters public CI logs/artifacts. No training API calls.
6. Demonstrate the update flow A -> prepare B -> switch stock Frigate to B -> fresh native worker -> cached restart, respecting the proven device/concurrency policy. Failed B never silently becomes A under B's metadata.
7. Verify offline cached inference, source-change detection, cold-miss recovery instructions, pruning/pinning, interruption recovery and persisted safety inhibition under Docker restart policy.
8. Repeat a controlled service soak when authorised, with complete request/response/timeout accounting and no second-client NPU polling. Separate native latency, total round-trip, pacing and saturation capacity. Preserve the prior stall/timeout qualification.
9. Produce release SBOM, full component hash/licence mapping and provenance. Verify no standalone vendor tar, private model or credential is published. Do not publish the image or claim universal XDNA/model compatibility without review.
10. Write concise operations/development guides and a compatibility matrix based only on tests actually run.

## Acceptance

All required gates in `docs/ACCEPTANCE.md` pass or are explicitly marked blocking. The final demonstration is:

```
clean supported environment
+ candidate image + NPU + /data
+ Plus credentials + new compatible model ID
-> download -> compile -> stock Frigate -> correct tracked objects
-> restart/offline -> no recompilation -> continued detection
```

No Frigate patches, SDK setup, AMD login/activation, maintainer precompile or hardcoded model hash entry.

## Stop

Deliver a release-candidate report and exact commands for user-approved staging. Do not alter live Frigate/Home Assistant, upgrade the host, or publish artifacts automatically.
