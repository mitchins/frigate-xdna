# Agent instructions — Frigate-XDNA

## Mission

Implement `SPEC.md` in the bounded tasks under `agent-tasks/`. This is product engineering on top of banked experiments, not a new SDK bring-up or model sweep.

## Non-negotiable product requirements

1. Stock Frigate `v0.18.0-rc2` works without source patches or a new detector type.
2. Users can prepare a newly issued compatible Frigate+ model/fine-tune themselves; no maintainer hash catalogue or new service image for new weights.
3. The default appliance includes the audited compile and inference payloads. No end-user AMD account, separate SDK, activation or manual conversion.
4. Multiple cached/prepared models; one resident native model/context per NPU initially.
5. Preserve exact source identity, input semantics, numeric class IDs and Plus metadata. Never normalize twice or assume COCO-80.
6. No fake readiness, fabricated detections or silent CPU fallback.
7. No autoloop after a suspected host reset/device fault. No experiments on production CT110, HA CT113, or the Proxmox host.

## Evidence discipline

Start by reading the actual evidence files named in SPEC §2.1, not by reconstructing them from chat summaries. Copy full hashes and recipe flags, not truncated prefixes. Preserve the original proof scripts and compiled artifacts.

The user reports a 24-hour resident soak and a clean offline compiler-appliance proof with an exact licence audit. Treat those as banked engineering inputs; carry the audited component licences/notices into packaging. Do not reopen general AMD licensing or substitute unrelated Vitis licences. Changes to binary versions or distribution scope require renewed component review.

The old reset cause is unproven. A process boundary is a lifecycle policy, not proof that firmware/kernel crashes are fixed. The 24-hour soak had 67 client timeouts during one host stall; retain that qualification.

Previous "CPU" sweep numbers used ORT CPU fallback, not production OpenVINO. Do not use those ratios as OpenVINO comparisons. The exact private Plus model still needs a production-contract test.

## Work style and permissions

* Implement one assigned task. Commit and report before continuing to the next task.
* Do not auto-start additional phases, model sweeps, benchmarks, NPU probes or dependency upgrades.
* No `apt upgrade`, kernel/DKMS/firmware changes, `/dev/kfd`, GPU use or NPU power/reset changes.
* Before any hardware test, verify that no soak/compiler/worker owns the device. Do not stop an unrelated process to make the test convenient.
* Keep large build/scratch files on the designated host-backed workspace, not the nearly-full historical root filesystem. Record and enforce free-space preflights.
* Allowed host-independent work includes tests with fake workers/Plus servers and standard compilation of the application's own code. Hardware tests are explicit and serialized.
* Do not accept agreements or authenticate as the user without authorisation. Runtime tests receive only the legitimate test credentials explicitly supplied; never print them.
* Do not publish proprietary payload tarballs, private model files, licence PDFs, or secrets as repository/CI artifacts. The release image's incorporated binary distribution follows the audited route.
* No claims of tested features when only stubs, docs or a plugin-level test exist.

## Required report format

```
Task / commit:
Implemented:
Tests passed:
Tests not run and why:
Hardware/device operations performed:
Changed model/toolchain hashes:
Security/licence packaging changes:
Known gaps:
Next bounded task:
```

## Immediate stop conditions

Stop hardware work on host reboot, device fault, nonfinite native results, evidence of wrong-model responses, interrupted unsafe operation, or filesystem safety failure. Preserve evidence and set persistent inhibition/quarantine. Do not reset hardware, delete locks blindly, retry a suspect artifact or "continue the mission" after a reboot.

## Technical facts that must survive refactors

* Frigate's model waits are 30 seconds; initial not-ready is not polled indefinitely.
* Inference wire messages do not carry a model ID/hash; bind identities to source/serving contract/generation.
* A new socket should transfer source bytes even if a same-name cache entry exists.
* `input_dtype: float` is already normalized by Frigate before the plugin receives it.
* The `.rai` mmap must outlive all native references; `fbs_buffer` uses the precise type required by the public FlexML C++ API.
* The compiler and native inference environments must remain separate.
