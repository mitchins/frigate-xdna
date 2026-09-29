# Compatibility observations (not a support matrix)

`frigate-xdna` runs on a fast-moving XDNA2 platform. This file
records **what has actually been observed**, working or failing, on
specific host/platform stacks. It is an observation matrix, not a
support or certification matrix.

A PASS here means "this explicitly stated workload completed on
this stack", not "this kernel/BIOS combination is certified
stable". No row may claim a stack is `stable`, `unstable`,
`supported` or `unsupported` — those words hide the actual
evidence class and duration.

Result classes:

```text
PASS       completed the explicitly stated test
LIMITED    useful successful observation, insufficient duration/scope
FAIL       recoverable test/runtime failure
RESET      host reset / hard loss
UNKNOWN    incomplete evidence
```

## Summary table

All rows are Framework Desktop / Ryzen AI Max+ 395 / XDNA2
`1022:17f0 rev 0x11` unless otherwise stated.

| ID | BIOS / AMD PI | Kernel | NPU stack | Workload | Result |
|---|---|---|---|---|---|
| [O-001](#o-001) | 03.02 / 1.0.0.1b | `7.0.14-14-pve` | FW 1.1.2.65 / amdxdna `4612EC55…` | v9s-320, 24h resident soak, 3,364,689 completions | **PASS — 24h** |
| [O-002](#o-002) | 03.05 / 1.0.0.2 | `7.0.14-14-pve` | FW 1.1.2.65 / same amdxdna srcversion | Frigate+ 320, fresh compile, live Frigate, real detection | **LIMITED — healthy, soak pending** |
| [O-003](#o-003) | 03.05 / 1.0.0.2 | Ubuntu `7.0.0-34-generic` | FW 1.1.2.65 / same amdxdna srcversion | C320/M320; Frigate; one run + GPU LLM load | **RESET — 0x08000800** |
| [O-004](#o-004) | 03.05 / 1.0.0.2 | `7.0.14-14-pve` | FW 1.1.2.65 / same amdxdna srcversion | C320 stability diagnostics: smoke, gentle (910 inferences/120 s), deliberate SIGKILL breadcrumb proof | **PASS — smoke+gentle** |

## Detailed observation records

### O-001

```text
Observation ID: O-001
Evidence class: user-reported (banked pre-project evidence, SPEC §2)

Hardware
  CPU/APU:          Ryzen AI Max+ 395 (Strix Halo)
  NPU PCI ID:       1022:17f0
  NPU revision:     0x11
  Subsystem ID:     f111:000a

BIOS
  AMD PI / AGESA:   1.0.0.1b (BIOS 03.02)

uname -a:           Linux <host> 7.0.14-14-pve #1 SMP PREEMPT_DYNAMIC PMX
                    7.0.14-14 (2026-08-22T15:01Z) x86_64
amdxdna srcversion: 4612EC552523E4C8FB4B5E5
amdxdna vermagic:   not recorded
NPU firmware:       1.1.2.65

power/control
  autosuspend_delay_ms: 5000 (auto)

frigate-xdna
  version:           pre-project (banked deployment runtime)
  image digest:      not recorded
  embedded XRT:      2.25.37
  embedded XDNA shim: 2.25.260102.56
  FlexMLRT:          1.8.0

Model
  family/variant:   YOLOv9s
  resolution:       320

Test/workload
  duration:          24 h
  inference count:   3,364,689 matched submissions/completions
  concurrent GPU:    none recorded

Result
  outcome:           PASS — 24h
  reset reason:      none; no inference errors, XDNA Err=0
  notes:             67 client timeouts occurred in one transient
                     host-stall window with immediate recovery;
                     retained as an availability qualification, not
                     relabelled as zero timeouts. A later project
                     soak on the 0.1.x appliance (docs/RELEASE-8.4.md:
                     1,234,208 completions, 4,069 person tracks, no
                     reset) reproduced 24 h stability on the same
                     host; the BIOS revision during that soak was not
                     separately recorded.
```

### O-002

```text
Observation ID: O-002
Evidence class: project-observed

Hardware
  CPU/APU:          Ryzen AI Max+ 395 (Strix Halo)
  NPU PCI ID:       1022:17f0
  NPU revision:     0x11
  Subsystem ID:     f111:000a

BIOS
  AMD PI / AGESA:   StrixHaloPI 1.0.0.2 (BIOS 03.05, 2026-01-21)

uname -a:           Linux <host> 7.0.14-14-pve #1 SMP PREEMPT_DYNAMIC PMX
                    7.0.14-14 (2026-08-22T15:01Z) x86_64
amdxdna srcversion: 4612EC552523E4C8FB4B5E5
amdxdna vermagic:   not exposed by this kernel
NPU firmware:       1.1.2.65 (sysfs fw_version)

power/control
  control:           auto
  runtime_status:    suspended (when idle)
  autosuspend_delay_ms: 5000

frigate-xdna
  version:           0.1.2 (stable)
  image digest:      sha256:f3ed72a929d97f2ccd4354ff53b03343485e48a5
                     52c5e82ab9cb85cc36aac9e5
  embedded XRT:      2.25.37
  embedded XDNA shim: 2.25.260102.56
  FlexMLRT:          1.8.0

Model
  family/variant:   Frigate+ YOLOv9 (Plus); variant not recorded
  resolution:       320

Test/workload
  duration:          single-session live use (not a soak)
  inference count:   not recorded (live traffic)
  concurrent GPU:    none

Result
  outcome:           LIMITED — healthy, soak pending
  reset reason:      none observed
  notes:             fresh compile, live Frigate serving real
                     detections. No 24 h-class soak has been run on
                     BIOS 03.05 yet; O-001's PASS is BIOS 03.02.
```

### O-003

```text
Observation ID: O-003
Evidence class: user-reported

Hardware
  CPU/APU:          Ryzen AI Max+ 395 (Strix Halo)
  NPU PCI ID:       1022:17f0
  NPU revision:     0x11 (as reported)
  Subsystem ID:     not recorded

BIOS
  AMD PI / AGESA:   1.0.0.2 (BIOS 03.05)

uname -a:           Ubuntu 7.0.0-34-generic (full string not recorded)
amdxdna srcversion: 4612EC552523E4C8FB4B5E5 (same in-tree driver)
amdxdna vermagic:   not recorded
NPU firmware:       1.1.2.65

power/control
  autosuspend_delay_ms: not recorded

frigate-xdna
  version:           0.1.x (as reported)
  image digest:      not recorded
  embedded XRT:      2.25.37
  embedded XDNA shim: 2.25.260102.56
  FlexMLRT:          1.8.0

Model
  family/variant:   YOLOv9 C320 and M320 (public exports)
  resolution:       320

Test/workload
  duration:          one run per model variant
  inference count:   not recorded
  concurrent GPU:    GPU LLM load running alongside

Result
  outcome:           RESET — 0x08000800
  reset reason:      host hard reset; 0x08000800 reported from host
                     evidence after reboot
  notes:             not reproduced by the project. Cause unproven;
                     see "Related platform observations" before
                     drawing conclusions. A concurrent GPU compute
                     load was active, so this is a coexistence
                     observation, not a pure NPU stability result.
```

### O-004

```text
Observation ID: O-004
Evidence class: project-observed

Hardware
  CPU/APU:          Ryzen AI Max+ 395 (Strix Halo)
  NPU PCI ID:       1022:17f0
  NPU revision:     0x11
  Subsystem ID:     f111:000a

BIOS
  AMD PI / AGESA:   StrixHaloPI 1.0.0.2 (BIOS 03.05)

uname -a:           Linux 7.0.14-14-pve #1 SMP PREEMPT_DYNAMIC PMX
                    7.0.14-14 (2026-08-22T15:01Z) x86_64
amdxdna srcversion: 4612EC552523E4C8FB4B5E5
NPU firmware:       1.1.2.65

frigate-xdna
  version:           development build of the stability-runner branch
                     (PR #39; not a release image)
  embedded XRT:      2.25.37
  embedded XDNA shim: 2.25.260102.56
  FlexMLRT:          1.8.0

Model
  family/variant:   YOLOv9-C (public export), source sha256
                    c7009a1c…, compile key ebb8220f…
  resolution:       320

Test/workload
  profile smoke:     E0/H1/H2 PASS; p50 13.7 ms, capacity 72.8 req/s
  profile gentle:    E0–H3 PASS; 910/910 inferences in 120 s at the
                     10% paced target, p95 14.3 ms, 0 errors, 0 timeouts
  breadcrumb proof:  container SIGKILLed mid-H3; journal left the last
                     durable position (H3 PHASE STARTED, request 72,
                     worker pid/RSS in the ≤1 s checkpoint); the next
                     run refused to start (exit 8) until acknowledged;
                     post-acknowledge smoke re-run PASS

Result
  outcome:           PASS — smoke+gentle
  reset reason:      none (the SIGKILL was the operator, not the host)
  notes:             journals under /mnt/downloads/fxdna-stability-proof
                     on the build host. Latency agrees with the C320
                     scaling sweep (14.09 ms p50 there, 13.7 ms here).
                     This is a diagnostic-tool observation, not a soak.
```

## Appliance runtime vs host tooling

Two different stacks are involved on any host:

```text
Host xrt-smi/XRT                diagnostic metadata only
frigate-xdna embedded runtime   the code that actually runs
```

The appliance image carries its own pinned XRT, XDNA shim and
FlexMLRT and does not use arbitrary host XRT libraries. A host
`xrt-smi` version therefore describes the host, never the
appliance. `fxdna host-info` reports both sides explicitly and
never mixes them.

## Related platform observations — not frigate-xdna compatibility results

External reports and vendor documentation that shape investigation
directions. They are **context, not evidence** that any row above
shares a cause.

- A Framework Desktop community report reproduces `0x08000800`
  (Data Fabric sync flood) through ACPI S5 power cycling with no
  XDNA workload at all — that reset code is not unique to XDNA.
- A separate Ryzen AI Max+ 395 / Ubuntu 26.04 report describes
  hard hangs under sustained Radeon (GPU) compute with no NPU
  workload.
- AMD has documented XDNA firmware compatibility boundaries around
  protocol-7 firmware, and an independent runtime-PM bug where
  idle/context-resume paths failed while continuously busy runs
  avoided the problematic window.

## Submitting an observation

```bash
fxdna host-info --json > host-info.json
fxdna stability report --last --json > stability.json
fxdna diagnose --out diagnose
# after a host reset, if possible:
sudo tools/collect-host-evidence.sh > host-evidence.txt
```

Use the compatibility-report issue template and attach those
artifacts. Maintainers promote sufficiently complete reports into
this file; there is no automatic crowdsourced table.
