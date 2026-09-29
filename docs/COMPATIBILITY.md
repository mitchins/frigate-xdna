# Compatibility observations (not a support matrix)

`frigate-xdna` runs on a fast-moving XDNA2 platform. This file
records **what has actually been observed**, working or failing, on
specific host/platform stacks. It is an observation matrix, not a
support or certification matrix.

A PASS here means only: "the explicitly stated workload completed on
the explicitly stated stack." It never means certified, supported,
universally stable or safe.

Result classes:

```text
PASS       completed the explicitly stated test
LIMITED    useful successful observation, insufficient duration/scope
FAIL       recoverable test/runtime failure
RESET      host reset / hard loss
UNKNOWN    incomplete evidence
```

Design rules:

- Each observation is self-contained. No compatibility row or
  detailed record is expressed relative to another observation.
- Every technical value is written literally; no value is expressed
  by reference to another row or record.
- Appliance runtimes are named by immutable R-IDs defined in the
  next section. An R-ID is defined once; a changed runtime gets a
  new ID, never a redefinition.

## Runtime IDs

Immutable appliance-runtime tuples (the payload inside the
frigate-xdna image, not anything installed on the host).

### R-001

```text
XRT:        2.25.37
XDNA shim:  2.25.260102.56
FlexMLRT:   1.8.0
recipe:     bf16-vaiml-v1 (VAIML BF16 partition via flexml 1.8.0)
```

## Summary table

| ID | Hardware / NPU | BIOS / AMD PI | Kernel | amdxdna | NPU FW | Appliance runtime | Workload | Result |
|---|---|---|---|---|---|---|---|---|
| [O-001](#o-001) | Framework Desktop, Ryzen AI Max+ 395 / 1022:17f0 rev 0x11, subsystem f111:000a | 03.02 / PI 1.0.0.1b | 7.0.14-14-pve | srcversion 4612EC55… | 1.1.2.65 | R-001 | YOLOv9s-320 resident soak, 24 h, 3,364,689 completions | **PASS — 24 h soak** |
| [O-002](#o-002) | Framework Desktop, Ryzen AI Max+ 395 / 1022:17f0 rev 0x11, subsystem f111:000a | 03.05 (INSYDE, 2026-01-21) / PI 1.0.0.2 | 7.0.14-14-pve | srcversion 4612EC55… | 1.1.2.65 | R-001 (image 0.1.2) | Frigate+ YOLOv9-320: fresh compile + live Frigate inference, real detections | **LIMITED — compile + live inference observed** |
| [O-003](#o-003) | Framework Desktop, Ryzen AI Max+ 395 / 1022:17f0 rev 0x11, subsystem f111:000a | 03.05 / StrixHaloPI-FP11 1.0.0.2 | Ubuntu 7.0.0-34-generic (Ubuntu 26.04.1) | srcversion 4612EC55… | 1.1.2.65 | R-001 (host XRT install 2.21.75 — diagnostic metadata only) | YOLOv9 C320 + M320: host reset immediately when a heavy Radeon/LLM workload started ~5 min into the first run; later runs reset after a few minutes with no intentional GPU workload | **RESET — 0x08000800 data fabric sync flood** |
| [O-004](#o-004) | Framework Desktop, Ryzen AI Max+ 395 / 1022:17f0 rev 0x11, subsystem f111:000a | 03.05 (INSYDE, 2026-01-21) / PI 1.0.0.2 | 7.0.14-14-pve | srcversion 4612EC55… | 1.1.2.65 | R-001 (development build of PR #39; released as 0.1.3-rc.1) | YOLOv9-C-320 stability diagnostics: smoke (p50 13.7 ms) + gentle (910/910 over 120 s, p95 14.3 ms) + intentional SIGKILL breadcrumb-durability proof | **PASS — smoke + gentle** |

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
  BIOS:             03.02
  AMD PI / AGESA:   1.0.0.1b

uname -a:           Linux 7.0.14-14-pve #1 SMP PREEMPT_DYNAMIC PMX
                    7.0.14-14 (2026-08-22T15:01Z) x86_64
amdxdna srcversion: 4612EC552523E4C8FB4B5E5
amdxdna vermagic:   not recorded
NPU firmware:       1.1.2.65

power/control
  control:           auto
  autosuspend_delay_ms: 5000

Appliance runtime:  R-001
frigate-xdna
  version:           pre-project (banked deployment runtime)
  image digest:      not recorded

Model
  family/variant:   YOLOv9s
  resolution:       320

Test/workload
  duration:          24 h
  inference count:   3,364,689 matched submissions/completions
  concurrent GPU:    none recorded

Result
  outcome:           PASS — 24 h soak
  reset reason:      none; no inference errors, XDNA Err=0
  notes:             67 client timeouts occurred in one transient
                     host-stall window with immediate recovery;
                     retained as an availability qualification, not
                     relabelled as zero timeouts. A separate project
                     soak (docs/RELEASE-8.4.md: 1,234,208 completions,
                     4,069 person tracks, no reset) later reproduced
                     24 h residence on a 0.1.x appliance; the BIOS
                     revision during that soak was not separately
                     recorded.
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
  BIOS:             03.05 (INSYDE, 2026-01-21)
  AMD PI / AGESA:   1.0.0.2 (PI build string not captured on this
                    host)

uname -a:           Linux 7.0.14-14-pve #1 SMP PREEMPT_DYNAMIC PMX
                    7.0.14-14 (2026-08-22T15:01Z) x86_64
amdxdna srcversion: 4612EC552523E4C8FB4B5E5
amdxdna vermagic:   not exposed by this kernel
NPU firmware:       1.1.2.65 (sysfs fw_version)

power/control
  control:           auto
  runtime_status:    suspended when idle, active under load
  autosuspend_delay_ms: 5000

Appliance runtime:  R-001
frigate-xdna
  version:           0.1.2 (stable)
  image digest:      sha256:f3ed72a929d97f2ccd4354ff53b03343485e48a5
                     52c5e82ab9cb85cc36aac9e5

Model
  family/variant:   Frigate+ YOLOv9 (Plus); variant not recorded
  resolution:       320

Test/workload
  duration:          single-session live use
  inference count:   not recorded (live traffic)
  concurrent GPU:    none

Result
  outcome:           LIMITED — compile + live inference observed
  reset reason:      none observed
  notes:             fresh compile, live Frigate serving real
                     detections. No 24 h-class soak has been observed
                     on this stack.
```

### O-003

```text
Observation ID: O-003
Evidence class: user-reported

Hardware
  CPU/APU:          Ryzen AI Max+ 395 (Strix Halo)
  NPU PCI ID:       1022:17f0
  NPU revision:     0x11
  Subsystem ID:     f111:000a

BIOS
  BIOS:             03.05
  AMD PI / AGESA:   StrixHaloPI-FP11 1.0.0.2

Host OS
  distribution:     Ubuntu 26.04.1
  uname:            Linux 7.0.0-34-generic
                    #34-Ubuntu SMP PREEMPT_DYNAMIC
amdxdna srcversion: 4612EC552523E4C8FB4B5E5
amdxdna vermagic:   not recorded
NPU firmware:       1.1.2.65

power/control
  control:           auto
  autosuspend_delay_ms: not recorded

Host XRT diagnostic installation: 2.21.75
  (host-side xrt-smi/XRT only — diagnostic metadata, never the
   runtime frigate-xdna executes)

Appliance runtime:  R-001
frigate-xdna
  version:           0.1.x (as reported)
  image digest:      not recorded

Model
  family/variant:   YOLOv9-C and YOLOv9-M (public exports)
  resolution:       320

Test/workload
  observation 1:     first run operated for about five minutes, then
                     the host reset immediately when a heavy
                     Radeon/LLM GPU workload was started
  observation 2:     subsequent runs also reset after a few minutes
                     with no intentional concurrent GPU workload
  models tried:      C320 and M320
  inference counts:  not recorded

Result
  outcome:           RESET — 0x08000800
  reset reason:      host hard reset; after reboot the reported
                     system reset reason was 0x08000800,
                     "uncorrected error caused a data fabric sync
                     flood event"
  notes:             not reproduced by the project; cause unproven.
                     This is NOT merely a GPU-coexistence failure:
                     resets recurred without any intentional GPU
                     load. See "Related platform observations" before
                     drawing conclusions.
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
  BIOS:             03.05 (INSYDE, 2026-01-21)
  AMD PI / AGESA:   1.0.0.2 (PI build string not captured on this
                    host)

uname -a:           Linux 7.0.14-14-pve #1 SMP PREEMPT_DYNAMIC PMX
                    7.0.14-14 (2026-08-22T15:01Z) x86_64
amdxdna srcversion: 4612EC552523E4C8FB4B5E5
amdxdna vermagic:   not exposed by this kernel
NPU firmware:       1.1.2.65 (sysfs fw_version)

power/control
  control:           auto
  runtime_status:    suspended when idle, active during the run
  autosuspend_delay_ms: 5000

Appliance runtime:  R-001
frigate-xdna
  version:           development build of the stability-runner branch
                     (PR #39); the runner is released in 0.1.3-rc.1
  image digest:      development image, not a release digest

Model
  family/variant:   YOLOv9-C (public export), source sha256
                    c7009a1c…, compile key ebb8220f…
  resolution:       320

Test/workload
  profile smoke:     E0/H1/H2 PASS; p50 13.7 ms, measured capacity
                     72.8 req/s
  profile gentle:    E0–H3 PASS; 910/910 inferences in 120 s at the
                     10% paced target, p95 14.3 ms, 0 errors,
                     0 timeouts
  breadcrumb proof:  the operator deliberately SIGKILLed the
                     container mid-H3 as an intentional
                     durability test of the journal (not a host
                     fault): the journal retained the last durable
                     position (H3 PHASE STARTED, request 72, worker
                     pid/RSS in the ≤1 s checkpoint); the next run
                     refused to start (exit 8) until acknowledged;
                     a post-acknowledge smoke re-run PASSed

Result
  outcome:           PASS — smoke + gentle
  reset reason:      none; the SIGKILL was operator-intentional and
                     is not a host-reset observation
  notes:             journals under /mnt/downloads/fxdna-stability-
                     proof on the build host. Latency agrees with the
                     C320 scaling sweep (14.09 ms p50 there, 13.7 ms
                     here). This is a diagnostic-tool observation,
                     not a soak.
```

## Appliance runtime vs host tooling

Two different stacks are involved on any host:

```text
Host xrt-smi/XRT                diagnostic metadata only
frigate-xdna embedded runtime   the code that actually runs
```

The appliance image carries its own pinned XRT, XDNA shim and
FlexMLRT (the R-ID tuple) and does not use arbitrary host XRT
libraries. A host `xrt-smi` version therefore describes the host,
never the appliance. `fxdna host-info` reports both sides explicitly
and never mixes them.

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
