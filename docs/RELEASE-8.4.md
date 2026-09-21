# Task 8.4 release-boundary record (staged candidate, NOT published)

Intended image name (defined now, never pushed during Task 8.4):

```text
ghcr.io/mitchins/frigate-xdna
```

No GHCR release workflow runs, no image is pushed, and no `:latest` or
version tag is assigned until Task 8.5. A subsequent automation needs
only: this file + `reports/sbom-8.4.json` + the digests below.

## 1. Candidate under test

| Field | Value |
|---|---|
| Local reference | `localhost/frigate-xdna:0.1-dev` |
| Image ID | `a465117727c5655e3c988ecc22934b54fe33018d9f30dc6e7946f4d28cb2e9f9` |
| Content digest (local) | `sha256:b4017d0434a6976442e1d5c034b567b29b6c19803d6cfb226a7ff3ee2cf6a223` |
| Uncompressed size | 4109585180 B (3.83 GiB) |
| Registry digest | none — never pushed; the digest above is local content only |
| Built | 2026-09-20 (Task 8.4, branch `feature/task-05-acceptance`) |
| NPU status | fix baked, NOT yet hardware-validated (no device sessions since build) |

This image is the first that actually contains the inference path:
`/opt/fxdna/native/fxdna-worker` (previously missing — the Task-04
native stage never built successfully), the supervision + activation
code, `pyzmq` in the manager venv, and the repaired recipe. ldd census:
only libflexmlrt/XRT/system libs (no ORT/VOE/VAIML/xcompiler/
pyflexmlrt in NEEDED).

## 2. Reproducing the build (private inputs stay private)

```sh
tools/prepare_vendor_input.py --payload-src <audited site-packages> \
    --xrt-src <audited xrt> --flexmlrt-lib <standalone lib> \
    --flexmlrt-include <audited wheel flexmlrt/include> \
    --legal-src <licence texts> --calib-src <calib dir>
podman build --security-opt apparmor=unconfined \
    -f packaging/Dockerfile -t frigate-xdna:0.1-dev .
```

* Required private path: `packaging/vendor-input/` (gitignored, COPY'd at
  build; verify against `packaging/vendor-files.manifest.json`).
* Vendor manifest digest:
  `5d5e7ac38b3861fac9e8a6f3588afa50c670de0a56338dc39712d93246894275`
  (`sha256sum packaging/vendor-files.manifest.json`; +1 file vs Task 03:
  `flexmlrt/include/FlexMLClient.h` from the audited 1.8.0 wheel, whose
  `libflexmlrt.so` is byte-identical to the staged lib — same version,
  same amd-eula licence, native-worker build requirement; flagged for
  maintainer component review).
* Base: `ubuntu:24.04`. The `apparmor=unconfined` flag is a build-host
  (LXC) workaround only — runtime uses `examples/compose.yaml` without it.

## 3. SBOM / licences / provenance

* `reports/sbom-8.4.json` (CycloneDX 1.5): 17 audited vendor components
  (hashes + licence IDs from `packaging/vendor.lock.json`) + 11 runtime
  + 11 dev/test PyPI pins (from `requirements.lock`; dev/test flagged,
  not shipped; pyzmq correctly scoped runtime). Generated offline by
  `tools/gen_sbom.py`; no syft/trivy on the build host.
  `sha256=b0713841223c79943922d9f1719a967aa4131b1af484b1ccd160ed58f30b699c`
* Licence inventory: `THIRD_PARTY_NOTICES.md` + file-level mapping
  `packaging/legal/component-map.json` (EULA flow-down notices under
  `/opt/fxdna/legal` in-image).
* No standalone vendor tar, private model, or credential is published or
  committed (`packaging/vendor-input/`, `*.sqlite3`, secrets gitignored).

## 4. Runtime contract (verified in 8.3.5, re-asserted here)

* UID:GID `10001:10001`, `NPU_GID` via `group_add`, single device
  `/dev/accel/accel0`; no `/dev/kfd`, no privileged, no Docker socket,
  no GPU compute, no published ZMQ port.
* `read_only: true` + tmpfs, `cap_drop: ALL`, `no-new-privileges:true`,
  `init: true`, `cpus 4.0 / mem 8g / pids 512`, `stop_grace_period 60s`.
* Liveness `fxdna health`; model readiness is separate (never inferred
  from `service_healthy`). Secrets via `PLUS_API_KEY_FILE` mount.
* Known-good Compose: `examples/compose.yaml` (+ `compose.plus.yaml`
  overlay for Frigate+ models). Named volume owned by image-init.
* LXC deviations recorded (test-host only, never for end users):
  `--security-opt apparmor=unconfined` (no profile in LXC),
  `--group-add 0` (host accel node is root-owned; production hosts use
  a dedicated NPU group), podman tmpfs accepts no uid/gid options
  (`/run` mode 0755 instead of 0700+uid).
* Observed defect: serve ignores SIGTERM past the 60 s grace twice
  (SIGKILL fallback, exit 137). Shutdown path needs a fix before release.

## 5. Acceptance evidence (Task 8.4)

Real Plus model: private yolov9s-320 (source `db3cfb8c…`, 320×320,
nchw/rgb/float, yolo-generic, zmq-capable, 46 labels, trained
2026-09-17). Model ID used exactly internally, withheld everywhere
else per the redaction policy.

| Gate (`docs/ACCEPTANCE.md`) | Status + evidence |
|---|---|
| §1 non-hardware CI | PASS — 225 hardware-free tests + ruff + Sonar gate OK (PR #5 CI) |
| §1 Plus stub layer | PASS — `FakePlus` + 25 client tests incl. republish (same-ID new bytes) and A+B coexistence; bearer never reaches download host |
| §1 external-observability privacy | PASS — `fxdna diagnose` bundle + 5 sentinel unit tests AND a real-data proof (bundle over the live sidecar registry: raw ID absent, HMAC alias present, no key/DB/bytes) |
| §2 compiler appliance | PARTIAL — fetch→quant→VAIML verified end to end on the real model (deterministic bf16 digest `9290cf47…`, 17 MB `.rai` produced); job reports COMPILE_FAILED only at the trailing HW-runner mapping (see §6 blocker 1) |
| §3 native runtime | PARTIAL — LOAD_OK + RUN_OK proven on the produced `.rai` (gen 1, cols 2100, 46-class geometry accepted, zeros → finite 480 B) via bounded out-of-band driver; in-product activation blocked by §6.1 |
| §4 full Frigate replay | PLUMBING PASS — stock rc2 (`80825459`) + deterministic COCO clip (`777e3f9b…`, manifest `runs[0]`, evidence `fa1c83cc…`): 9 detector stat samples, config + events API reachable, 5 fps camera, 10 ms not-ready round-trip; 0 tracked objects (sidecar not-ready), zero device FDs held |
| §4 real Plus model | FETCH+CACHE PASS, ACTIVATE BLOCKED — metadata/labels preserved through stock Frigate (46 labels + logo attributes visible in `/api/config`); activation awaits §6.1 |
| §5 A→B update / offline / failure | STUB GREEN (client layers), hardware runs blocked by §6.1 |
| §6 24 h soak | NOT STARTED — needs quiet host after §6 |
| §7 release gate | THIS FILE — inputs complete, publication explicitly withheld |

Post-acceptance correction (2026-09-20): the "CMA exhaustion" verdict
below was built on container-view counters (`CmaTotal: 0 kB` proves
filtering) and is WITHDRAWN as a device diagnosis. The exact failure —
a 512 MiB anonymous VA reservation returning ENOMEM after a valid
`.rai` was generated — matches process address-space pressure from
the compiler's `RLIMIT_AS=6GiB` cap (ORT reserves multi-GB virtual
arenas; RSS was only ~1.8 GB). Fixed on the branch: AS limiting
removed (container/cgroup `mem_limit 8g` is the bound), VmSize/VmPeak
instrumented per phase into `compile_stats`, in-child probe already
moved out. No NPU runs until this fix is validated; CMA must be
re-recorded from the raw Proxmox host (see `docs/OPERATIONS.md`).

Fixes landed during Task 8.4 (all committed on the branch):

* Real-Plus contract: `inputShape: "nchw"` layout tag accepted (was
  int-list-only); width/height remain the numeric claim.
* Recipe: `EnableVaimlBF16` + `restore_float_inputs` (BF16-fed Cast
  graphs load; numerically exact, checker-verified, no-op otherwise).
* Worker: `class_count` in LOAD (required, OOB-guarded, generation
  never advances on refusal) + channel-agnostic geometry (was COCO-84).
* Supervision: spawn/LOAD/INFER/retire, activate/activate_worker,
  inhibition on death/fault (never respawn), `XILINX_XRT` child env.
* Image: native stage actually builds (linker symlinks, SYSTEM
  third-party includes, `-Werror` first-party fix), flexmlrt headers
  staged, pyzmq shipped, staged-verify SKIP semantics restored.
* Tooling: `diagnose` bundle, HMAC pseudonyms, replay harness + clip,
  offline SBOM generator, Sonar triage to gate OK.

## 5b. Acceptance run 2026-09-20/21 (branch, post-rlimit-fix)

- Fresh empty-volume compile (vol `frigate-xdna-087`): fetch →
  source validation → BF16 prep (deterministic `9290cf47…`,
  `collapsed_inputs=1`) → VAIML 525.8 s → 17 MB `.rai` →
  out-of-child probe → validate → PUBLISHED `PREPARED`.
  `compile_stats`: wall 544 s, peak_rss 1.87 GB, **vm_peak 6.39 GB**
  (print: AS cap would have killed this; RSS never near any limit).
- Real-Plus contract fix verified live (`inputShape: nchw` accepted).
- Activation through product supervisor: ACTIVE, worker gen 1→2→3→2
  across restart/switch cycles, finite outputs, no inhibition.
- Full replay: stock rc2 + walk2 clip → **person tracks @0.78**
  (evidence `/mnt/downloads/fxdna-087-replay-evidence.json`: 22 events;
  B-replay 10 events; A-replay 8 events). Steady 10–18 ms inference.
- A→B with one artifact (local alias A shares Plus B's bytes — no
  second model exists): gens advanced, old worker reaped (PID gone),
  sessions rebound, replay continuity on both sides. Documented limit:
  same-bytes switch proves lifecycle mechanics, not cross-model
  isolation (covered structurally by generation guards + unit tests).
- Restart: both containers exit clean (sidecar **0** in ~1 s —
  SIGTERM fix proven twice); artifact reused, zero new compiles,
  fresh worker, replay passes.
- Offline (no key, `FXDNA_OFFLINE`): startup honestly refuses
  acquisition; activate/replay from cache (8 person events);
  **zero Plus packets** (empty pcap); no jobs beyond handshake
  provenance rows.
- Harness lessons (fixture/test-side, not product): file inputs need
  `input_args` override (rc2 `user_agent` default breaks them);
  DB event inserts require snapshots or clips enabled;
  `detect.enabled` defaults false in rc2 config surface used here.
- Product bugs found+fixed in-run: real-backend publish identity
  (`BACKEND_ID` class attr — real publish NEVER worked before),
  status dict-active crash, bf16 digest parse, cache-hit provenance
  rows for activate-by-ref. All with regression tests (250 green).

## 6. Verdict

**BLOCKERS before Task 8.5** (i.e. NOT `RELEASE_AUTOMATION_READY`):

1. **Rlimit attribution: PROVEN.** VmPeak 6.39 GB vs RSS 1.87 GB
   on the clean post-fix compile — the old 6 GiB AS cap would have
   killed exactly this mapping. No mmap failure, no segfaults in the
   full product run since.
2. **24 h soak outstanding (running).** Started 2026-09-21T06:13:47Z
   on the exact state above (offline B active gen 2, replay
   looping, 200 ms detector timeout, production thresholds);
   baseline in `/mnt/downloads/fxdna-087-soak-baseline.json`.
   Verdict flips to `RELEASE_AUTOMATION_READY` only when the soak
   closes with full accounting and no stop events.
3. **30-min confidence: PASSED 06:09Z.** +26k requests, zero errors
   of any kind, worker RSS byte-flat (208860 kB ×4 samples),
   same worker/gen throughout, tracks flowing.
4. Component review renewal: +1 header file (same amd-eula 1.8.0;
   digests above) + recipe behavior change (documented in
   `recipes/bf16-vaiml-v1/recipe.json`) + cache-hit provenance rows
   + BACKEND_ID class attr (all committed, all tested).
5. Pre-reboot session segfaults remain unexplained — any recurrence
   during soak is a stop event, not background noise.

When 1–5 close, the release workflow (Task 8.5) may: rebuild with the
pinned inputs above, `podman push ghcr.io/mitchins/frigate-xdna:<version>`,
and attach `reports/sbom-8.4.json` + this file to the release.
