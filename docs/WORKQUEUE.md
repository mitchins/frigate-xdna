# Implementation work queue

Gaps recorded as implementation work, not research hypotheses. Owners: later
agent tasks per SPEC §15.

Done:
- **Task 01** (commit `5cd26ba`): foundation, contracts, fixtures, fakes.
- **Task 02** (commit `af729ea` + follow-ups): Plus client, cache, daemon
  CLI, etc. Live token exchange verified (no model fetch).
- **Task 03** (this branch): compiler appliance — vendor payload pinned
  (`vendor.lock.json` + `vendor-files.manifest.json` + `component-map.json`
  + `verify_payload.py`), Docker candidate (4.12 GB image, 1.30 GB
  compressed, podman --security-opt apparmor=unconfined due to LXC),
  isolated launcher (env allowlist, secret stripping, 4 threads / 6 GiB /
  45 min, one-at-a-time), real backend `bf16-vaiml-v1` behind the manager
  (fake retained for tests), offline fresh compile (yolov8n 640 → 7.96 MB
  .rai, 1.4s BF16 + 610s VAIML, peak 1.70 GB, scratch 6.7 MB, wall 619s
  with device), device-required classification (without /dev/accel →
  HW context fail; with device → PREPARED), bounded HW validation
  (Active, 396 submissions, Err 0, 101 FPS), cache-hit (same key →
  PREPARED, zero compiler) and key-versioning (different source →
  distinct key c5a40..., no overwrite) proofs. Image prefixes verified
  byte-for-byte at build (vendor manifest) and at runtime (launcher env).

- **Task 04** (PR #3, commit `df1a44a`): resident native worker
  (`native/src`, mmap lifetime preserved) + stock ZMQ ROUTER frontend
  (`src/frigate_xdna/transport/`), per-identity source/generation
  binding with forced transfer, device lease + safety journal, fake-v0
  activation refusal, bounded HW test (LOAD_OK/RUN_OK, Err 0). Soak
  qualification from Phase 7.6 retained (67 client timeouts in one
  host stall window).
1. **Task 05 — acceptance**: done (PR #5) — full Frigate rc2 replay
   with tracked objects, private Plus fetch/compile/activate, A→B,
   restart/offline gates, 24 h soak clean, SBOM, release report;
   verdict `RELEASE_AUTOMATION_READY` in `docs/RELEASE-8.4.md`.
2. **Task 8.5 — release automation** (this branch): GHCR workflow
   (`.github/workflows/release.yml`, stock runners + private B2 vendor
   bundle, no self-hosted runners), release docs (`docs/RELEASE.md`),
   README install. No tag created or pushed yet — first tag
   (`v0.1.0-rc.2`) is an explicit owner action.
3. **Project LICENSE owner**: done — `Copyright (c) 2026 Mitchell Currie`.
4. **Docker availability**: podman 4.9.3 with
   --security-opt apparmor=unconfined (LXC needs the bypass; image builds
   and runs; storage at /mnt/downloads/podman-data to avoid root fill).
5. **Plus credentials/model ID**: operator key present in /root/.env
   (token exchange verified); no test model ID supplied yet — Task 05
   private-model acceptance stays gated on an explicit ID.
6. **`input_dtype: float_denorm`**: contract covers only normalized float32;
   non-normalized float inputs need an explicit serving-contract decision
   (Task 02/04), not silent acceptance.
7. **`FXDNA_ALLOW_UPLOADS` default**: stays `true` per the issued spec
   (operator decision 2026-09-18; CodeRabbit's opt-in suggestion recorded
   here for a future product review, not applied silently).
