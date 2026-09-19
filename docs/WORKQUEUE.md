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

1. **Task 04 — native/ZMQ**: C++ private-IPC child (from
   `native/reference/`), Python ROUTER frontend, identity/generation
   binding, activation/validation, device lease + safety journal. Must
   refuse `backend: fake-v0` artifacts at activation (see
   `Supervisor._publish_fake_artifact`); set `COMPILER_BACKEND` to the
   audited backend id when the real compiler lands (stale fake rows
   auto-invalidate on next prepare). Needs:
   exclusive device window; short one-context hardware test only.
2. **Task 05 — acceptance**: full Frigate rc2 container test, private Plus
   model fetch (needs user-supplied key/ID at runtime, never in repo),
   A→B update flow, 24 h service soak, SBOM, release report.
   Native ZMQ product path (Task 04) not yet claimed.
3. **Project LICENSE owner**: done — `Copyright (c) 2026 Mitchell Currie`.
4. **Docker availability**: podman 4.9.3 with
   --security-opt apparmor=unconfined (LXC needs the bypass; image builds
   and runs; storage at /mnt/downloads/podman-data to avoid root fill).
6. **Plus credentials/model ID**: operator key present in /root/.env
   (token exchange verified); no test model ID supplied yet — Task 05
   private-model acceptance stays gated on an explicit ID.
7. **`input_dtype: float_denorm`**: contract covers only normalized float32;
   non-normalized float inputs need an explicit serving-contract decision
   (Task 02/04), not silent acceptance.
8. **`FXDNA_ALLOW_UPLOADS` default**: stays `true` per the issued spec
   (operator decision 2026-09-18; CodeRabbit's opt-in suggestion recorded
   here for a future product review, not applied silently).
