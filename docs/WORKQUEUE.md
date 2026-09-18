# Implementation work queue

Gaps recorded as implementation work, not research hypotheses. Owners: later
agent tasks per SPEC §15.

Done:
- **Task 01** (commit `5cd26ba`): foundation, contracts, fixtures, fakes.
- **Task 02** (this commit): Plus read-only client (verified against pinned
  rc2 `plus.py` + rc2 `const.py` values), content-addressed cache (SQLite
  registry, atomic publish, recovery, pins/prune), serve daemon with
  private admin socket, full CLI, local ONNX/RAI import, offline behaviour,
  fake compile transitions (published through the real atomic path).
  Live Plus token exchange verified with the operator-supplied key (token
  endpoint only; no model ID supplied so no model fetch, list, or download
  performed).

1. **Task 03 — compiler appliance**: vendor payload import with full
   per-file `vendor.lock.json` + `component-map.json`, Docker candidate,
   launcher with limits, offline-compile gate, cache-hit proof, with/without
   device classification. Needs: docker on a build host, private payload
   input, exclusive device window (no soak running).
2. **Task 04 — native/ZMQ**: C++ private-IPC child (from
   `native/reference/`), Python ROUTER frontend, identity/generation
   binding, activation/validation, device lease + safety journal. Must
   refuse `backend: fake-v0` artifacts at activation (see
   `Supervisor._publish_fake_artifact`); set `COMPILER_BACKEND` to the
   audited backend id when the real compiler lands (stale fake rows
   auto-invalidate on next prepare). Needs:
   exclusive device window; short one-context hardware test only.
3. **Task 05 — acceptance**: full Frigate rc2 container test, private Plus
   model fetch (needs user-supplied key/ID at runtime, never in repo),
   A→B update flow, 24 h service soak, SBOM, release report.
4. **Project LICENSE owner**: replace `[PROJECT OWNER]` in `LICENSE` with
   the copyright holder's name before any release.
5. **Docker availability**: no docker/podman in this CT; install at Task 03
   (user pre-approved mid-way install).
6. **Plus credentials/model ID**: operator key present in /root/.env
   (token exchange verified); no test model ID supplied yet — Task 05
   private-model acceptance stays gated on an explicit ID.
7. **`input_dtype: float_denorm`**: contract covers only normalized float32;
   non-normalized float inputs need an explicit serving-contract decision
   (Task 02/04), not silent acceptance.
