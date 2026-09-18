# Implementation work queue (Task 01 exit)

Gaps recorded as implementation work, not research hypotheses. Owners: later
agent tasks per SPEC §15.

1. **Task 02 — manager**: `serve` daemon, admin Unix socket, Plus client
   (from `tests/upstream/plus.py.src`), SQLite registry, cache store,
   compile-job state machine (fake backend exists), pins/prune, offline
   behaviour. Needs: Plus API host/constants verified from pinned source.
2. **Task 03 — compiler appliance**: vendor payload import with full
   per-file `vendor.lock.json` + `component-map.json`, Docker candidate,
   launcher with limits, offline-compile gate, cache-hit proof, with/without
   device classification. Needs: docker on a build host, private payload
   input, exclusive device window (no soak running).
3. **Task 04 — native/ZMQ**: C++ private-IPC child (from
   `native/reference/`), Python ROUTER frontend, identity/generation
   binding, activation/validation, device lease + safety journal. Needs:
   exclusive device window; short one-context hardware test only.
4. **Task 05 — acceptance**: full Frigate rc2 container test, private Plus
   model fetch (needs user-supplied key/ID at runtime, never in repo),
   A→B update flow, 24 h service soak, SBOM, release report.
5. **Project LICENSE owner**: replace `[PROJECT OWNER]` in `LICENSE` with
   the copyright holder's name before any release.
6. **Docker availability**: no docker/podman in this CT; install at Task 03
   (user pre-approved mid-way install).
7. **Plus credentials**: no test key/ID supplied; Task 02 live check and
   Task 05 acceptance stay mock-gated until provided.
8. **`input_dtype: float_denorm`**: contract covers only normalized float32;
   non-normalized float inputs need an explicit serving-contract decision
   (Task 02/04), not silent acceptance.
