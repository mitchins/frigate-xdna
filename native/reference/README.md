# Proven native implementation — reference import (Task 01, no rewrite)

These two files are byte-identical copies of the banked research
implementation. They are the reuse baseline for Task 04; FlexML lifetime and
mmap semantics must be preserved, not re-derived from memory.

| File | Origin | SHA-256 |
|---|---|---|
| `worker-phase7.cc` | `/root/xdna/phase7/xdna-zmq-worker/worker.cc` (persistent ZMQ worker, Phase 7; stocked Frigate protocol + in-process YOLO→[20,6]) | `b65eccb424f1f481e885d484d85e9dbbfa823f228de335b1743d1afe91ff55b0` |
| `yolo_flexml-phase5.cc` | `/root/xdna/phase5/runner/yolo_flexml.cc` (standalone FlexMLRT runner, Phase 5/6/D5 deployment proof) | `d4189e505905cbedf3f4e07d3bb5d9636cd0943a9af7a9549582d7695a3dd1ba` |

Related soak variant (not imported; superseded by the above two plus a
resolution parameter): `/root/xdna/phase7-soak/worker_v9s320.cc`
(`665930458e09ff9f646fa3b069a54a5d9d62d27cf70a4e4edfb8ea64d4a5ead8`),
binary `4b1b00da74beb0cb5bd71fd5ca5ffda4024f4f546ebfacbc70578e6806441ca4`.

## Reuse plan (Task 04)

REUSE (semantics frozen):
- `.rai` mmap-once via `map_rai` (`uint8_t*` buffer handed to
  `extOptions["fbs_buffer"]`, munmap only after Model destruction);
- single `flexmlrt::client::Model` per process, one HW context, REQ/REP
  forever; process exit (not unload) to change models;
- Frigate `[20,6]` row order `[class, score, ymin, xmin, ymax, xmax]`,
  conf 0.25 / NMS 0.45 as the banked YOLO-raw defaults (product decoder
  parameters come from the serving contract, not these literals);
- stock-protocol model_available/model_loaded/model_saved replies.

Behavioural difference Task 04 must not blindly reuse: the reference
worker's NMS (`worker-phase7.cc` postprocess) is class-agnostic (suppresses
across classes), while SPEC §7.1 and the product decoder
(`models/contracts.py`) are class-aware (different classes are retained).
Detector thresholds likewise move from code literals to the serving
contract.

REPLACE / EXTEND (product requirements):
- output anchor count must come from model metadata (8400@640 vs 2100@320),
  never a hardcoded constant — the soak variant proved the pattern;
- input shape check must accept the serving contract's geometry, never a
  hardcoded 640;
- protocol layer replaced by the private supervisor IPC (INTERFACES.md §4)
  with request IDs and worker generations; the stock ZMQ frontend moves to
  the Python ROUTER (Task 04), not the C++ child;
- structured error replies instead of silent zero frames on the private IPC
  (zero frames remain the stock-wire behaviour for malformed tensors).
