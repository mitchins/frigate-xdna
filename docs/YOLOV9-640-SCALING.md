# YOLOv9-640 scaling qualification (Strix Halo, 2026-09-24/25)

Paired repeat of the 320 sweep at 640×640: same public YOLOv9
family, same Frigate/WongKinYiu export route, same toolchain and
deviations, same benchmark method, same walk2 replay. Production
Frigate stayed on ONNX throughout (separate test rig, test-only
network/volumes/configs; NPU verified free before and after; no
device faults, no inhibition).

Question: how does doubling spatial resolution change XDNA
behaviour, and does C-640 cross a capacity boundary that C-320
does not? All five 640 graphs inspect identically: input
`images [1,3,640,640]` fp32 → `output0 [1,84,8400]` fp32, opset 18,
IR 10, 80 classes, `yolo-raw` compatible. No Plus key anywhere.

Method per variant (canonical 4-CPU/8-GiB envelope, fresh volume):
inspect → cold compile → validate → PREPARED → worker load →
known-crop inference (same motion-crop content as 320, resized to
640) → fixed-tensor sequential bench (20 warmup + 300 measured) →
short Frigate 0.18.0 replay (same thresholds; model block 640).
p50/p95/p99 are client-observed; sustained req/s is N/total. No
reciprocated peak FPS.

## Paired results

320 values from `docs/YOLOV9-320-SCALING.md`.

| Var | 320 p50 | 640 p50 | 640/320 | 320 req/s | 640 req/s | 640 compile s | 640 peak RSS GiB | 640 .rai MB | Frigate 640 | Result |
|---|---|---|---|---|---|---|---|---|---|---|
| T | 7.37 | — | — | 134.0 | — | 725–840 (two runs, both refused) | — | — | — | FAILS at validation (probe non-finite, twice) |
| S | 8.98 | 20.00 | 2.23x | 110.3 | 49.8 | 906.1 | 1.87 | 17.1 | 8 person, 23.3 ms | passes |
| M | 13.13 | 34.44 | 2.62x | 75.4 | 28.9 | 1190.8 | 2.13 | 46.5 | 8 person, 39.7 ms | passes |
| C | 14.09 | 35.19 | 2.50x | 70.6 | 28.3 | 941.3 | 2.27 | 54.6 | 8 person, 40.6 ms | passes |
| E | 69.36 | 224.81 | 3.24x | 14.4 | 4.4 | 1498.5 | 5.19 | 123.7 | 0 events at 200 ms; 8 person at diagnostic 1000 ms | runs; not live-usable at canonical budget |

Full 640 numbers (p95/p99, VmPeak, keys, SHAs):
`/mnt/downloads/fxdna-012-scale640/scale640-results.json`.
Binaries, logs, bench harness, charts (paired + per-run): same
directory. 640 ONNX SHAs: T `7455f159…`, S `5cfb0b36…`, M
`7081c645…`, C `2e9224eb…`, E `e1e9d4c7…` (weights are the same
files as the 320 sweep).

Charts: `yolov9-320-vs-640.png` (paired latency + throughput).

## Conclusion

1. **Largest 640 variant passing end-to-end (canonical settings):
   C640.** E640 runs correctly but cannot serve a live Frigate
   detector at the canonical 200 ms budget.
2. **640 costs ~2.2–2.6x for S/M/C** (20.00/8.98, 34.44/13.13,
   35.19/14.09), 3.24x for E. 4x pixels → ~2.5x latency:
   sublinear, smooth.
3. **Yes, roughly proportional across S/M/C** — no second cliff
   in that band. The surprise from 320 repeats: M640→C640 is
   +0.75 ms (+2%), the same flat step as M320→C320.
4. **C640 stays practical with caveats:** 28.3 req/s against
   8×5 fps = 40/s of motion-ungated demand — viable where
   motion-gating applies (the normal Frigate case), without the
   headroom C320 enjoys. Not a recommendation change.
5. **No evidence of a new capacity regime at C640.** The
   M→C flat step is the strongest signal: a mapping cliff
   between them would break it. E's step (both resolutions) is
   the known capacity behavior, steeper at 640.
6. **E640 is unusably slow live, not broken:** 0 events with the
   detector stalling at the 200 ms budget, but 8 person tracks at
   a diagnostic 1000 ms budget (232 ms reported) with zero
   transport errors. Specialist batch use at best.
7. **The existing recommendation holds: C-320 remains the
   quality/performance inflection; 640 is a deliberate
   higher-resolution tradeoff (~2.5x latency for S/M/C, larger
   for E); E stays a specialist throughput tradeoff.** The 640
   experiment adds one asymmetry: T640 fails validation while
   S–E640 pass, so 640 support starts at S, not T.

Two anomalies, both bounded and recorded:

- **T640 fails the in-compile native probe** (non-finite output
  on zeros input), twice on fresh volumes, kind=permanent, no
  retry, no inhibition. The device was proven healthy immediately
  after (T320 artifact: person 0.938 on hardware). T-specific
  mapping issue, not a device fault and not resolution-systematic
  (S–E640 all pass the same probe). Not diagnosed further here.
- **A bench-harness header bug** (stale 320 shape with a 640
  tensor) caused 21 rejected S640 requests mid-run; caught via
  the frontend counters, fixed, re-ran clean. The counters doing
  their job is itself a small verification of the observability
  path. 320 numbers used the correct header throughout.

No accuracy claims: replay/person scores are correctness checks,
not mAP. No power data. Method for 640 Frigate blocks is 640
throughout (model block width/height 640, same thresholds).
