# YOLOv9-320 scaling qualification (Strix Halo, 2026-09-24)

Scope: public YOLOv9 **T/S/M/C/E at 320 only**, exported via the
documented Frigate/WongKinYiu route with the recorded toolchain
(C4 deviations apply: native recipe run, onnx-simplifier 0.5.0, CPU
torch 2.14.0). T is the C7-qualified baseline. No other model
families, no 640, no power benchmarking.

Method per variant, under the canonical 4-CPU/8-GiB appliance
envelope: fresh volume (cold compile) → inspect → compile →
validate (in-compile probe) → ZMQ handshake → known-crop inference
(person check) → fixed-tensor sequential benchmark (N=300 + 20
warmup, one 1x3x320x320 crop) → short stock-Frigate 0.18.0 walk2
replay. No `PLUS_API_KEY` anywhere. p50/p95/p99 are client-observed
per-request latencies; sustained req/s is directly measured
(N/total). No peak FPS is derived by reciprocating latency.

## Weights (upstream, same release page as C4)

| Variant | Weights SHA-256 | .pt bytes |
|---|---|---|
| T | `61e080e9…` (C4) | 4,632,858 |
| S | `09bf9ca4…` | 15,044,442 |
| M | `4f60eef3…` | 40,675,619 |
| C | `39b6b490…` | 51,477,927 |
| E | `6763139d…` | 117,160,386 |

`https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-<t,s,m,c,e>-converted.pt`,
`export.py --weights … --imgsz 320 --simplify --include onnx`.
All five exports inspect identically: input `images [1,3,320,320]`
fp32 → `output0 [1,84,2100]` fp32, opset 18, IR 10, 80 classes,
`yolo-raw` compatible. ONNX SHAs/sizes below.

## Results

| Var | ONNX SHA-256 | ONNX MB | Compile s | Peak RSS GiB | .rai MB | ZMQ p50 / p95 / p99 (ms) | Sustained req/s | Person top | Frigate events | Frigate-reported ms | Errors |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T | `338ba4ad…` | 8.1 | 434.0 | 1.71 | 7.2 | 7.37 / 7.87 / 10.42 | 134.0 | 0.941 | 48 person | 8.4–9.5 | none |
| S | `1f3121d6…` | 28.6 | 547.4 | 1.88 | 17.1 | 8.98 / 9.66 / 10.38 | 110.3 | 0.953 | 8 person | 10.36 | none |
| M | `9a85f17b…` | 80.0 | 595.8 | 2.13 | 46.3 | 13.13 / 14.14 / 16.05 | 75.4 | 0.957 | 8 person | 14.79 | none |
| C | `c7009a1c…` | 101.3 | 631.8 | 2.28 | 54.5 | 14.09 / 14.71 / 15.61 | 70.6 | 0.957 | 8 person | 15.64 | none |
| E | `3e894d2a…` | 229.6 | 968.9 | 3.56 | 123.5 | 69.36 / 69.90 / 70.11 | 14.4 | 0.973 | 8 person | 70.96 | none |

Chart: `/mnt/downloads/fxdna-012-scale/yolov9-320-scaling.png`
(raw numbers: `/mnt/downloads/fxdna-012-scale/scale-results.json`;
binaries, logs, bench harness: same directory). That path is the
build host's local evidence area, not a durable public location —
but every byte above is reproducible from public inputs, and the
full digests below identify exactly which files were measured.

Full SHA-256 (weights `.pt` / exported `.onnx`):

- T: `61e080e964e65e32b884477c5e6344c607c7e02103d64649de810edaeb869803`
  / `338ba4addc585da4d19ca59624cc51d9ff8aa8ef1e7675b5b5f9e456fa5ea4ce`
- S: `09bf9ca4adef37944f4406455b5b81b451c1e67307916e4702da68cda4d3e46d`
  / `1f3121d6372d8eefc5c91f911772ea313a8e66938f63c7bf0f82e4a38a10e949`
- M: `4f60eef3aca520d4a74e80d1c246c6b094ba40228f22bea13e864cd7a49c2349`
  / `9a85f17b63951c5d3702318637fb2cc7009b74237c76de03dcfb64d2511cf30c`
- C: `39b6b490b1e4034b4fefdcdac81380155e0a30ba6c938a9751e2c351f81a517f`
  / `c7009a1cf132aa2659214b9143f9d5d580da3e95e82b9b39baaa94d7d485c3d4`
- E: `6763139daa09adfe3c694226725b460211f1f717b971bfade10a548b600716e2`
  / `3e894d2a17330595c42650dc00f8077a37cda87039d6f8b1e94624b3679f2fc9`

## Verdict

- **Largest variant passing end-to-end: E.** It compiles, loads,
  and runs safely inside the envelope (peak RSS 3.56 GiB < 8 GiB),
  detects persons (0.973), and completes a Frigate replay with
  person tracks. No faults, no inhibition, no unsafe behavior.
- **C is the largest variant staying in the ~15 ms band.**
  T/S sit ~7–10 ms, M/C ~13–16 ms, E steps to ~69–71 ms with a
  tight distribution (min 68.5 / max 70.2): a capacity step, not
  jitter and not a fault. The E artifact (123.5 MB) is ~2.3x C's;
  the step is consistent with NPU-capacity pressure, recorded as
  observed behavior, not diagnosed further here.
- No model hit the stop rule (nothing failed to compile, load, or
  run safely), so no larger-memory diagnostic run was needed and
  the supported default envelope is unchanged.

## Recommendation

**YOLOv9-C 320 is the recommended quality/performance inflection
on the qualified Strix Halo platform.** It is the largest YOLOv9
variant that remains in the ~15 ms detector-latency band,
sustaining ~71 sequential detector requests/s in qualification.
This provides substantial headroom for a typical motion-driven
multi-camera Frigate deployment at the recommended 5 detect fps
(8 cameras × 5 fps = 40 camera frames/s of demand, before
motion-gating reduces it further). No claim is made about eight
cameras at 15 fps unconstrained: simultaneous demand from all
cameras, and multiple regions per frame, can exceed any single
number — size the detector from the measured ~71 req/s against
your own camera count and frame rates.

YOLOv9-E also runs correctly, but incurs a large
latency/throughput step (~69 ms / 14 req/s) and is better treated
as a specialist quality-over-throughput option.

The recommendation combines frigate-xdna's measured 320 inference
performance with the upstream YOLOv9 model family's published
accuracy scaling (M → C → E: 51.4 → 53.0 → 55.6 COCO AP at 640 —
C takes most of the family's quality headroom for almost no
latency cost over M here, while E buys 2.6 further AP points at
roughly a 5x latency penalty in this 320 deployment).
frigate-xdna did not independently measure COCO mAP at 320.
- Caveats: the benchmark replays one fixed crop tensor, so
  sustained req/s has no preprocessing or scene variance in it;
  all Frigate replays used the walk2 clip with the C7 rig config
  (`min_score` 0.3, `min_area` 500 — set for the tiny T and kept
  for comparability). Latency is client-observed round trip
  (Frigate-side preprocessing excluded, ZMQ transport included).
