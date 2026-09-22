# Replay fixture (Task 05 full-app acceptance)

Deterministic local moving video + stock Frigate `v0.18.0` Debug Replay
path, exercising the whole chain rather than pushing prepared tensors:

```text
real Frigate app -> FFmpeg decode -> motion regions -> detector scheduler
-> stock ZMQ -> XDNA sidecar -> tracker -> observable detections/events/API
```

## Clip (stored outside the repo)

The clip is measurement input, not source: keep it under
`/mnt/downloads/frigate-xdna-replay/` (host-backed, 5.5 TB mount), never
commit it. Prefer a short (30–60 s) clip with persons/cars and steady
lighting. Record its identity in `manifest.json`:

```sh
sha256sum clip.mp4
ffprobe -v error -show_entries format=duration \
  -show_entries stream=avg_frame_rate,width,height -of default=noprint_wrappers=1 clip.mp4
```

No encoder on the build host: generate or trim the clip on a machine with
`ffmpeg`, e.g. trim a longer capture deterministically:

```sh
ffmpeg -ss 00:01:00 -i source.mp4 -t 45 -c copy replay-45s.mp4
```

A synthetic motion-only clip (moving rectangles, no real objects) is
acceptable for pipeline-shape checks (motion regions -> scheduler calls ->
ZMQ -> sidecar accounting) but NOT for detection evidence: record
`"labels": []` and a note in that case.

## Frigate side (operator, gated run)

1. Pull the pinned image: `ghcr.io/blakeblackshear/frigate:0.18.0`.
2. Mount the clip read-only into the Frigate container and configure a
   replay camera whose input is the file, carrying across the production
   camera's detect config (detect resolution, object filters, zones,
   motion config); recording/snapshots/review/audio stay disabled for
   the replay camera.
3. Point the camera's detector at the sidecar (`tcp://xdna:5555`) with
   the prepared model's tensor geometry and label map.
4. Start replay and run `tools/replay_acceptance.py` (below) for the
   clip duration.

## Driver

```sh
PYTHONPATH=src python3 tools/replay_acceptance.py \
  --frigate http://127.0.0.1:5000 --camera replay_xdna \
  --data-dir /data --duration-s 60 --out /tmp/replay-evidence.json
```

The driver only uses Frigate's long-stable HTTP surface (`/api/config`,
`/api/stats`, `/api/events`); it fails visibly with the actual HTTP
status if Frigate ever drifts. Evidence JSON is appended to
`manifest.json` `runs[]` (redacted: no Plus IDs/tokens).

## Regression use

After release, every change to cache, compiler recipe, FlexML runtime,
ZMQ lifecycle, or Frigate version replays the same clip and diffs the
new evidence against `expected_observations`.
