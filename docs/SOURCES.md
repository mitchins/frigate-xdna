# Source and evidence register

Retrieved 2026-09-18. Tag-specific files are the compatibility baseline; current documentation is explanatory and must not silently override the rc2 code contract. URLs are recorded as source identifiers, not executable shell instructions.

## S1 — Stock ZMQ plugin

Repository/tag: `blakeblackshear/frigate`, `v0.18.0-rc2`
Path: `frigate/detectors/plugins/zmq_ipc.py`
Git blob: `cc9a538c81160562184901395dafd3988559c4f1`

```
https://github.com/blakeblackshear/frigate/blob/v0.18.0-rc2/frigate/detectors/plugins/zmq_ipc.py
```

Establishes basename-only model handshake, missing per-inference model identity, 30-second model timeout, default 200-ms inference timeout, `yologeneric` enum-name header and initial-not-ready behaviour.

## S2 — Frigate tensor transformation/scheduling boundary

Path: `frigate/object_detection/base.py`, same tag.
Git blob: `bc7910e4d09a13c66fe07e67f20ff87774da1a41`

```
https://github.com/blakeblackshear/frigate/blob/v0.18.0-rc2/frigate/object_detection/base.py
```

Establishes transpose/float normalization before the detector, `[20,6]` shared result and sorted confidence handling. Do not normalize again.

## S3 — Frigate+ client

Path: `frigate/plus.py`, same tag.
Git blob: `d528aa1757f26bfdd4c285fdd0da8b3d6fe9ec84`

```
https://github.com/blakeblackshear/frigate/blob/v0.18.0-rc2/frigate/plus.py
```

Establishes API-key/token flow and read-only model metadata/download/list methods. Reimplement narrowly or preserve MIT attribution when reusing source; do not import unrelated image-upload functionality into the sidecar.

## S4 — Model configuration and Plus metadata

Path: `frigate/detectors/detector_config.py`, same tag.
Git blob: `52d75ff8f728f205e76cb556223e6efb8803c3e2`

```
https://github.com/blakeblackshear/frigate/blob/v0.18.0-rc2/frigate/detectors/detector_config.py
```

Establishes `supportedDetectors`, input metadata, `labelMap`, attributes and the separate Frigate MD5 model hash.

## S5 — Official Frigate detector documentation

```
https://docs.frigate.video/configuration/object_detectors/
https://docs.frigate.video/integrations/plus/
```

Explains region-based detection, model sizes, Plus and Apple ZMQ use. Current docs may describe features newer than the compatibility tag.

## S6 — ZeroMQ socket contract

```
https://libzmq.readthedocs.io/en/latest/zmq_socket.html
```

REQ/ROUTER compatibility, routing envelopes, send/receive semantics and socket thread-safety. Does not grant application authentication or model identity.

## S7 — Docker Compose service configuration

```
https://docs.docker.com/reference/compose-file/services/
```

Device mappings, supplementary groups, secrets, resource controls, healthchecks and network exposure. Validate platform/LXC-specific behaviour rather than assuming it.

## S8 — Frigate source licence

Path: `LICENSE`, `v0.18.0-rc2`
Git blob: `924cb4148cda90ee9d7f7953dcbd9fea31a05874`

```
https://github.com/blakeblackshear/frigate/blob/v0.18.0-rc2/LICENSE
```

MIT licence for copied Frigate source/tests; not a licence for AMD components or private model weights.

## Local experimental evidence (supplied by the user)

These paths are on the user's worker/host-backed data, not files inspected in this spec-writing environment. Task 01 must import the underlying exact manifests/reports before claiming reproduction:

```
/mnt/downloads/xdna-phase5/20260916/
/mnt/downloads/xdna-phase5-license/20260916/
/mnt/downloads/xdna-phase7/20260916/
/mnt/downloads/xdna-phase7-soak/20260917/REPORT.md
/mnt/downloads/xdna-compiler-audit/20260918/COMPILER-AUDIT.md
/root/xdna/UNWIND.md
```

Do not use this spec as an independent legal opinion. The exact EULA/TPN PDF texts and binary hashes are in the user's completed audit. This design adopts that audit for the specific incorporated-unmodified-appliance scope and requires its conditions to be reproduced in release packaging.
