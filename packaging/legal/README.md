# Packaging: legal inputs and image flow-down

## Inputs (maintainer-side, private)

`tools/prepare_vendor_input.py` stages `packaging/vendor-input/` (gitignored)
from the audited Phase-7.7 payload:

* `compile-site-packages/` — the exact audited compiler files, verified
  byte-for-byte against `packaging/vendor-files.manifest.json` at stage
  time and again inside the Docker build.
* `xrt/` — minimal XRT userspace + shim + version record.
* `flexmlrt/` — standalone `libflexmlrt.so` (runtime only).
* `legal/` — AMD EULA + Ryzen AI TPN texts shipped inside the appliance
  as flow-down notices (the §2.3 incorporated-distribution route adopted
  for engineering; see THIRD_PARTY_NOTICES.md).

Nothing under `vendor-input/` is committed. The public release artifact is
the incorporated appliance image, never the loose payload.

## In-image material (`/opt/fxdna/legal/`)

* `THIRD_PARTY_NOTICES.md` (project file, committed)
* `component-map.json` — per-component package/version/hash/licence/purpose
  mapping (committed; hashes verify the incorporated binaries)
* AMD EULA + TPN texts (private input, image only)
* `vendor.lock.json` + `vendor-files.manifest.json` (committed hashes)

## Rules (from the audit)

* Binaries are incorporated unmodified; never patch/strip them.
* Do not relabel the image MIT: source and vendor binaries keep separate
  applicable terms.
* Any binary-version or distribution-scope change renews component review.
