# Upstream contract-test sources

Byte-identical Frigate `v0.18.0-rc2` files (verified by `git hash-object`
against the blobs in `upstream.lock.json`) plus minimal test shims.

Pinned-verbatim (assert byte-identical in `tests/contract/test_pinned.py`):
* `frigate/detectors/plugins/zmq_ipc.py` — the stock plugin under test.
* `frigate/detectors/detector_config.py` — stock config models.

Reference-only (source assertions, never imported):
* `base.py.src` — input transform / normalization evidence (S2).
* `plus.py.src` — Plus client study input for Task 02 (S3).

Shims (original test code, MIT project licence, never upstream behaviour):
* `frigate/const.py`, `frigate/plus.py`, `frigate/util/builtin.py`,
  `frigate/detectors/detection_api.py`.

Upstream licence: MIT, `blakeblackshear/frigate` `LICENSE` blob
`924cb4148cda90ee9d7f7953dcbd9fea31a05874`. This notice does not extend to
AMD binaries or private models.
