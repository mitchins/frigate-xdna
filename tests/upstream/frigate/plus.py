"""TEST SHIM — not upstream Frigate code.

Placeholder for `frigate.plus.PlusApi`, imported by the pinned
`detector_config.py` only for type annotation. The real Plus client
(`tests/upstream/plus.py.src`, pinned blob d528aa17...) is studied in
Task 02; this shim implements no API behaviour.
"""


class PlusApi:  # pragma: no cover - import-time placeholder only
    pass
