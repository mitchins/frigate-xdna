"""TEST SHIM — not upstream Frigate code.

Minimal `DetectionApi` base class sufficient to import the pinned
`zmq_ipc.ZmqIpcDetector` in contract tests. The real
`frigate/detectors/detection_api.py` is not in the pinned source set; this
shim provides only what the plugin observably uses: config storage via
`super().__init__(detector_config)` and the `detect_raw` override point.
"""


class DetectionApi:
    def __init__(self, detector_config):
        self.detector_config = detector_config

    def detect_raw(self, tensor_input):  # pragma: no cover - overridden
        raise NotImplementedError
