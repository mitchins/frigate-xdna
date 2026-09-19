"""Exit codes and error taxonomy (docs/INTERFACES.md §1, normative)."""

SUCCESS = 0
INVALID_ARGS = 2
NOT_READY = 3
ACQUISITION_FAILED = 4
UNSUPPORTED_CONTRACT = 5
COMPILE_FAILED = 6
VALIDATION_FAILED = 7
DEVICE_UNAVAILABLE = 8
OWNERSHIP_CONFLICT = 9
CACHE_CORRUPT = 10

# Stable error-code strings (wire/status vocabulary, not exit codes).
NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
MODEL_NOT_PREPARED = "MODEL_NOT_PREPARED"
MODEL_IN_USE = "MODEL_IN_USE"
INVALID_MODEL = "INVALID_MODEL"

# Extended registry/job vocabulary (CACHE.md states + control errors).
# These travel in status/job records and admin replies; only the wire codes
# above go to stock Frigate clients.
ERROR_CODES = frozenset({
    "INVALID_ARGS", "INVALID_CONFIG", "INVALID_REF", "INVALID_MODEL",
    "NOT_IMPLEMENTED", "WAIT_TIMEOUT", "UNKNOWN_JOB",
    "MODEL_NOT_PREPARED", "MODEL_IN_USE",
    "ACQUISITION_FAILED", "AUTH_FAILED", "DOWNLOAD_FAILED",
    "UNSUPPORTED_CONTRACT", "SOURCE_CHANGED",
    "COMPILE_FAILED", "RESOURCE_EXCEEDED", "VALIDATION_FAILED",
    "CACHE_CORRUPT", "INTERRUPTED", "QUARANTINED",
    "DEVICE_BUSY", "DEVICE_FAULT", "SAFETY_INHIBITED",
    "OWNERSHIP_CONFLICT", "ACTIVATION_UNAVAILABLE",
    "DAEMON_UNREACHABLE",
})

ERROR_NAMES = {
    0: "SUCCESS",
    2: "INVALID_ARGS",
    3: "NOT_READY",
    4: "ACQUISITION_FAILED",
    5: "UNSUPPORTED_CONTRACT",
    6: "COMPILE_FAILED",
    7: "VALIDATION_FAILED",
    8: "DEVICE_UNAVAILABLE",
    9: "OWNERSHIP_CONFLICT",
    10: "CACHE_CORRUPT",
}


class FxdnaError(Exception):
    """Structured service error with a CLI exit code and stable error code."""

    def __init__(self, exit_code: int, error_code: str, message: str):
        super().__init__(message)
        self.exit_code = exit_code
        self.error_code = error_code
        self.message = message
