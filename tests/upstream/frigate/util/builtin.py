"""TEST SHIM — not upstream Frigate code.

Minimal stand-ins for `frigate.util.builtin` helpers used at import/call
time by the pinned `detector_config.py`. Signatures mirror the real ones;
bodies are trivial. Never cite as upstream behaviour.
"""


def generate_color_palette(n: int):  # pragma: no cover - trivial shim
    return [(0, 0, 0)] * max(0, n)


def load_labels(path: str, encoding: str = "utf-8"):  # pragma: no cover
    labels: dict = {}
    try:
        with open(path, encoding=encoding) as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(None, 1)
                    if len(parts) == 2:
                        labels[int(parts[0])] = parts[1]
    except OSError:
        pass
    return labels
