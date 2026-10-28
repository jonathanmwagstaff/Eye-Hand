"""Audio feedback for calibration: the person holding the board or camera can't watch a UI."""

from __future__ import annotations

import subprocess
from pathlib import Path

SOUNDS = Path("/usr/share/sounds/freedesktop/stereo")


def play(name: str) -> None:
    """Play a freedesktop sound ("camera-shutter", "complete", "dialog-warning") without
    blocking. Silently does nothing if no player is available."""
    try:
        subprocess.Popen(["paplay", str(SOUNDS / f"{name}.oga")],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass
