"""Frame timing: de-jittered timestamps, frame period, dropped frames, inter-camera phase.

Kernel timestamps jitter by about +/-2 ms (measured: 32 / 36 ms alternating intervals
around a true 33.3 ms period). A camera's frames are really evenly spaced, so a straight
line t = a + b*n fitted to (frame number, timestamp) over a sliding window gives a much
cleaner time for each frame. Frame numbers are inferred from the timestamps, so gaps
(dropped frames) are detected rather than silently mis-paired.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrameTime:
    n: int              # inferred frame number (gaps = dropped frames)
    t_raw_ms: float     # kernel timestamp
    t_fit_ms: float     # de-jittered timestamp from the fitted line
    dropped_before: int


class FrameClock:
    """Online model t = a + b*n of one camera's frame timestamps."""

    def __init__(self, nominal_period_ms: float, window: int = 300):
        self.nominal_period_ms = nominal_period_ms
        self._n: deque[int] = deque(maxlen=window)
        self._t: deque[float] = deque(maxlen=window)
        self.dropped = 0
        self._a = 0.0
        self._b = nominal_period_ms

    @property
    def period_ms(self) -> float:
        return self._b

    @property
    def count(self) -> int:
        return len(self._n)

    def predict(self, n: float) -> float:
        return self._a + self._b * n

    def add(self, t_ms: float) -> FrameTime:
        if self._n:
            gap = max(1, round((t_ms - self._t[-1]) / self._b))
            n = self._n[-1] + gap
        else:
            gap, n = 1, 0
        dropped_before = gap - 1
        self.dropped += dropped_before
        self._n.append(n)
        self._t.append(t_ms)
        if len(self._n) >= 3:
            # Centred least squares: timestamps are ~1e6 ms, so fit deviations from the mean.
            nn = np.asarray(self._n, float)
            tt = np.asarray(self._t)
            dn = nn - nn.mean()
            self._b = float(dn @ (tt - tt.mean()) / (dn @ dn))
            self._a = float(tt.mean() - self._b * nn.mean())
        else:
            self._a = t_ms - self._b * n
        return FrameTime(n, t_ms, self.predict(n), dropped_before)

    def residuals_ms(self) -> np.ndarray:
        n = np.asarray(self._n, float)
        return np.asarray(self._t) - self.predict(n)

    def residual_rms_ms(self) -> float:
        r = self.residuals_ms()
        return float(np.sqrt(np.mean(r**2))) if r.size else float("nan")


def phase_ms(a: FrameClock, b: FrameClock, at_ms: float) -> float:
    """Offset of camera b's frame grid relative to camera a's near time `at_ms`.

    Result is in [-P/2, P/2) with P = a's period: 0 means the two cameras start their
    exposures at the same moment. It drifts when the two clocks run at different rates.
    """
    na = round((at_ms - a._a) / a.period_ms)
    nb = round((at_ms - b._a) / b.period_ms)
    d = b.predict(nb) - a.predict(na)
    p = a.period_ms
    return (d + p / 2) % p - p / 2


def drift_ppm(a: FrameClock, b: FrameClock) -> float:
    """Relative frame-rate difference of b vs a, in parts per million."""
    return (b.period_ms - a.period_ms) / a.period_ms * 1e6
