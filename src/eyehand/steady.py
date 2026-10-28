"""Is a camera physically steady? Measured from the image itself, no board needed.

Features of the static scene (ORB, matched against reference frames) give the image shift
and in-plane rotation with RANSAC, so a person moving in part of the view is rejected as
outliers. A distant background is the best reference: a camera turn shifts it like everything
else, and small sideways movements barely affect it. A person is a poor reference: a 1 cm
head movement at 60 cm looks like a 1 degree turn. A camera that tilts by 1 degree shifts its image by about f * tan(1 deg) pixels:
16 px for the laptop camera, 28 px for the Feeltek (C1/C2). The mount must hold the camera
to well under 0.1 deg between stereo calibration and use (docs/RECALIBRATION.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Motion:
    dx_px: float
    dy_px: float
    roll_deg: float
    tilt_deg: float          # total pan/tilt implied by the shift, using the focal length
    inliers: int


class SteadyMeter:
    """Image motion of one camera relative to reference frame(s).

    Features are taken evenly from a 4 x 4 grid (a high-contrast object such as a calibration
    board would otherwise take all of them), matched with a ratio test, and fitted with RANSAC.
    With several references (for example every frame of a calibration capture), the median
    over them is reported, which removes most of the noise of a single comparison.
    """

    GRID = 4
    PER_TILE = 400

    def __init__(self, references_bgr: list[np.ndarray] | np.ndarray, focal_px: float):
        if isinstance(references_bgr, np.ndarray):
            references_bgr = [references_bgr]
        self.f = focal_px
        self.orb = cv2.ORB_create(self.PER_TILE)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        h, w = references_bgr[0].shape[:2]
        self.centre = np.array([w / 2, h / 2])
        self.refs = [self._features(r) for r in references_bgr]

    def _features(self, bgr):
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
        g = cv2.createCLAHE(2.0, (8, 8)).apply(g)      # robust to exposure changes
        h, w = g.shape
        kps = []
        for ty in range(self.GRID):
            for tx in range(self.GRID):
                mask = np.zeros_like(g)
                mask[ty * h // self.GRID:(ty + 1) * h // self.GRID,
                     tx * w // self.GRID:(tx + 1) * w // self.GRID] = 255
                kps += self.orb.detect(g, mask)
        return self.orb.compute(g, kps)

    def _one(self, ref, cur) -> Motion | None:
        (ka, da), (kb, db) = ref, cur
        if da is None or db is None or len(da) < 2 or len(db) < 2:
            return None
        good = [m[0] for m in self.matcher.knnMatch(da, db, k=2)
                if len(m) == 2 and m[0].distance < 0.8 * m[1].distance]
        if len(good) < 20:
            return None
        a = np.float32([ka[m.queryIdx].pt for m in good])
        b = np.float32([kb[m.trainIdx].pt for m in good])
        M, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2.0,
                                             maxIters=5000, confidence=0.999)
        if M is None or inl.sum() < 20:
            return None
        # Shift of the image centre: a pure roll moves the corners but not the centre.
        dx, dy = (M @ np.append(self.centre, 1.0)) - self.centre
        roll = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
        tilt = float(np.degrees(np.arctan(np.hypot(dx, dy) / self.f)))
        return Motion(float(dx), float(dy), roll, tilt, int(inl.sum()))

    def measure(self, bgr) -> Motion | None:
        cur = self._features(bgr)
        found = [m for m in (self._one(r, cur) for r in self.refs) if m is not None]
        if not found:
            return None
        dx = float(np.median([m.dx_px for m in found]))
        dy = float(np.median([m.dy_px for m in found]))
        roll = float(np.median([m.roll_deg for m in found]))
        tilt = float(np.degrees(np.arctan(np.hypot(dx, dy) / self.f)))
        return Motion(dx, dy, roll, tilt, int(np.median([m.inliers for m in found])))
