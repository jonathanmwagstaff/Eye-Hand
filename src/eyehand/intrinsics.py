"""Intrinsic calibration (C1, C2): ChArUco views -> camera matrix K and distortion d.

Pipeline, as specified in docs/CALIBRATION.md:

1. `ViewSelector` keeps a view only when the board is held still (no motion blur or
   rolling-shutter skew) and differs from every view already kept. Corners are averaged over
   the still frames.
2. `Coverage` tracks which image cells have corners; distortion is only determined where
   corners were seen, so the image corners must be covered too.
3. `solve()`: fit, reject outlier views, choose the distortion model on held-out views,
   check consistency between two random halves, and evaluate the gates.
4. `install()` writes `calib/intrinsics_<camera>.json`; `load_intrinsics()` refuses a result
   whose camera mode or focus differs from the current config (v1 calibrated full frames and
   ran on a crop).

Object points are in mm. Uniform scale does not affect intrinsics at all; it is recorded so
the same detections can be reused for stereo (C5).
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from eyehand.boards import BoardSpec
from eyehand.config import PROJECT_ROOT, CameraConfig

# Controls that change how the camera forms images. A calibration is only valid while they
# match the configuration it was made with.
GEOMETRY_CONTROLS = ("focus_absolute", "focus_automatic_continuous", "zoom_absolute")

# Acceptance gates (docs/CALIBRATION.md, C1). Values are limits that must not be exceeded,
# except the coverage ones, which are minimums.
#
# The main gate is the bootstrap ray uncertainty inside the tracking area: what the cursor
# actually feels (~0.7 mm of cursor per px, docs/ARCHITECTURE.md error budget). The first
# version gated on reprojection RMS < 0.3 px, std(fx) < 0.5 % and split-half agreement < 1 px.
# On the real laptop camera (2026-10-27, 84 views) that was wrong twice over: its corner noise
# floor is ~0.45 px even with 4-frame averaging, so 0.3 px was unreachable; and OpenCV's
# formal std(fx) (0.06 %) was 4x smaller than the bootstrap spread, because per-view errors
# are correlated within a view.
GATES = {
    "rms_px": 0.6,               # sanity: blur, bad detections
    "holdout_ratio": 1.15,       # hold-out RMS / fit RMS: overfitting
    "ray_uncertainty_rms_px": 1.0,
    "ray_uncertainty_max_px": 2.0,
    "coverage_min": 0.85,
}
TRACKING_MARGIN = 0.08           # the live views keep eye and fingertip inside this margin
BOOTSTRAP_RESAMPLES = 16

CRITERIA = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-12)
MIN_CORNERS = 12


@dataclass
class View:
    """One accepted board view: averaged corners and their board object points."""
    name: str
    ids: np.ndarray          # (N,) int32 ChArUco corner ids
    corners: np.ndarray      # (N, 2) float32 pixels, OpenCV convention (pixel centre = integer)
    obj: np.ndarray          # (N, 3) float32 mm, board frame (z = 0)
    t_soe_ms: float = 0.0
    frames_averaged: int = 1

    def to_json(self) -> dict:
        return {"name": self.name, "ids": self.ids.tolist(), "corners": self.corners.tolist(),
                "obj": self.obj.tolist(), "t_soe_ms": self.t_soe_ms,
                "frames_averaged": self.frames_averaged}

    @classmethod
    def from_json(cls, d: dict) -> "View":
        return cls(d["name"], np.asarray(d["ids"], np.int32),
                   np.asarray(d["corners"], np.float32), np.asarray(d["obj"], np.float32),
                   d.get("t_soe_ms", 0.0), d.get("frames_averaged", 1))


@dataclass(frozen=True)
class TargetScale:
    """mm per board unit along the board's x and y.

    Print board: measured square size / nominal (printers can scale x and y differently).
    Screen board: the panel's pixel pitch.
    """
    x: float
    y: float
    source: str


def object_points(spec: BoardSpec, ids: np.ndarray, scale: TargetScale) -> np.ndarray:
    obj = spec.board().getChessboardCorners()[np.asarray(ids).ravel()].astype(np.float64)
    obj[:, 0] *= scale.x
    obj[:, 1] *= scale.y
    return obj.astype(np.float32)


def detect(detector: cv2.aruco.CharucoDetector, image: np.ndarray):
    """Returns (corners (N,2), ids (N,)) or (None, None)."""
    corners, ids, _, _ = detector.detectBoard(image)
    if ids is None or len(ids) == 0:
        return None, None
    return corners.reshape(-1, 2).astype(np.float32), ids.ravel().astype(np.int32)


# ---------------------------------------------------------------------------------------------
# Live view selection


class ViewSelector:
    """Decides, frame by frame, whether the current detection becomes a calibration view.

    still_px:    median corner motion between consecutive frames must stay below this for
                 `still_frames` frames in a row. Camera L's exposure can be a full frame long,
                 so motion of 0.5 px/frame means at most ~0.5 px of blur.
    novel_px:    a view is a duplicate if its shared corners lie within this median distance
                 of an existing view's.
    """

    def __init__(self, board: cv2.aruco.CharucoBoard, still_px: float = 0.5,
                 still_frames: int = 4, novel_px: float = 40.0, min_corners: int = MIN_CORNERS):
        self.board = board
        self.still_px = still_px
        self.still_frames = still_frames
        self.novel_px = novel_px
        self.min_corners = min_corners
        self.window: deque[tuple[np.ndarray, np.ndarray, float]] = deque(maxlen=still_frames)
        self.accepted: list[dict[int, np.ndarray]] = []
        self.status = "no board"

    def reset_window(self) -> None:
        self.window.clear()

    def offer(self, corners, ids, t_soe_ms: float = 0.0):
        """Feed one frame's detection. Returns (ids, averaged corners, frames) when accepted."""
        if ids is None:
            self.status = "no board"
            self.window.clear()
            return None
        if len(ids) < self.min_corners:
            self.status = f"only {len(ids)} corners"
            self.window.clear()
            return None
        if self.board.checkCharucoCornersCollinear(ids.reshape(-1, 1)):
            self.status = "corners collinear"
            self.window.clear()
            return None
        if self.window:
            prev_c, prev_ids, _ = self.window[-1]
            motion = _median_shift(prev_ids, prev_c, ids, corners)
            if motion is None or motion > self.still_px:
                self.window.clear()
                self.window.append((corners, ids, t_soe_ms))
                self.status = "moving" if motion is not None else "board changed"
                return None
        self.window.append((corners, ids, t_soe_ms))
        if len(self.window) < self.still_frames:
            self.status = "hold still"
            return None

        # Average over the still window, using corners seen in every frame.
        common = set(self.window[0][1].tolist())
        for _, i, _ in self.window:
            common &= set(i.tolist())
        common_ids = np.array(sorted(common), np.int32)
        if len(common_ids) < self.min_corners:
            self.status = "corners flicker"
            return None
        stack = []
        for c, i, _ in self.window:
            lookup = dict(zip(i.tolist(), c))
            stack.append([lookup[k] for k in common_ids])
        mean = np.mean(np.asarray(stack, np.float64), axis=0).astype(np.float32)
        t_mid = float(np.mean([t for _, _, t in self.window]))

        candidate = dict(zip(common_ids.tolist(), mean))
        for view in self.accepted:
            shared = [k for k in candidate if k in view]
            if len(shared) >= 0.5 * min(len(candidate), len(view)):
                d = np.median([np.linalg.norm(candidate[k] - view[k]) for k in shared])
                if d < self.novel_px:
                    self.status = "already have this view: move or tilt the board"
                    return None
        self.accepted.append(candidate)
        self.window.clear()
        self.status = "captured"
        return common_ids, mean, t_mid, self.still_frames

    def undo(self) -> None:
        if self.accepted:
            self.accepted.pop()


def _median_shift(ids_a, c_a, ids_b, c_b) -> float | None:
    a = dict(zip(ids_a.tolist(), c_a))
    shared = [(a[k], c) for k, c in zip(ids_b.tolist(), c_b) if k in a]
    if len(shared) < 0.8 * len(ids_b):
        return None
    return float(np.median([np.linalg.norm(p - q) for p, q in shared]))


class Coverage:
    """Counts corners per image cell (about 80 px square) and per image-corner zone.

    The corner zones are the outer CORNER_ZONE fraction of the width and height at each image
    corner: where distortion is strongest. The tracking views keep the eye and fingertip
    inside an 8 % margin, so corners reaching into a 12 % x 15 % zone cover everything used.
    """

    CORNER_ZONE = (0.12, 0.15)

    def __init__(self, image_size: tuple[int, int], cell_px: int = 80):
        w, h = image_size
        self.size = image_size
        self.cols, self.rows = max(1, round(w / cell_px)), max(1, round(h / cell_px))
        self.counts = np.zeros((self.rows, self.cols), np.int32)
        self.zones = {"top-left": 0, "top-right": 0, "bottom-left": 0, "bottom-right": 0}

    def _cells(self, corners: np.ndarray):
        w, h = self.size
        cx = np.clip((corners[:, 0] / w * self.cols).astype(int), 0, self.cols - 1)
        cy = np.clip((corners[:, 1] / h * self.rows).astype(int), 0, self.rows - 1)
        return cy, cx

    def add(self, corners: np.ndarray, sign: int = 1) -> None:
        np.add.at(self.counts, self._cells(corners), sign)
        w, h = self.size
        zx, zy = self.CORNER_ZONE[0] * w, self.CORNER_ZONE[1] * h
        x, y = corners[:, 0], corners[:, 1]
        left, right, top, bottom = x < zx, x > w - zx, y < zy, y > h - zy
        for name, m in (("top-left", top & left), ("top-right", top & right),
                        ("bottom-left", bottom & left), ("bottom-right", bottom & right)):
            self.zones[name] += sign * int(m.any())

    def fraction(self) -> float:
        return float((self.counts > 0).mean())

    def missing_corners(self) -> list[str]:
        """Image corner zones (in the real, unmirrored image) with no board corner yet."""
        return [name for name, n in self.zones.items() if n <= 0]

    def complete(self) -> bool:
        return self.fraction() >= GATES["coverage_min"] and not self.missing_corners()

    def overlay(self, image: np.ndarray) -> np.ndarray:
        """Tint covered cells green, outline empty ones red, outline the corner zones."""
        out = image.copy()
        h, w = image.shape[:2]
        tint = np.zeros_like(out)
        for r in range(self.rows):
            for c in range(self.cols):
                x0, x1 = c * w // self.cols, (c + 1) * w // self.cols
                y0, y1 = r * h // self.rows, (r + 1) * h // self.rows
                if self.counts[r, c] > 0:
                    tint[y0:y1, x0:x1] = (0, min(255, 60 + 30 * int(self.counts[r, c])), 0)
                else:
                    cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 200), 1)
        out = cv2.addWeighted(out, 1.0, tint, 0.35, 0)
        zx, zy = int(self.CORNER_ZONE[0] * w), int(self.CORNER_ZONE[1] * h)
        for name, (x0, y0) in (("top-left", (0, 0)), ("top-right", (w - zx, 0)),
                               ("bottom-left", (0, h - zy)), ("bottom-right", (w - zx, h - zy))):
            colour = (0, 220, 0) if self.zones[name] > 0 else (0, 0, 255)
            cv2.rectangle(out, (x0 + 1, y0 + 1), (x0 + zx - 2, y0 + zy - 2), colour, 3)
        return out


TILT_MIN_DEG = 20.0
TILT_PER_DIRECTION = 3


def board_tilt(spec: BoardSpec, ids: np.ndarray, corners: np.ndarray,
               image_size: tuple[int, int]) -> dict[str, float] | None:
    """How far each edge of the board is turned away from the camera, in degrees.

    Uses a rough camera (f = 0.75 * width, no distortion): good to a few degrees, which is
    all the guidance needs. "left"/"right" are the sides as seen in the **mirrored** live
    view, "top"/"bottom" as seen. Returns e.g. {"left": 27.0, "top": 0.0, ...}, where a
    positive value means that edge is farther from the camera than the opposite one.
    """
    w, h = image_size
    K = np.array([[0.75 * w, 0, w / 2], [0, 0.75 * w, h / 2], [0, 0, 1]])
    obj = spec.board().getChessboardCorners()[ids].astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, corners.astype(np.float64), K, None,
                                  flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    R = cv2.Rodrigues(rvec)[0]
    lo, hi = obj.min(0), obj.max(0)
    mid = (lo + hi) / 2
    ends = {"x0": [lo[0], mid[1], 0], "x1": [hi[0], mid[1], 0],
            "y0": [mid[0], lo[1], 0], "y1": [mid[0], hi[1], 0]}
    P = {k: R @ np.array(v) + tvec.ravel() for k, v in ends.items()}

    def turned(a, b):          # degrees that point a is farther than point b
        d = P[a] - P[b]
        return float(np.degrees(np.arctan2(d[2], np.hypot(d[0], d[1]))))

    # Which board end appears on the left of the mirrored view: the one with larger image x.
    a, b = ("x1", "x0") if P["x1"][0] / P["x1"][2] > P["x0"][0] / P["x0"][2] else ("x0", "x1")
    c, d = ("y0", "y1") if P["y0"][1] / P["y0"][2] < P["y1"][1] / P["y1"][2] else ("y1", "y0")
    side, vert = turned(a, b), turned(c, d)
    return {"left": max(side, 0.0), "right": max(-side, 0.0),
            "top": max(vert, 0.0), "bottom": max(-vert, 0.0)}


class TiltTracker:
    """Counts captured views turned at least TILT_MIN_DEG in each of the four directions.
    Frontal views alone leave the focal length poorly determined."""

    def __init__(self):
        self.counts = {"left": 0, "right": 0, "top": 0, "bottom": 0}

    def add(self, tilt: dict[str, float] | None, sign: int = 1) -> None:
        if tilt:
            for k, v in tilt.items():
                if v >= TILT_MIN_DEG:
                    self.counts[k] += sign

    def complete(self) -> bool:
        return all(n >= TILT_PER_DIRECTION for n in self.counts.values())

    def summary(self) -> str:
        return "  ".join(f"{k} away {min(n, TILT_PER_DIRECTION)}/{TILT_PER_DIRECTION}"
                         for k, n in self.counts.items())


# ---------------------------------------------------------------------------------------------
# Solving


@dataclass
class Fit:
    K: np.ndarray
    dist: np.ndarray
    rms: float
    std_intrinsics: np.ndarray
    per_view: np.ndarray
    rvecs: list
    tvecs: list
    fix_k3: bool

    @property
    def model(self) -> str:
        return "k1 k2 p1 p2" if self.fix_k3 else "k1 k2 p1 p2 k3"


def fit(views: list[View], image_size: tuple[int, int], fix_k3: bool = True) -> Fit:
    flags = cv2.CALIB_FIX_K3 if fix_k3 else 0
    obj = [v.obj.reshape(-1, 1, 3) for v in views]
    img = [v.corners.reshape(-1, 1, 2) for v in views]
    rms, K, dist, rvecs, tvecs, std_in, _, per_view = cv2.calibrateCameraExtended(
        obj, img, image_size, None, None, flags=flags, criteria=CRITERIA)
    return Fit(K, dist.ravel()[:5], float(rms), std_in.ravel(), per_view.ravel(),
               list(rvecs), list(tvecs), fix_k3)


def reprojection_rms(views: list[View], K: np.ndarray, dist: np.ndarray) -> float:
    """RMS over all corners after fitting only each view's pose (for views not in the fit)."""
    sq, n = 0.0, 0
    for v in views:
        ok, rvec, tvec = cv2.solvePnP(v.obj, v.corners, K, dist, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            continue
        rvec, tvec = cv2.solvePnPRefineLM(v.obj, v.corners, K, dist, rvec, tvec)
        proj, _ = cv2.projectPoints(v.obj, rvec, tvec, K, dist)
        sq += float(np.sum((proj.reshape(-1, 2) - v.corners) ** 2))
        n += len(v.corners)
    return float(np.sqrt(sq / n)) if n else float("nan")


def distort_normalized(xy: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """OpenCV's k1 k2 p1 p2 k3 model applied to normalised coordinates."""
    k1, k2, p1, p2, k3 = (list(np.ravel(dist)) + [0.0] * 5)[:5]
    x, y = xy[:, 0], xy[:, 1]
    r2 = x * x + y * y
    radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
    return np.stack([x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
                     y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y], 1)


def undistort_normalized(px: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Pixels -> normalised image coordinates (x/z, y/z).

    OpenCV's fixed-point undistortion stops ~0.01 px short with strong distortion, so it is
    only the starting point for Newton iterations on the exact model (to < 1e-9 px).
    """
    px = np.asarray(px, np.float64).reshape(-1, 2)
    target = (px - K[:2, 2]) / np.diag(K)[:2]       # calibrateCamera never fits skew
    xy = cv2.undistortPoints(px.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    if not np.any(np.ravel(dist)):
        return xy
    h = 1e-7
    for _ in range(10):
        r = distort_normalized(xy, dist) - target
        jx = (distort_normalized(xy + [h, 0], dist) - distort_normalized(xy - [h, 0], dist)) / (2 * h)
        jy = (distort_normalized(xy + [0, h], dist) - distort_normalized(xy - [0, h], dist)) / (2 * h)
        det = jx[:, 0] * jy[:, 1] - jy[:, 0] * jx[:, 1]
        dx = (jy[:, 1] * r[:, 0] - jy[:, 0] * r[:, 1]) / det
        dy = (-jx[:, 1] * r[:, 0] + jx[:, 0] * r[:, 1]) / det
        xy = xy - np.stack([dx, dy], 1)
        if np.abs(r).max() < 1e-13:
            break
    return xy


def model_difference_px(K1, d1, K2, d2, image_size, step: int = 16,
                        allow_rotation: bool = True,
                        margin: float = 0.0) -> tuple[float, float]:
    """How differently two calibrations map pixels to rays: take each pixel's ray under
    model 1, project it with model 2, and measure how far it lands (max, RMS) in pixels.

    With `allow_rotation`, a best-fit small camera rotation is removed first. A difference
    that is a pure rotation (for example a principal point shift with matching focal
    lengths) only re-aims the camera, and the extrinsic calibrations (C4, C5), made with
    the same intrinsics, absorb it. What remains warps the angles between rays, and that
    does reach the 3D points.
    """
    w, h = image_size
    gx, gy = np.meshgrid(np.arange(0, w, step) + 0.5, np.arange(0, h, step) + 0.5)
    px = np.stack([gx.ravel(), gy.ravel()], 1)
    if margin > 0:       # only the tracking area: margin * size in from every edge
        keep = ((px[:, 0] > margin * w) & (px[:, 0] < (1 - margin) * w)
                & (px[:, 1] > margin * h) & (px[:, 1] < (1 - margin) * h))
        px = px[keep]
    n = undistort_normalized(px, K1, d1)
    rays = np.hstack([n, np.ones((len(n), 1))])

    def err(rvec):
        proj, _ = cv2.projectPoints(rays, rvec, np.zeros(3), K2, d2)
        return (proj.reshape(-1, 2) - px).ravel()

    rvec = np.zeros(3)
    if allow_rotation:
        from scipy.optimize import least_squares
        rvec = least_squares(err, rvec, x_scale=1e-3).x
    e = np.linalg.norm(err(rvec).reshape(-1, 2), axis=1)
    return float(e.max()), float(np.sqrt(np.mean(e**2)))


def view_tilt_deg(rvec) -> float:
    """Angle between the board normal and the camera's optical axis."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64))
    return float(np.degrees(np.arccos(min(1.0, abs(R[2, 2])))))


def _reject_outliers(views, image_size, fix_k3, max_rounds: int = 3):
    """Drop views whose own RMS is far above the median. At most 10 % of views are dropped."""
    rejected: list[str] = []
    limit = max(1, len(views) // 10)
    for _ in range(max_rounds):
        f = fit(views, image_size, fix_k3)
        med = float(np.median(f.per_view))
        bad = [i for i, e in enumerate(f.per_view) if e > max(3 * med, med + 0.3)]
        bad = sorted(bad, key=lambda i: -f.per_view[i])[: limit - len(rejected)]
        if not bad:
            return views, rejected, f
        rejected += [views[i].name for i in bad]
        views = [v for i, v in enumerate(views) if i not in set(bad)]
    return views, rejected, fit(views, image_size, fix_k3)


def bootstrap_ray_uncertainty(views, image_size, fix_k3: bool, ref: Fit, rng,
                              n: int = BOOTSTRAP_RESAMPLES) -> tuple[float, float]:
    """How much the pixel-to-ray mapping inside the tracking area moves when the views are
    resampled with replacement (rotation removed, as in `model_difference_px`).
    Returns (RMS over the area, max over the area) of the per-point RMS spread, in px."""
    w, h = image_size
    step = 32
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(views), len(views))
        f = fit([views[i] for i in idx], image_size, fix_k3)
        gx, gy = np.meshgrid(np.arange(0, w, step) + 0.5, np.arange(0, h, step) + 0.5)
        px = np.stack([gx.ravel(), gy.ravel()], 1)
        m = TRACKING_MARGIN
        px = px[(px[:, 0] > m * w) & (px[:, 0] < (1 - m) * w)
                & (px[:, 1] > m * h) & (px[:, 1] < (1 - m) * h)]
        rays = np.hstack([undistort_normalized(px, ref.K, ref.dist), np.ones((len(px), 1))])

        def err(r):
            p, _ = cv2.projectPoints(rays, r, np.zeros(3), f.K, f.dist)
            return p.reshape(-1, 2) - px

        from scipy.optimize import least_squares
        r = least_squares(lambda r: err(r).ravel(), np.zeros(3), x_scale=1e-3).x
        diffs.append(np.linalg.norm(err(r), axis=1))
    spread = np.sqrt(np.mean(np.square(diffs), axis=0))
    return float(np.sqrt(np.mean(spread**2))), float(spread.max())


def solve(views: list[View], image_size: tuple[int, int], seed: int = 0) -> dict:
    """Full C1/C2 solve. Returns a JSON-ready report including gates and `passed`."""
    if len(views) < 10:
        raise ValueError(f"need at least 10 views, have {len(views)}")
    rng = np.random.default_rng(seed)
    views, rejected, _ = _reject_outliers(views, image_size, fix_k3=True)

    # Model choice on held-out views: k3 only if it predicts unseen views better.
    order = rng.permutation(len(views))
    n_hold = max(3, round(0.2 * len(views)))
    hold = [views[i] for i in order[:n_hold]]
    train = [views[i] for i in order[n_hold:]]
    holdout = {}
    for fix_k3 in (True, False):
        f = fit(train, image_size, fix_k3)
        holdout[fix_k3] = reprojection_rms(hold, f.K, f.dist)
    fix_k3 = not (holdout[False] < 0.98 * holdout[True] - 0.005)

    final = fit(views, image_size, fix_k3)

    # Consistency: two disjoint random halves should describe the same lens.
    order = rng.permutation(len(views))
    half_a = fit([views[i] for i in order[: len(views) // 2]], image_size, fix_k3)
    half_b = fit([views[i] for i in order[len(views) // 2:]], image_size, fix_k3)
    split_max, split_rms = model_difference_px(half_a.K, half_a.dist, half_b.K, half_b.dist,
                                               image_size)
    split_raw_max, _ = model_difference_px(half_a.K, half_a.dist, half_b.K, half_b.dist,
                                           image_size, allow_rotation=False)

    coverage = Coverage(image_size)
    for v in views:
        coverage.add(v.corners)
    tilts = [view_tilt_deg(r) for r in final.rvecs]

    ray_rms, ray_max = bootstrap_ray_uncertainty(views, image_size, fix_k3, final, rng)

    K, std = final.K, final.std_intrinsics
    gates = [
        _gate("ray uncertainty in the tracking area, RMS (px)", ray_rms,
              GATES["ray_uncertainty_rms_px"]),
        _gate("ray uncertainty in the tracking area, max (px)", ray_max,
              GATES["ray_uncertainty_max_px"]),
        _gate("reprojection RMS (px)", final.rms, GATES["rms_px"]),
        _gate("hold-out RMS / fit RMS", holdout[fix_k3] / final.rms, GATES["holdout_ratio"]),
        _gate("image coverage (fraction of cells)", coverage.fraction(), GATES["coverage_min"],
              minimum=True),
        _gate("image corner zones missing", len(coverage.missing_corners()), 0),
    ]
    return {
        "model": final.model,
        "image_size": list(image_size),
        "K": K.tolist(),
        "dist": final.dist.tolist(),
        "std": {"fx": std[0], "fy": std[1], "cx": std[2], "cy": std[3],
                "dist": std[4:9].tolist()},
        "rms_px": final.rms,
        "holdout_rms_px": {"k1 k2 p1 p2": holdout[True], "k1 k2 p1 p2 k3": holdout[False]},
        "ray_uncertainty_px": {"rms": ray_rms, "max": ray_max, "margin": TRACKING_MARGIN,
                               "resamples": BOOTSTRAP_RESAMPLES},
        "split_half_px": {"max": split_max, "rms": split_rms, "max_before_rotation": split_raw_max},
        "views_used": len(views),
        "views_rejected": rejected,
        "corners_used": int(sum(len(v.ids) for v in views)),
        "coverage": {"fraction": coverage.fraction(), "missing_corners": coverage.missing_corners()},
        "tilt_deg": {"median": float(np.median(tilts)), "max": float(np.max(tilts)),
                     "views_over_20": int(np.sum(np.array(tilts) > 20))},
        "per_view": [{"name": v.name, "rms_px": float(e), "tilt_deg": t}
                     for v, e, t in zip(views, final.per_view, tilts)],
        "gates": gates,
        "passed": all(g["pass"] for g in gates),
    }


def _gate(name: str, value: float, limit: float, minimum: bool = False) -> dict:
    ok = value >= limit if minimum else value <= limit
    return {"name": name, "value": float(value), "limit": limit,
            "kind": "min" if minimum else "max", "pass": bool(ok)}


def format_report(r: dict) -> str:
    K = np.asarray(r["K"])
    lines = [
        f"model {r['model']}, {r['views_used']} views, {r['corners_used']} corners"
        + (f", rejected {len(r['views_rejected'])}: {', '.join(r['views_rejected'])}"
           if r["views_rejected"] else ""),
        f"fx {K[0, 0]:.2f} ± {r['std']['fx']:.2f}   fy {K[1, 1]:.2f} ± {r['std']['fy']:.2f}   "
        f"cx {K[0, 2]:.2f} ± {r['std']['cx']:.2f}   cy {K[1, 2]:.2f} ± {r['std']['cy']:.2f}",
        "dist " + "  ".join(f"{d:+.5f}" for d in r["dist"]),
        "hold-out RMS: " + ", ".join(f"[{m}] {v:.3f} px" for m, v in r["holdout_rms_px"].items()),
        f"board tilt: median {r['tilt_deg']['median']:.0f}°, max {r['tilt_deg']['max']:.0f}°, "
        f"{r['tilt_deg']['views_over_20']} views over 20°",
        "",
    ]
    for g in r["gates"]:
        op = "≥" if g["kind"] == "min" else "≤"
        lines.append(f"  {'PASS' if g['pass'] else 'FAIL'}  {g['name']}: {g['value']:.3f}"
                     f"  (needs {op} {g['limit']})")
    lines.append(f"\n{'ALL GATES PASSED' if r['passed'] else 'GATES FAILED'}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# Session files


@dataclass
class Session:
    """Everything captured in one calibration run, saved as <raw_dir>/session.json."""
    camera: str
    usb_id: str
    image_size: tuple[int, int]
    fourcc: str
    controls: dict[str, int]          # verified read-back at capture time
    board: dict
    scale: dict
    views: list[View] = field(default_factory=list)
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def save(self, raw_dir: Path) -> Path:
        raw_dir.mkdir(parents=True, exist_ok=True)
        d = asdict(self)
        d["views"] = [v.to_json() for v in self.views]
        path = raw_dir / "session.json"
        path.write_text(json.dumps(d, indent=1))
        return path

    @classmethod
    def load(cls, raw_dir: Path) -> "Session":
        d = json.loads((raw_dir / "session.json").read_text())
        d["views"] = [View.from_json(v) for v in d["views"]]
        d["image_size"] = tuple(d["image_size"])
        return cls(**d)


def result_path(camera: str) -> Path:
    return PROJECT_ROOT / "calib" / f"intrinsics_{camera}.json"


def build_result(session: Session, report: dict, raw_dir: Path) -> dict:
    try:
        raw = str(raw_dir.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        raw = str(raw_dir)
    return {
        "camera": session.camera,
        "usb_id": session.usb_id,
        "mode": {"width": session.image_size[0], "height": session.image_size[1],
                 "fourcc": session.fourcc},
        "geometry_controls": {k: v for k, v in session.controls.items() if k in GEOMETRY_CONTROLS},
        "controls_at_capture": session.controls,
        "board": session.board,
        "scale": session.scale,
        "captured": session.created,
        "solved": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "raw_dir": raw,
        **report,
    }


def install(result: dict) -> Path:
    """Write calib/intrinsics_<camera>.json, moving any previous result to calib/history/."""
    path = result_path(result["camera"])
    if path.exists():
        history = path.parent / "history"
        history.mkdir(exist_ok=True)
        path.rename(history / f"{path.stem}.{time.strftime('%Y%m%d-%H%M%S')}.json")
    path.write_text(json.dumps(result, indent=1))
    return path


@dataclass(frozen=True)
class Intrinsics:
    camera: str
    image_size: tuple[int, int]
    K: np.ndarray
    dist: np.ndarray


class StaleCalibration(RuntimeError):
    pass


def check_matches(result: dict, config: CameraConfig) -> None:
    """Raise if the camera is not configured exactly as it was when calibrated."""
    problems = []
    mode = result["mode"]
    if (mode["width"], mode["height"], mode["fourcc"]) != (config.width, config.height,
                                                           config.fourcc):
        problems.append(f"mode {mode} != config {config.width}x{config.height} {config.fourcc}")
    if result["usb_id"] != config.usb_id:
        problems.append(f"USB id {result['usb_id']} != config {config.usb_id}")
    for name in GEOMETRY_CONTROLS:
        was, now = result["geometry_controls"].get(name), config.controls.get(name)
        if was != now:
            problems.append(f"{name} was {was} at calibration, config has {now}")
    if problems:
        raise StaleCalibration(f"intrinsics_{result['camera']}.json does not match the camera "
                               "config: " + "; ".join(problems)
                               + " (see docs/RECALIBRATION.md)")


def load_intrinsics(config: CameraConfig, path: Path | None = None,
                    allow_failed: bool = False) -> Intrinsics:
    result = json.loads((path or result_path(config.name)).read_text())
    check_matches(result, config)
    if not result["passed"] and not allow_failed:
        raise StaleCalibration(f"intrinsics_{config.name}.json did not pass its gates")
    return Intrinsics(config.name, tuple(result["image_size"]),
                      np.asarray(result["K"], np.float64), np.asarray(result["dist"], np.float64))
