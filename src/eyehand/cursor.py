"""Pure eye/fingertip triangulation and screen intersection for the cursor runtime."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from eyehand import stereo
from eyehand.intrinsics import Intrinsics, undistort_normalized

MAX_REPROJECTION_PX = 4.0
MIN_RAY_ANGLE_DEG = 5.0


class InvalidPointer(ValueError):
    pass


@dataclass(frozen=True)
class PointerPosition:
    x_norm: float
    y_norm: float
    reprojection_px: float
    ray_angle_deg: float
    hand_mm: tuple[float, float, float]


def _project(points: np.ndarray, intr: Intrinsics) -> np.ndarray:
    px, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), intr.K, intr.dist)
    return px.reshape(-1, 2)


def _triangulate_checked(px_L, px_F, intr_L, intr_F, T_F_L):
    if not np.isfinite(px_L).all() or not np.isfinite(px_F).all():
        raise InvalidPointer("non-finite landmarks")
    n_L = undistort_normalized(px_L, intr_L.K, intr_L.dist)
    n_F = undistort_normalized(px_F, intr_F.K, intr_F.dist)
    points = stereo.triangulate(T_F_L, n_L, n_F)
    points_F = (T_F_L[:3, :3] @ points.T).T + T_F_L[:3, 3]
    if not np.isfinite(points).all() or np.any(points[:, 2] <= 0) or np.any(points_F[:, 2] <= 0):
        raise InvalidPointer("triangulation is behind a camera")
    error = max(float(np.linalg.norm(_project(points, intr_L) - px_L, axis=1).max()),
                float(np.linalg.norm(_project(points_F, intr_F) - px_F, axis=1).max()))
    if error > MAX_REPROJECTION_PX:
        raise InvalidPointer(f"reprojection error {error:.1f} px")
    rays_L = np.column_stack((n_L, np.ones(len(n_L))))
    rays_F = (T_F_L[:3, :3].T @ np.column_stack((n_F, np.ones(len(n_F)))).T).T
    rays_L /= np.linalg.norm(rays_L, axis=1, keepdims=True)
    rays_F /= np.linalg.norm(rays_F, axis=1, keepdims=True)
    angle = float(np.degrees(np.arccos(np.clip(np.sum(rays_L * rays_F, axis=1), -1, 1))).min())
    if angle < MIN_RAY_ANGLE_DEG:
        raise InvalidPointer(f"camera ray angle {angle:.1f} deg")
    return points, error, angle


class FingertipStabilizer:
    """Carry fingertips with the palm; gently update their measured palm-relative offsets."""

    def __init__(self):
        self.hands = {}

    def stabilize(self, hand, t_ms, palm_L, palm_F, px_L, px_F, intr_L, intr_F, T_F_L):
        previous = self.hands.get(hand)
        try:
            palm, _, _ = _triangulate_checked(palm_L, palm_F, intr_L, intr_F, T_F_L)
            if np.linalg.svd(palm - palm.mean(axis=0), compute_uv=False)[1] < 5:
                raise InvalidPointer("palm landmarks are degenerate")
        except InvalidPointer:
            self.hands.pop(hand, None)
            return px_L, px_F, False

        if previous is not None and not 0 <= t_ms - previous[0] <= 150:
            previous = None
        predicted = None
        if previous is not None:
            last_t, good_t, last_palm, last_tips = previous
            a, b = last_palm - last_palm.mean(axis=0), palm - palm.mean(axis=0)
            U, _, Vt = np.linalg.svd(a.T @ b)
            R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
            # Reject a hand mismatch or a palm pose that cannot be explained by rigid motion.
            if np.linalg.norm(a @ R - b, axis=1).max() <= 15:
                predicted = (last_tips - last_palm.mean(axis=0)) @ R + palm.mean(axis=0)
            else:
                previous = None
        try:
            tips, _, _ = _triangulate_checked(px_L[1:], px_F[1:], intr_L, intr_F, T_F_L)
        except InvalidPointer:
            if previous is None or predicted is None or t_ms - good_t > 150:
                self.hands.pop(hand, None)
                return px_L, px_F, False
            tips = predicted
            predicting = True
        else:
            good_t = t_ms
            predicting = False
            if predicted is not None:
                alpha = -np.expm1(-(t_ms - last_t) / 60)
                tips = predicted + alpha * (tips - predicted)
        self.hands[hand] = (t_ms, good_t, palm.copy(), tips.copy())
        tips_F = (T_F_L[:3, :3] @ tips.T).T + T_F_L[:3, 3]
        return (np.vstack((px_L[0], _project(tips, intr_L))),
                np.vstack((px_F[0], _project(tips_F, intr_F))), predicting)


def solve_pointer(px_L: np.ndarray, px_F: np.ndarray, intr_L: Intrinsics,
                  intr_F: Intrinsics, T_F_L: np.ndarray, T_S_L: np.ndarray,
                  screen_mm: tuple[float, float], *, thumb_weight: float = 0.75) -> PointerPosition:
    """Triangulate eye and tips, then aim through the estimated pinch point.

    Inputs are [eye, index] or [eye, index, thumb], in camera pixels.
    """
    if not np.isfinite(thumb_weight) or not 0 <= thumb_weight <= 1:
        raise InvalidPointer("thumb weight must be between 0 and 1")
    px_L = np.asarray(px_L, float).reshape(-1, 2)
    px_F = np.asarray(px_F, float).reshape(-1, 2)
    if px_L.shape != px_F.shape or len(px_L) not in (2, 3):
        raise InvalidPointer("expected matching eye and fingertip observations")
    points_L, reprojection, ray_angle = _triangulate_checked(px_L, px_F, intr_L, intr_F, T_F_L)

    points_S = (T_S_L[:3, :3] @ points_L.T).T + T_S_L[:3, 3]
    eye, tip = points_S[:2]
    if len(points_S) == 3:
        tip = (1 - thumb_weight) * tip + thumb_weight * points_S[2]
    if eye[2] >= tip[2] or tip[2] >= 0:
        raise InvalidPointer("finger is not between eye and screen")
    s = eye[2] / (eye[2] - tip[2])
    hit = eye + s * (tip - eye)
    x, y = float(hit[0] / screen_mm[0]), float(hit[1] / screen_mm[1])
    if not np.isfinite((x, y)).all() or not (0 <= x <= 1 and 0 <= y <= 1):
        raise InvalidPointer("aim is outside the screen")
    return PointerPosition(x, y, reprojection, ray_angle, tuple(float(v) for v in tip))
