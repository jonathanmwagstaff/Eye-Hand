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


def _project(points: np.ndarray, intr: Intrinsics) -> np.ndarray:
    px, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), intr.K, intr.dist)
    return px.reshape(-1, 2)


def solve_pointer(px_L: np.ndarray, px_F: np.ndarray, intr_L: Intrinsics,
                  intr_F: Intrinsics, T_F_L: np.ndarray, T_S_L: np.ndarray,
                  screen_mm: tuple[float, float]) -> PointerPosition:
    """Triangulate eye and tips, then aim through the estimated pinch point.

    Inputs are [eye, index] or [eye, index, thumb], in camera pixels.
    """
    px_L = np.asarray(px_L, float).reshape(-1, 2)
    px_F = np.asarray(px_F, float).reshape(-1, 2)
    if px_L.shape != px_F.shape or len(px_L) not in (2, 3):
        raise InvalidPointer("expected matching eye and fingertip observations")
    if not np.isfinite(px_L).all() or not np.isfinite(px_F).all():
        raise InvalidPointer("non-finite landmarks")
    n_L = undistort_normalized(px_L, intr_L.K, intr_L.dist)
    n_F = undistort_normalized(px_F, intr_F.K, intr_F.dist)
    points_L = stereo.triangulate(T_F_L, n_L, n_F)
    points_F = (T_F_L[:3, :3] @ points_L.T).T + T_F_L[:3, 3]
    if not np.isfinite(points_L).all() or np.any(points_L[:, 2] <= 0) or np.any(points_F[:, 2] <= 0):
        raise InvalidPointer("triangulation is behind a camera")

    reprojection = max(float(np.linalg.norm(_project(points_L, intr_L) - px_L, axis=1).max()),
                       float(np.linalg.norm(_project(points_F, intr_F) - px_F, axis=1).max()))
    if reprojection > MAX_REPROJECTION_PX:
        raise InvalidPointer(f"reprojection error {reprojection:.1f} px")

    rays_L = np.column_stack((n_L, np.ones(len(n_L))))
    rays_F_in_L = (T_F_L[:3, :3].T @ np.column_stack((n_F, np.ones(len(n_F)))).T).T
    rays_L /= np.linalg.norm(rays_L, axis=1, keepdims=True)
    rays_F_in_L /= np.linalg.norm(rays_F_in_L, axis=1, keepdims=True)
    angles = np.degrees(np.arccos(np.clip(np.sum(rays_L * rays_F_in_L, axis=1), -1, 1)))
    ray_angle = float(angles.min())
    if ray_angle < MIN_RAY_ANGLE_DEG:
        raise InvalidPointer(f"camera ray angle {ray_angle:.1f} deg")

    points_S = (T_S_L[:3, :3] @ points_L.T).T + T_S_L[:3, 3]
    eye, tip = points_S[:2]
    if len(points_S) == 3:
        tip = 0.25 * tip + 0.75 * points_S[2]
    if eye[2] >= tip[2] or tip[2] >= 0:
        raise InvalidPointer("finger is not between eye and screen")
    s = eye[2] / (eye[2] - tip[2])
    hit = eye + s * (tip - eye)
    x, y = float(hit[0] / screen_mm[0]), float(hit[1] / screen_mm[1])
    if not np.isfinite((x, y)).all() or not (0 <= x <= 1 and 0 <= y <= 1):
        raise InvalidPointer("aim is outside the screen")
    return PointerPosition(x, y, reprojection, ray_angle)
