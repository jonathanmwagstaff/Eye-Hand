"""Stereo extrinsics (C5): where camera F is relative to camera L.

This is the calibration that has to be redone whenever the tripod, the laptop or the lid
angle moves (docs/RECALIBRATION.md). Intrinsics from C1/C2 are held fixed.

- `StereoSelector` accepts a view only when the board is still in **both** cameras at the
  same time. The cameras are not synchronised (30.07 vs 30.00 fps), but a board that is still
  in both looks the same whichever frames are paired.
- `solve()` runs a bundle adjustment of `T_F_L` and every board pose together, minimising
  pixel reprojection error in both cameras, then evaluates the gates: reprojection RMS,
  epipolar error of held-out views, and the known-length test.

`T_A_B` maps points in frame B into frame A: X_A = R_A_B X_B + t_A_B (docs/ARCHITECTURE.md).
"""

from __future__ import annotations

import hashlib
import json
import time
import warnings
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from eyehand.config import PROJECT_ROOT
from eyehand.intrinsics import Intrinsics, _median_shift, undistort_normalized

# The first version gated on reprojection RMS < 0.5 px and held-out epipolar RMS < 0.5 px,
# chosen in simulation with 0.15 px corner noise. These webcams have ~0.33 px per axis
# (C1/C2), which puts both near 0.5 px for perfectly good data (2026-10-27, run
# 20261027-222013: 0.48 / 0.45 px, epipolar 0.60 px, known lengths within 0.30 mm). So
# epipolar consistency is now judged against the fit's own in-sample level, and a separate
# gate catches a camera that moves during the capture, which is what actually went wrong.
GATES = {
    "rms_px": 0.7,                    # sanity: blur, bad detections
    "holdout_epipolar_ratio": 1.3,    # held-out epipolar RMS / in-sample epipolar RMS
    "epipolar_px": 1.0,               # sanity on the held-out value itself
    "drift_deg": 0.3,                 # camera rotation between first and last third of views
    "length_p95_mm": 1.0,
    "min_views": 20,
}
MIN_CORNERS = 12
LENGTH_MIN_SQUARES = 6          # corner pairs at least this far apart are used for lengths


def intrinsics_fingerprint(intr: Intrinsics) -> str:
    data = json.dumps([np.round(intr.K, 6).tolist(), np.round(intr.dist, 9).tolist()])
    return hashlib.sha256(data.encode()).hexdigest()[:16]


@dataclass
class StereoView:
    name: str
    ids_L: np.ndarray
    corners_L: np.ndarray
    ids_F: np.ndarray
    corners_F: np.ndarray
    t_L_ms: float = 0.0
    t_F_ms: float = 0.0

    def to_json(self) -> dict:
        return {"name": self.name, "ids_L": self.ids_L.tolist(), "corners_L": self.corners_L.tolist(),
                "ids_F": self.ids_F.tolist(), "corners_F": self.corners_F.tolist(),
                "t_L_ms": self.t_L_ms, "t_F_ms": self.t_F_ms}

    @classmethod
    def from_json(cls, d: dict) -> "StereoView":
        return cls(d["name"], np.asarray(d["ids_L"], np.int32), np.asarray(d["corners_L"], np.float32),
                   np.asarray(d["ids_F"], np.int32), np.asarray(d["corners_F"], np.float32),
                   d.get("t_L_ms", 0.0), d.get("t_F_ms", 0.0))


class _Still:
    """Board stillness in one camera over the last `n` frames; averages the corners."""

    def __init__(self, n: int, still_px: float):
        self.n, self.still_px = n, still_px
        self.window: deque = deque(maxlen=n)

    def offer(self, corners, ids, t_ms) -> str:
        if ids is None or len(ids) < MIN_CORNERS:
            self.window.clear()
            return "no board"
        if self.window:
            c0, i0, _ = self.window[-1]
            motion = _median_shift(i0, c0, ids, corners)
            if motion is None or motion > self.still_px:
                self.window.clear()
                self.window.append((corners, ids, t_ms))
                return "moving"
        self.window.append((corners, ids, t_ms))
        return "still" if len(self.window) == self.n else "hold still"

    def average(self):
        if len(self.window) < self.n:
            return None
        common = set(self.window[0][1].tolist())
        for _, i, _ in self.window:
            common &= set(i.tolist())
        ids = np.array(sorted(common), np.int32)
        if len(ids) < MIN_CORNERS:
            return None
        stack = [[dict(zip(i.tolist(), c))[k] for k in ids] for c, i, _ in self.window]
        return ids, np.mean(np.asarray(stack, np.float64), 0).astype(np.float32), \
            float(np.mean([t for _, _, t in self.window])), self.window[0][2], self.window[-1][2]


class StereoSelector:
    """Feeds per-camera detections (from any thread, under the caller's lock) and returns a
    StereoView when the board is still in both cameras over overlapping time spans."""

    def __init__(self, still_px: float = 0.5, still_frames: int = 4, novel_px: float = 40.0):
        self.still = {"L": _Still(still_frames, still_px), "F": _Still(still_frames, still_px)}
        self.status = {"L": "no board", "F": "no board"}
        self.novel_px = novel_px
        self.accepted: list[dict[str, dict[int, np.ndarray]]] = []

    def offer(self, camera: str, corners, ids, t_ms: float) -> StereoView | None:
        self.status[camera] = self.still[camera].offer(corners, ids, t_ms)
        if self.status["L"] != "still" or self.status["F"] != "still":
            return None
        a, b = self.still["L"].average(), self.still["F"].average()
        if a is None or b is None:
            return None
        # The two still spans must overlap in time, so both show the same board pose.
        if a[3] > b[4] or b[3] > a[4]:
            return None
        cand = {"L": dict(zip(a[0].tolist(), a[1])), "F": dict(zip(b[0].tolist(), b[1]))}
        for view in self.accepted:
            if all(self._same(cand[c], view[c]) for c in ("L", "F")):
                self.status = {"L": "already have this view", "F": "already have this view"}
                return None
        self.accepted.append(cand)
        for s in self.still.values():
            s.window.clear()
        self.status = {"L": "captured", "F": "captured"}
        return StereoView(f"{len(self.accepted) - 1:03d}", a[0], a[1], b[0], b[1], a[2], b[2])

    def _same(self, x: dict, y: dict) -> bool:
        shared = [k for k in x if k in y]
        if len(shared) < 0.5 * min(len(x), len(y)):
            return False
        return float(np.median([np.linalg.norm(x[k] - y[k]) for k in shared])) < self.novel_px

    def undo(self) -> None:
        if self.accepted:
            self.accepted.pop()


# ---------------------------------------------------------------------------------------------
# Geometry helpers


def to_matrix(rvec, tvec) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, np.float64))[0]
    T[:3, 3] = np.ravel(tvec)
    return T


def from_matrix(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return cv2.Rodrigues(T[:3, :3])[0].ravel(), T[:3, 3].copy()


def rotation_mean(Rs: list[np.ndarray]) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.sum(Rs, axis=0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def board_pose(obj, corners, intr: Intrinsics) -> np.ndarray | None:
    ok, rvec, tvec = cv2.solvePnP(obj, corners, intr.K, intr.dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(obj, corners, intr.K, intr.dist, rvec, tvec)
    return to_matrix(rvec, tvec)


def triangulate(T_F_L: np.ndarray, n_L: np.ndarray, n_F: np.ndarray) -> np.ndarray:
    """Linear (DLT) triangulation from normalised coordinates; points in frame L."""
    P_L = np.hstack([np.eye(3), np.zeros((3, 1))])
    P_F = T_F_L[:3]
    out = []
    for (xl, yl), (xf, yf) in zip(n_L, n_F):
        A = np.stack([xl * P_L[2] - P_L[0], yl * P_L[2] - P_L[1],
                      xf * P_F[2] - P_F[0], yf * P_F[2] - P_F[1]])
        X = np.linalg.svd(A)[2][-1]
        out.append(X[:3] / X[3])
    return np.asarray(out)


def epipolar_px(T_F_L: np.ndarray, n_L: np.ndarray, n_F: np.ndarray, f_L: float,
                f_F: float) -> np.ndarray:
    """Symmetric epipolar distance per correspondence, in pixels (normalised * focal)."""
    R, t = T_F_L[:3, :3], T_F_L[:3, 3]
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    E = tx @ R                                  # x_F^T E x_L = 0
    xl = np.hstack([n_L, np.ones((len(n_L), 1))])
    xf = np.hstack([n_F, np.ones((len(n_F), 1))])
    lf = xl @ E.T                               # epipolar lines in F
    ll = xf @ E                                 # epipolar lines in L
    r = np.sum(xf * lf, axis=1)
    d_F = np.abs(r) / np.hypot(lf[:, 0], lf[:, 1]) * f_F
    d_L = np.abs(r) / np.hypot(ll[:, 0], ll[:, 1]) * f_L
    return 0.5 * (d_L + d_F)


# ---------------------------------------------------------------------------------------------
# Solve


def _bundle_adjust(views, obj_of, intr_L, intr_F, T_F_L0, poses0):
    """Refine T_F_L and every board pose T_L_B by minimising pixel error in both cameras."""
    nv = len(views)

    def unpack(p):
        T_F_L = to_matrix(p[:3], p[3:6])
        poses = [to_matrix(p[6 + 6 * i: 9 + 6 * i], p[9 + 6 * i: 12 + 6 * i]) for i in range(nv)]
        return T_F_L, poses

    def residuals(p):
        T_F_L, poses = unpack(p)
        res = []
        for v, T_L_B in zip(views, poses):
            for ids, corners, T, intr in ((v.ids_L, v.corners_L, T_L_B, intr_L),
                                          (v.ids_F, v.corners_F, T_F_L @ T_L_B, intr_F)):
                rvec, tvec = from_matrix(T)
                proj, _ = cv2.projectPoints(obj_of(ids), rvec, tvec, intr.K, intr.dist)
                res.append((proj.reshape(-1, 2) - corners).ravel())
        return np.concatenate(res)

    sparsity_rows = sum(2 * (len(v.ids_L) + len(v.ids_F)) for v in views)
    S = lil_matrix((sparsity_rows, 6 + 6 * nv), dtype=int)
    row = 0
    for i, v in enumerate(views):
        nl, nf = 2 * len(v.ids_L), 2 * len(v.ids_F)
        S[row: row + nl, 6 + 6 * i: 12 + 6 * i] = 1
        row += nl
        S[row: row + nf, 6 + 6 * i: 12 + 6 * i] = 1
        S[row: row + nf, :6] = 1
        row += nf

    p0 = np.concatenate([np.concatenate(from_matrix(T_F_L0))]
                        + [np.concatenate(from_matrix(T)) for T in poses0])
    # x_scale="jac" converges in ~15 evaluations instead of ~500. SciPy's sparse solver (LSMR)
    # warns about an overflow in its condition-number estimate, which is diagnostic only.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "overflow encountered in cast", RuntimeWarning)
        sol = least_squares(residuals, p0, jac_sparsity=S, x_scale="jac", loss="soft_l1",
                            f_scale=1.0, method="trf")
    T_F_L, poses = unpack(sol.x)
    return T_F_L, poses, residuals(sol.x)


def _split_residuals(views, res):
    rl, rf, row = [], [], 0
    for v in views:
        nl, nf = 2 * len(v.ids_L), 2 * len(v.ids_F)
        rl.append(res[row: row + nl])
        rf.append(res[row + nl: row + nl + nf])
        row += nl + nf
    rms = lambda r: float(np.sqrt(np.mean(np.concatenate(r).reshape(-1, 2) ** 2) * 2))
    return rms(rl), rms(rf)


def fit(views, obj_of, intr_L, intr_F):
    poses_L, rel = [], []
    usable = []
    for v in views:
        T_L_B = board_pose(obj_of(v.ids_L), v.corners_L, intr_L)
        T_F_B = board_pose(obj_of(v.ids_F), v.corners_F, intr_F)
        if T_L_B is None or T_F_B is None:
            continue
        usable.append(v)
        poses_L.append(T_L_B)
        rel.append(T_F_B @ np.linalg.inv(T_L_B))
    T0 = np.eye(4)
    T0[:3, :3] = rotation_mean([T[:3, :3] for T in rel])
    T0[:3, 3] = np.median([T[:3, 3] for T in rel], axis=0)
    T_F_L, poses, res = _bundle_adjust(usable, obj_of, intr_L, intr_F, T0, poses_L)
    return usable, T_F_L, poses, res


def _common(v: StereoView):
    lookup = dict(zip(v.ids_F.tolist(), v.corners_F))
    ids = np.array([k for k in v.ids_L.tolist() if k in lookup], np.int32)
    cl = np.array([c for k, c in zip(v.ids_L.tolist(), v.corners_L) if k in lookup])
    cf = np.array([lookup[k] for k in ids])
    return ids, cl, cf


def length_errors(views, T_F_L, obj_of, intr_L, intr_F, squares_x: int):
    """Known-length test: triangulate corner pairs >= LENGTH_MIN_SQUARES apart in a row or
    column and compare their 3D distance with the board's true distance (mm)."""
    errors, true_lengths, positions = [], [], []
    for v in views:
        ids, cl, cf = _common(v)
        if len(ids) < 2:
            continue
        X = triangulate(T_F_L, undistort_normalized(cl, intr_L.K, intr_L.dist),
                        undistort_normalized(cf, intr_F.K, intr_F.dist))
        obj = obj_of(ids)
        cols, rows = ids % (squares_x - 1), ids // (squares_x - 1)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                same_row = rows[i] == rows[j] and abs(cols[i] - cols[j]) >= LENGTH_MIN_SQUARES
                same_col = cols[i] == cols[j] and abs(rows[i] - rows[j]) >= LENGTH_MIN_SQUARES
                if same_row or same_col:
                    true = float(np.linalg.norm(obj[i] - obj[j]))
                    errors.append(float(np.linalg.norm(X[i] - X[j])) - true)
                    true_lengths.append(true)
                    positions.append(0.5 * (X[i] + X[j]))
    return np.array(errors), np.array(true_lengths), np.array(positions)


def epipolar_rms(views, T_F_L, intr_L, intr_F) -> float:
    errs = []
    for v in views:
        _, cl, cf = _common(v)
        if len(cl):
            errs.append(epipolar_px(T_F_L, undistort_normalized(cl, intr_L.K, intr_L.dist),
                                    undistort_normalized(cf, intr_F.K, intr_F.dist),
                                    intr_L.K[0, 0], intr_F.K[0, 0]))
    e = np.concatenate(errs) if errs else np.array([np.nan])
    return float(np.sqrt(np.mean(e**2)))


def per_view_rotation_deg(views, obj_of, intr_L, intr_F, T_L_F) -> list[np.ndarray]:
    """For each view, the rotation (rotation vector, degrees, in F's axes) between the camera
    pose that view alone implies and the joint solution. A camera that turns during the
    capture shows up as a trend over the views."""
    out = []
    for v in views:
        T_L_B = board_pose(obj_of(v.ids_L), v.corners_L, intr_L)
        T_F_B = board_pose(obj_of(v.ids_F), v.corners_F, intr_F)
        if T_L_B is None or T_F_B is None:
            continue
        T_view = T_L_B @ np.linalg.inv(T_F_B)
        out.append(np.degrees(cv2.Rodrigues(T_L_F[:3, :3].T @ T_view[:3, :3])[0].ravel()))
    return out


def drift_deg(rotations: list[np.ndarray]) -> float:
    """Rotation between the median per-view pose of the first and the last third of views."""
    n = len(rotations) // 3
    if n < 3:
        return float("nan")
    a = np.median(rotations[:n], axis=0)
    b = np.median(rotations[-n:], axis=0)
    return float(np.linalg.norm(b - a))


def solve_emergency(views, obj_of, intr_L, intr_F, squares_x) -> dict:
    """Provisional three-pose fit; no held-out or capture-drift validation."""
    if len(views) != 3:
        raise ValueError("emergency recalibration needs exactly 3 board poses")
    usable, T_F_L, _, res = fit(views, obj_of, intr_L, intr_F)
    if len(usable) != 3:
        raise ValueError("could not solve all 3 board poses")
    rms_L, rms_F = _split_residuals(usable, res)
    epi = epipolar_rms(usable, T_F_L, intr_L, intr_F)
    errors, _, _ = length_errors(usable, T_F_L, obj_of, intr_L, intr_F, squares_x)
    p95 = float(np.percentile(np.abs(errors), 95)) if len(errors) else float("nan")
    gates = [_gate("reprojection RMS, camera L (px)", rms_L, GATES["rms_px"]),
             _gate("reprojection RMS, camera F (px)", rms_F, GATES["rms_px"]),
             _gate("in-sample epipolar RMS (px)", epi, GATES["epipolar_px"]),
             _gate("known-length error (mm)", p95, GATES["length_p95_mm"])]
    T_L_F = np.linalg.inv(T_F_L)
    return {"emergency": True, "T_L_F": {"R": T_L_F[:3, :3].tolist(),
                                          "t_mm": T_L_F[:3, 3].tolist()},
            "baseline_mm": float(np.linalg.norm(T_L_F[:3, 3])),
            "views_used": 3, "length_test": {"p95_mm": p95},
            "gates": gates, "passed": all(g["pass"] for g in gates)}


def solve(views: list[StereoView], obj_of, intr_L: Intrinsics, intr_F: Intrinsics,
          squares_x: int, seed: int = 0) -> dict:
    if len(views) < 6:
        raise ValueError(f"need at least 6 views, have {len(views)}")
    usable, T_F_L, poses, res = fit(views, obj_of, intr_L, intr_F)
    rms_L, rms_F = _split_residuals(usable, res)

    # Held-out epipolar error: fit without 20 % of the views, test on their shared corners.
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(usable))
    n_hold = max(2, round(0.2 * len(usable)))
    hold = [usable[i] for i in order[:n_hold]]
    _, T_train, _, _ = fit([usable[i] for i in order[n_hold:]], obj_of, intr_L, intr_F)
    epi = []
    for v in hold:
        _, cl, cf = _common(v)
        if len(cl):
            epi.append(epipolar_px(T_train, undistort_normalized(cl, intr_L.K, intr_L.dist),
                                   undistort_normalized(cf, intr_F.K, intr_F.dist),
                                   intr_L.K[0, 0], intr_F.K[0, 0]))
    epi = np.concatenate(epi) if epi else np.array([np.nan])
    epi_rms = float(np.sqrt(np.mean(epi**2)))
    epi_in = epipolar_rms(usable, T_F_L, intr_L, intr_F)
    rotations = per_view_rotation_deg(sorted(usable, key=lambda v: v.t_L_ms), obj_of,
                                      intr_L, intr_F, np.linalg.inv(T_F_L))
    drift = drift_deg(rotations)

    errs, true_len, _ = length_errors(usable, T_F_L, obj_of, intr_L, intr_F, squares_x)
    scale = float(np.sum((errs + true_len) * true_len) / np.sum(true_len**2))  # measured/true
    corrected = (errs + true_len) / scale - true_len
    p95 = float(np.percentile(np.abs(corrected), 95)) if len(corrected) else float("nan")

    T_L_F = np.linalg.inv(T_F_L)
    centres = np.array([T[:3, 3] for T in poses])
    angles = []
    for v, T in zip(usable, poses):
        c = T[:3, :3] @ obj_of(v.ids_L).mean(0) + T[:3, 3]      # board centre in L
        angles.append(np.degrees(np.arccos(np.dot(-c, T_L_F[:3, 3] - c)
                                           / np.linalg.norm(c) / np.linalg.norm(T_L_F[:3, 3] - c))))
    gates = [
        _gate("reprojection RMS, camera L (px)", rms_L, GATES["rms_px"]),
        _gate("reprojection RMS, camera F (px)", rms_F, GATES["rms_px"]),
        _gate("camera movement during the capture (deg)", drift, GATES["drift_deg"]),
        _gate("held-out / in-sample epipolar RMS", epi_rms / epi_in, GATES["holdout_epipolar_ratio"]),
        _gate("held-out epipolar RMS (px)", epi_rms, GATES["epipolar_px"]),
        _gate("known-length error, 95th percentile (mm)", p95, GATES["length_p95_mm"]),
        _gate("views used", len(usable), GATES["min_views"], minimum=True),
    ]
    return {
        "T_L_F": {"R": T_L_F[:3, :3].tolist(), "t_mm": T_L_F[:3, 3].tolist()},
        "baseline_mm": float(np.linalg.norm(T_L_F[:3, 3])),
        "rms_px": {"L": rms_L, "F": rms_F},
        "holdout_epipolar_rms_px": epi_rms,
        "insample_epipolar_rms_px": epi_in,
        "drift_deg": drift,
        "length_test": {"pairs": int(len(errs)), "scale": scale,
                        "rms_mm": float(np.sqrt(np.mean(corrected**2))) if len(errs) else None,
                        "p95_mm": p95,
                        "max_mm": float(np.max(np.abs(corrected))) if len(errs) else None},
        "views_used": len(usable),
        "board_positions_mm": {"min": centres.min(0).tolist(), "max": centres.max(0).tolist()},
        "angle_between_cameras_deg": {"median": float(np.median(angles)),
                                      "min": float(np.min(angles)), "max": float(np.max(angles))},
        "gates": gates,
        "passed": all(g["pass"] for g in gates),
    }


def _gate(name, value, limit, minimum=False):
    ok = value >= limit if minimum else value <= limit
    return {"name": name, "value": float(value), "limit": limit,
            "kind": "min" if minimum else "max", "pass": bool(ok)}


def format_report(r: dict) -> str:
    lt = r["length_test"]
    lines = [
        f"{r['views_used']} views, baseline {r['baseline_mm']:.1f} mm, angle between cameras at"
        f" the board: median {r['angle_between_cameras_deg']['median']:.0f}°",
        f"reprojection RMS: L {r['rms_px']['L']:.3f} px, F {r['rms_px']['F']:.3f} px;"
        f" epipolar RMS held-out {r['holdout_epipolar_rms_px']:.3f} px,"
        f" in-sample {r['insample_epipolar_rms_px']:.3f} px; camera movement {r['drift_deg']:.2f}°",
        f"known lengths ({lt['pairs']} pairs): scale {lt['scale']:.5f},"
        f" after scale correction RMS {lt['rms_mm']:.3f} mm, 95 % {lt['p95_mm']:.3f} mm,"
        f" max {lt['max_mm']:.3f} mm",
        "",
    ]
    for g in r["gates"]:
        op = "≥" if g["kind"] == "min" else "≤"
        lines.append(f"  {'PASS' if g['pass'] else 'FAIL'}  {g['name']}: {g['value']:.3f}"
                     f"  (needs {op} {g['limit']})")
    lines.append(f"\n{'ALL GATES PASSED' if r['passed'] else 'GATES FAILED'}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# Files


STEREO_PATH = PROJECT_ROOT / "calib" / "stereo.json"


@dataclass
class StereoSession:
    intrinsics: dict                  # camera -> fingerprint
    board: dict
    scale: dict
    views: list[StereoView] = field(default_factory=list)
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def save(self, raw_dir) -> None:
        raw_dir.mkdir(parents=True, exist_ok=True)
        d = {"intrinsics": self.intrinsics, "board": self.board, "scale": self.scale,
             "created": self.created, "views": [v.to_json() for v in self.views]}
        (raw_dir / "session.json").write_text(json.dumps(d, indent=1))

    @classmethod
    def load(cls, raw_dir) -> "StereoSession":
        d = json.loads((raw_dir / "session.json").read_text())
        d["views"] = [StereoView.from_json(v) for v in d["views"]]
        return cls(**d)


def build_result(session: StereoSession, report: dict, raw_dir) -> dict:
    try:
        raw = str(raw_dir.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        raw = str(raw_dir)
    return {"intrinsics": session.intrinsics, "board": session.board, "scale": session.scale,
            "captured": session.created, "solved": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "raw_dir": raw, **report}


def install(result: dict, path=None) -> None:
    path = path or STEREO_PATH
    if path.exists():
        history = path.parent / "history"
        history.mkdir(exist_ok=True)
        path.rename(history / f"{path.stem}.{time.strftime('%Y%m%d-%H%M%S')}.json")
    path.write_text(json.dumps(result, indent=1))


def load_stereo(intr_L: Intrinsics, intr_F: Intrinsics, path=None) -> np.ndarray:
    """T_L_F (4x4). Refuses a result made with different intrinsics or one that failed."""
    r = json.loads((path or STEREO_PATH).read_text())
    want = {"L": intrinsics_fingerprint(intr_L), "F": intrinsics_fingerprint(intr_F)}
    if r["intrinsics"] != want:
        raise RuntimeError("stereo.json was made with different intrinsics; redo C5")
    if not r["passed"]:
        raise RuntimeError("stereo.json did not pass its gates")
    T = np.eye(4)
    T[:3, :3] = np.asarray(r["T_L_F"]["R"])
    T[:3, 3] = np.asarray(r["T_L_F"]["t_mm"])
    return T
