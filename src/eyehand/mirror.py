"""C4: where the screen is relative to camera L (`T_S_L`), seen through a plane mirror.

Camera L faces out of the screen and can never see it directly. The screen shows a horizontally
flipped ChArUco board; in a mirror it reads normally, and camera L detects it. Every mirror
pose i has an unknown plane m_i . X = e_i (frame L); the screen's pose is shared by all of
them. The solver fits T_L_S and every plane together, minimising pixel reprojection error of
the reflected corners (docs/CALIBRATION.md, C4; Rodrigues et al. ECCV 2010, Takahashi et al.
CVPR 2012: three or more poses whose normals are not coplanar give a unique solution).

A back-silvered mirror behaves, to first order, like an ideal mirror ~0.66 x glass thickness
behind its front surface. The planes are free unknowns, so that shift is absorbed
(tools/mirror_glass_sim.py: <= 0.1 mm of cursor effect for 6 mm glass).

Frames: S = screen (mm, origin at the outer top-left pixel edge, x right, y down, z into the
screen); L = camera L (OpenCV). `T_A_B` maps B coordinates into A.
"""

from __future__ import annotations

import hashlib
import json
import time
import warnings
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from eyehand import boards
from eyehand.config import PROJECT_ROOT
from eyehand.intrinsics import Intrinsics

RESULT_PATH = PROJECT_ROOT / "calib" / "screen_from_L.json"
GATES = {
    "rms_px": 0.7,                  # sanity: camera L's corner noise floor is ~0.45 px
    "worst_view_ratio": 2.5,        # worst view RMS / median view RMS: a bent mirror region
    # Jackknife (leave-one-pose-out) standard error of the cursor position over the screen. The
    # largest single leave-one-out shift underestimates the uncertainty by about sqrt(n).
    "cursor_uncertainty_mm": 0.7,
    "roll_uncertainty_deg": 0.1,
    "normal_spread_deg": 3.0,       # mirror normals must vary in BOTH directions
    "prior_mm": 25.0,               # agreement with "camera at the top centre, facing out"
    "prior_deg": 6.0,
    "min_views": 8,
}

# Prior: camera L at the top centre of the lid, just above the active area, looking straight
# out of the screen. Image right is the user's left, so x_L = -x_S, y_L = y_S, z_L = -z_S.
PRIOR_R_L_S = np.diag([-1.0, 1.0, -1.0])
PRIOR_ABOVE_MM = 8.0
PRIOR_BEHIND_MM = 3.0

# Pointing geometry for judging T_S_L errors at the cursor (docs/ARCHITECTURE.md): eye 650 mm
# out in front of the screen centre, fingertip 450 mm from the eye toward each target.
EYE_DISTANCE_MM = 650.0
FINGER_FROM_EYE_MM = 450.0


@dataclass
class MirrorView:
    name: str
    ids: np.ndarray
    corners: np.ndarray       # px in camera L, as detected (reflection of the flipped board)
    t_ms: float = 0.0

    def to_json(self):
        return {"name": self.name, "ids": self.ids.tolist(), "corners": self.corners.tolist(),
                "t_ms": self.t_ms}

    @classmethod
    def from_json(cls, d):
        return cls(d["name"], np.asarray(d["ids"], np.int32), np.asarray(d["corners"], np.float32),
                   d.get("t_ms", 0.0))


@dataclass
class ScreenTarget:
    """Corner positions of the displayed (flipped) screen board: in the board's own frame
    (for PnP of the virtual board) and in the real screen frame S (mm)."""
    spec: boards.BoardSpec
    placement: boards.ScreenPlacement
    pitch: tuple[float, float]

    def board_mm(self, ids) -> np.ndarray:
        """Board frame, mm (as the camera sees it in the mirror: a normal, unflipped board)."""
        obj = self.spec.board().getChessboardCorners()[np.asarray(ids)].astype(np.float64)
        obj[:, 0] *= self.pitch[0]
        obj[:, 1] *= self.pitch[1]
        return obj

    def screen_mm(self, ids) -> np.ndarray:
        """Real positions on the screen, frame S, mm (z = 0)."""
        b = self.spec.board().getChessboardCorners()[np.asarray(ids), :2]
        px = self.placement.board_to_screen_px(b)      # pixel-edge coordinates
        return np.column_stack([px[:, 0] * self.pitch[0], px[:, 1] * self.pitch[1],
                                np.zeros(len(px))])


def prior_T_L_S(screen_size_mm: tuple[float, float]) -> np.ndarray:
    c_S = np.array([screen_size_mm[0] / 2, -PRIOR_ABOVE_MM, -PRIOR_BEHIND_MM])
    T = np.eye(4)
    T[:3, :3] = PRIOR_R_L_S
    T[:3, 3] = -PRIOR_R_L_S @ c_S
    return T


def reflect(X: np.ndarray, m: np.ndarray, e: float) -> np.ndarray:
    return X - 2 * (X @ m - e)[:, None] * m


def _basis(n):
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(n, a)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def _plane_from_pairs(X, Xv):
    """Mirror plane that best maps real points X onto virtual points Xv (both frame L)."""
    d = X - Xv
    m = d.mean(0)
    m /= np.linalg.norm(m)
    return m, float(np.mean((X + Xv) @ m) / 2)


def virtual_pose(view: MirrorView, target: ScreenTarget, intr: Intrinsics) -> np.ndarray | None:
    obj = target.board_mm(view.ids)
    ok, rvec, tvec = cv2.solvePnP(obj, view.corners.astype(np.float64), intr.K, intr.dist,
                                  flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(obj, view.corners.astype(np.float64), intr.K, intr.dist,
                                      rvec, tvec)
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    return T


def fit(views: list[MirrorView], target: ScreenTarget, intr: Intrinsics,
        T0: np.ndarray, planes0: list | None = None):
    """Joint fit of T_L_S and one mirror plane per view. Returns (T_L_S, planes, residuals)."""
    obj_S = [target.screen_mm(v.ids) for v in views]
    if planes0 is None:
        planes0 = []
        for v, X_S in zip(views, obj_S):
            Tv = virtual_pose(v, target, intr)
            X = X_S @ T0[:3, :3].T + T0[:3, 3]
            Xv = target.board_mm(v.ids) @ Tv[:3, :3].T + Tv[:3, 3]
            planes0.append(_plane_from_pairs(X, Xv))
    bases = [_basis(m) for m, _ in planes0]
    n = len(views)

    def unpack(p):
        R = cv2.Rodrigues(p[:3])[0]
        planes = []
        for i, ((m0, _), (u, w)) in enumerate(zip(planes0, bases)):
            a, b, e = p[6 + 3 * i: 9 + 3 * i]
            m = m0 + a * u + b * w
            planes.append((m / np.linalg.norm(m), e))
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, p[3:6]
        return T, planes

    def residuals(p):
        T, planes = unpack(p)
        res = []
        for v, X_S, (m, e) in zip(views, obj_S, planes):
            X = reflect(X_S @ T[:3, :3].T + T[:3, 3], m, e)
            proj, _ = cv2.projectPoints(X, np.zeros(3), np.zeros(3), intr.K, intr.dist)
            res.append((proj.reshape(-1, 2) - v.corners).ravel())
        return np.concatenate(res)

    rows = sum(2 * len(v.ids) for v in views)
    S = lil_matrix((rows, 6 + 3 * n), dtype=int)
    r = 0
    for i, v in enumerate(views):
        S[r: r + 2 * len(v.ids), :6] = 1
        S[r: r + 2 * len(v.ids), 6 + 3 * i: 9 + 3 * i] = 1
        r += 2 * len(v.ids)
    p0 = np.concatenate([cv2.Rodrigues(T0[:3, :3])[0].ravel(), T0[:3, 3]]
                        + [[0.0, 0.0, e] for _, e in planes0])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "overflow encountered in cast", RuntimeWarning)
        sol = least_squares(residuals, p0, jac_sparsity=S, x_scale="jac", loss="soft_l1",
                            f_scale=1.0)
    T, planes = unpack(sol.x)
    return T, planes, residuals(sol.x)


def cursor_shift_mm(T_L_S_a: np.ndarray, T_L_S_b: np.ndarray, screen_mm, n: int = 9) -> float:
    """Largest cursor difference over a grid of screen targets when eye and fingertip are
    measured in frame L and mapped to S with transform b instead of a."""
    return float(np.linalg.norm(cursor_shifts(T_L_S_a, T_L_S_b, screen_mm, n), axis=1).max())


def cursor_shifts(T_L_S_a: np.ndarray, T_L_S_b: np.ndarray, screen_mm, n: int = 9) -> np.ndarray:
    """Cursor displacement vectors (mm) over an n x n grid of targets, as above."""
    W, H = screen_mm
    E = np.array([W / 2, H / 2, -EYE_DISTANCE_MM])
    Tb_inv = np.linalg.inv(T_L_S_b)
    out = []
    for x in np.linspace(0.05 * W, 0.95 * W, n):
        for y in np.linspace(0.05 * H, 0.95 * H, n):
            C = np.array([x, y, 0.0])
            Fp = E + (C - E) * (FINGER_FROM_EYE_MM / np.linalg.norm(C - E))
            pts_L = [T_L_S_a[:3, :3] @ P + T_L_S_a[:3, 3] for P in (E, Fp)]   # what L measures
            Eb, Fb = [Tb_inv[:3, :3] @ P + Tb_inv[:3, 3] for P in pts_L]
            s = Eb[2] / (Eb[2] - Fb[2])
            out.append((Eb + s * (Fb - Eb))[:2] - C[:2])
    return np.array(out)


def roll_deg(T_a, T_b) -> float:
    """Rotation difference about the screen normal (the most sensitive direction)."""
    R = T_a[:3, :3].T @ T_b[:3, :3]          # in S
    return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))


def solve(views: list[MirrorView], target: ScreenTarget, intr: Intrinsics) -> dict:
    if len(views) < 3:
        raise ValueError("need at least 3 mirror poses")
    screen_mm = (target.placement.screen_width * target.pitch[0],
                 target.placement.screen_height * target.pitch[1])
    prior = prior_T_L_S(screen_mm)
    T, planes, res = fit(views, target, intr, prior)

    # Per-view RMS (a bent part of the mirror shows up as one bad view).
    per_view, r = [], 0
    for v in views:
        k = 2 * len(v.ids)
        per_view.append(float(np.sqrt(np.mean(res[r: r + k].reshape(-1, 2) ** 2) * 2)))
        r += k
    rms = float(np.sqrt(np.mean(res.reshape(-1, 2) ** 2) * 2))

    # Jackknife: refit without each pose; the spread of the results estimates the uncertainty.
    n = len(views)
    shifts, rolls = [], []
    for i in range(n):
        keep = [j for j in range(n) if j != i]
        Ti, _, _ = fit([views[j] for j in keep], target, intr, T, [planes[j] for j in keep])
        shifts.append(cursor_shifts(T, Ti, screen_mm))
        rolls.append(roll_deg(T, Ti))
    shifts = np.array(shifts)                                   # (n, grid, 2)
    dev = shifts - shifts.mean(0)
    cursor_se = float(np.sqrt((n - 1) / n * np.sum(dev**2, axis=(0, 2))).max())
    rolls = np.array(rolls)
    roll_se = float(np.sqrt((n - 1) / n * np.sum((rolls - rolls.mean()) ** 2)))

    # Mirror normals must vary in two directions (coplanar normals leave the pose undetermined).
    normals = np.array([m if m[2] < 0 else -m for m, _ in planes])
    yaw = np.degrees(np.arctan2(normals[:, 0], -normals[:, 2]))
    pitch = np.degrees(np.arctan2(normals[:, 1], -normals[:, 2]))
    spread = min(np.std(yaw), np.std(pitch))

    T_S_L = np.linalg.inv(T)
    d_prior = np.linalg.inv(prior) @ T
    prior_mm = float(np.linalg.norm(T_S_L[:3, 3] - np.linalg.inv(prior)[:3, 3]))
    prior_deg = float(np.degrees(np.linalg.norm(cv2.Rodrigues(d_prior[:3, :3])[0])))
    med = float(np.median(per_view))
    gates = [
        _gate("reprojection RMS (px)", rms, GATES["rms_px"]),
        _gate("worst mirror pose / median pose RMS", max(per_view) / med, GATES["worst_view_ratio"]),
        _gate("cursor uncertainty from this calibration, worst point on screen (mm)", cursor_se,
              GATES["cursor_uncertainty_mm"]),
        _gate("roll uncertainty (deg)", roll_se, GATES["roll_uncertainty_deg"]),
        _gate("mirror tilt variety, smaller of the two directions (deg)", spread,
              GATES["normal_spread_deg"], minimum=True),
        _gate("camera position vs. 'top centre of the lid' (mm)", prior_mm, GATES["prior_mm"]),
        _gate("camera direction vs. 'straight out of the screen' (deg)", prior_deg,
              GATES["prior_deg"]),
        _gate("mirror poses used", len(views), GATES["min_views"], minimum=True),
    ]
    return {
        "T_S_L": {"R": T_S_L[:3, :3].tolist(), "t_mm": T_S_L[:3, 3].tolist()},
        "camera_in_screen_mm": T_S_L[:3, 3].tolist(),
        "camera_tilt_vs_screen_deg": prior_deg,
        "rms_px": rms,
        "per_view_rms_px": dict(zip([v.name for v in views], per_view)),
        "cursor_uncertainty_mm": cursor_se,
        "roll_uncertainty_deg": roll_se,
        "mirror_tilt_deg": {"yaw_std": float(np.std(yaw)), "pitch_std": float(np.std(pitch))},
        "mirror_distance_mm": [float(abs(e)) for _, e in planes],
        "views_used": len(views),
        "pitch_mm": list(target.pitch),
        "gates": gates,
        "passed": all(g["pass"] for g in gates),
    }


def _gate(name, value, limit, minimum=False):
    ok = value >= limit if minimum else value <= limit
    return {"name": name, "value": float(value), "limit": limit,
            "kind": "min" if minimum else "max", "pass": bool(ok)}


def format_report(r: dict) -> str:
    c = r["camera_in_screen_mm"]
    lines = [
        f"{r['views_used']} mirror poses; camera L at x {c[0]:.1f}, y {c[1]:.1f}, z {c[2]:.1f} mm"
        f" in screen coordinates, tilted {r['camera_tilt_vs_screen_deg']:.2f}° from straight out",
        f"reprojection RMS {r['rms_px']:.3f} px; uncertainty: cursor {r['cursor_uncertainty_mm']:.2f} mm,"
        f" roll {r['roll_uncertainty_deg']:.3f}°; mirror tilt spread {r['mirror_tilt_deg']['yaw_std']:.1f}°"
        f" / {r['mirror_tilt_deg']['pitch_std']:.1f}°",
        "",
    ]
    for g in r["gates"]:
        op = "≥" if g["kind"] == "min" else "≤"
        lines.append(f"  {'PASS' if g['pass'] else 'FAIL'}  {g['name']}: {g['value']:.3f}"
                     f"  (needs {op} {g['limit']})")
    lines.append(f"\n{'ALL GATES PASSED' if r['passed'] else 'GATES FAILED'}")
    return "\n".join(lines)


def intrinsics_fingerprint(intr: Intrinsics) -> str:
    data = json.dumps([np.round(intr.K, 6).tolist(), np.round(intr.dist, 9).tolist()])
    return hashlib.sha256(data.encode()).hexdigest()[:16]


@dataclass
class MirrorSession:
    intrinsics_L: str
    pitch: list
    placement: dict
    board: str = "screen_mirror"
    views: list[MirrorView] = field(default_factory=list)
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def save(self, raw_dir) -> None:
        raw_dir.mkdir(parents=True, exist_ok=True)
        d = {"intrinsics_L": self.intrinsics_L, "pitch": self.pitch, "placement": self.placement,
             "board": self.board, "created": self.created, "views": [v.to_json() for v in self.views]}
        (raw_dir / "session.json").write_text(json.dumps(d, indent=1))

    @classmethod
    def load(cls, raw_dir) -> "MirrorSession":
        d = json.loads((raw_dir / "session.json").read_text())
        d["views"] = [MirrorView.from_json(v) for v in d["views"]]
        return cls(**d)


def install(result: dict) -> None:
    if RESULT_PATH.exists():
        history = RESULT_PATH.parent / "history"
        history.mkdir(exist_ok=True)
        RESULT_PATH.rename(history / f"{RESULT_PATH.stem}.{time.strftime('%Y%m%d-%H%M%S')}.json")
    RESULT_PATH.write_text(json.dumps(result, indent=1))


def load_T_S_L(intr_L: Intrinsics) -> np.ndarray:
    r = json.loads(RESULT_PATH.read_text())
    if r["intrinsics_L"] != intrinsics_fingerprint(intr_L):
        raise RuntimeError("screen_from_L.json was made with different camera L intrinsics; redo C4")
    if not r["passed"]:
        raise RuntimeError("screen_from_L.json did not pass its gates")
    T = np.eye(4)
    T[:3, :3] = np.asarray(r["T_S_L"]["R"])
    T[:3, 3] = np.asarray(r["T_S_L"]["t_mm"])
    return T
