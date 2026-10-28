"""How much does an unknown mirror-glass thickness bias the C4 mirror calibration?

A household mirror is back-silvered: light refracts into the glass, reflects off the silver
layer, and refracts out again. The C4 solver models an ideal (zero-thickness) plane mirror
whose plane is a free unknown for every mirror pose. This script ray-traces the real
thick-glass mirror, fits the ideal model to the result, and reports how wrong `T_S_L` comes out.

Geometry: camera L just above the top edge of a 344 x 194 mm screen, f = 1000 px, 1280 x 720.
The mirror is held 250-350 mm in front of the lid, facing it, tilted differently for each pose.

Run: uv run python tools/mirror_glass_sim.py
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

F_PX, W, H = 1000.0, 1280, 720
N_GLASS = 1.52
rng = np.random.default_rng(1)

# Screen-board corners in S (mm): an 18 x 10 board of 100 px squares, 17 x 9 inner corners.
pitch = 344 / 1920
gx, gy = np.meshgrid(np.arange(1, 18) * 100 + 60, np.arange(1, 10) * 100 + 40)
X_S = np.stack([gx.ravel() * pitch, gy.ravel() * pitch, np.zeros(gx.size)], axis=1)

# True T_L_S: camera centred 8 mm above the active area, looking out of the screen, rotated
# by about a degree so the test is not at a special pose.
R_true = Rotation.from_rotvec(np.deg2rad([0.8, -0.6, 1.0])).as_matrix() @ np.diag([-1.0, 1.0, -1.0])
cam_S = np.array([172.0, -8.0, -3.0])
t_true = -R_true @ cam_S


def to_L(R, t, X):
    return X @ R.T + t


def project(Xc):
    return F_PX * Xc[:, :2] / Xc[:, 2:3] + [W / 2, H / 2]


def plane_normal(a, b):
    """Unit normal pointing away from the camera (into the mirror), from two tilt angles."""
    return Rotation.from_euler("xy", [a, b]).apply([0.0, 0.0, 1.0])


def ideal_project(R, t, m, e):
    X = to_L(R, t, X_S)
    Xv = X - 2 * (X @ m - e)[:, None] * m
    return project(Xv)


def refract(d, n, eta):
    """Snell refraction of unit direction d at a surface with unit normal n facing d."""
    c = -d @ n
    k = 1 - eta**2 * (1 - c**2)
    return eta * d + (eta * c - np.sqrt(k)) * n


def trace(u, m, e, tg, R, t):
    """Camera ray u -> thick mirror (front plane m.x = e, silver at e + tg) -> screen plane.
    Returns the hit point in screen coordinates (x, y)."""
    u = u / np.linalg.norm(u)
    p = u * (e / (u @ m))                          # front surface
    v = refract(u, -m, 1 / N_GLASS)                # into glass
    p = p + v * ((e + tg - p @ m) / (v @ m))       # silver layer
    v = v - 2 * (v @ m) * m                        # reflect
    p = p + v * ((e - p @ m) / (v @ m))            # front surface again
    v = refract(v, m, N_GLASS)                     # out of glass
    # Screen plane z_S = 0 in L: points X_L = R X_S + t, so z_S = (R^T (X_L - t))_z
    rz = R.T[2]
    s = (rz @ t - rz @ p) / (rz @ v)
    hit = R.T @ (p + s * v - t)
    return hit[:2]


def thick_project(m, e, tg):
    """Pixel of every screen corner seen through the thick mirror (inverse ray trace)."""
    guess = ideal_project(R_true, t_true, m, e + tg / N_GLASS)
    out = np.empty_like(guess)
    for i, (Xs, g) in enumerate(zip(X_S, guess)):
        f = lambda q: trace(np.array([(q[0] - W / 2) / F_PX, (q[1] - H / 2) / F_PX, 1.0]),
                            m, e, tg, R_true, t_true) - Xs[:2]
        out[i] = least_squares(f, g, xtol=1e-12, ftol=1e-12).x
    return out


def make_poses(n):
    poses = []
    while len(poses) < n:
        a, b = np.deg2rad(rng.uniform(-14, -2)), np.deg2rad(rng.uniform(-12, 12))
        e = rng.uniform(250, 350)
        uv = ideal_project(R_true, t_true, plane_normal(a, b), e)
        vis = (uv[:, 0] > 20) & (uv[:, 0] < W - 20) & (uv[:, 1] > 20) & (uv[:, 1] < H - 20)
        if vis.sum() >= 40:
            poses.append((a, b, e, vis))
    return poses


def solve(obs, poses):
    """Joint fit: T_L_S (6) + an ideal mirror plane per pose (3 each)."""
    def resid(p):
        R = Rotation.from_rotvec(p[:3]).as_matrix() @ np.diag([-1.0, 1.0, -1.0])
        r = []
        for k, ((_, _, _, vis), uv) in enumerate(zip(poses, obs)):
            a, b, e = p[6 + 3 * k: 9 + 3 * k]
            r.append((ideal_project(R, p[3:6], plane_normal(a, b), e) - uv)[vis].ravel())
        return np.concatenate(r)

    p0 = np.concatenate([np.zeros(3), t_true + [5, -5, 5]] + [[a, b, e] for a, b, e, _ in poses])
    sol = least_squares(resid, p0, x_scale="jac")
    R = Rotation.from_rotvec(sol.x[:3]).as_matrix() @ np.diag([-1.0, 1.0, -1.0])
    cam = -R.T @ sol.x[3:6]
    dang = np.rad2deg(Rotation.from_matrix(R @ R_true.T).magnitude())
    rms = np.sqrt(np.mean(sol.fun**2))
    return np.linalg.norm(cam - cam_S), dang, rms, sol.x[6:].reshape(-1, 3)[:, 2], (R, sol.x[3:6])


def cursor_error(R, t):
    """Worst cursor shift (mm) over the screen when eye/finger are measured in L and mapped to
    S with the estimated T_S_L instead of the true one. Eye 650 mm out, finger 450 mm from eye."""
    E = np.array([172.0, 97.0, -650.0])
    worst = 0.0
    for C in X_S:
        Fp = E + (C - E) * (450 / 650)
        Ee, Fe = [R.T @ (R_true @ X + t_true - t) for X in (E, Fp)]
        s = Ee[2] / (Ee[2] - Fe[2])
        worst = max(worst, np.linalg.norm((Ee + s * (Fe - Ee))[:2] - C[:2]))
    return worst


print("glass  poses  noise | camera pos  camera angle  reproj  plane shift | worst cursor")
print(" (mm)          (px)  |  err (mm)    err (deg)    (px)     (mm)      |  error (mm)")
for npose in [12, 24]:
    poses = make_poses(npose)
    for tg in [0.0, 3.0, 6.0]:
        clean = [thick_project(plane_normal(a, b), e, tg) for a, b, e, _ in poses]
        for noise in [0.0, 0.3, 0.1]:
            trials = 1 if noise == 0 else 5
            res = []
            for _ in range(trials):
                obs = [c + rng.normal(0, noise, c.shape) for c in clean]
                dpos, dang, rms, e_fit, Rt = solve(obs, poses)
                res.append((dpos, dang, rms, np.mean(e_fit - [p[2] for p in poses]), cursor_error(*Rt)))
            r = np.sqrt(np.mean(np.square(res), axis=0))
            print(f"{tg:5.1f}  {npose:5d}  {noise:4.1f}  | {r[0]:9.3f}  {r[1]:11.4f}  {r[2]:7.3f}  {r[3]:8.2f}    | {r[4]:8.2f}")
print(f"(noisy rows: RMS over 5 trials. Paraxial theory: plane shift = t/n = t x {1/N_GLASS:.3f})")
