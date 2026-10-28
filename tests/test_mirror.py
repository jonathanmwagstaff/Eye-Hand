import cv2
import numpy as np

from eyehand import boards, intrinsics, mirror
from eyehand.intrinsics import Intrinsics

INTR = Intrinsics("L", (1280, 720), np.array([[909.0, 0, 650], [0, 909.0, 356], [0, 0, 1]]),
                  np.array([0.07, -0.41, 0.0006, 0.002, 0.53]))
PITCH = (344 / 1920, 344 / 1920)


def target():
    _, placement = boards.render_screen(boards.SCREEN_BOARD, mirrored=True)
    return mirror.ScreenTarget(boards.SCREEN_BOARD, placement, PITCH)


def true_T_L_S():
    prior = mirror.prior_T_L_S((344.0, 193.5))
    d = np.eye(4)
    d[:3, :3] = cv2.Rodrigues(np.deg2rad([1.2, -0.8, 1.5]))[0]   # camera ~2 deg off "straight"
    d[:3, 3] = [4.0, -3.0, 2.0]
    return d @ prior


def make_views(tilts, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    tg, T = target(), true_T_L_S()
    ids_all = np.arange(17 * 9, dtype=np.int32)
    X = tg.screen_mm(ids_all) @ T[:3, :3].T + T[:3, 3]
    views = []
    for k, (yaw, pitch, dist) in enumerate(tilts):
        m = cv2.Rodrigues(np.deg2rad([pitch, yaw, 0.0]))[0] @ np.array([0, 0, 1.0])
        Xv = mirror.reflect(X, m, dist)
        px, _ = cv2.projectPoints(Xv, np.zeros(3), np.zeros(3), INTR.K, INTR.dist)
        px = px.reshape(-1, 2) + rng.normal(0, noise, (len(ids_all), 2))
        keep = (Xv[:, 2] > 0) & (px[:, 0] > 5) & (px[:, 0] < 1275) & (px[:, 1] > 5) & (px[:, 1] < 715)
        if keep.sum() >= 20:
            views.append(mirror.MirrorView(f"{k:03d}", ids_all[keep], px[keep].astype(np.float32)))
    return views


GOOD = [(y, p, d) for y in (-8, 0, 8) for p in (-12, -5, 2) for d in (280, 340)][:14]


def test_recovers_screen_pose_and_passes():
    views = make_views(GOOD)
    assert len(views) >= 10
    r = mirror.solve(views, target(), INTR)
    T_S_L = np.eye(4)
    T_S_L[:3, :3], T_S_L[:3, 3] = r["T_S_L"]["R"], r["T_S_L"]["t_mm"]
    err = mirror.cursor_shift_mm(true_T_L_S(), np.linalg.inv(T_S_L), (344.0, 193.5))
    assert err < 0.8, err      # noise-limited: 0.3 px noise, 14 poses (rms ~0.46 mm)
    assert r["passed"], mirror.format_report(r)


def test_mirror_tilted_only_one_way_is_rejected():
    views = make_views([(y, -6, d) for y in (-10, -5, 0, 5, 10) for d in (280, 340)])
    r = mirror.solve(views, target(), INTR)
    spread = next(g for g in r["gates"] if "variety" in g["name"])
    assert not spread["pass"]


def test_live_preview_does_not_disturb_board_detection():
    from eyehand.cli import mirror_cmd as M
    img, pl = boards.render_screen(boards.MIRROR_BOARD, mirrored=True, top=M.BOARD_TOP)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # Worst case for the preview: the camera sees the whole reflected screen, board included.
    frame = cv2.resize(np.ascontiguousarray(bgr[:, ::-1]), (1280, 720))
    comp = M.compose(bgr, pl, frame, None, None, ["Mirror poses: 3 of 20", "status"], (0, 170, 0))
    assert comp.shape == bgr.shape
    # What camera L sees in the mirror: the flipped screen, ~0.35 camera px per screen px.
    seen = cv2.resize(np.ascontiguousarray(comp[:, ::-1]), None, fx=0.35, fy=0.35,
                      interpolation=cv2.INTER_AREA)
    corners, ids = intrinsics.detect(boards.MIRROR_BOARD.detector(), seen)
    n_inner = (boards.MIRROR_BOARD.squares_x - 1) * (boards.MIRROR_BOARD.squares_y - 1)
    assert ids is not None and len(ids) == n_inner and len(set(ids.tolist())) == n_inner
