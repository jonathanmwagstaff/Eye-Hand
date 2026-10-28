import cv2
import numpy as np
import pytest

from eyehand import boards, stereo
from eyehand.intrinsics import Intrinsics, TargetScale, object_points

SPEC = boards.PRINT_BOARD
SCALE = TargetScale(1.0, 1.0, "test")
INTR_L = Intrinsics("L", (1280, 720), np.array([[1000.0, 0, 642], [0, 1001.0, 357], [0, 0, 1]]),
                    np.array([-0.05, 0.08, 0.0005, -0.0003, 0.0]))
INTR_F = Intrinsics("F", (1280, 960), np.array([[920.0, 0, 633], [0, 921.0, 485], [0, 0, 1]]),
                    np.array([-0.12, 0.10, -0.0004, 0.0006, 0.0]))


def obj_of(ids):
    return object_points(SPEC, ids, SCALE)


def true_T_F_L():
    # F sits ~45 cm to the right of L, ~25 cm closer to the user, turned ~40 deg toward the scene.
    R_L_F = cv2.Rodrigues(np.array([0.05, -np.deg2rad(40), 0.02]))[0]
    t_L_F = np.array([450.0, 60.0, 250.0])
    T_L_F = np.eye(4)
    T_L_F[:3, :3], T_L_F[:3, 3] = R_L_F, t_L_F
    return np.linalg.inv(T_L_F)


def project(T_cam_B, intr, ids):
    rvec, tvec = stereo.from_matrix(T_cam_B)
    px, _ = cv2.projectPoints(obj_of(ids), rvec, tvec, intr.K, intr.dist)
    return px.reshape(-1, 2)


def make_views(n=28, noise=0.15, seed=1):
    rng = np.random.default_rng(seed)
    T_F_L = true_T_F_L()
    all_ids = np.arange((SPEC.squares_x - 1) * (SPEC.squares_y - 1), dtype=np.int32)
    centre = np.array([SPEC.squares_x, SPEC.squares_y, 0]) * SPEC.square / 2
    views = []
    while len(views) < n:
        # Board between face and screen, roughly facing a point between the two cameras.
        pos = np.array([rng.uniform(-150, 250), rng.uniform(-120, 150), rng.uniform(400, 700)])
        R = cv2.Rodrigues(np.array([rng.uniform(-0.4, 0.4), np.deg2rad(160) + rng.uniform(-0.4, 0.4),
                                    rng.uniform(-0.3, 0.3)]))[0]
        T_L_B = np.eye(4)
        T_L_B[:3, :3], T_L_B[:3, 3] = R, pos - R @ centre
        sides = {}
        for cam, T, intr in (("L", T_L_B, INTR_L), ("F", T_F_L @ T_L_B, INTR_F)):
            Xc = (T[:3, :3] @ obj_of(all_ids).T).T + T[:3, 3]
            normal_z = T[2, 2]
            px = project(T, intr, all_ids) + rng.normal(0, noise, (len(all_ids), 2))
            w, h = intr.image_size
            keep = (Xc[:, 2] > 0) & (px[:, 0] > 5) & (px[:, 0] < w - 5) & (px[:, 1] > 5) & (px[:, 1] < h - 5)
            if normal_z < 0.2 and keep.sum() >= 20:   # board faces this camera
                sides[cam] = (all_ids[keep], px[keep].astype(np.float32))
        if len(sides) == 2:
            views.append(stereo.StereoView(f"{len(views):03d}", *sides["L"], *sides["F"]))
    return views


def test_bundle_adjustment_recovers_relative_pose_and_passes_gates():
    views = make_views()
    report = stereo.solve(views, obj_of, INTR_L, INTR_F, SPEC.squares_x)
    T_true = np.linalg.inv(true_T_F_L())
    t_err = np.linalg.norm(np.asarray(report["T_L_F"]["t_mm"]) - T_true[:3, 3])
    R_err = np.degrees(np.linalg.norm(cv2.Rodrigues(
        np.asarray(report["T_L_F"]["R"]) @ T_true[:3, :3].T)[0]))
    assert t_err < 1.0 and R_err < 0.05
    assert report["passed"], stereo.format_report(report)
    assert 30 < report["angle_between_cameras_deg"]["median"] < 60


def test_wrong_focal_length_fails_the_length_test():
    views = make_views()
    K = INTR_F.K.copy()
    K[0, 0] *= 1.03
    K[1, 1] *= 1.03
    bad_F = Intrinsics("F", INTR_F.image_size, K, INTR_F.dist)
    report = stereo.solve(views, obj_of, INTR_L, bad_F, SPEC.squares_x)
    assert not report["passed"]


def test_emergency_three_pose_fit_and_invalid_capture_count():
    views = make_views(n=3, noise=0.05)
    report = stereo.solve_emergency(views, obj_of, INTR_L, INTR_F, SPEC.squares_x)
    assert report["emergency"] and report["passed"]
    assert report["views_used"] == 3
    truth = np.linalg.inv(true_T_F_L())
    assert np.linalg.norm(np.array(report["T_L_F"]["t_mm"]) - truth[:3, 3]) < 1
    with pytest.raises(ValueError, match="exactly 3"):
        stereo.solve_emergency(views[:2], obj_of, INTR_L, INTR_F, SPEC.squares_x)
    noisy = make_views(n=3, noise=2.0)
    assert not stereo.solve_emergency(noisy, obj_of, INTR_L, INTR_F, SPEC.squares_x)["passed"]


def test_epipolar_distance_is_zero_for_true_correspondences():
    T_F_L = true_T_F_L()
    X = np.array([[0, 0, 500.0], [100, -50, 600], [-80, 40, 450]])
    n_L = X[:, :2] / X[:, 2:]
    Xf = (T_F_L[:3, :3] @ X.T).T + T_F_L[:3, 3]
    n_F = Xf[:, :2] / Xf[:, 2:]
    assert stereo.epipolar_px(T_F_L, n_L, n_F, 1000, 900).max() < 1e-9
    assert np.allclose(stereo.triangulate(T_F_L, n_L, n_F), X)


def _grid(offset):
    ids = np.arange(30, dtype=np.int32)
    return (SPEC.board().getChessboardCorners()[ids, :2] * 2 + offset).astype(np.float32), ids


def test_selector_needs_both_cameras_still_at_overlapping_times():
    sel = stereo.StereoSelector(still_frames=3)
    cl, ids = _grid(100.0)
    cf, _ = _grid(300.0)
    # L still at t = 0..66 ms, F still only much later: no overlap, no view.
    for t in (0, 33, 66):
        assert sel.offer("L", cl, ids, t) is None
    for t in (500, 533, 566):
        assert sel.offer("F", cf, ids, t) is None
    # L stays still while F is still too: the spans now overlap and one view comes out.
    got = [sel.offer("L", cl, ids, t) for t in (500, 533, 566)]
    views = [v for v in got if v is not None]
    assert len(views) == 1 and len(views[0].ids_L) == 30
    # Same place again: duplicate.
    for t in (600, 633, 666):
        sel.offer("L", cl, ids, t)
        out = sel.offer("F", cf, ids, t + 5)
    assert out is None and "already" in sel.status["L"]
