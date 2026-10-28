import json

import cv2
import numpy as np
import pytest

from eyehand import boards, intrinsics
from eyehand.config import CameraConfig
from eyehand.intrinsics import TargetScale, View

SIZE = (640, 480)
K_TRUE = np.array([[600.0, 0, 331.0], [0, 603.0, 236.0], [0, 0, 1]])
D_TRUE = np.array([-0.28, 0.11, 0.0008, -0.0005, 0.0])
MM_PER_PX = 0.25        # rendering resolution of the synthetic board


def _board_texture():
    spec = boards.PRINT_BOARD
    w, h = spec.squares_x * spec.square / MM_PER_PX, spec.squares_y * spec.square / MM_PER_PX
    img = spec.board().generateImage((int(w), int(h)), marginSize=0, borderBits=1)
    pad = 40  # white margin around the board
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    return cv2.GaussianBlur(img, (0, 0), 1.0), pad


def _render(texture, pad, rvec, tvec, rays):
    """Image of the board seen by the true camera at pose (rvec, tvec)."""
    R, _ = cv2.Rodrigues(rvec)
    n, t = R[:, 2], tvec.ravel()
    s = (n @ t) / (rays @ n)
    Xb = (s[:, None] * rays - t) @ R            # board coordinates (mm), z ~ 0
    mx = (Xb[:, 0] / MM_PER_PX + pad - 0.5).reshape(SIZE[1], SIZE[0]).astype(np.float32)
    my = (Xb[:, 1] / MM_PER_PX + pad - 0.5).reshape(SIZE[1], SIZE[0]).astype(np.float32)
    behind = (s < 0).reshape(SIZE[1], SIZE[0])
    mx[behind] = -1
    img = cv2.remap(texture, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                    borderValue=255)
    return img


def make_synthetic_views(seed=3):
    rng = np.random.default_rng(seed)
    texture, pad = _board_texture()
    gx, gy = np.meshgrid(np.arange(SIZE[0]), np.arange(SIZE[1]))
    px = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float64)
    nrm = intrinsics.undistort_normalized(px, K_TRUE, D_TRUE)
    rays = np.hstack([nrm, np.ones((len(nrm), 1))])

    spec = boards.PRINT_BOARD
    detector = spec.detector()
    scale = TargetScale(1.0, 1.0, "nominal")
    centre = np.array([spec.squares_x, spec.squares_y, 0]) * spec.square / 2
    views = []
    targets = [(u, v) for u in (0.2, 0.5, 0.8) for v in (0.2, 0.5, 0.8)] * 3
    for k, (u, v) in enumerate(targets):
        tilt = np.deg2rad(rng.uniform(-35, 35, 2))
        R = cv2.Rodrigues(np.array([tilt[0], tilt[1], rng.uniform(-0.3, 0.3)]))[0]
        dist = rng.uniform(380, 550)
        ray = np.array([(u * SIZE[0] - K_TRUE[0, 2]) / K_TRUE[0, 0],
                        (v * SIZE[1] - K_TRUE[1, 2]) / K_TRUE[1, 1], 1.0])
        t = ray / np.linalg.norm(ray) * dist - R @ centre
        img = _render(texture, pad, cv2.Rodrigues(R)[0], t, rays)
        img = np.clip(img + rng.normal(0, 2, img.shape), 0, 255).astype(np.uint8)
        corners, ids = intrinsics.detect(detector, img)
        if ids is None or len(ids) < intrinsics.MIN_CORNERS:
            continue
        views.append(View(f"v{k:02d}", ids, corners, intrinsics.object_points(spec, ids, scale)))
    return views


@pytest.fixture(scope="module")
def synthetic_views():
    return make_synthetic_views()


def test_solve_recovers_true_camera(synthetic_views):
    assert len(synthetic_views) >= 20
    report = intrinsics.solve(synthetic_views, SIZE)
    K = np.asarray(report["K"])
    assert abs(K[0, 0] / K_TRUE[0, 0] - 1) < 0.005
    assert abs(K[1, 1] / K_TRUE[1, 1] - 1) < 0.005
    assert np.abs(K[:2, 2] - K_TRUE[:2, 2]).max() < 2.0
    # What matters is the pixel-to-ray mapping (up to a rotation, which extrinsics absorb),
    # not the individual coefficients.
    worst, rms = intrinsics.model_difference_px(K_TRUE, D_TRUE, K, np.asarray(report["dist"]),
                                                SIZE)
    assert rms < 0.2 and worst < 0.6
    assert report["passed"], intrinsics.format_report(report)


def test_model_difference_is_zero_for_identical_and_grows_with_focal_error():
    worst, _ = intrinsics.model_difference_px(K_TRUE, D_TRUE, K_TRUE, D_TRUE, SIZE)
    assert worst < 1e-3
    K2 = K_TRUE.copy()
    K2[0, 0] *= 1.01
    worst, _ = intrinsics.model_difference_px(K_TRUE, D_TRUE, K2, D_TRUE, SIZE,
                                              allow_rotation=False)
    assert 2.5 < worst < 4.5   # ~1 % of the largest x distance from the principal point
    # A principal point shift is (almost) a pure rotation: nearly nothing remains.
    K3 = K_TRUE.copy()
    K3[0, 2] += 3.0
    raw, _ = intrinsics.model_difference_px(K_TRUE, D_TRUE, K3, D_TRUE, SIZE, allow_rotation=False)
    rotated, _ = intrinsics.model_difference_px(K_TRUE, D_TRUE, K3, D_TRUE, SIZE)
    assert raw > 2.9 and rotated < 0.5


def _grid(offset=(0.0, 0.0)):
    spec = boards.PRINT_BOARD
    ids = np.arange(30, dtype=np.int32)
    obj = spec.board().getChessboardCorners()[ids, :2]
    return (obj * 2 + 100 + np.asarray(offset)).astype(np.float32), ids


def test_view_selector_needs_stillness_and_novelty():
    sel = intrinsics.ViewSelector(boards.PRINT_BOARD.board(), still_frames=3)
    c, ids = _grid()
    assert sel.offer(c, ids) is None and sel.offer(c + 0.1, ids) is None
    got = sel.offer(c + 0.2, ids)
    assert got is not None
    got_ids, mean, _, n = got
    assert n == 3 and np.allclose(mean, c[np.searchsorted(ids, got_ids)] + 0.1, atol=1e-4)
    for _ in range(3):                       # same place again: duplicate
        assert sel.offer(c, ids) is None
    assert "already" in sel.status
    c2, _ = _grid((5.0, 0.0))                # jumps 5 px between frames: moving
    assert sel.offer(c2, ids) is None and sel.status == "moving"
    far, _ = _grid((200.0, 50.0))
    results = [sel.offer(far, ids) for _ in range(3)]
    assert results[-1] is not None


def test_coverage_counts_and_corner_zones():
    cov = intrinsics.Coverage((640, 480), cell_px=80)
    assert (cov.cols, cov.rows) == (8, 6)
    # Zones are 12 % x 15 %: x < 76.8 or > 563.2, y < 72 or > 408.
    cov.add(np.array([[70, 70], [570, 60]], np.float32))
    assert cov.missing_corners() == ["bottom-left", "bottom-right"]
    assert cov.fraction() == pytest.approx(2 / 48)
    cov.add(np.array([[70, 70], [570, 60]], np.float32), sign=-1)
    assert len(cov.missing_corners()) == 4


def _config(**controls):
    return CameraConfig("F", "", "/dev/null", "1bcf:0b15", 1280, 960, "MJPG", 30,
                        controls=controls)


def test_load_refuses_changed_focus_or_mode(tmp_path):
    result = {"camera": "F", "usb_id": "1bcf:0b15",
              "mode": {"width": 1280, "height": 960, "fourcc": "MJPG"},
              "geometry_controls": {"focus_absolute": 120, "focus_automatic_continuous": 0},
              "image_size": [1280, 960], "K": K_TRUE.tolist(), "dist": D_TRUE.tolist(),
              "passed": True}
    path = tmp_path / "intrinsics_F.json"
    path.write_text(json.dumps(result))
    ok = _config(focus_automatic_continuous=0, focus_absolute=120)
    assert intrinsics.load_intrinsics(ok, path).K[0, 0] == 600.0
    with pytest.raises(intrinsics.StaleCalibration, match="focus_absolute"):
        intrinsics.load_intrinsics(_config(focus_automatic_continuous=0, focus_absolute=128), path)
    cropped = CameraConfig("F", "", "/dev/null", "1bcf:0b15", 1280, 720, "MJPG", 30,
                           controls={"focus_automatic_continuous": 0, "focus_absolute": 120})
    with pytest.raises(intrinsics.StaleCalibration, match="mode"):
        intrinsics.load_intrinsics(cropped, path)


def _tilted_view(yaw_deg, pitch_deg):
    """Board corners as seen by a 640x480 camera, board turned by yaw (about the vertical)
    and pitch (about the horizontal), 500 mm away."""
    spec = boards.PRINT_BOARD
    ids = np.arange((spec.squares_x - 1) * (spec.squares_y - 1), dtype=np.int32)
    obj = spec.board().getChessboardCorners()[ids].astype(np.float64)
    centre = np.array([spec.squares_x, spec.squares_y, 0]) * spec.square / 2
    R = cv2.Rodrigues(np.deg2rad([pitch_deg, yaw_deg, 0.0]))[0]
    t = np.array([0, 0, 500.0]) - R @ centre
    K = np.array([[480.0, 0, 320], [0, 480, 240], [0, 0, 1]])
    px, _ = cv2.projectPoints(obj, cv2.Rodrigues(R)[0], t, K, None)
    return ids, px.reshape(-1, 2).astype(np.float32)


def test_board_tilt_names_the_far_edge_in_the_mirrored_view():
    spec = boards.PRINT_BOARD
    flat = intrinsics.board_tilt(spec, *_tilted_view(0, 0), (640, 480))
    assert max(flat.values()) < 2
    # +30 deg about the camera's y axis maps the board's x axis to (cos, 0, -sin): its
    # image-right end comes closer, so the image-LEFT end is farther. Image-left is the
    # RIGHT of the mirrored view.
    t = intrinsics.board_tilt(spec, *_tilted_view(30, 0), (640, 480))
    assert 25 < t["right"] < 35 and t["left"] == 0
    t = intrinsics.board_tilt(spec, *_tilted_view(-30, 0), (640, 480))
    assert 25 < t["left"] < 35 and t["right"] == 0
    # +30 deg about x pushes the image-bottom end away (y down, z forward).
    t = intrinsics.board_tilt(spec, *_tilted_view(0, 30), (640, 480))
    assert 25 < t["bottom"] < 35 and t["top"] == 0


def test_tilt_tracker_needs_every_direction():
    tr = intrinsics.TiltTracker()
    for _ in range(3):
        tr.add({"left": 25, "right": 0, "top": 30, "bottom": 0})
        tr.add({"left": 0, "right": 22, "top": 0, "bottom": 5})
    assert not tr.complete()
    for _ in range(3):
        tr.add({"left": 0, "right": 0, "top": 0, "bottom": 21})
    assert tr.complete()
    tr.add({"left": 0, "right": 0, "top": 0, "bottom": 21}, sign=-1)
    tr.add({"left": 0, "right": 0, "top": 0, "bottom": 21}, sign=-1)
    assert not tr.complete()
