import numpy as np
import pytest

from eyehand import browser_cursor, cursor
from eyehand.intrinsics import Intrinsics
from eyehand.landmarks import Landmarks, LEFT_IRIS, RIGHT_IRIS


K = np.array([[500.0, 0, 640], [0, 500.0, 480], [0, 0, 1]])
INTR_L = Intrinsics("L", (1280, 960), K, np.zeros(5))
INTR_F = Intrinsics("F", (1280, 960), K, np.zeros(5))
T_F_L = np.eye(4)
T_F_L[0, 3] = -100.0
T_S_L = np.diag([-1.0, 1.0, -1.0, 1.0])
SCREEN_MM = (344.0, 194.0)


def test_browser_page_is_utf8():
    assert "Eye-Hand browser cursor" in browser_cursor.PAGE.decode()


def test_pairing_prefers_fresh_aligned_frames():
    old_L, old_F = type("L", (), {"t_ms": 0.0})(), type("L", (), {"t_ms": 0.0})()
    new_L, new_F = type("L", (), {"t_ms": 100.0})(), type("L", (), {"t_ms": 115.0})()
    assert browser_cursor._best_pair([old_L, new_L], [old_F, new_F]) == (new_L, new_F)


def project(points, transform=np.eye(4)):
    camera = (transform[:3, :3] @ points.T).T + transform[:3, 3]
    return camera[:, :2] / camera[:, 2:] * 500 + [640, 480]


def observations():
    # Eye and fingertip in screen coordinates. Their ray hits the screen at (175, 65) mm.
    points_S = np.array([[100.0, 50.0, -600.0], [150.0, 60.0, -200.0]])
    points_L = (T_S_L[:3, :3].T @ points_S.T).T
    return project(points_L), project(points_L, T_F_L)


def test_pointer_intersects_screen():
    px_L, px_F = observations()
    point = cursor.solve_pointer(px_L, px_F, INTR_L, INTR_F, T_F_L, T_S_L, SCREEN_MM)
    assert point.x_norm == pytest.approx(175 / 344)
    assert point.y_norm == pytest.approx(65 / 194)
    assert point.reprojection_px < 1e-6


def test_pointer_aims_through_weighted_3d_pinch_point():
    points_S = np.array([[100., 50., -600.], [150., 60., -200.], [110., 80., -240.]])
    points_L = (T_S_L[:3, :3].T @ points_S.T).T
    point = cursor.solve_pointer(project(points_L), project(points_L, T_F_L),
                                 INTR_L, INTR_F, T_F_L, T_S_L, SCREEN_MM)
    pinch = .25 * points_S[1] + .75 * points_S[2]
    hit = points_S[0] + (pinch - points_S[0]) * (-points_S[0, 2] / (pinch[2] - points_S[0, 2]))
    assert point.x_norm == pytest.approx(hit[0] / SCREEN_MM[0])
    assert point.y_norm == pytest.approx(hit[1] / SCREEN_MM[1])


def test_pointer_rejects_mismatched_landmarks():
    px_L, px_F = observations()
    px_F[1] += [80, 50]
    with pytest.raises(cursor.InvalidPointer, match="reprojection"):
        cursor.solve_pointer(px_L, px_F, INTR_L, INTR_F, T_F_L, T_S_L, SCREEN_MM)


def two_hand_observations():
    # Independent hands with different screen intersections, plus both iris positions.
    points_S = np.array([[100., 50., -600.], [110., 50., -600.],
                         [150., 60., -200.], [145., 60., -200.],
                         [120., 70., -200.], [115., 70., -200.]])
    points_L = (T_S_L[:3, :3].T @ points_S.T).T
    observations = []
    for transform in (np.eye(4), T_F_L):
        px = project(points_L, transform)
        face = np.zeros((478, 2))
        face[RIGHT_IRIS], face[LEFT_IRIS] = px[:2]
        hands = []
        for label, indices in (("Right", (2, 3)), ("Left", (4, 5))):
            hand = np.zeros((21, 2))
            hand[[8, 4]] = px[list(indices)]
            hands.append((label, 0.99, hand))
        observations.append(Landmarks(100, face, hands))
    return observations


def test_independent_hands_and_selected_eye():
    a, b = two_hand_observations()
    intr = {"L": INTR_L, "F": INTR_F}
    right_eye = browser_cursor._pointer_samples(a, b, "R", intr, T_F_L, T_S_L, SCREEN_MM)
    assert right_eye["L"]["valid"] and right_eye["R"]["valid"]
    assert right_eye["L"]["x"] != right_eye["R"]["x"]
    left_eye = browser_cursor._pointer_samples(a, b, "L", intr, T_F_L, T_S_L, SCREEN_MM)
    assert left_eye["R"]["valid"]
    assert left_eye["R"]["x"] != right_eye["R"]["x"]
    b.hands = [h for h in b.hands if h[0] == "Right"]
    missing = browser_cursor._pointer_samples(a, b, "R", intr, T_F_L, T_S_L, SCREEN_MM)
    assert missing["R"]["valid"] and not missing["L"]["valid"]


def test_pinch_points_preserves_default_and_chooses_same_hand():
    a, _ = two_hand_observations()
    np.testing.assert_array_equal(a.pinch_points(), a.right_hand()[[8, 4]])
    np.testing.assert_array_equal(a.pinch_points("L"), a.hand("L")[[8, 4]])
    with pytest.raises(ValueError):
        a.pinch_points("invalid")


def test_settings_validate_sides_and_return_a_copy():
    samples = browser_cursor.Samples()
    assert samples.settings() == {"eye": "R", "primary_hand": "R"}
    samples.set_settings({"eye": "L", "primary_hand": "L"})
    copy = samples.settings()
    copy["eye"] = "R"
    assert samples.settings()["eye"] == "L"
    for invalid in ({"eye": "X", "primary_hand": "R"}, {"eye": "L"}, [],
                    {"eye": "L", "primary_hand": ["R"]}):
        with pytest.raises(ValueError):
            samples.set_settings(invalid)


def test_settings_endpoint_rejects_foreign_origins_and_malformed_payloads():
    import io
    import json
    from types import SimpleNamespace
    from unittest.mock import Mock

    samples = browser_cursor.Samples()
    handler_type = browser_cursor._handler(samples)
    handler = handler_type.__new__(handler_type)
    handler.path = "/settings"
    handler.server = SimpleNamespace(server_port=8765)
    handler.send_error, handler.send_response, handler.end_headers = Mock(), Mock(), Mock()
    body = json.dumps({"eye": "L", "primary_hand": "L"}).encode()
    handler.headers = {"Origin": "http://127.0.0.1:8765", "Content-Length": str(len(body))}
    handler.rfile = io.BytesIO(body)
    handler.do_POST()
    handler.send_response.assert_called_once_with(204)
    assert samples.settings() == {"eye": "L", "primary_hand": "L"}
    handler.headers["Origin"] = "https://example.com"
    handler.do_POST()
    handler.send_error.assert_called_with(403)
    handler.headers = {"Content-Length": "999999"}
    handler.do_POST()
    handler.send_error.assert_called_with(400)
    handler.headers = {"Content-Length": "1"}
    handler.rfile = io.BytesIO(b"{")
    handler.do_POST()
    handler.send_error.assert_called_with(400)
