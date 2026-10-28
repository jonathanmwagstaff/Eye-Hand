import cv2
import numpy as np

from eyehand import boards


def detect(spec, image):
    corners, ids, _, _ = spec.detector().detectBoard(image)
    return corners, ids


def test_screen_board_detected_with_all_corners():
    spec = boards.SCREEN_BOARD
    image, placement = boards.render_screen(spec)
    corners, ids = detect(spec, image)
    n_inner = (spec.squares_x - 1) * (spec.squares_y - 1)
    assert ids is not None and len(ids) == n_inner


def test_screen_board_corners_map_to_screen_pixels():
    """Detected corner positions equal the board->screen mapping of their object points."""
    spec = boards.SCREEN_BOARD
    image, placement = boards.render_screen(spec)
    corners, ids = detect(spec, image)
    obj = spec.board().getChessboardCorners()[ids.ravel(), :2]
    expected = placement.board_to_screen_px(obj)
    # OpenCV pixel centres are at integer coordinates; our screen coordinates put pixel
    # (u, v) at [u, u+1), so detected positions are 0.5 px lower.
    err = corners.reshape(-1, 2) + 0.5 - expected
    assert np.abs(err).max() < 0.1


def test_mirrored_board_is_rejected_directly_but_reads_in_a_mirror():
    spec = boards.SCREEN_BOARD
    mirrored, placement = boards.render_screen(spec, mirrored=True)
    _, ids_direct = detect(spec, mirrored)
    assert ids_direct is None or len(ids_direct) == 0

    reflected = np.ascontiguousarray(mirrored[:, ::-1])  # what a camera sees in a mirror
    corners, ids = detect(spec, reflected)
    assert ids is not None and len(ids) == (spec.squares_x - 1) * (spec.squares_y - 1)
    # The mapping must give the real screen location of each corner on the mirrored display.
    obj = spec.board().getChessboardCorners()[ids.ravel(), :2]
    on_screen = placement.board_to_screen_px(obj)
    seen_in_mirror = corners.reshape(-1, 2) + 0.5
    assert np.abs(on_screen[:, 0] - (placement.screen_width - seen_in_mirror[:, 0])).max() < 0.1
    assert np.abs(on_screen[:, 1] - seen_in_mirror[:, 1]).max() < 0.1


def test_print_board_square_rounding():
    image, square_mm = boards.render_print(dpi=600)
    spec = boards.PRINT_BOARD
    assert abs(square_mm - spec.square) < 25.4 / 600
    assert image.shape == (spec.squares_y * round(spec.square / 25.4 * 600),
                           spec.squares_x * round(spec.square / 25.4 * 600))
    corners, ids = detect(spec, cv2.resize(image, None, fx=0.25, fy=0.25))
    assert ids is not None and len(ids) == (spec.squares_x - 1) * (spec.squares_y - 1)


def test_fullscreen_is_requested_after_first_frame_and_verified(monkeypatch):
    calls = []
    monkeypatch.setattr(cv2, "namedWindow", lambda *args: calls.append("named"))
    monkeypatch.setattr(cv2, "imshow", lambda *args: calls.append("show"))
    monkeypatch.setattr(cv2, "waitKey", lambda *args: calls.append("wait"))
    monkeypatch.setattr(cv2, "setWindowProperty", lambda *args: calls.append("fullscreen"))
    monkeypatch.setattr(cv2, "getWindowImageRect", lambda *args: (0, 0, 1920, 1080))

    boards.open_fullscreen("board", np.zeros((1080, 1920), np.uint8))

    assert calls[:4] == ["named", "show", "wait", "fullscreen"]


def test_fullscreen_rejects_rescaled_board(monkeypatch):
    monkeypatch.setattr(cv2, "namedWindow", lambda *args: None)
    monkeypatch.setattr(cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(cv2, "waitKey", lambda *args: None)
    monkeypatch.setattr(cv2, "setWindowProperty", lambda *args: None)
    monkeypatch.setattr(cv2, "getWindowImageRect", lambda *args: (0, 0, 1600, 900))
    closed = []
    monkeypatch.setattr(cv2, "destroyWindow", lambda *args: closed.append(True))

    with np.testing.assert_raises_regex(RuntimeError, "1600x900"):
        boards.open_fullscreen("board", np.zeros((1080, 1920), np.uint8))
    assert closed
