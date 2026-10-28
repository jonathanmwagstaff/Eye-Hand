import shutil

import numpy as np

from eyehand import boards, intrinsics
from eyehand.cli import intrinsics_cmd
from eyehand.config import DEFAULT_CAMERA_CONFIG, load_camera_configs


def test_margin_status_text_does_not_hide_board_corners():
    spec = boards.SCREEN_BOARD
    image, placement = boards.render_screen(spec)
    text = "views 49/50  coverage 100%  missing: top-left, bottom-right  hold still"
    shown = intrinsics_cmd.draw_margin_status(image, placement, text)
    assert not np.array_equal(shown, image)
    corners, ids = intrinsics.detect(spec.detector(), shown)
    assert len(ids) == (spec.squares_x - 1) * (spec.squares_y - 1)
    # The text stays in the margin above the board.
    assert np.array_equal(shown[placement.offset_y:], image[placement.offset_y:])


def test_write_focus_changes_only_that_camera(tmp_path):
    path = tmp_path / "cameras.toml"
    shutil.copy(DEFAULT_CAMERA_CONFIG, path)
    before = load_camera_configs(path)
    intrinsics_cmd._write_focus("F", 97, path)
    after = load_camera_configs(path)
    assert after["F"].controls["focus_absolute"] == 97
    assert list(after["F"].controls) == list(before["F"].controls)   # order kept
    assert after["L"] == before["L"]


def test_session_roundtrip_solve_and_install(tmp_path, monkeypatch):
    from test_intrinsics import SIZE, make_synthetic_views

    monkeypatch.setattr(intrinsics, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(intrinsics_cmd, "PROJECT_ROOT", tmp_path)
    (tmp_path / "calib").mkdir()
    config = load_camera_configs()["F"]
    session = intrinsics.Session("F", config.usb_id, SIZE, "MJPG", dict(config.controls),
                                 {"name": "print_a4"}, {"x": 1.0, "y": 1.0, "source": "test"},
                                 make_synthetic_views())
    raw = tmp_path / "calib" / "raw" / "intrinsics_F" / "run1"
    session.save(raw)
    loaded = intrinsics.Session.load(raw)
    assert len(loaded.views) == len(session.views)
    assert np.array_equal(loaded.views[0].corners, session.views[0].corners)

    assert intrinsics_cmd._solve_and_report(loaded, raw) == 0
    assert (raw / "result.json").exists()
    installed = tmp_path / "calib" / "intrinsics_F.json"
    # A second install keeps the first in calib/history/.
    assert intrinsics_cmd._solve_and_report(loaded, raw) == 0
    assert installed.exists() and len(list((tmp_path / "calib" / "history").iterdir())) == 1
    # The installed result loads against a config with the same mode and focus.
    import dataclasses
    same = dataclasses.replace(config, width=SIZE[0], height=SIZE[1])
    assert intrinsics.load_intrinsics(same, installed).image_size == SIZE
