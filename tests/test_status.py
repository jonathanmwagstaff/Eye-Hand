import json

import numpy as np

from eyehand import status, stereo
from eyehand.config import load_camera_configs
from eyehand.intrinsics import Intrinsics

CONFIGS = load_camera_configs()


def write_intrinsics(calib, name, fx=1000.0):
    c = CONFIGS[name]
    K = [[fx, 0, c.width / 2], [0, fx, c.height / 2], [0, 0, 1]]
    result = {"camera": name, "usb_id": c.usb_id,
              "mode": {"width": c.width, "height": c.height, "fourcc": c.fourcc},
              "geometry_controls": {k: v for k, v in c.controls.items()
                                    if k in ("focus_absolute", "focus_automatic_continuous")},
              "image_size": [c.width, c.height], "K": K, "dist": [0, 0, 0, 0, 0],
              "passed": True, "rms_px": 0.2, "views_used": 50, "solved": "2026-10-28T10:00:00"}
    (calib / f"intrinsics_{name}.json").write_text(json.dumps(result))
    return Intrinsics(name, (c.width, c.height), np.array(K), np.zeros(5))


def write_stereo(calib, iL, iF, solved="2026-10-28T11:00:00"):
    result = {"intrinsics": {"L": stereo.intrinsics_fingerprint(iL),
                             "F": stereo.intrinsics_fingerprint(iF)},
              "passed": True, "solved": solved, "length_test": {"p95_mm": 0.4}}
    (calib / "stereo.json").write_text(json.dumps(result))


def c5(calib):
    return next(s for s in status.calibration_status(CONFIGS, calib) if s.id == "C5")


def test_fresh_project_points_to_c1(tmp_path):
    statuses = status.calibration_status(CONFIGS, tmp_path)
    assert status.next_action(statuses).id == "C1"
    assert c5(tmp_path).state == status.MISSING


def test_stereo_ok_then_moved_then_recalibrated(tmp_path):
    iL, iF = write_intrinsics(tmp_path, "L"), write_intrinsics(tmp_path, "F")
    write_stereo(tmp_path, iL, iF, solved="2000-01-01T00:00:00")
    assert c5(tmp_path).state == status.OK
    status.log_event("moved", "test", tmp_path / "events.jsonl")
    assert c5(tmp_path).state == status.STALE
    # A newer stereo result clears the "moved" event.
    write_stereo(tmp_path, iL, iF, solved="2999-01-01T00:00:00")
    assert c5(tmp_path).state == status.OK


def test_redoing_a_lens_calibration_makes_stereo_stale(tmp_path):
    iL, iF = write_intrinsics(tmp_path, "L"), write_intrinsics(tmp_path, "F")
    write_stereo(tmp_path, iL, iF)
    write_intrinsics(tmp_path, "F", fx=1003.0)
    s = c5(tmp_path)
    assert s.state == status.STALE and "lens" in s.detail


def test_health_alarm_asks_for_a_check(tmp_path):
    iL, iF = write_intrinsics(tmp_path, "L"), write_intrinsics(tmp_path, "F")
    write_stereo(tmp_path, iL, iF, solved="2000-01-01T00:00:00")
    status.log_event("c7_alarm", "face epipolar 9 px", tmp_path / "events.jsonl")
    s = c5(tmp_path)
    assert s.state == status.CHECK and s.usable


def test_screen_pitch_evaluation():
    from eyehand import screen_calib
    r = screen_calib.evaluate([143.35, 143.40, 143.30], [143.70, 143.75, 143.65])
    assert abs(r["pitch_x_mm"] - 143.35 / 800) < 1e-9
    assert r["passed"]
    bad = screen_calib.evaluate([143.0, 143.6, 143.3], [143.7, 143.7, 143.7])
    assert not bad["passed"]   # 0.6 mm spread


def test_emergency_calibration_is_usable_but_provisional(tmp_path):
    iL, iF = write_intrinsics(tmp_path, "L"), write_intrinsics(tmp_path, "F")
    write_stereo(tmp_path, iL, iF)
    path = tmp_path / "stereo.json"
    result = json.loads(path.read_text())
    result["emergency"] = True
    path.write_text(json.dumps(result))
    s = c5(tmp_path)
    assert s.usable and s.state == status.CHECK
    assert "provisional" in s.detail
