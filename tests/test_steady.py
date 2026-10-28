import cv2
import numpy as np

from eyehand.steady import SteadyMeter


def test_measures_a_known_shift_and_roll_despite_a_moving_object():
    rng = np.random.default_rng(0)
    scene = cv2.GaussianBlur((rng.random((960, 1280)) * 255).astype(np.uint8), (0, 0), 2)
    scene = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
    meter = SteadyMeter(scene, focal_px=1611.0)
    M = cv2.getRotationMatrix2D((640, 480), 0.5, 1.0)     # roll about the centre ...
    M[:, 2] += (14.0, -6.0)                                # ... plus a turn of the camera
    moved = cv2.warpAffine(scene, M, (1280, 960), borderMode=cv2.BORDER_REFLECT)
    moved[300:700, 400:700] = 255 - moved[300:700, 400:700]   # a "person" moving in view
    m = meter.measure(moved)
    assert abs(abs(m.roll_deg) - 0.5) < 0.05
    assert abs(m.dx_px - 14.0) < 0.5 and abs(m.dy_px + 6.0) < 0.5
    assert abs(m.tilt_deg - np.degrees(np.arctan(np.hypot(14, 6) / 1611))) < 0.02
    still = meter.measure(scene)
    assert still.tilt_deg < 0.01 and abs(still.roll_deg) < 0.01
