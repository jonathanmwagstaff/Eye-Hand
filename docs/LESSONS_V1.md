# Lessons from v1 (`~/hand-eye`)

v1 (2025, built on TemugeB's `python_stereo_camera_calibrate` and `handpose3d`) never produced
good 3D positions. A code review on 2026-10-27 found these causes. Each one has a matching
safeguard in this project.

| # | v1 problem | Location in v1 | Effect | Safeguard here |
|---|---|---|---|---|
| 1 | **Wrong calibration loaded.** `utils.py` read `c0.dat`, `c1.dat`, `rot_trans_c*.dat`, the sample files shipped with TemugeB's repo. The user's own results (`camera*_intrinsics.dat`, `camera*_rot_trans.dat`) were never read. | `cursor_control/utils.py` | Triangulated with someone else's cameras | Calibration files record camera identity (USB ID, mode, focus) and are checked against the live camera at load |
| 2 | **Crop vs calibration mismatch.** Frames were center-cropped 1280 → 720 wide, but the intrinsics assumed the full frame. | `cursor_control.py`, `handpose3d.py` | Principal point off by 280 px → ~150 mm cursor error (simulated) | Mode is part of C0. Any resize or crop must update K, or is forbidden. |
| 3 | **No undistortion** before triangulation | `utils.py DLT` | Large errors toward image edges | Pixel → normalised-ray conversion is mandatory |
| 4 | **Under-sampled, overfit intrinsics.** 10 frames per camera. Distortion k3 ≈ 1.2, k2 ≈ −1.7. Cam1 fx = 1686 vs cam0 982 at the same nominal resolution, cy off by 79 px. No RMS recorded. (The fx difference itself was real: C2 in 2026 measured the Feeltek at fx = 1611 px with a narrow ~43° lens, against 909 px for the laptop camera.) | `calibration_settings.yaml` | Unreliable projection, worst at the edges | 40–60 views, coverage map, simpler distortion model, uncertainty and hold-out gates |
| 5 | **No time synchronisation.** `read()` called on each camera in turn through OpenCV's buffer; timestamps faked as `+= 33` ms | `cursor_control.py` | Views up to ~100 ms apart | Kernel exposure timestamps, drained queues, time alignment (C6) |
| 6 | **Probably the wrong eye.** Index 473 labelled `IRIS_RIGHT`; in MediaPipe 473 is the subject's left iris | `cursor_control.py` | ~14 mm error (simulated) | Dominant-eye check and one-eye-closed index test (C8) |
| 7 | **Geometry replaced by curve fitting.** 28-term polynomial fitted to 18 samples (9 targets × 2 almost identical), with the head nearly still | `gaze_calibrator.py` | A lookup table that fails when the head moves | Physical geometry first; only a low-order user model, checked on held-out targets |
| 8 | Camera auto settings left on (autofocus, auto exposure, dynamic framerate) | — | Intrinsics and timing change at runtime | C0 applies and verifies fixed controls |
| 9 | No acceptance checks at any stage | — | Errors stacked up unseen | Every calibration has a gate ([CALIBRATION.md](CALIBRATION.md)) |
