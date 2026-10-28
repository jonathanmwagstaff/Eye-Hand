# Eye-Hand

A cursor you aim like sighting along your finger. Two webcams find the 3D positions of your
**dominant (right) eye** and your **index fingertip**. The cursor goes where the line from the
eye through the fingertip meets the laptop screen.

The goal is **absolute accuracy from cheap hardware**: careful spatial and temporal calibration,
with a measured acceptance check at every step.

## Documents

| Document | Contents |
|---|---|
| [docs/SETUP.md](docs/SETUP.md) | Hardware (measured), camera settings and their quirks, placement, lighting, equipment |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Coordinate frames, the math, error budget, time model, software structure |
| [docs/CALIBRATION.md](docs/CALIBRATION.md) | Every calibration (C0–C9): procedure, result file, acceptance gate |
| [docs/RECALIBRATION.md](docs/RECALIBRATION.md) | What to redo when the Feeltek, the laptop or the lid moves, and anything else |
| [docs/LESSONS_V1.md](docs/LESSONS_V1.md) | Why the 2025 attempt (`~/hand-eye`) failed, and the safeguard for each cause |

## Hardware

- ASUS Vivobook 15 X1503ZA: Samsung OLED panel, 1920×1080, 344 × 194 mm
- **Camera L:** built-in webcam, 1280×720 @ 30 fps
- **Camera F:** Feeltek FHD Pro 1080p on a tripod to the user's right, 1280×960 @ 30 fps

## The app

`uv run eyehand install-launcher` adds **Eye-Hand** to the applications menu (already done on
this laptop). The app shows both cameras live for placement (right eye, right fingertip, the
angle between the cameras), the state of every calibration, and a button that redoes the
camera-position calibration after anything moves, with no terminal needed. See
[docs/RECALIBRATION.md](docs/RECALIBRATION.md).

## Quick start

```bash
uv sync                     # Python 3.12, MediaPipe, OpenCV, NumPy, SciPy
uv run pytest               # unit tests
uv run eyehand probe        # C0: cameras found, controls verified, 30 fps, no drops
uv run eyehand timing       # both cameras: frame periods, drift, phase; saves raw stamps
uv run eyehand preview      # live view (add --board print to see ChArUco detection)
uv run eyehand boards       # write ChArUco boards to boards/ (print one at 100 %)
uv run eyehand show-board   # full-screen board on the laptop (--mirrored for C4)
uv run eyehand focus F      # C2 step 1: does focus do anything? (this unit: fixed-focus)
uv run eyehand intrinsics capture L --square-mm 24.0      # C1, printed board
uv run eyehand intrinsics capture F --square-mm 24.0      # C2, printed board
uv run eyehand intrinsics solve calib/raw/intrinsics_L/<timestamp>   # re-solve a capture
uv run eyehand intrinsics compare a.json b.json           # do two calibrations agree?
uv run eyehand status       # which calibrations are valid, stale or missing, and what's next
uv run eyehand app          # GUI; choose Cursor only or Cursor with videos (localhost:8765)
uv run eyehand browser-cursor  # development cursor at http://127.0.0.1:8765
```

In the GUI, **Emergency camera position recalibration** captures just three distinct board
poses, then applies a provisional camera-position estimate. Keep cameras fixed; hold the
board still for each click, then move or tilt it. Basic reprojection, epipolar and known-length
checks must pass; this skips full held-out, coverage and capture-drift validation and the face
baseline wait. Previous calibration is archived. Run full calibration later for accuracy and
face-monitor baseline. Laptop camera movement relative to the screen
also requires C4 recalibration.
Starting a recalibration stops the browser cursor server.

Choose **Cursor with videos** to share GUI tracking while keeping previews and calibration
checks active, or **Cursor only** for the direct pipeline with previews and checks paused.
Both buttons open the browser. **Stop cursor** returns from cursor-only mode to the GUI;
stop first to switch back to videos. Both modes share the same eye/hand settings and calibration.
Smoothing and hysteresis are not applied in either mode.

In the cursor page, expand **Cursor controls** to choose the aiming eye (L/R) and dominant
clicker hand (L/R). Both hands use that eye: red is the primary cursor, blue is the independent
secondary cursor. Each hand needs its thumb and index tips visible to both cameras. Choices
last for the current app/server session. Clicking is not implemented; the secondary hand is intended
for secondary/right-click actions later.

## Status

| Stage | State |
|---|---|
| Project skeleton, camera control and verification (C0), timestamped capture, timing analysis | done, tested on the hardware |
| ChArUco boards (print, screen, mirrored screen) | done, tested |
| C1/C2 intrinsics: focus sweep, live capture with coverage map, solve with gates | built, tested on synthetic images; not yet run with the board |
| App: dual live view, placement aids, calibration status, guided C5, C7 face check | built; ran on both cameras (30 fps each with tracking) |
| C5 stereo solver and gates | built, tested on synthetic data |
| C3 screen, C4 mirror | next |
| C6 timing (LED) | planned |
| MediaPipe landmarks (models pinned) | built |
| Development browser cursor: live landmarks, triangulation, screen intersection | built; C6/C8 still required for accuracy |
| Production process workers and Linux cursor output | planned |
| C8 user model, C9 validation | planned |
