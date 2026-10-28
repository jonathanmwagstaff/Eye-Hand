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
- **Camera F:** Feeltek FHD Pro 1080p on a tripod to the user's left, 1280×960 @ 30 fps

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
uv run eyehand app          # GUI; choose Cursor only or Cursor + hand/eye videos (localhost:8765)
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

Choose **Cursor + hand/eye videos** to share GUI tracking while keeping previews and calibration
checks active, or **Cursor only** for the direct pipeline with previews and checks paused.
Both buttons open the browser. **Stop cursor** returns from cursor-only mode to the GUI;
either mode button switches after the previous pipeline releases its cameras. Both modes share the same eye/hand settings and calibration.
Neither mode smooths or interpolates movement. Each cursor updates immediately on a valid
position and holds its last valid position for up to three seconds during tracking loss,
then hides. Missing/invalid updates do not extend that timeout; each hand expires independently.

The cursor page has in-page **Move shapes** and **3D Viewer** demo tabs. Move shapes
contains a circle, triangle, a photo icon with an Upload photos drop zone,
a Two of Spades with an Ace of Spades target,
and a junk files folder with a Recycle Bin target. Hold the mouse button or pinch
and hold with the primary claw to drag; release to drop. Shapes stay inside the play
area. Tracking loss pauses a grabbed shape without releasing it; movement resumes with
valid tracking. The dotted shape outlines only change color: they never snap, lock, move,
or release shapes. Focused shapes also move with arrow keys.

**3D Viewer** displays the supplied toroidal propeller by Akshay Venugopal. Hold and
drag anywhere in the viewer with the mouse to rotate; scroll to zoom. With the primary
claw, pinch anywhere in the viewer (including empty space), then move the tracked hand
left/right or up/down to rotate with the cursor using a virtual trackball. Move toward
the screen to zoom out, away to zoom in.
Release to hold the orientation. Each new pinch starts from the current view. Tracking
loss pauses motion. **Reset view** restores the initial orientation and zoom. Arrow keys
rotate a focused viewer; +/− zoom. The mesh is centered at the rotation origin and all
viewer assets are served locally. The user-provided archive includes no mesh license;
source attribution is retained in `src/eyehand/viewer/ATTRIBUTION.txt`.

In the cursor page, expand **Cursor controls** to choose the aiming eye (L/R) and dominant
clicker hand (L/R). Both hands use that eye: red is the primary cursor, blue is the independent
secondary cursor. Cursors are mirrored left/right crab claws without labels. Each hand
needs its thumb and index tips visible to both cameras.
The **Thumb / index ratio** slider controls the aiming point: 0% aims through the index tip, 50% through the
3D midpoint, and 100% through the thumb tip. The default is 75% toward the thumb, matching
the previous behavior; the ratio applies to both cursors in either mode. Changes apply while
dragging the slider. Settings last for the current app/server session.
The **Applied** value confirms the ratio saved by the server. Cursor tabs share one tracking
connection so multiple open tabs cannot block settings requests.

Close thumb and index to press and hold; open them clearly to release. This sends separate
mouse-down and mouse-up actions; ordinary clicks activate on release, and shape drags
remain held between those actions. The claw closes
and turns green during a pinch. The dominant hand sends a normal click and the other hand
sends a secondary (`contextmenu`) event to the element under its claw, inside the cursor
page only. This does not control other pages or the desktop, or open the browser's native
right-click menu. Either camera can detect the pinch or release independently, even if the
other camera disagrees or cannot see that hand. Each camera must see an opening before
it can trigger another press, preventing disagreement from producing repeated clicks;
the pinch stays latched through partial opening and missing observations until a clearly
open gap is observed. **Release sensitivity** adjusts how far the fingers must open:
0% requires a wide opening (0.85 palm widths), 100% releases with a small opening (0.35),
and the default 50% releases at 0.60. The closing threshold stays at 0.30. Changes apply
while dragging and are shared by both modes.
Losing page focus cancels a drag. A fresh pinch can click a held position until its three-second
timeout, but stale gesture data cannot click. The page must be focused.

**Space** is a backup click: it prioritizes the currently tracked dominant hand. If only
the other hand is tracked, Space sends a secondary click instead. Holding Space keeps
that claw closed and green without repeated presses; releasing Space releases the button
unless the physical pinch is still held.
Claws held at old positions do not count as currently tracked hands. Space retains its
normal behavior while a form control or menu summary is focused.

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
