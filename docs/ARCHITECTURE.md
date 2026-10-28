# Architecture

How a cursor position is computed, which coordinate frames are involved, and how the
software is split up.

## The idea in one line

Find the 3D positions of the **dominant eye** E and the **index fingertip** F with two
calibrated cameras. The cursor is where the line from E through F hits the screen plane.

## Coordinate frames

All lengths are in **millimetres**. All frames are right-handed.

| Frame | Origin | Axes | Notes |
|---|---|---|---|
| **S** (screen) | Outer top-left corner of the active pixel area | x along pixel rows (right), y along pixel columns (down), z = x × y (**into** the screen) | The user is at z < 0. Pixel (u, v) has its centre at ((u + ½)·pₓ, (v + ½)·p_y, 0). |
| **L** (laptop camera) | Optical centre | OpenCV convention: x right, y down, z forward (out of the screen toward the user) | Rigidly attached to S: both are in the lid |
| **F** (Feeltek camera) | Optical centre | OpenCV convention | On a tripod; its pose relative to L changes when anything moves |

Rigid transforms are written `T_A_B`, meaning "maps points in frame B into frame A":
`X_A = R_A_B · X_B + t_A_B`.

| Transform | Found by | Changes when |
|---|---|---|
| `T_S_L` | Mirror calibration (C4) | Never (rigid in the lid) |
| `T_L_F` | Stereo calibration (C5) | Tripod, laptop or lid angle moves |
| `T_S_F = T_S_L · T_L_F` | derived | as `T_L_F` |

The world frame **is S**. Each camera's projection matrix is expressed in S, so triangulated
points come out directly in screen coordinates.

## The math

### 1. From pixels to rays

Each camera c has intrinsics K_c and distortion d_c (from C1/C2). A MediaPipe landmark at
pixel (u, v) is **undistorted** to normalised image coordinates:

  `x̂ = undistort(K_c, d_c, (u, v))`  → a ray direction `(x̂, ŷ, 1)` in camera frame c.

Undistortion always happens before any geometry. That step was missing in v1
(see [LESSONS_V1.md](LESSONS_V1.md)).

### 2. Triangulation

With the normalised projection matrices `P_c = [R_c_S | t_c_S]` (that is, camera-from-screen),
each 3D point X (in frame S) is found in two stages:

1. **Linear (DLT)** on normalised coordinates, for a starting value.
2. **Gauss–Newton refinement** that minimises the reprojection error in **pixels**, the
   statistically correct cost for pixel noise.

Two quality numbers come out of every triangulation and are logged with it:

- reprojection error per camera (px)
- the angle between the two viewing rays (small angle = poor depth)

Points that fail the gates (for example reprojection error > 4 px) are rejected rather than
fed into the cursor.

### 3. The cursor

With E and F in frame S (screen plane z = 0, user at z < 0):

  `s = E_z / (E_z − F_z)`, `C = E + s·(F − E)`, cursor pixel = `(C_x / pₓ − ½, C_y / p_y − ½)`

How errors propagate (to first order, sideways components):

  `δC ≈ s·δF + (1 − s)·δE`

At a desk (eye about 650 mm from the screen, finger about 450 mm in front of the eye),
**s ≈ 1.44**. A finger error is amplified 1.44×; an eye error is scaled by −0.44.

**Uniform scale cancels.** If the whole reconstruction (eye, finger and the camera-to-screen
transform) is scaled by a common factor about camera L, the line still hits the same screen
point. So the printed square size is not critical for the cursor. Distortion and focal-length
errors are non-linear and do **not** cancel.

### 4. Which "eye point"

- The ray runs from the target, through the fingertip, to the **entrance pupil** of the
  **right** eye, which is about 3 mm behind the cornea.
- MediaPipe gives the iris centre: index **468**, which is the subject's right eye in
  MediaPipe's numbering. *To verify with a one-eye-closed test before relying on it.*
- The few-millimetre difference between iris centre and entrance pupil lies roughly along
  the viewing direction. It is scaled by (1 − s) ≈ −0.44, so it contributes about 1 mm.
  The user model (C8) absorbs what remains.

### 5. Human aiming bias

Even with perfect tracking, people do not aim exactly along the eye–finger line. Mayer et
al. (CHI 2015, OptiTrack) found systematic, user-specific offsets. Calibration C8 fits a
**low-order** correction: a constant angular offset, or at most an affine one. It is checked
on held-out targets. It never replaces the geometry.

## Error budget

Simulated for this geometry (two cameras ~70 cm apart, f ≈ 1000 px, 720p; script:
`tools/error_budget.py`):

| Source | Cursor error |
|---|---|
| Keypoint noise, finger 2 px / eye 1 px | 1.6 mm RMS |
| Keypoint noise, finger 4 px / eye 1.5 px | 3.2 mm RMS |
| 3 % focal-length error in one camera | 7.6 mm bias |
| Inter-camera time skew 16 / 33 ms, finger moving at 0.3 m/s | 2.8 / 5.8 mm |
| 1 mm error in `T_S_L` translation | ~1 mm |
| 1° error in `T_S_L` roll | up to ~4.6 mm (screen corners) |
| Wrong eye | ~14 mm |
| Human aiming bias (scaled from the literature) | probably 10–30 mm before correction |

One screen pixel is about 0.18 mm, so 1 mm ≈ 5.6 px.

## Time model

Each frame k of camera c carries a kernel timestamp `τ_k` (start of exposure, `CLOCK_MONOTONIC`,
exposed by OpenCV's V4L2 backend as `CAP_PROP_POS_MSEC`). The frame number `n_k` is inferred
from the timestamps, so gaps show up as dropped frames.

1. **De-jitter.** Over a sliding window, fit `τ ≈ a_c + b_c·n`. The fitted value replaces the raw
   ±2 ms stamp. Gaps in `n` reveal dropped frames.
2. **Mid-exposure.** The moment the frame represents is `t_k = τ̂_k + e_c/2 + δ_c`, where `e_c` is
   the exposure time and `δ_c` is the camera's residual timestamp bias, measured in C6.
3. **Alignment.** The two cameras run on independent clocks (measured: L 30.07 fps, F 30.00
   fps), so their frames slide past each other: a full frame every ~14 s. Frames are **never** paired by index.
   - v1: for each frame of one camera, linearly interpolate the other camera's 2D landmarks
     to the same `t`, then triangulate.
   - v2: an asynchronous Kalman filter in 3D that takes each camera's 2D observations at
     their own `t`, handles missed detections and different rates, and predicts forward to
     the display time to cancel latency.
4. **Rolling shutter** (second-order): row y of a frame is exposed at about `t_k + (y/H)·r_c`,
   where r_c is the readout time. Modelled only if the measured error says so.

## Software

### Stack

Python 3.12 (managed by `uv`), MediaPipe Tasks API, OpenCV (ChArUco), NumPy and SciPy.
Camera controls are set with `v4l2-ctl`. Python is not the timing bottleneck: timing comes
from kernel timestamps taken at exposure start, not from when Python receives the frame.

### Processes

```
 ┌────────────────────────┐        ┌────────────────────────┐
 │ worker L (process)     │        │ worker F (process)     │
 │  V4L2 capture          │        │  V4L2 capture          │
 │  apply+verify controls │        │  apply+verify controls │
 │  kernel ts + seq       │        │  lock focus after start│
 │  MediaPipe hand + face │        │  MediaPipe hand + face │
 └──────────┬─────────────┘        └───────────┬────────────┘
            │  Observation(cam, seq, t, landmarks px, scores)
            └──────────────┬───────────────────┘
                           ▼
              ┌───────────────────────────┐
              │ fusion (main process)     │
              │  undistort                │
              │  time-align (interp / KF) │
              │  triangulate + gates      │
              │  E,F in screen frame S    │
              │  ray ∩ screen → cursor    │
              │  user model, filtering    │
              │  health monitor (C7)      │
              └───────────┬───────────────┘
                          ▼
              cursor output / overlay / logs
```

- **One process per camera** gets around Python's global interpreter lock (GIL), so each
  camera's MediaPipe runs on its own CPU cores.
- **Only the newest frame matters.** A reader thread per camera dequeues every frame as it
  arrives (4 driver buffers), so the queue never backs up and the driver never runs out of
  buffers. Frame age is measured, not assumed: about 40 ms from exposure start to Python.
  v1 read frames in turn through OpenCV's buffering, which lets frames age by up to ~100 ms.
- **MediaPipe in VIDEO mode** gets each frame's real timestamp (integer ms, monotonic), not
  an invented counter.
- **Everything is logged** (raw timestamps, landmarks, triangulation quality) so problems can
  be replayed offline instead of guessed at.

### Repository layout

```
Eye-Hand/
  README.md
  docs/            SETUP, ARCHITECTURE, CALIBRATION, RECALIBRATION, LESSONS_V1
  config/
    cameras.toml   camera devices, modes, controls
  calib/           calibration results (versioned, one file per calibration)
  src/eyehand/
    config.py      load config/cameras.toml
    v4l2.py        list cameras by stable USB id; set + verify controls via v4l2-ctl
    capture.py     camera stream: reader thread, kernel exposure-start timestamps,
                   auto-lock exposure, focus lock after stream start
    timebase.py    timestamp de-jitter, frame-period and drift estimation
    boards.py      ChArUco board definitions, printable and screen (normal and mirrored)
    intrinsics.py  C1/C2: view selection, coverage, solve + gates, save/load with mode check
    stereo.py      C5: synchronised-stillness view selection, bundle adjustment, gates
    landmarks.py   MediaPipe hand + face (pinned, hash-checked models), pixel convention
    status.py      which calibrations are valid/stale and why; calib/events.jsonl
    app.py         GUI: both cameras live, placement aids, status, guided C5, C7 monitor
    sound.py       audio feedback during calibration
    cli/           command-line tools (probe, timing, preview, boards, focus, intrinsics,
                   stereo, status, app, install-launcher)
  tools/           one-off analysis scripts (error budget, mirror-glass simulation)
  tests/
```
