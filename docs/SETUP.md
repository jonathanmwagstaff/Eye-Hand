# Setup

The physical setup: hardware, camera settings, placement and lighting.
Everything here was measured on the actual machine on 2026-10-27 unless marked *to verify*.

## Hardware

### Laptop: ASUS Vivobook 15 X1503ZA

| Item | Value | Source |
|---|---|---|
| CPU | Intel i7-12700H (14 cores / 20 threads) | `lscpu` |
| RAM | 16 GB | `free` |
| GPU | Intel Iris Xe (Alder Lake-P GT2) | `lspci` |
| OS | Linux 7.0, X11 session | `uname`, `$XDG_SESSION_TYPE` |
| Panel | Samsung OLED **ATNA56YX03-0**, 1920×1080 @ 60 Hz, glossy | EDID |
| Panel active area | **344 × 194 mm** (EDID, rounded to whole mm) | EDID, *to verify with calipers* |
| Nominal pixel pitch | 344/1920 = 0.1792 mm, 194/1080 = 0.1796 mm | derived |

The EDID size is rounded to 1 mm, so it is only good to about ±0.3 %. The calipers measurement in
[CALIBRATION.md § C3](CALIBRATION.md#c3--screen-geometry) replaces it.

### Camera L: built-in laptop webcam

| Item | Value |
|---|---|
| USB ID | `13d3:5463` (Azurewave "USB2.0 HD UVC WebCam"), **internal USB 2.0**, `uvcvideo` driver |
| Stable path | `/dev/v4l/by-id/usb-Azurewave_USB2.0_HD_UVC_WebCam_0x0001-video-index0` |
| Mode used | **1280×720 MJPEG @ 30 fps** (the best mode it has; YUYV 720p is only 10 fps) |
| Focus | **Fixed focus**: no focus control exists, so intrinsics are stable |
| Exposure | `exposure_time_absolute` 50–10000 (units of 100 µs; the scale is correct, brightness grows in proportion) |
| **Hidden gain** | Auto exposure applies roughly **20× extra gain** that no control exposes. In manual mode it is gone: 8.3 ms gives a black image (0.5/255) and 33 ms with maximum `gain` gives 2.8/255, where auto gives 145/255 at the same frame rate. **Workaround (measured to work):** let auto exposure settle, then set `auto_exposure=1` *without* writing an exposure value. The camera keeps the auto-chosen exposure and gain (brightness 77.7 → 77.3, still 30 fps). This is `exposure_mode = "auto-lock"`. The cost: the real exposure time is unknown (≤ 33 ms) and can differ per session. |
| Problem control | `exposure_dynamic_framerate` is **on by default**. In dim light it silently halved the frame rate (68 ms frame interval measured). **Must be 0.** |
| White balance | Auto by default; lock it (`white_balance_automatic=0`) |
| Timestamps | Kernel buffer timestamps flagged **`ts-monotonic, ts-src-soe`** (start of exposure, `CLOCK_MONOTONIC`). Measured jitter ±2 ms (frame intervals alternate 32 / 36 ms). |
| Metadata node | `video-index1` streams raw UVC payload headers (device PTS/SCR clock), a finer timing source if ever needed |

### Camera F: Feeltek FHD Pro 1080p (external, on tripod)

| Item | Value |
|---|---|
| USB ID | `1bcf:0b15` (Sunplus Innovation, "Feeltek Full HD Webcam 1080P"), USB 2.0 |
| Stable path | `/dev/v4l/by-id/usb-SunplusIT_Inc_Feeltek_Full_HD_Webcam_1080P-video-index0` |
| Sensor | 5 MP, 2592×1944 (4:3) |
| MJPEG modes | 2592×1944 @ 15, 1920×1080 @ **20**, **1280×960 @ 30**, 1280×720 @ 30, smaller @ 30 |
| Mode used | **1280×960 MJPEG @ 30 fps**: the full 4:3 field of view at full frame rate. Measured 2026-10-27 by matching frames against the 2592×1944 mode: 1280×960 covers the **whole sensor** (scaled), while 1280×720 and 1920×1080 cut off the top and bottom. |
| Field of view | **Narrow: about 43° × 33°** (fx = 1611 px at 1280×960, from C2). The laptop camera sees about 70° × 43° (fx = 909 px at 1280×720). From 60 cm away, F sees a patch about 47 × 36 cm, so if the eye and fingertip don't both fit inside the box in the app, move the tripod back. |
| Focus | **Continuous autofocus on by default.** `focus_absolute` (0–256) can only be set **after streaming has started**. Setting it on an idle device returns `Input/output error`. |
| Exposure | Manual available: `exposure_time_absolute` 3–2047 (100 µs units). There is no dynamic-framerate control. |
| Timestamps | Same as camera L: `ts-monotonic, ts-src-soe`, ±2 ms jitter |

**Measured 2026-10-27:** `focus_absolute` has no detectable effect. Across 0–256 the image scale showed no trend (0.06 %, against 0.18 % random scatter), so the lens behaves as fixed-focus. The settings below are kept anyway, in case firmware honours them.

Autofocus is the single most dangerous setting on this camera. Every focus change moves the
focal length and distortion, which invalidates the intrinsic calibration. The capture code
disables autofocus, sets a fixed `focus_absolute`, and verifies both by reading them back on
every stream start.

Both cameras streamed together at 30 fps on the same USB 2 controller with no bandwidth problem
(MJPEG at ~25–100 KB per frame).

### Measured timing (`eyehand probe`, `eyehand timing`, 2026-10-27)

| | L | F |
|---|---|---|
| Frame period | **33.254 ms (30.07 fps)** | **33.333 ms (30.00 fps)** |
| Dropped frames (20 s) | 0 | 0 |
| Raw timestamp jitter (RMS about the fitted line) | 1.1 ms | 1.1 ms |
| Exposure start → frame in Python | median 39 ms | median 41 ms |

**The two cameras do not run at the same rate.** The phase between them drifts by
**2.36 ms per second** and sweeps through a full frame (±16.7 ms) about every **14 s**. Frames
must be aligned by timestamp, never paired by index.

**V4L2 buffers:** with only one driver buffer (`CAP_PROP_BUFFERSIZE=1`, a common
"low-latency" tip), the driver drops every frame that ends while Python holds the buffer. That
halved the frame rate in testing. The capture code uses 4 buffers and a reader thread that
dequeues every frame immediately, so the queue never backs up.

## Camera settings (applied by software, never by hand)

All settings live in [`config/cameras.toml`](../config/cameras.toml). The capture module applies
them, reads every value back, and refuses to run if any read-back differs.

| Control | L (laptop) | F (Feeltek) | Why |
|---|---|---|---|
| Format | 1280×720 MJPG | 1280×960 MJPG | Highest resolution that still gives 30 fps |
| Exposure mode | **auto-lock** (see camera L table) | manual | Running auto exposure changes exposure time, and with it motion blur and latency, so it is never left running |
| `exposure_time_absolute` | (left as auto chose) | 83 (8.3 ms) | Short to limit blur. **1/120 s = 8.33 ms** is a whole number of flicker periods for 60 Hz mains lighting, so it avoids banding. |
| `exposure_dynamic_framerate` | 0 | n/a | Keeps the frame rate fixed at 30 fps |
| `gain` | 64 (the visible gain; the hidden auto gain is locked by auto-lock) | n/a (no control) | Brightness comes from lighting first |
| `white_balance_automatic` | 0 | 0 | Stable colours for MediaPipe |
| `focus_automatic_continuous` | n/a | 0 (after stream start) | Stable intrinsics |
| `focus_absolute` | n/a | fixed value chosen once (after stream start) | Must equal the value used during intrinsic calibration |
| `power_line_frequency` | 2 (60 Hz) | 2 (60 Hz) | Already the default on both |

Motion blur at 0.3 m/s fingertip speed: 33 ms exposure → 10 mm smear, 8.3 ms → 2.5 mm.

## Placement

```
                     ┌──────────────── screen (lid) ─────────────────┐
                     │                 [L] laptop cam                │
                     │                                               │
                     └───────────────────────────────────────────────┘
                                       keyboard
 [F] Feeltek on tripod     ✋ right index finger (~45 cm from eye)
 to the user's LEFT,
 ~40–60 cm from the              👁  right (dominant) eye, ~60–70 cm
 user's line of sight,               from the screen
 toed in ~30–45°
```

- **Place the cameras with the Eye-Hand app open** (applications menu, or `uv run eyehand app`).
  It shows both cameras live and marks your right eye ("R eye") and right index fingertip
  ("R index"), with a box to keep them inside, away from the image edges where distortion is
  largest. Hold the printed board where your hand will be: the side panel shows the angle
  between the two cameras at the board (aim for 30–60°) and each camera's distance. Before
  C1/C2 exist that angle is approximate (nominal lenses), but good enough for placement.
- **Camera F is on the user's left.** The app shows F on the left and L (the laptop) on
  the right. Camera IDs identify hardware, not sides. Check that your chosen eye and
  pointing hand remain visible in both views; re-aim F if the nose or hand hides them.
  Redo C5 after moving the tripod; keep each camera's lens calibration and focus settings.
- **Camera L sees the finger almost end-on.** That is unavoidable with a lid camera, and it is
  MediaPipe's worst view. L still measures the finger's left/right and up/down position well;
  F supplies the depth. Prefer pointing poses where the finger is not hidden behind the hand
  in L's view.
- **Angle between the cameras' views of the working volume: 30–60°.** A smaller angle gives
  poor depth. A larger one makes the two views look too different for consistent landmarks.
- **F's height:** about eye level or a little below, so that the face and the extended hand
  are both in frame.
- **Mount stiffness (measured problem, 2026-10-27):** during the first stereo capture the
  Feeltek crept ~0.5° in 4 minutes, then tilted ~6° in the 47 s before the second capture,
  with nobody touching it. The second capture still showed ~0.3° of creep. The suspects are
  the hinge between the Feeltek and its monitor clip (it pivots on exactly that axis) and the
  USB cable's weight pulling on it. Fix: lock or tape the clip hinge, or mount the camera by its
  thread directly on the tripod head; tighten the head; tape the cable to a tripod leg with a
  slack loop so it can't pull. Then check with `uv run eyehand steady F --seconds 300`
  (no board needed): the largest turn should stay below 0.05°. 0.1° of Feeltek rotation is
  roughly 1 mm at the cursor.
- **Rigidity:** the tripod must not be touched after stereo calibration. Moving the tripod,
  the laptop, **or the lid angle** invalidates the stereo calibration (see
  [RECALIBRATION.md](RECALIBRATION.md)).
- **Warm-up:** let both cameras stream for about 5 minutes before any calibration. Webcam
  optics drift slightly as they warm up.

## Lighting

- 8.3 ms exposure needs **bright, diffuse front light** on the face and hand. A desk lamp
  bounced off a wall works well.
- Avoid windows or bright lamps behind the user (backlight). Avoid point lights reflecting off
  the glossy OLED during screen-based calibration.
- Camera F: raise brightness with light (it has no gain control). Camera L: more light makes
  its locked auto exposure shorter, which reduces both blur and the timing uncertainty.

## Equipment for calibration

| Item | Used in |
|---|---|
| Printed ChArUco board on a rigid flat backing (foam board or clipboard) | C1, C5 |
| Calipers | C3 (screen size), C4 (camera position prior), board square size |
| Flat mirror, any shape, about 15 cm or larger, propped steady. Glass thickness is not needed (see C4) | C4 |
| Optional: 3–4 printed ArUco markers to stick on the mirror | C4 |
| Arduino (Uno R3 on hand) + LED + 220–330 Ω resistor | C6 |
| Tripod for camera F | runtime, C2, C5 |

## Software environment

- Python 3.12, managed by `uv` (`uv sync`)
- MediaPipe Tasks API, OpenCV (with ArUco/ChArUco), NumPy, SciPy
- `v4l2-ctl` (package `v4l-utils`) for camera controls
- The user must be in the `video` group (already true)
