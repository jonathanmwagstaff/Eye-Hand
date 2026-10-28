# Calibration

Every calibration the system needs: what it finds, how it is done, what equipment it needs,
where the result is stored, and the **acceptance gate** it must pass before the next step
builds on it.

**Rule:** no step runs on top of a step that failed its gate. v1 failed largely because
nothing was checked (see [LESSONS_V1.md](LESSONS_V1.md)).

## Overview

| ID | Calibration | Finds | Done | Result file |
|---|---|---|---|---|
| C0 | Camera configuration | fixed modes, exposure, focus | every start (automatic) | `config/cameras.toml` |
| C1 | Intrinsics, camera L | K_L, d_L | once | `calib/intrinsics_L.json` |
| C2 | Intrinsics, camera F | K_F, d_F (at a fixed focus value) | once per focus value | `calib/intrinsics_F.json` |
| C3 | Screen geometry | pixel pitch pₓ, p_y | once | `calib/screen.json` |
| C4 | Laptop camera ↔ screen | `T_S_L` | once (rigid in the lid) | `calib/screen_from_L.json` |
| C5 | Stereo extrinsics | `T_L_F` | **every time F, the laptop or the lid moves** | `calib/stereo.json` |
| C6 | Time calibration | timestamp model, offsets δ_L, δ_F, latency | once per camera mode + periodic check | `calib/timing.json` |
| C7 | Health monitor | detects that C5 has gone stale | continuous (automatic) | log |
| C8 | User model | dominant eye, aiming bias | once per user, quick check per session | `calib/user_<name>.json` |
| C9 | End-to-end validation | cursor accuracy report | after any recalibration | `calib/reports/` |

Order: C0 → C1, C2, C3 (independent of each other) → C4 → C5 → C6 → C8 → C9.

General rules for every board-based calibration:

- Let the cameras warm up for about 5 minutes first.
- Use the exact camera mode and controls from C0. Calibrating at one resolution and running
  at another (v1's crop) is invalid.
- Keep **every** captured image and detection in `calib/raw/<calibration>/<timestamp>/`, so
  any result can be recomputed and audited.

---

## C0 — Camera configuration

**Finds:** nothing. It **fixes** everything that would otherwise move the calibrations:
resolution, pixel format, exposure, white balance, focus, frame rate.

**How:** automatic (`eyehand probe` runs and reports it). The capture module applies
`config/cameras.toml`, reads every control back, and checks the real frame size and frame
interval.
- Camera F: autofocus is disabled and `focus_absolute` set **after the stream starts**,
  because the device rejects it while idle.
- Camera L: **auto-lock** exposure. Auto exposure settles for 60 frames, then is switched to
  manual without writing a value, freezing its choice. Its manual mode lacks a hidden gain, so
  a plain manual exposure would be black (see [SETUP.md](SETUP.md)).

**Gate:**
- every read-back equals the configured value
- measured frame interval 33.3 ms ± 1 ms with no dropped frames over 10 s
- timestamp flags are `monotonic` + `start-of-exposure`

---

## C1 — Intrinsics, camera L (laptop)

**Finds:** camera matrix K_L (fx, fy, cx, cy) and distortion d_L.

**Target:** a **printed ChArUco board** on a rigid flat backing. Camera L cannot see the laptop
screen, so the screen can't be the target here. A phone or tablet showing a ChArUco board is an
alternative, if its pixel pitch is known.

**Procedure:**
0. Print `boards/print_a4_a4_landscape.pdf` (or `..._letter_landscape.pdf`) at **Actual size**.
   The board is centred with 9–12 mm of white on each side. Printing the bare PNG let the
   print program crop it; one print lost its bottom 3 rows, another the bottom 4 mm.
1. Measure the printed square size with calipers. Measure across **6 squares (nominally 144 mm)** in
   **both** directions: printers can scale x and y differently, and a 6-square span is 6× more
   precise than one square. Uniform scale cancels in the cursor; an x/y difference does not
   fully cancel. (2026-10-27: one square measured 24 mm; the 6-square spans are still to do.)
2. Run `uv run eyehand intrinsics capture L --square-x-mm <span_x/6> --square-y-mm <span_y/6>`.
   Hold the board in front of the laptop; the window shows the camera image, the corners found,
   and the **coverage map** (covered cells green, empty ones outlined red).
3. A view is captured automatically, with a shutter sound, when the board has been **still**
   for 4 frames (median corner motion ≤ 0.5 px per frame) **and** is in a new place. The corners
   of those 4 frames are averaged. Hold still at each spot until you hear the shutter, then move.
   Stillness matters more for camera L than usual: its locked exposure can be up to a whole
   frame (33 ms) long.
4. Aim for **40–60 views**: the board near each image corner and edge, at the working
   distance (40–80 cm) and somewhat closer, and **turned**. The window shows the live tilt and
   counts views turned **≥ 20°** in each direction (left, right, top or bottom edge away, as
   seen in the mirrored window). It needs 3 of each. **Frontal views alone cannot fix the
   focal length:** the first real capture (51 views, median tilt 6°) gave fx ± 1.2 % and two
   halves that disagreed by 51 px.
   Capture stops by itself at 50 views once the corner zones are reached and the tilts are
   done, or press Q (or Ctrl+C) to finish and solve. U undoes the last view; space pauses.
   `--resume calib/raw/intrinsics_L/<timestamp>` continues an earlier capture, keeping its
   views.
5. The solve runs automatically:
   - `cv.calibrateCameraExtended` with distortion model k1, k2, p1, p2.
   - Views whose own error is above 3× the median (and more than 0.3 px above it) are
     rejected and the fit is repeated. At most 10 % of the views are rejected.
   - **Model choice on held-out views:** 20 % of the views are left out, and k3 is used only if
     it predicts those views at least 2 % better.
   - Final fit on all remaining views, then a **split-half check**: two disjoint random halves
     are fitted separately and compared.
6. Images, averaged corners and the result are kept in `calib/raw/intrinsics_L/<timestamp>/`.
   If all gates pass, the result is installed as `calib/intrinsics_L.json`, and any previous
   one is moved to `calib/history/`. `eyehand intrinsics solve <folder> [--drop 007 012]`
   re-solves a saved capture.

**Gate** (chosen for what reaches the cursor: about 0.7 mm of cursor per pixel of ray error):
- **ray uncertainty in the tracking area** (the 8 % margin box the live views keep the eye and
  fingertip in): the views are resampled 16 times with replacement and refitted. Each time,
  the pixel-to-ray mapping is compared with the main fit after removing a best-fit rotation
  (C4/C5, solved with the same intrinsics, absorb a rotation). The spread must be
  **≤ 1.0 px RMS** and **≤ 2.0 px** at its worst point.
- reprojection RMS **≤ 0.6 px** (sanity check for blur and bad detections)
- hold-out RMS **≤ 1.15 ×** the fit RMS (overfitting check)
- **coverage:** at least **85 %** of the ~80 px image cells contain a corner, and corners reach
  into each of the four **corner zones** (the outer 12 % of the width × 15 % of the height).
- a re-run on a fresh set of views agrees: `eyehand intrinsics compare a.json b.json` (within
  √2 × 2.0 px in the tracking area, after removing a rotation)

**Why not the usual "RMS < 0.3 px".** That was the first version of these gates, and the real
camera showed it was the wrong test. 84 views of camera L (2026-10-27):
- Corner noise is about 0.33 px per axis even after averaging 4 frames, so the fit RMS bottoms
  out near 0.45 px. That is detection noise: the lens model accounts for all but ~4 % of the
  residual, the board is flat to ~0.08 mm, and within-view patterns from bending or motion
  are near the pure-noise level. Hold-out RMS equals fit RMS, so it doesn't overfit.
- OpenCV's formal std(fx) said 0.06 %, but the bootstrap spread is about 0.2 %. Errors within a
  view are correlated, which the formal estimate ignores.
- Uncertainty grows evenly from the image centre (0.1 px) to the edges of the tracking area
  (~1.5 px): a focal-length-like uncertainty worth about 0.5 mm at the cursor.

**Getting the top image corners with camera L:** the top of its image looks up at the ceiling,
so a board held there is backlit by ceiling lights and isn't detected. Tilt the lid forward
for those views, so a board at a comfortable height lands in the top of the image. The lid
angle doesn't affect intrinsics; it only matters for C5, which is done afterwards.

---

## C2 — Intrinsics, camera F (Feeltek)

**Finds:** K_F and d_F, **valid only for the recorded `focus_absolute` value**.

**Target:** the **printed ChArUco board**, as in C1, with the Feeltek on its tripod and the board
moved in front of it. The first plan used the laptop OLED screen as the target: perfectly
flat, with a known pixel pitch. In practice the person moving the camera got almost no
feedback: no live view, only sounds and a line of text. Meanwhile C1 showed the print is flat
to ~0.08 mm, far below this class of webcam's detection noise. The screen board stays
available (`--board screen`) but is not the default.

**Procedure:**
1. Check the focus. Prop the printed board at the working distance (where face and hand
   will be, ~50–80 cm from F) and run `uv run eyehand focus F`. It measures sharpness and the
   board's **size in the image** at each `focus_absolute` value. If the lens really moves, the
   size changes measurably ("focus breathing"). The first sweep (2026-10-27) showed no
   sharpness change beyond noise, and a direct test at 0/64/128/192/256 agreed. So this unit
   may behave as fixed-focus; the size test will settle it. Original description: It steps
   `focus_absolute` through its range, measures sharpness inside the board at each value,
   re-measures the best few values (focus motors have backlash), and prints the best one.
   `eyehand focus F --write` puts it in `config/cameras.toml`. It is never changed afterwards.
2. Run `uv run eyehand intrinsics capture F --square-mm <measured>`. The same mirrored live
   window as C1, with coverage map, corner zones, live tilt and the tilt counters. Move the
   board in front of the Feeltek at 40–80 cm.
3. 2026-10-27 focus check: across focus_absolute 0–256, the board's size in the image showed
   no trend (0.06 % against 0.18 % random scatter), and sharpness was flat at 7 of 9 settings.
   **The Feeltek behaves as fixed-focus.** The configured value (144) is kept anyway, so
   nothing changes if the firmware ever honours it.
4. Solve and gates as in C1, with results in `calib/raw/intrinsics_F/` and
   `calib/intrinsics_F.json`. The result records `focus_absolute`, and loading refuses it if
   the config's value differs.

**Gate:** same as C1.

**Note:** an OLED screen gives an ideal flat target. A small error in its pixel pitch only
rescales the target, which affects distances but not the distortion or the angles between rays.

---

## C3 — Screen geometry

**Finds:** horizontal and vertical pixel pitch, pₓ and p_y, in mm/pixel.

**Procedure:**
1. Display a test pattern: thin lines at known pixel columns and rows, near the edges of
   the active area.
2. With calipers, measure the distance between two lines far apart, horizontally and
   vertically. Measure the pattern, not the bezel or glass edge.
3. pₓ = measured mm / pixel distance; same for p_y.

**Gate:**
- three repeated measurements agree within 0.2 mm
- the result is within 0.5 % of the EDID value (344 × 194 mm)

**Optional (decided 2026-10-27).** The EDID size, 344 × 194 mm rounded to 1 mm, is good to
±0.15 %. The cursor maps to pixels by scaling about camera L's position at the top centre of
the screen, so that error moves the cursor by at most ~0.4 mm, at the farthest corner. In C4,
a scale error of the screen board mostly moves the screen along camera L's axis, to which the
cursor is insensitive because camera L sits almost in the screen plane. Both are far below
keypoint noise (1.6–3 mm). Without `calib/screen.json` the EDID size is used.
`uv run eyehand screen` measures it anyway: a full-screen pattern of lines whose edges are
exactly 800 px apart, checked pixel-exact by a screen grab. Measure each span three times;
the spread must be ≤ 0.3 mm.

---

## C4 — Laptop camera ↔ screen (`T_S_L`)

**Finds:** where camera L sits and how it is oriented relative to the screen pixels. Camera and
panel are both fixed in the lid, so this **never changes**. It is the most important one-time
measurement, because every error here moves the cursor directly: 1 mm of position error gives
1 mm of cursor error, and 1° of roll gives up to ~4.6 mm at the screen corners.

**Why a mirror:** camera L faces out of the screen and can never see the screen itself. A mirror
lets it see the screen's reflection. This is an established method: the MPIIGaze laptop
gaze dataset (Zhang et al., CVPR 2015) calibrated each laptop's camera to its screen exactly this
way, using Rodrigues, Barreto & Nunes (ECCV 2010). Takahashi, Nobuhara & Matsuyama (CVPR 2012)
proved that three mirror poses give a unique solution.

**The logic:**
1. A plane mirror with unit normal **m** and distance e (plane m·X = e, in frame L) maps every
   point to its reflection: `X' = X − 2(m·X − e)·m`. The reflection flips handedness.
2. The screen shows a **horizontally flipped** board, so its reflection is a normal,
   right-handed board. The ArUco detector rejects mirror-image markers, and PnP needs a proper
   rotation. PnP then gives the pose of the **virtual** screen behind the mirror.
3. One view is not enough. The real screen is the reflection of the virtual one about a plane
   we don't know, which leaves 3 unknowns per view. Two views still leave a one-parameter
   family of solutions.
4. Three or more views with **different tilt directions** fix it. The real screen pose is
   shared by every view; only the mirror moves. The relative motion between two virtual
   screens is a rotation about the axis m_i × m_j, which is perpendicular to both mirror
   normals. With three poses this gives each normal, then each distance, then the real pose
   (Takahashi's orthogonality constraint).
   - The three normals must **not lie in one plane**. Tilting the mirror only left and right is
     degenerate; also tilt it up and down.
5. A joint least-squares fit refines `T_S_L` and every mirror plane against all detected
   corners at once.

**What the mirror must be:** flat and clean. The shape doesn't matter: round is fine,
because ChArUco works from partial views. Mirror size: roughly half the width of the screen
area you want to see in one view. A 15 cm mirror is enough, because the poses together cover
the screen.

**Glass thickness does not need to be measured** for method (b) below. A back-silvered mirror
acts, to first order, like an ideal mirror placed t/n ≈ 0.66·t behind the front surface. The
solver treats each plane as unknown, so that shift is absorbed. `tools/mirror_glass_sim.py`
ray-traces a real thick mirror through this solver. The remaining bias is at most **0.1 mm of
cursor error, even with 6 mm glass**. By contrast, 0.3 px of corner noise costs about 0.3–0.6 mm.

**Procedure (`uv run eyehand mirror capture`):**
1. Screen at full brightness. The command shows the flipped board (18 × 7 squares) across the
   top of the screen. Below it: a live, mirror-like view of camera L with the detected corners,
   framed red (no board), orange (board seen) or green (pose captured), and the status in large
   text. The preview is drawn at half contrast. Through the mirror, camera L sees it at about
   3 px per square, far too small to detect, and a test confirms the real board is still found
   completely with the preview present.
2. Prop the mirror 25–35 cm in front of the lid, facing it, so that camera L sees the screen
   in it. A pose is captured, with a click, once the reflection has been still for 4 frames
   and differs from earlier poses. Then re-prop the mirror at a different tilt or distance.
3. **20 poses**, tilted both sideways and up/down. The status line shows the tilt variety:
   half the spread of the reflected board's direction, since a mirror tilt of a turns the
   reflection by 2a. It finishes by itself once both directions reach 3°, or press Q.
4. Solve: a prior of "camera at the top centre of the lid, facing straight out", then for
   each pose the virtual board's PnP pose and the mirror plane between the real (prior) and
   virtual corners, then a joint robust least-squares fit of `T_L_S` and all planes. Glass
   thickness is absorbed.
5. Raw data: `calib/raw/mirror/<timestamp>/`. A passing result is installed as
   `calib/screen_from_L.json`; `eyehand mirror solve <folder> [--drop 003]` re-solves.

**First real result (2026-10-27).** Two independent 20-pose runs with the round mirror:
- Run 1: tilt spread 5.5° / 8.1°; cursor uncertainty 0.71 mm, just failing.
- Run 2: tilt spread 7.7° / 9.8°; cursor uncertainty 0.43 mm. **Installed.** Camera L sits at
  x 172.5, y −3.7, z 1.9 mm in frame S, tilted 2.7° from straight out of the screen.
- The two runs differ by ≤ 1.17 mm at the cursor, against ~0.83 mm expected from their
  uncertainties; roll differs by 0.009°.
- Merging them gave a *larger* jackknife uncertainty (0.64 mm) than run 2 alone, presumably
  because of run 1's weaker poses. Run 2 is kept, and the realistic C4 contribution is taken
  as ≈ 0.5–1 mm at the worst screen point.

**Flatness check:** a bent mirror acts as a weak lens. A pose whose RMS is far above the rest
fails the "worst pose" gate.

**Gate:**
- reprojection RMS ≤ 0.7 px; worst pose ≤ 2.5 × the median pose
- **cursor uncertainty ≤ 0.7 mm** at the worst point on the screen: jackknife standard error
  over leave-one-pose-out refits, pushed through the pointing geometry (eye 650 mm out,
  fingertip 450 mm from the eye). In simulation (0.3 px noise), the largest single
  leave-one-out shift underestimated the true error about 2×; the jackknife was within about
  1.4×. 14 poses gave ~0.46 mm RMS true error, 24 poses ~0.39 mm.
- roll uncertainty ≤ 0.1°
- mirror tilt variety ≥ 3° (standard deviation) in **both** directions. Tilting only sideways
  makes the normals coplanar and leaves the pose undetermined; simulation confirms the gate
  catches it.
- sanity vs. the prior: camera within 25 mm of the top centre and 6° of straight out
- at least 8 poses

---

## C5 — Stereo extrinsics (`T_L_F`)

**Finds:** where camera F is relative to camera L. **This is the calibration that has to be
redone whenever the tripod, the laptop or the lid angle moves.** It is run from the Eye-Hand
app, with no command line (see [RECALIBRATION.md](RECALIBRATION.md)).

**Target:** the printed ChArUco board held in the working volume (between face and screen),
seen by **both** cameras at the same time.

**Procedure (app: "Recalibrate camera positions"):**
1. Intrinsics from C1/C2 are held fixed. The button is disabled until both are valid.
2. A view is captured, with a click, when the board is **still in both cameras over
   overlapping time spans** (median corner motion ≤ 0.5 px per frame for 4 frames in each).
   The cameras are not synchronised, but a board that is still in both looks the same
   whichever frames are paired. So C5 does not depend on C6.
3. Views must be new: corners at least 40 px from every earlier view in one camera or the other.
4. Coverage guidance: camera L's image is split into 3 × 2 areas, each with a near (< 45 cm)
   and a far band. The capture finishes by itself at **25 views** once at least 5 of the 6
   areas and both depth bands have views. "Finish now" solves earlier (minimum 6 views).
5. Solve: an initial `T_F_L` from each view's two board poses (rotation mean, median
   translation), then **bundle adjustment** of `T_F_L` and every board pose together,
   minimising pixel reprojection error in both cameras with a soft-L1 robust loss.
6. Images and corners are kept in `calib/raw/stereo/<timestamp>/`. A passing result is
   installed as `calib/stereo.json`, with the previous one moved to `calib/history/`. It
   records fingerprints of the intrinsics it was made with.
   `eyehand stereo solve <folder> [--install]` re-solves a saved capture.

**Gate:**
- reprojection RMS **≤ 0.7 px** in both cameras (sanity; these webcams' corner noise floor is ~0.45 px)
- **camera movement during the capture ≤ 0.3°.** Each view also gives its own estimate of the
  camera pose; the median of the first third of views is compared with the last third.
  First real capture: 0.47°, from a creeping clip hinge.
- **held-out epipolar consistency:** fit without 20 % of the views; the held-out views'
  epipolar RMS must be ≤ 1.3 × the in-sample value, and ≤ 1.0 px. (The first version used an
  absolute 0.5 px, chosen in simulation with 0.15 px noise. Good real data gives ~0.6 px.)
- **known-length test:** triangulate pairs of board corners at least 6 squares (≥ 144 mm) apart
  in a row or column, everywhere the board went. Divide out the overall scale (the print
  scale cancels in the cursor). The 95th percentile of the remaining length error must be
  **< 1 mm**. In simulation, a 3 % focal-length error in one camera fails this gate.
- at least **20 views** used

In simulation (0.15 px corner noise, 28 views), the camera position comes out within 1 mm and
0.05°, and the lengths within 0.2 mm (95 %).

Time: about 1–2 minutes.

---

## C6 — Time calibration

**Finds:**
- (a) **per camera:** the timestamp model (frame period, de-jittering line `τ ≈ a + b·n`) and
  any dropped frames
- (b) **between cameras:** δ_L − δ_F, the difference in how each camera's timestamps relate to
  the real moment of exposure. This is what matters for triangulation.
- (c) **absolute latency:** the time from exposure to the result being available. This matters
  for cursor lag and prediction, not for accuracy.

**Why it matters:** the two cameras run on separate clocks. Their frames slide past each other,
so the gap between "matching" frames keeps changing. A 16 ms mismatch with the finger moving at
0.3 m/s gives about 2.8 mm of cursor error.

**Procedure:**
1. **(a) Free:** computed continuously from kernel timestamps and sequence numbers.
2. **(b) LED test:** an Arduino switches an LED that both cameras see, at **random** moments,
   and reports each switch time over USB serial (host-side jitter about 1 ms).
   - Each switch falls between two consecutive frames of a camera: the last frame with the old
     state and the first with the new one. Each such pair is an *interval constraint* on that
     camera's timestamp bias δ_c.
   - Frames with partial brightness (the switch happened during the exposure) pin the moment
     down further.
   - Because the switches land at random phases relative to the frames, intersecting a few
     hundred constraints narrows δ_c to about a millisecond, much finer than one frame.
   - Doing this for both cameras gives δ_L − δ_F.
3. **(c) Absolute latency:** the same data gives each camera's delay from exposure to frame
   arrival, plus the extra delay through MediaPipe and the fusion step.
4. **Online tracking (required for camera L):** camera L's locked exposure is unknown
   (≤ 33 ms) and can differ every session. Its effective timestamp bias δ_L can therefore
   shift by up to about half a frame between sessions, so the LED value is a starting point,
   not a constant.
   - While running, hand and head motion let the offset be re-estimated continuously: the
     epipolar error of the landmarks, as a function of a trial time offset, is smallest at the
     true offset.
   - A drift alarm is raised if the estimate moves more than 3 ms from the LED value within a
     session.

Measured so far: the cameras run at 30.07 fps (L) and 30.00 fps (F), so their relative phase
cycles through a full frame every ~14 s. Pairing by index would mismatch frames by up to
16.7 ms (see [SETUP.md](SETUP.md)).

**Gate:**
- de-jittered timestamps have residual RMS < 1 ms
- offset (b) repeatable within **±2 ms** across three runs
- the online estimate agrees with (b) within 3 ms

---

## C7 — Health monitor (continuous)

**Finds:** that something has moved since C5.

**How:** the app runs MediaPipe's face landmarks on both cameras. With a valid `T_L_F`, the
**epipolar error** of the 478 face landmarks seen by both stays small. Only frame pairs taken
less than 25 ms apart are used, and only while the head is still (median landmark motion
≤ 1.5 px per frame), so neither time skew nor motion inflates it.

- **Baseline:** recorded right after each successful C5. You sit as for pointing and look at
  the screen for a few seconds. The baseline is the median over 60 measurements, stored in
  `calib/stereo.json`. It is several pixels, because MediaPipe's face landmarks from two very
  different viewpoints don't land on exactly the same skin points.
- **Alarm:** the median stays above max(2 × baseline, baseline + 3 px) for 2 s. That means
  the tripod, the laptop or the lid moved. The app plays a warning sound, shows it in red, and
  logs `c7_alarm` in `calib/events.jsonl`. C5 then shows "check" until it is redone.

A bump of the tripod is caught within seconds instead of silently degrading accuracy.

---

## C8 — User model

**Finds:** dominant eye, the eye-point offset, and the user's systematic aiming bias.

**Procedure:**
1. **Dominant eye:** stated as **right**. Confirmed by a sighting test: point at a target with
   both eyes open; the eye whose line passes through the finger to the target is dominant.
   The software compares right-eye and left-eye ray fits.
2. **Aim samples:** targets appear on the screen (a 5×5 grid plus random positions). The user
   points with **no cursor shown** and holds still. Repeat from **3–4 different head positions**,
   so the eye–finger lines are not all from one viewpoint.
3. **Fit:** a constant angular offset first; an affine correction only if the hold-out error
   drops noticeably.
4. **Hold-out:** 25 % of the targets are never used for fitting.

**Gate:**
- hold-out error improves with the model compared to raw geometry
- hold-out error does not depend on head position (otherwise the geometry, not the user, is wrong)

**Also:** the iris-landmark index for the right eye is verified here by a one-eye-closed test.

---

## C9 — End-to-end validation

**Finds:** how good the whole system actually is.

**Procedure:** a fixed target grid, pointed at with no cursor shown, from several head positions.
For each target, record the error in mm and px, the timing, and the triangulation quality.

**Report:**
- mean and 95th-percentile error, overall and per screen region
- static jitter (RMS while holding still)
- latency

Every report is kept in `calib/reports/` so changes can be compared over time.
