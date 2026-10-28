# Recalibration

Which calibrations go stale after which change. Calibration IDs refer to
[CALIBRATION.md](CALIBRATION.md).

## The short version: camera F (Feeltek) moved

If the tripod, the laptop, or the **lid angle** has changed by any amount, only **C5** (the
camera positions) needs redoing. It is done entirely in the **Eye-Hand app**: open it from the
applications menu, with no terminal and no AI needed.

1. If the app hasn't noticed by itself, press **"I moved a camera"**. The status list marks C5
   stale and records the event in `calib/events.jsonl`.
2. Use the live views to re-aim the cameras if needed. Both should see your right eye and your
   right index fingertip inside the box. Hold the printed board where your hand will be: the
   panel shows the angle between the cameras (aim for 30–60°).
3. Press **"Recalibrate camera positions"**. **Step 1 (20 s, hands off):** the app checks that
   both cameras hold still: nothing to hold up, just don't touch the laptop, tripod or cables.
   If either camera turns more than 0.05°, it stops and says which one. **Step 2:** follow the
   instructions at the top. Hold the
   printed board still at a spot until you hear the click, then move or tilt it. The camera L
   panel shows which areas still need a view, near and far. After 25 views covering the
   space, it solves by itself (or press "Finish now").
4. If the result passes its gates, it is installed and the old one is kept in
   `calib/history/`. If accuracy checks fail, nothing changes automatically, and the app says
   why. Press "Try again", or **"Use calibration anyway (less accurate)"** to apply the
   estimate explicitly. The previous calibration is kept in history; the new one retains
   its failed checks and shows a warning in status, but can be used by the cursor.
   This option is unavailable if the solver errored without producing an estimate.
5. **Last step:** sit as you will when pointing and look at the screen for a few seconds.
   This records the face-check baseline that C7 compares against from then on.

Lens calibrations, screen geometry, the mirror calibration and timing are all kept.

Nothing else is needed, **provided F's focus value and camera mode did not change.** The
software sets focus itself, so the only way it changes is by editing the config. If it does
change, the app shows the C2 lens calibration as stale, and C5 after it.

Why the **lid angle** counts: camera L and the screen are fixed in the lid, but camera F is on
the tripod. Tilting the lid moves L relative to F, exactly as if F had moved.

### How the system keeps track

Nobody has to remember what is stale. `eyehand.status` recomputes it on every start from the
files themselves:

| Result | Becomes stale when |
|---|---|
| `calib/intrinsics_L.json`, `intrinsics_F.json` | the camera's mode, USB id or focus in `config/cameras.toml` differs from the one recorded in the result |
| `calib/stereo.json` | a lens calibration was redone (each stereo result records fingerprints of the intrinsics it used), **or** a "moved" event was logged after it was solved |
| `calib/stereo.json` (needs a check) | the health monitor (C7) logged an alarm after it was solved |

The app shows this list with coloured dots. `uv run eyehand status` prints the same thing.
Every loader also refuses stale results: the lens calibration won't load with a different
focus, and the stereo result won't load with different intrinsics. So a stale calibration
can't be used by accident.

`uv run eyehand steady` measures whether the cameras are physically still, from the images
alone. `--since-stereo` compares them with the last frames of the latest stereo capture,
which shows whether anything has moved since.

**Room check (in the app):** right after a successful calibration, the app saves a snapshot of
each camera's view. While the app is open, it compares the cameras' current views with those
snapshots every 5 s. A turn of more than 0.1° in two readings in a row is a "moved" alarm:
a warning sound, a red line in the panel, and C5 marked "check". It needs no face in view.
Typing on the laptop keyboard measurably moved the lid (0.35°) on 2026-10-27, so use an
external keyboard and mouse.

C7 (health monitor) notices movement by itself while the app is open, even if you didn't:
a sustained jump in the face check logs an alarm and plays a warning sound.

## Full matrix

✔ = must redo · (✔) = quick check, redo only if the check fails · blank = still valid

| Change | C0 config | C1 intr. L | C2 intr. F | C3 screen | C4 L↔screen | C5 stereo | C6 timing | C8 user | C9 validate |
|---|---|---|---|---|---|---|---|---|---|
| **Tripod moved or bumped** | | | | | | ✔ | | | ✔ |
| **Lid angle changed** | | | | | | ✔ | | | ✔ |
| **Laptop moved on the desk** | | | | | | ✔ | | | ✔ |
| Camera F rotated on the tripod head | | | | | | ✔ | | | ✔ |
| Camera F's `focus_absolute` changed | ✔ | | ✔ | | | ✔ | | | ✔ |
| Camera F's resolution or format changed | ✔ | | ✔ | | | ✔ | ✔ | | ✔ |
| Camera L's resolution or format changed | ✔ | ✔ | | | ✔ | ✔ | ✔ | | ✔ |
| Exposure time changed | ✔ | | | | | | (✔) | | (✔) |
| Gain / brightness / lighting changed | | | | | | | (✔) online | | (✔) |
| Every session start (camera L re-locks its exposure) | auto | | | | | | (✔) online | | |
| Camera F unplugged and replugged | auto | | | | | (✔) | (✔) | | |
| USB port, hub, kernel or driver changed | | | | | | | ✔ | | (✔) |
| Display resolution or scaling changed | | | | ✔ (pixel mapping) | | | | | ✔ |
| Camera F replaced with another unit (even same model) | ✔ | | ✔ | | | ✔ | ✔ | | ✔ |
| Laptop screen or lid hinge repaired | | (✔) | | ✔ | ✔ | ✔ | | | ✔ |
| New user | | | | | | | | ✔ | ✔ |
| Same user, new session | | | | | | (✔ via C7) | | (✔) | |
| Camera warm-up (first 5 minutes) | | | | | | | | | wait |

Notes:

- **Unplug and replug:** C0 is reapplied automatically, including the focus lock. The cable
  tug may have nudged the camera, which is why C5 gets a quick C7 check.
- **Exposure change:** timestamps mark the *start* of exposure, and the model adds half the
  exposure time. For camera F (manual, known exposure) the offset should carry over, and one
  LED run confirms it. Camera L re-locks an unknown exposure at every start, so its offset is
  tracked online (C6 step 4).
- **Display scaling:** only the mapping from mm to pixels changes. The physical screen
  geometry is the same.
- **A new unit of the same camera model** needs its own intrinsics. Lenses differ from unit
  to unit.

## Rule of thumb

> Anything that moves **F relative to L** → redo C5.
> Anything that changes **how a camera forms images** (focus, mode) → redo that camera's
> intrinsics, then C5.
> The screen and camera L are fixed in the lid, so **C3 and C4 essentially never need redoing**.

## How a stale calibration shows itself

| Symptom | Likely stale |
|---|---|
| C7 alarm: face-landmark epipolar error jumped | C5 (something moved) |
| Triangulated fingertip "breathes" in depth while you hold still | C5, or focus drift (check C0 read-back) |
| Cursor error grows toward the screen edges and corners | C1/C2 (distortion) or C4 roll |
| Cursor offset is constant everywhere | C4 translation, or C8 (aiming bias) |
| Error appears only while moving, worse at higher speed | C6 (time offset) |
| Error depends on head position | C4 or C5 geometry, not C8 |
