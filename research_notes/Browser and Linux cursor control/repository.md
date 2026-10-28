# Existing Eye Hand integration surface

## What camera, vision, hand, or gaze signals exist today, and in what coordinates and cadence?

### Takeaway
The executable code currently stops at configured, timestamped camera frames: two 30 fps BGR images plus exposure-start and arrival times. Eye landmarks, fingertip landmarks, stereo calibration, triangulated 3D points, ray/screen intersection, filtering, and cursor samples are documented designs, not implemented signals.

### Cited Findings
- `Camera.read()` returns the newest `Frame(camera, image, t_soe_ms, t_arrival_ms, index, skipped)` from a four-buffer reader thread, so the concrete runtime boundary today is a camera image and monotonic timing metadata rather than a pointer coordinate. — [capture.py](file:///home/jonathan-wagstaff/Eye-Hand/src/eyehand/capture.py)
- Both configured modes are 30 fps: camera L is 1280×720 MJPEG and camera F is 1280×960 MJPEG. The measured frame periods are 33.254 ms and 33.333 ms, and the streams drift through a full relative phase in about 14 seconds. — [cameras.toml](file:///home/jonathan-wagstaff/Eye-Hand/config/cameras.toml); [SETUP.md](file:///home/jonathan-wagstaff/Eye-Hand/docs/SETUP.md)
- `FrameClock` de-jitters kernel timestamps, infers dropped-frame sequence gaps, and exposes fitted timestamps; it does not align landmarks or create cursor coordinates. — [timebase.py](file:///home/jonathan-wagstaff/Eye-Hand/src/eyehand/timebase.py)
- The intended geometry uses screen frame S in millimetres, with origin at the active panel’s top-left, x right, y down, z into the screen. The planned ray/screen result converts to pixels as `(C_x / p_x - 0.5, C_y / p_y - 0.5)`. — [ARCHITECTURE.md](file:///home/jonathan-wagstaff/Eye-Hand/docs/ARCHITECTURE.md)
- The README explicitly marks C1/C2 intrinsics as next, C3–C6 as planned, and MediaPipe workers, triangulation, and cursor as planned. — [README.md](file:///home/jonathan-wagstaff/Eye-Hand/README.md)
- The dependency list contains MediaPipe, Pillow, NumPy, and SciPy; it contains no browser server, IPC, or Linux input-injection package. — [pyproject.toml](file:///home/jonathan-wagstaff/Eye-Hand/pyproject.toml)

### Inferences
- A 30 Hz pointer is the initial ceiling if each displayed position is based on a newly fused camera observation. Rendering can run at display rate, but interpolation/prediction would be required to make that more than repeated 30 Hz samples.
- A browser or Linux sink should not be built directly on raw camera frames. The stable seam is after geometry, C8 correction, quality gates, and filtering, where a valid screen-space cursor result exists.

### Gaps
- No runtime MediaPipe model assets, landmark worker, `Observation` type, calibration loaders for C1–C6, triangulator, ray/screen function, or cursor filter exists in the repository yet. — [Repository layout](file:///home/jonathan-wagstaff/Eye-Hand/src/eyehand)
- The planned right-eye iris index 468 remains explicitly marked for a one-eye-closed verification, and there is no implemented click/gesture signal. — [ARCHITECTURE.md](file:///home/jonathan-wagstaff/Eye-Hand/docs/ARCHITECTURE.md)

## Is there an existing browser, user interface, server, or IPC layer?

### Takeaway
No browser or IPC layer exists. The only UI is OpenCV windows for camera preview and full-screen calibration boards, while all commands are registered in a single `argparse` CLI module.

### Cited Findings
- The CLI exposes only `probe`, `timing`, `preview`, `boards`, and `show-board`; `preview` and `show-board` use `cv2.imshow`, and there is no browser command. — [cli/main.py](file:///home/jonathan-wagstaff/Eye-Hand/src/eyehand/cli/main.py)
- The architecture diagram ends in the unimplemented generic label `cursor output / overlay / logs`; it does not select a transport or platform backend. — [ARCHITECTURE.md](file:///home/jonathan-wagstaff/Eye-Hand/docs/ARCHITECTURE.md)
- A repository-wide dependency and source search found no Flask, FastAPI, aiohttp, WebSocket, Selenium, Playwright, pyautogui, pynput, python-xlib, evdev, uinput, xdotool, or ydotool integration. — [pyproject.toml](file:///home/jonathan-wagstaff/Eye-Hand/pyproject.toml); [source tree](file:///home/jonathan-wagstaff/Eye-Hand/src)

### Inferences
- The smallest browser proof should be a page-local dot driven by a localhost stream from the Python process, not browser automation and not a browser-side duplicate of MediaPipe/geometry.
- A dependency-free one-client prototype can use Python’s standard-library threaded HTTP server plus Server-Sent Events (`EventSource`) to push latest-value cursor samples. If production latency or bidirectional controls later justify it, that implementation can be replaced by WebSocket without changing the sample contract.

### Gaps
- Browser networking behavior and Linux injection mechanisms are intentionally covered by the other research notes; this repository-only review does not establish their platform support.

## Where should pointer coordinates and gestures enter with the least architecture?

### Takeaway
Add one narrow output seam after the existing planned fusion stages: a latest valid cursor sample in normalized active-screen coordinates. Feed that same sample first to a browser-page sink, then to a Linux sink; defer gestures/clicking entirely until movement and loss-of-tracking safety are validated.

### Cited Findings
- The planned fusion order is undistortion → time alignment → triangulation and gates → E/F in screen frame → ray/screen intersection → user model and filtering → output. — [ARCHITECTURE.md](file:///home/jonathan-wagstaff/Eye-Hand/docs/ARCHITECTURE.md)
- Poor triangulations are supposed to be rejected using per-camera reprojection error and inter-ray angle rather than fed into the cursor. — [ARCHITECTURE.md](file:///home/jonathan-wagstaff/Eye-Hand/docs/ARCHITECTURE.md)
- C7 is designed to invalidate operation when stereo geometry goes stale, and C9 records accuracy, jitter, and latency after recalibration. — [CALIBRATION.md](file:///home/jonathan-wagstaff/Eye-Hand/docs/CALIBRATION.md)

### Inferences
- The minimum cross-sink value is `(x_norm, y_norm, t_ms, valid)`, where `x_norm/y_norm` derive from the calibrated active panel, not the current OpenCV image. Normalization keeps browser CSS pixels and Linux desktop pixels out of the geometry code.
- Invalid tracking should produce `valid=false` and freeze or hide the page cursor; it should not clamp a bad 3D estimate onto an edge. The Linux backend needs the same fail-closed behavior.
- Movement and activation are separate trust levels. A gesture/click field would create accidental-action risk before the repository has implemented quality gates, so it is not part of the first complete pointer-movement slice.

### Gaps
- The exact filter and prediction method are undecided: the architecture lists landmark interpolation as v1 and an asynchronous 3D Kalman filter as v2. — [ARCHITECTURE.md](file:///home/jonathan-wagstaff/Eye-Hand/docs/ARCHITECTURE.md)

## What calibration, smoothing, safety, and test coverage already exist or remain absent?

### Takeaway
The design is unusually explicit about physical calibration and failure gates, but implementation is only at C0/timing and ChArUco board generation. Cursor-specific safety, smoothing, loss handling, display mapping, and end-to-end validation are absent.

### Cited Findings
- The calibration plan requires C0 through C9, including intrinsics, screen geometry, camera/screen pose, stereo pose, time offset, stale-calibration health monitoring, user bias, and end-to-end error/jitter/latency reporting. — [CALIBRATION.md](file:///home/jonathan-wagstaff/Eye-Hand/docs/CALIBRATION.md)
- Moving the tripod, laptop, or lid angle invalidates C5, while changing camera mode or focus also invalidates intrinsics; the design requires C7 checks and C9 revalidation. — [RECALIBRATION.md](file:///home/jonathan-wagstaff/Eye-Hand/docs/RECALIBRATION.md)
- The current automated suite has eight passing tests, covering board rendering/detection/mapping and timing de-jitter/drop/phase behavior; it has no cursor contract, geometry, browser transport, or output-backend tests. — [test_boards.py](file:///home/jonathan-wagstaff/Eye-Hand/tests/test_boards.py); [test_timebase.py](file:///home/jonathan-wagstaff/Eye-Hand/tests/test_timebase.py)
- The architecture promises rejection of poor triangulations and mentions filtering/prediction, but no corresponding source module exists. — [ARCHITECTURE.md](file:///home/jonathan-wagstaff/Eye-Hand/docs/ARCHITECTURE.md); [source tree](file:///home/jonathan-wagstaff/Eye-Hand/src/eyehand)

### Inferences
- The first output implementation should include one small pure-function test for screen-pixel → normalized-coordinate mapping and invalid/out-of-bounds behavior. Transport and Linux injection can consume that tested contract without duplicating calibration math.
- An emergency stop, immediate disable on C7/tracking failure, bounded update rate, and startup-disabled Linux injection are requirements for the real-pointer stage even though they are not currently specified in code.

### Gaps
- Hardware calibration files other than raw placeholders are not present, so a real camera-derived cursor cannot yet satisfy the repository’s own acceptance gates. — [calib directory](file:///home/jonathan-wagstaff/Eye-Hand/calib)
- The worktree has no commits and all project files are untracked, so there is no repository history from which to infer an earlier cursor implementation or compatibility contract. — [Git metadata](file:///home/jonathan-wagstaff/Eye-Hand/.git)

## What staged proof of concept best fits this repository?

### Takeaway
Build the output path only after it has a real or deterministic normalized sample source: (1) pure cursor mapping and synthetic samples, (2) localhost browser dot, (3) connect the calibrated fusion result, (4) add opt-in Linux movement. Do not introduce a UI framework, browser automation, gesture system, or general plugin architecture.

### Cited Findings
- The package already centralizes commands in `eyehand.cli.main:main`, so a proof command belongs beside the existing subcommands rather than in a second application. — [pyproject.toml](file:///home/jonathan-wagstaff/Eye-Hand/pyproject.toml); [cli/main.py](file:///home/jonathan-wagstaff/Eye-Hand/src/eyehand/cli/main.py)
- Python 3.12 is the project runtime, making the standard-library HTTP server available without changing dependencies. — [pyproject.toml](file:///home/jonathan-wagstaff/Eye-Hand/pyproject.toml)
- The project’s stated priority is measured absolute accuracy, with an acceptance check at every step; current status places calibration and geometry ahead of the final cursor. — [README.md](file:///home/jonathan-wagstaff/Eye-Hand/README.md)

### Inferences
- Minimal future reach: add one cursor-sample/mapping module and its unit test; add one `browser-cursor` CLI path with an embedded/static minimal page and localhost SSE; then add one Linux backend selected explicitly by a separate CLI flag or command. Existing camera/config/timebase modules should remain unchanged except where the fusion loop eventually calls the output seam.
- For the browser milestone, a deterministic Lissajous/grid sweep is a better first producer than unfinished vision: it proves coordinate orientation, bounds, transport, loss behavior, and rendering independently. The next producer should be the real post-C8 fusion output.
- The Linux stage should reuse the exact same normalized samples and validity handling. Platform-specific display enumeration and injection should stay in the backend, not leak into triangulation or user calibration.
- Tests that future changes must reach are: a cursor mapping/invalidity unit test; a lightweight localhost response/event test if transport logic becomes non-trivial; and manual acceptance runs for browser edge/corner mapping, disconnect freeze/hide, Linux emergency stop, display bounds, and C7 loss.

### Gaps
- This review does not choose the Linux injection API; that choice depends on the current X11/Wayland environment and security model researched separately.
- No code change was requested or made. The recommended filenames are an implementation sketch, not an established repository convention.
