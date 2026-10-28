"""`eyehand intrinsics capture|solve|compare` (C1, C2) and `eyehand focus` (C2 step 1)."""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from eyehand import boards, intrinsics, sound, v4l2
from eyehand.capture import Camera
from eyehand.config import DEFAULT_CAMERA_CONFIG, PROJECT_ROOT, load_camera_configs
from eyehand.intrinsics import (TILT_MIN_DEG, TILT_PER_DIRECTION, Coverage, Session,
                                TargetScale, TiltTracker, View, ViewSelector, board_tilt)

EDID_PITCH_MM = 344 / 1920      # until C3 measures it; pixels assumed square
MAX_VIEWS = 120


def _camera(name: str):
    configs = load_camera_configs()
    if name not in configs:
        sys.exit(f"unknown camera {name!r}; configured: {', '.join(configs)}")
    return configs[name]


def _target_scale(args, spec: boards.BoardSpec) -> TargetScale:
    if spec.unit == "px":
        screen = PROJECT_ROOT / "calib" / "screen.json"
        if screen.exists():
            s = json.loads(screen.read_text())
            return TargetScale(s["pitch_x_mm"], s["pitch_y_mm"], "calib/screen.json (C3)")
        return TargetScale(EDID_PITCH_MM, EDID_PITCH_MM, "EDID 344 mm / 1920 px, square pixels")
    sx = args.square_x_mm or args.square_mm
    sy = args.square_y_mm or args.square_mm
    if sx is None or sy is None:
        return TargetScale(1.0, 1.0, f"nominal {spec.square} mm squares (not measured)")
    return TargetScale(sx / spec.square, sy / spec.square,
                       f"calipers: square {sx:.3f} x {sy:.3f} mm")


class LiveView:
    """Print board: camera image with detected corners, coverage map and status."""

    name = "eyehand intrinsics (Q: finish and solve, U: undo, space: pause)"

    def __init__(self, image_size):
        w, h = image_size
        self.scale = min(1600 / w, 900 / h, 1.0)
        cv2.namedWindow(self.name, cv2.WINDOW_AUTOSIZE)

    def show(self, image, corners, ids, lines: list[str], coverage: Coverage) -> None:
        img = coverage.overlay(image)
        if ids is not None:
            cv2.aruco.drawDetectedCornersCharuco(img, corners.reshape(-1, 1, 2),
                                                 ids.reshape(-1, 1), (255, 0, 0))
        # Mirrored like a mirror, so moving the board feels natural. Display only: the
        # detections and saved images are the real, unmirrored ones.
        img = cv2.flip(cv2.resize(img, None, fx=self.scale, fy=self.scale), 1)
        for k, text in enumerate(lines):
            y = 30 + 30 * k
            cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4)
            cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        cv2.imshow(self.name, img)

    def close(self) -> None:
        cv2.destroyWindow(self.name)


class ScreenBoardView:
    """Screen board: the laptop shows the board full-screen, so the only on-screen feedback is
    one line of text in the top margin. Sounds mark each captured view and completion."""

    name = "eyehand screen board (Q: finish and solve, U: undo, space: pause)"

    def __init__(self, spec: boards.BoardSpec):
        self.board, self.placement = boards.render_screen(spec)
        self.last_text = None
        boards.open_fullscreen(self.name, self.board)

    def show(self, image, corners, ids, lines: list[str], coverage: Coverage) -> None:
        text = "   ".join(lines)
        if text != self.last_text:
            cv2.imshow(self.name, draw_margin_status(self.board, self.placement, text))
            self.last_text = text

    def close(self) -> None:
        cv2.destroyWindow(self.name)


def draw_margin_status(board_img: np.ndarray, placement: boards.ScreenPlacement,
                       text: str) -> np.ndarray:
    """Board image with grey status text inside the top margin, clear of the board."""
    img = board_img.copy()
    height = placement.offset_y
    scale = min(0.7, (height - 12) / 30)
    cv2.putText(img, text, (placement.offset_x, height - 12), cv2.FONT_HERSHEY_SIMPLEX,
                scale, 120, 1, cv2.LINE_AA)
    return img


def _mirrored_names(names: list[str]) -> list[str]:
    swap = {"left": "right", "right": "left"}
    return ["-".join(swap.get(part, part) for part in n.split("-")) for n in names]


def cmd_capture(args) -> int:
    config = _camera(args.camera)
    spec = boards.PRINT_BOARD if args.board == "print" else boards.SCREEN_BOARD
    scale = _target_scale(args, spec)
    size = (config.width, config.height)
    detector = spec.detector()
    selector = ViewSelector(spec.board(), still_px=args.still_px)
    coverage = Coverage(size)
    tilts = TiltTracker()
    view_tilt: dict[str, dict | None] = {}

    if args.resume:
        captures = PROJECT_ROOT / "calib" / "raw" / f"intrinsics_{config.name}"
        if args.resume == "latest":
            found = sorted(p for p in captures.glob("*/session.json")
                           if json.loads(p.read_text())["board"].get("name") == spec.name)
            if not found:
                sys.exit(f"no earlier capture of camera {config.name} in {captures}")
            raw_dir = found[-1].parent
        else:
            raw_dir = Path(args.resume).resolve()
        if not (raw_dir / "session.json").exists():
            names = ", ".join(sorted(p.parent.name for p in captures.glob("*/session.json")))
            sys.exit(f"no capture at {raw_dir}. Captures of camera {config.name}: {names or 'none'}."
                     " Use --resume on its own for the latest.")
        session = Session.load(raw_dir)
        if session.camera != config.name or tuple(session.image_size) != size:
            sys.exit(f"{raw_dir} is a capture of another camera or mode")
        if session.board.get("name") != spec.name:
            sys.exit(f"{raw_dir} was captured with the {session.board.get('name')} board, not"
                     f" {spec.name}; start a new capture (leave out --resume)")
        for v in session.views:
            coverage.add(v.corners)
            view_tilt[v.name] = board_tilt(spec, v.ids, v.corners, size)
            tilts.add(view_tilt[v.name])
            selector.accepted.append(dict(zip(v.ids.tolist(), v.corners)))
    else:
        raw_dir = (PROJECT_ROOT / "calib" / "raw" / f"intrinsics_{config.name}"
                   / time.strftime("%Y%m%d-%H%M%S"))
        session = None

    print(f"[{config.name}] intrinsic calibration with the {spec.name} board")
    print(f"  target scale: {scale.source}")
    print(f"  raw data: {raw_dir.relative_to(PROJECT_ROOT)}"
          + (f" (resuming, {len(session.views)} views so far)" if session else ""))
    print("  Reminder: let the cameras warm up ~5 minutes first.")
    print(f"  Views are captured automatically when the board is still and in a new place.")
    print(f"  Finishes by itself at {args.views}+ views once the image corners are reached and"
          f" the board has been turned {TILT_MIN_DEG:.0f}°+ each way {TILT_PER_DIRECTION} times.")
    print("  Press Q in the window (or Ctrl+C here) to finish early and solve what you have.")

    with Camera(config) as cam:
        if session is None:
            session = Session(config.name, config.usb_id, size, config.fourcc,
                              dict(cam.controls), asdict(spec), asdict(scale))
        ui = LiveView(size) if args.board == "print" else ScreenBoardView(spec)
        paused = False
        try:
            while True:
                f = cam.read()
                corners, ids = intrinsics.detect(detector, f.image)
                got = None if paused else selector.offer(corners, ids, f.t_soe_ms)
                live_tilt = None
                if ids is not None and len(ids) >= 6:
                    live_tilt = board_tilt(spec, ids, corners, size)
                if got is not None:
                    got_ids, mean, t_mid, n = got
                    view_name = f"{len(session.views):03d}"
                    while any(v.name == view_name for v in session.views):
                        view_name = f"{int(view_name) + 1:03d}"
                    raw_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(raw_dir / f"{view_name}.png"), f.image)
                    session.views.append(View(view_name, got_ids, mean,
                                              intrinsics.object_points(spec, got_ids, scale),
                                              t_mid, n))
                    coverage.add(mean)
                    view_tilt[view_name] = board_tilt(spec, got_ids, mean, size)
                    tilts.add(view_tilt[view_name])
                    session.save(raw_dir)
                    print(f"  view {view_name}: {len(got_ids)} corners, "
                          f"coverage {100 * coverage.fraction():.0f}%, {tilts.summary()}")
                    sound.play("camera-shutter")

                missing = coverage.missing_corners()
                if args.board == "print":
                    missing = _mirrored_names(missing)
                now = ("tilt now: " + ", ".join(f"{k} {v:.0f}°" for k, v in live_tilt.items()
                                                if v >= 3) if live_tilt else "")
                lines = [f"views {len(session.views)}/{args.views}   coverage"
                         f" {100 * coverage.fraction():.0f}%"
                         + (f"   corners still needed: {', '.join(missing)}" if missing else
                            "   corners: done"),
                         f"turned {TILT_MIN_DEG:.0f}°+:  {tilts.summary()}",
                         ("PAUSED" if paused else selector.status) + (f"   ({now})" if now else ""),
                         "Q: finish and solve   U: undo   space: pause"]
                ui.show(f.image, corners, ids, lines, coverage)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                if key == ord(" "):
                    paused = not paused
                    selector.reset_window()
                if key in (ord("u"), ord("U")) and session.views:
                    last = session.views.pop()
                    selector.undo()
                    coverage.add(last.corners, sign=-1)
                    tilts.add(view_tilt.pop(last.name, None), sign=-1)
                    (raw_dir / f"{last.name}.png").unlink(missing_ok=True)
                    session.save(raw_dir)
                    print(f"  view {last.name} removed")
                done = (len(session.views) >= args.views and coverage.complete()
                        and tilts.complete())
                if done or len(session.views) >= MAX_VIEWS:
                    sound.play("complete")
                    break
        except KeyboardInterrupt:
            print("\n  stopped; solving what was captured")
        finally:
            ui.close()

    if len(session.views) < 10:
        print(f"only {len(session.views)} views captured; need at least 10. Nothing solved.")
        return 1
    if not tilts.complete():
        print(f"  note: not enough turned views ({tilts.summary()}); the focal length will be"
              " poorly determined")
    return _solve_and_report(session, raw_dir)


def _solve_and_report(session: Session, raw_dir: Path) -> int:
    print(f"\nsolving {len(session.views)} views ...")
    report = intrinsics.solve(session.views, session.image_size)
    result = intrinsics.build_result(session, report, raw_dir)
    (raw_dir / "result.json").write_text(json.dumps(result, indent=1))
    print(intrinsics.format_report(report))
    if report["passed"]:
        path = intrinsics.install(result)
        print(f"installed: {path.relative_to(PROJECT_ROOT)}")
        return 0
    print(f"not installed (result kept in {(raw_dir / 'result.json').relative_to(PROJECT_ROOT)}).")
    print("Usual fixes: cover the missing image areas, tilt the board more, hold it stiller,"
          " or remove bad views and run `eyehand intrinsics solve` on the folder again.")
    return 1


def cmd_solve(args) -> int:
    raw_dir = Path(args.raw_dir).resolve()
    session = Session.load(raw_dir)
    if args.drop:
        drop = set(args.drop)
        session.views = [v for v in session.views if v.name not in drop]
    return _solve_and_report(session, raw_dir)


def cmd_compare(args) -> int:
    a, b = (json.loads(Path(p).read_text()) for p in (args.a, args.b))
    if a["image_size"] != b["image_size"]:
        sys.exit("different image sizes: not comparable")
    Ka, Kb = np.asarray(a["K"]), np.asarray(b["K"])
    da, db = np.asarray(a["dist"]), np.asarray(b["dist"])
    size = tuple(a["image_size"])
    for label, i, j, key in (("fx", 0, 0, "fx"), ("fy", 1, 1, "fy"),
                             ("cx", 0, 2, "cx"), ("cy", 1, 2, "cy")):
        diff = Kb[i, j] - Ka[i, j]
        sigma = np.hypot(a["std"][key], b["std"][key])
        print(f"{label}: {Ka[i, j]:9.2f} -> {Kb[i, j]:9.2f}  diff {diff:+7.2f}"
              f"  ({abs(diff) / sigma:.1f} sigma)")
    m = intrinsics.TRACKING_MARGIN
    raw = intrinsics.model_difference_px(Ka, da, Kb, db, size, allow_rotation=False, margin=m)
    rot = intrinsics.model_difference_px(Ka, da, Kb, db, size, margin=m)
    print(f"ray difference in the tracking area: max {raw[0]:.2f} px, RMS {raw[1]:.2f} px;"
          f" after removing a rotation: max {rot[0]:.2f} px, RMS {rot[1]:.2f} px")
    # Two independent calibrations each within the gate differ by up to sqrt(2) x the gate.
    ok = rot[0] <= np.sqrt(2) * intrinsics.GATES["ray_uncertainty_max_px"]
    print("AGREE" if ok else "DISAGREE")
    return 0 if ok else 1


# ---------------------------------------------------------------------------------------------
# Focus (C2 step 1)


def _measure_frames(images: list[np.ndarray], detector, squares_x: int):
    """(sharpness, corners found, board size in px) from a few frames of a still board.

    Frames are averaged first: in dim light, single-frame sensor noise dominates sharpness.
    Sharpness = mean squared gradient (Tenengrad) of the lightly smoothed image inside the
    board. Board size = median distance between neighbouring ChArUco corners; if the lens
    moves when focus changes, the image scale changes ("focus breathing"), so this shows
    whether the focus control does anything at all.
    """
    mean = np.mean(images, axis=0).astype(np.uint8)
    gray = cv2.cvtColor(mean, cv2.COLOR_BGR2GRAY) if mean.ndim == 3 else mean
    corners, ids = intrinsics.detect(detector, gray)
    mask = np.zeros(gray.shape, np.uint8)
    size = float("nan")
    if ids is not None and len(ids) >= 4:
        cv2.fillConvexPoly(mask, cv2.convexHull(corners.astype(np.int32)), 1)
        lookup = dict(zip(ids.tolist(), corners))
        gaps = [np.linalg.norm(lookup[i] - lookup[i + 1]) for i in lookup
                if i + 1 in lookup and (i + 1) % (squares_x - 1) != 0]
        if gaps:
            size = float(np.median(gaps))
    else:
        h, w = gray.shape
        mask[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = 1
    smooth = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
    gx, gy = cv2.Sobel(smooth, cv2.CV_32F, 1, 0), cv2.Sobel(smooth, cv2.CV_32F, 0, 1)
    sharp = float(np.mean((gx**2 + gy**2)[mask > 0]))
    return sharp, 0 if ids is None else len(ids), size


def cmd_focus(args) -> int:
    config = _camera(args.camera)
    if "focus_absolute" not in config.controls:
        sys.exit(f"camera {config.name} has no focus_absolute in its config")
    spec = boards.PRINT_BOARD if args.board == "print" else boards.SCREEN_BOARD
    detector = spec.detector()
    print(f"[{config.name}] focus sweep. Prop the board still at the working distance"
          " (where face and hand will be). Don't touch it until the sweep ends.")

    def measure(cam, value):
        v4l2.apply_and_verify(config.device, {"focus_absolute": value})
        t_end = time.monotonic() + args.settle_s     # let the focus motor arrive
        while time.monotonic() < t_end:
            cam.read()
        return _measure_frames([cam.read().image for _ in range(8)], detector, spec.squares_x)

    results: dict[int, tuple[float, int, float]] = {}
    with Camera(config) as cam:
        for value in range(args.start, args.stop + 1, args.step):
            results[value] = measure(cam, value)
            print(f"  {value:4d}: sharpness {results[value][0]:8.1f}, {results[value][1]:2d}"
                  f" corners, square {results[value][2]:6.2f} px")
        repeat = measure(cam, args.start)

    sharp = np.array([r[0] for r in results.values()])
    sizes = np.array([r[2] for r in results.values()])
    noise = abs(repeat[0] - results[args.start][0]) / results[args.start][0]
    spread = (sharp.max() - sharp.min()) / sharp.max()
    print(f"\nrepeat of {args.start}: sharpness {repeat[0]:.1f} (first time"
          f" {results[args.start][0]:.1f}, so measurement noise ~{100 * noise:.0f} %)")
    values = np.array(list(results), float)
    ok = np.isfinite(sizes)
    lens_moves = None
    if ok.sum() >= 4:
        # A moving lens changes the image scale steadily with the focus value ("breathing").
        # Random scatter of the size measurement is not lens motion, so look for a trend that
        # stands out from the scatter about it.
        slope, icpt = np.polyfit(values[ok], sizes[ok], 1)
        trend = abs(slope) * (values[ok].max() - values[ok].min()) / np.median(sizes[ok])
        scatter = np.std(sizes[ok] - (slope * values[ok] + icpt)) / np.median(sizes[ok])
        print(f"board size: trend across the focus range {100 * trend:.2f} %, random scatter"
              f" {100 * scatter:.2f} %")
        lens_moves = trend > max(3 * scatter, 0.003)
    else:
        print("board not detected well enough to measure its size")
    typical = np.median(sharp)
    flat = np.mean(np.abs(sharp - typical) / typical < max(3 * noise, 0.05))
    if lens_moves is False and flat >= 0.6:
        print("\nRESULT: the focus control has no measurable effect: no scale trend, and sharpness"
              f" is the same at {100 * flat:.0f} % of the settings.\nThis camera behaves as"
              " fixed-focus, which is good for calibration. Keep the configured value.")
        return 0
    best = max(results, key=lambda v: results[v][0])
    print(f"\nbest focus_absolute = {best}")
    if args.write:
        _write_focus(config.name, best)
        print(f"written to {DEFAULT_CAMERA_CONFIG.relative_to(PROJECT_ROOT)}."
              " Any intrinsics made with another focus value are now invalid.")
    else:
        print(f"to use it: eyehand focus {config.name} --write")
    return 0


def _write_focus(camera: str, value: int, path: Path = DEFAULT_CAMERA_CONFIG) -> None:
    text = path.read_text()
    section = f"[cameras.{camera}.controls]"
    start = text.index(section)
    end = text.find("\n[", start + len(section))
    end = len(text) if end < 0 else end
    block, n = re.subn(r"^focus_absolute\s*=\s*\d+.*$",
                       f"focus_absolute = {value:<15d} # chosen with `eyehand focus` on "
                       f"{time.strftime('%Y-%m-%d')}; changing it invalidates C2 and C5",
                       text[start:end], count=1, flags=re.M)
    if n != 1:
        raise RuntimeError(f"focus_absolute not found under {section}")
    path.write_text(text[:start] + block + text[end:])


def add_parsers(sub) -> None:
    p = sub.add_parser("intrinsics", help="intrinsic calibration (C1 laptop, C2 Feeltek)")
    isub = p.add_subparsers(dest="intrinsics_cmd", required=True)

    s = isub.add_parser("capture", help="capture board views live, then solve")
    s.add_argument("camera")
    s.add_argument("--board", choices=["print", "screen"], default="print",
                   help="print: hold the printed board (C1). screen: the laptop shows the board"
                        " and you move the camera (C2)")
    s.add_argument("--views", type=int, default=50, help="stop after this many (default 50)")
    s.add_argument("--square-mm", type=float, help="measured print square size (both axes)")
    s.add_argument("--square-x-mm", type=float, help="measured square size along the board's x")
    s.add_argument("--square-y-mm", type=float, help="measured square size along the board's y")
    s.add_argument("--resume", metavar="RAW_DIR", nargs="?", const="latest",
                   help="continue an earlier capture, keeping its views (on its own: the latest)")
    s.add_argument("--still-px", type=float, default=0.5,
                   help="max corner motion per frame to count as still (default 0.5)")
    s.set_defaults(func=cmd_capture)

    s = isub.add_parser("solve", help="re-solve a saved capture folder")
    s.add_argument("raw_dir")
    s.add_argument("--drop", nargs="*", help="view names to leave out, e.g. 007 012")
    s.set_defaults(func=cmd_solve)

    s = isub.add_parser("compare", help="compare two intrinsics results")
    s.add_argument("a")
    s.add_argument("b")
    s.set_defaults(func=cmd_compare)

    s = sub.add_parser("focus", help="sweep focus_absolute and find the sharpest value (C2)")
    s.add_argument("camera", nargs="?", default="F")
    s.add_argument("--board", choices=["print", "screen"], default="print")
    s.add_argument("--start", type=int, default=0)
    s.add_argument("--stop", type=int, default=256)
    s.add_argument("--step", type=int, default=32)
    s.add_argument("--settle-s", type=float, default=1.0)
    s.add_argument("--write", action="store_true", help="write the best value into the config")
    s.set_defaults(func=cmd_focus)
