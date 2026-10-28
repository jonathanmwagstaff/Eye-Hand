"""`eyehand` command line: cameras, calibration, app, and browser cursor."""

from __future__ import annotations

import argparse
import csv
import sys
import threading
import time
from pathlib import Path

import numpy as np

from eyehand import boards, v4l2
from eyehand.capture import Camera, CameraError
from eyehand.cli import intrinsics_cmd, mirror_cmd
from eyehand.config import PROJECT_ROOT, load_camera_configs
from eyehand.timebase import FrameClock, drift_ppm, phase_ms


def _select(configs, names):
    if not names:
        return list(configs.values())
    missing = [n for n in names if n not in configs]
    if missing:
        sys.exit(f"unknown camera(s): {', '.join(missing)}; configured: {', '.join(configs)}")
    return [configs[n] for n in names]


def cmd_probe(args) -> int:
    """C0 check: devices present, controls applied and verified, real fps and timing."""
    configs = load_camera_configs()
    print("Video devices (by-id):")
    for d in v4l2.list_video_devices():
        print(f"  {d.usb_id or '?':9}  {d.node:12}  {d.by_id}")
    ok = True
    for c in _select(configs, args.camera):
        print(f"\n[{c.name}] {c.description}")
        try:
            with Camera(c) as cam:
                print(f"  mode {c.width}x{c.height} {c.fourcc} @ {c.fps}: OK")
                print(f"  controls verified: {cam.controls}")
                clock = FrameClock(c.frame_period_ms)
                ages, brightness = [], []
                for _ in range(args.frames):
                    f = cam.read()
                    clock.add(f.t_soe_ms)
                    ages.append(f.t_arrival_ms - f.t_soe_ms)
                    brightness.append(float(f.image.mean()))
        except (CameraError, v4l2.ControlError) as e:
            print(f"  FAIL: {e}")
            ok = False
            continue
        period_ok = abs(clock.period_ms - c.frame_period_ms) < 1.0 and clock.dropped == 0
        ok &= period_ok
        print(f"  frame period {clock.period_ms:.3f} ms (nominal {c.frame_period_ms:.3f}), "
              f"dropped {clock.dropped}  {'OK' if period_ok else 'FAIL'}")
        print(f"  timestamp jitter (residual RMS) {clock.residual_rms_ms():.2f} ms")
        print(f"  exposure-start -> Python arrival: median {np.median(ages):.1f} ms, "
              f"max {np.max(ages):.1f} ms")
        b = float(np.mean(brightness))
        hint = "  (too dark: add light or raise gain)" if b < 60 else ""
        print(f"  mean brightness {b:.0f}/255{hint}")
    return 0 if ok else 1


def cmd_timing(args) -> int:
    """Run both cameras together; report periods, drift, phase, drops, and save raw stamps."""
    configs = load_camera_configs()
    cams = _select(configs, args.camera)
    if len(cams) != 2:
        sys.exit("timing needs exactly two cameras")
    clocks = {c.name: FrameClock(c.frame_period_ms) for c in cams}
    rows: list[tuple[str, int, float, float]] = []
    lock = threading.Lock()
    stop = threading.Event()
    errors: list[str] = []

    def worker(c):
        try:
            with Camera(c) as cam:
                while not stop.is_set():
                    f = cam.read()
                    with lock:
                        ft = clocks[c.name].add(f.t_soe_ms)
                        rows.append((c.name, ft.n, f.t_soe_ms, f.t_arrival_ms))
        except (CameraError, v4l2.ControlError) as e:
            errors.append(f"{c.name}: {e}")
            stop.set()

    threads = [threading.Thread(target=worker, args=(c,), daemon=True) for c in cams]
    for t in threads:
        t.start()
    a, b = (clocks[c.name] for c in cams)
    t_end = time.monotonic() + args.seconds
    print(f"{'t[s]':>5}  {'phase ' + cams[1].name + '-' + cams[0].name + ' [ms]':>18}  dropped")
    t0 = time.monotonic()
    while time.monotonic() < t_end and not stop.is_set():
        time.sleep(1.0)
        with lock:
            if a.count > 30 and b.count > 30:
                now = rows[-1][2]
                print(f"{time.monotonic() - t0:5.0f}  {phase_ms(a, b, now):18.2f}  "
                      f"{a.dropped}/{b.dropped}")
    stop.set()
    for t in threads:
        t.join(timeout=2)
    if errors:
        print("FAIL:", *errors, sep="\n  ")
        return 1

    for c, clk in zip(cams, (a, b)):
        print(f"[{c.name}] period {clk.period_ms:.4f} ms, jitter {clk.residual_rms_ms():.2f} ms, "
              f"dropped {clk.dropped}")
    print(f"drift {cams[1].name} vs {cams[0].name}: {drift_ppm(a, b):+.0f} ppm "
          f"({drift_ppm(a, b) * 1e-6 * 1000:+.3f} ms per second)")

    out_dir = PROJECT_ROOT / "calib" / "raw" / "timing"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / time.strftime("%Y%m%d-%H%M%S.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["camera", "n", "t_soe_ms", "t_arrival_ms"])
        w.writerows(rows)
    print(f"raw timestamps: {out}")
    return 0


def cmd_preview(args) -> int:
    """Live view of the cameras with timing and brightness, optionally detecting a board."""
    import cv2

    configs = load_camera_configs()
    cams = [Camera(c).open() for c in _select(configs, args.camera)]
    spec = {"print": boards.PRINT_BOARD, "screen": boards.SCREEN_BOARD}.get(args.board)
    detector = spec.detector() if spec else None
    clocks = [FrameClock(c.config.frame_period_ms) for c in cams]
    try:
        while True:
            views = []
            for cam, clock in zip(cams, clocks):
                f = cam.read()
                clock.add(f.t_soe_ms)
                img = f.image
                if detector is not None:
                    corners, ids, _, _ = detector.detectBoard(img)
                    if ids is not None:
                        cv2.aruco.drawDetectedCornersCharuco(img, corners, ids)
                text = (f"{cam.config.name}  {1000 / clock.period_ms:5.2f} fps  "
                        f"drop {clock.dropped}  age {f.t_arrival_ms - f.t_soe_ms:4.0f} ms  "
                        f"bright {img.mean():3.0f}")
                cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                scale = 480 / img.shape[0]
                views.append(cv2.resize(img, None, fx=scale, fy=scale))
            cv2.imshow("eyehand preview (ESC quits)", np.hstack(views))
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        for cam in cams:
            cam.close()
        cv2.destroyAllWindows()
    return 0


def cmd_boards(args) -> int:
    out = PROJECT_ROOT / "boards"
    png = boards.save_print(out, dpi=args.dpi)
    print(f"print board: {png}  (print at 100 %, then measure a square with calipers)")
    for pdf in boards.save_print_pdfs(out, dpi=args.dpi):
        print(f"print board, centred page: {pdf}  (print at 'Actual size')")
    for p in boards.save_screen(out):
        print(f"screen board: {p}")
    return 0


def cmd_show_board(args) -> int:
    """Show the screen board full-screen, normal (C2) or mirrored (C4)."""
    import cv2

    image, placement = boards.render_screen(mirrored=args.mirrored)
    name = "eyehand board (ESC quits)"
    boards.open_fullscreen(name, image)
    while cv2.waitKey(50) & 0xFF != 27:
        pass
    cv2.destroyAllWindows()
    return 0


def cmd_status(args) -> int:
    from eyehand import status

    statuses = status.calibration_status(load_camera_configs())
    print(status.format_status(statuses))
    return 0


def cmd_stereo_solve(args) -> int:
    """Re-solve a saved C5 capture (normally the app does this)."""
    import json

    from eyehand import intrinsics, stereo
    from eyehand.app import print_board_scale

    configs = load_camera_configs()
    raw_dir = Path(args.raw_dir).resolve()
    session = stereo.StereoSession.load(raw_dir)
    intr = {n: intrinsics.load_intrinsics(configs[n]) for n in ("L", "F")}
    want = {n: stereo.intrinsics_fingerprint(intr[n]) for n in ("L", "F")}
    if session.intrinsics != want:
        sys.exit("this capture was made with different lens calibrations; capture again")
    from eyehand.intrinsics import TargetScale

    scale = TargetScale(**session.scale)
    obj_of = lambda ids: intrinsics.object_points(boards.PRINT_BOARD, ids, scale)
    report = stereo.solve(session.views, obj_of, intr["L"], intr["F"], boards.PRINT_BOARD.squares_x)
    result = stereo.build_result(session, report, raw_dir)
    (raw_dir / "result.json").write_text(json.dumps(result, indent=1))
    print(stereo.format_report(report))
    if report["passed"] and args.install:
        stereo.install(result)
        print(f"installed: {stereo.STEREO_PATH.relative_to(PROJECT_ROOT)}")
    return 0 if report["passed"] else 1


def cmd_steady(args) -> int:
    """Watch cameras for physical movement (no board needed); optionally compare with the
    last frames of the latest stereo capture."""
    import cv2

    from eyehand import intrinsics
    from eyehand.steady import SteadyMeter

    configs = load_camera_configs()
    cams = _select(configs, args.camera)
    stereo_dir = None
    if args.since_stereo:
        runs = sorted((PROJECT_ROOT / "calib" / "raw" / "stereo").glob("*/session.json"))
        if not runs:
            sys.exit("no stereo capture found")
        stereo_dir = runs[-1].parent
    opened = {}
    meters = {}
    try:
        for c in cams:
            # Measurement settings: the same exposure as the calibration frames it may be
            # compared with.
            cam = Camera(c, profile="measurement").open()
            opened[c.name] = cam
            try:
                f = intrinsics.load_intrinsics(c).K[0, 0]
            except Exception:
                f = 1000.0 * c.width / 1280
            if stereo_dir is not None:
                ref_imgs = sorted(stereo_dir.glob(f"*_{c.name}.png"))
                refs = [cv2.imread(str(p)) for p in ref_imgs]
                print(f"[{c.name}] reference: all {len(refs)} frames of"
                      f" {stereo_dir.relative_to(PROJECT_ROOT)}")
            else:
                refs = [cam.read().image]
            meters[c.name] = SteadyMeter(refs, f)
        print("\nThis test needs NOTHING held up: no board, no face, no hands. It watches the")
        print("room behind (walls, furniture, ceiling) and measures whether that scenery shifts")
        print("in the picture, which would mean the camera itself turned. Step out of view or")
        print("sit still, and don't touch the cameras or the laptop until it finishes.")
        print("A window shows each camera with the scenery points it is tracking (green).\n")
        print(f"{'t[s]':>5}  " + "  ".join(f"{n}: shift px / turned deg / roll deg".ljust(40)
                                          for n in opened))
        t0 = time.monotonic()
        worst = {n: 0.0 for n in opened}
        readings = {n: 0 for n in opened}
        while True:
            row = []
            panels = []
            for n, cam in opened.items():
                img = cam.read().image
                m = meters[n].measure(img)
                panels.append(_steady_panel(n, img, meters[n], m))
                if m is None:
                    row.append("(not enough matching features)".ljust(40))
                    continue
                worst[n] = max(worst[n], m.tilt_deg)
                readings[n] += 1
                row.append(f"{np.hypot(m.dx_px, m.dy_px):6.1f} / {m.tilt_deg:6.3f} / "
                           f"{m.roll_deg:+6.3f}  ({m.inliers} pts)".ljust(40))
            print(f"{time.monotonic() - t0:5.0f}  " + "  ".join(row), flush=True)
            cv2.imshow("eyehand steady (nothing to hold up; Q stops)", np.hstack(panels))
            if time.monotonic() - t0 > args.seconds:
                break
            t_next = time.monotonic() + args.interval
            stop = False
            while time.monotonic() < t_next:                  # keep the window responsive
                if cv2.waitKey(50) & 0xFF in (27, ord("q"), ord("Q")):
                    stop = True
                    break
                for cam in opened.values():
                    cam.read()
            if stop:
                break
        for n, w in worst.items():
            if readings[n] == 0:
                print(f"[{n}] NO READING: not enough matching scenery. Nothing is known about"
                      " this camera; add light or point it at a more detailed background.")
                continue
            verdict = "steady" if w < 0.05 else "MOVED" if w > 0.2 else "slight movement"
            print(f"[{n}] largest turn {w:.3f}° over {readings[n]} readings: {verdict}")
    finally:
        for cam in opened.values():
            cam.close()
        cv2.destroyAllWindows()
    return 0


def _steady_panel(name, img, meter, motion):
    """Camera picture (mirrored) with the tracked scenery points and the current reading."""
    import cv2

    k = 480 / img.shape[0]
    out = cv2.resize(img, None, fx=k, fy=k)
    kp, _ = meter._features(img)
    for p in kp[:600]:
        x, y = p.pt[0] * k, p.pt[1] * k
        cv2.circle(out, (int(x), int(y)), 2, (0, 220, 0), -1)
    out = cv2.flip(out, 1)
    text = (f"{name}: turned {motion.tilt_deg:.3f} deg" if motion else f"{name}: no reading")
    colour = (0, 220, 0) if motion and motion.tilt_deg < 0.05 else (0, 0, 255)
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
    return out


def cmd_screen(args) -> int:
    from eyehand import screen_calib

    return screen_calib.main()


def cmd_app(args) -> int:
    from eyehand import app

    return app.main()


def cmd_browser_cursor(args) -> int:
    from eyehand import browser_cursor

    return browser_cursor.run(args.port)


LAUNCHER = Path.home() / ".local" / "share" / "applications" / "eyehand.desktop"


def cmd_install_launcher(args) -> int:
    """Desktop launcher, so the app opens from the applications menu without a terminal."""
    import shutil

    uv = shutil.which("uv")
    if uv is None:
        sys.exit("uv not found on PATH")
    LAUNCHER.parent.mkdir(parents=True, exist_ok=True)
    LAUNCHER.write_text(
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=Eye-Hand\n"
        "Comment=Camera placement, calibration status and camera-position recalibration\n"
        f"Exec={uv} run --project {PROJECT_ROOT} eyehand app\n"
        f"Path={PROJECT_ROOT}\n"
        "Icon=camera-web\n"
        "Terminal=false\n"
        "Categories=Utility;\n")
    print(f"launcher written: {LAUNCHER}")
    print("Find 'Eye-Hand' in the applications menu (it can be pinned to the dock).")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="eyehand")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("probe", help="check cameras, controls, frame rate and timestamps (C0)")
    s.add_argument("camera", nargs="*", help="camera names from config (default: all)")
    s.add_argument("--frames", type=int, default=90)
    s.set_defaults(func=cmd_probe)

    s = sub.add_parser("timing", help="run both cameras; measure periods, drift and phase")
    s.add_argument("camera", nargs="*")
    s.add_argument("--seconds", type=float, default=20)
    s.set_defaults(func=cmd_timing)

    s = sub.add_parser("preview", help="live camera view")
    s.add_argument("camera", nargs="*")
    s.add_argument("--board", choices=["print", "screen"], help="detect and draw a ChArUco board")
    s.set_defaults(func=cmd_preview)

    s = sub.add_parser("boards", help="write printable and screen ChArUco boards to boards/")
    s.add_argument("--dpi", type=int, default=600)
    s.set_defaults(func=cmd_boards)

    s = sub.add_parser("show-board", help="display the screen ChArUco board full-screen")
    s.add_argument("--mirrored", action="store_true", help="flipped, for the mirror (C4)")
    s.set_defaults(func=cmd_show_board)

    intrinsics_cmd.add_parsers(sub)
    mirror_cmd.add_parsers(sub)

    s = sub.add_parser("status", help="which calibrations are valid, stale or missing")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("stereo", help="stereo extrinsics (C5); normally run from the app")
    ssub = s.add_subparsers(dest="stereo_cmd", required=True)
    t = ssub.add_parser("solve", help="re-solve a saved capture folder")
    t.add_argument("raw_dir")
    t.add_argument("--install", action="store_true", help="install the result if it passes")
    t.set_defaults(func=cmd_stereo_solve)

    s = sub.add_parser("steady", help="measure whether cameras are physically steady (no board)")
    s.add_argument("camera", nargs="*")
    s.add_argument("--seconds", type=float, default=120)
    s.add_argument("--interval", type=float, default=5)
    s.add_argument("--since-stereo", action="store_true",
                   help="compare with the last frames of the latest stereo capture")
    s.set_defaults(func=cmd_steady)

    s = sub.add_parser("screen", help="C3: measure the screen's pixel pitch with calipers")
    s.set_defaults(func=cmd_screen)

    s = sub.add_parser("app", help="placement view, calibration status, recalibration (GUI)")
    s.set_defaults(func=cmd_app)

    s = sub.add_parser("browser-cursor", help="development eye-hand cursor in a localhost page")
    s.add_argument("--port", type=int, default=8765)
    s.set_defaults(func=cmd_browser_cursor)

    s = sub.add_parser("install-launcher", help="add Eye-Hand to the desktop applications menu")
    s.set_defaults(func=cmd_install_launcher)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
