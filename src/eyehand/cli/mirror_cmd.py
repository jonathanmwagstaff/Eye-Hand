"""`eyehand mirror` (C4): capture mirror poses of the flipped screen board, solve T_S_L."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from eyehand import boards, intrinsics, mirror, sound
from eyehand.capture import Camera
from eyehand.config import PROJECT_ROOT, load_camera_configs
from eyehand.screen_calib import screen_pitch_mm

BOARD_TOP = 40
INSET_W, INSET_H = 512, 288
TARGET_POSES = 20
MAX_POSES = 60
SPREAD_DEG = mirror.GATES["normal_spread_deg"]


def _virtual_angles(view, target, intr):
    """Sideways and up/down angles of the virtual board's normal. A mirror tilt of a degrees
    turns the reflection by 2a, so mirror variety = half the variety of these."""
    T = mirror.virtual_pose(view, target, intr)
    if T is None:
        return None
    n = T[:3, 2] if T[2, 2] < 0 else -T[:3, 2]
    return np.degrees(np.arctan2(n[0], -n[2])), np.degrees(np.arctan2(n[1], -n[2]))


def compose(board_bgr, placement, frame, corners, ids, lines, colour) -> np.ndarray:
    """Board at the top; below it a live, mirror-like view of camera L and the status."""
    out = board_bgr.copy()
    h, w = out.shape[:2]
    y0 = BOARD_TOP + int(boards.MIRROR_BOARD.squares_y * boards.MIRROR_BOARD.square) + 16
    k = INSET_W / frame.shape[1]
    inset = cv2.resize(frame, (INSET_W, int(frame.shape[0] * k)))
    if ids is not None:
        for c in corners:
            cv2.circle(inset, (int(c[0] * k), int(c[1] * k)), 3, (0, 200, 0), -1)
    inset = cv2.flip(inset, 1)
    # Half contrast, so the small copy of the board inside the preview can never compete with
    # the real board when the camera sees the screen in the mirror.
    inset = cv2.addWeighted(inset, 0.5, np.full_like(inset, 160), 0.5, 0)
    inset = inset[: min(INSET_H, h - y0 - 4)]
    x0 = placement.offset_x
    out[y0: y0 + inset.shape[0], x0: x0 + INSET_W] = inset
    cv2.rectangle(out, (x0 - 3, y0 - 3), (x0 + INSET_W + 2, y0 + inset.shape[0] + 2), colour, 3)
    for i, text in enumerate(lines):
        cv2.putText(out, text, (x0 + INSET_W + 30, y0 + 40 + 44 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.95, (40, 40, 40), 2, cv2.LINE_AA)
    return out


def cmd_capture(args) -> int:
    config = load_camera_configs()["L"]
    intr = intrinsics.load_intrinsics(config)
    spec = boards.MIRROR_BOARD
    board_img, placement = boards.render_screen(spec, mirrored=True, top=BOARD_TOP)
    board_bgr = cv2.cvtColor(board_img, cv2.COLOR_GRAY2BGR)
    px, py, pitch_source = screen_pitch_mm((placement.screen_width, placement.screen_height))
    target = mirror.ScreenTarget(spec, placement, (px, py))
    selector = intrinsics.ViewSelector(spec.board(), novel_px=args.novel_px)
    detector = spec.detector()
    angles = []
    captures = PROJECT_ROOT / "calib" / "raw" / "mirror"
    if args.resume:
        found = sorted(captures.glob("*/session.json"))
        if not found:
            print("no earlier mirror capture to resume")
            return 1
        raw_dir = found[-1].parent
        session = mirror.MirrorSession.load(raw_dir)
        if session.intrinsics_L != mirror.intrinsics_fingerprint(intr) or \
                session.board != spec.name or session.placement != asdict(placement):
            print(f"{raw_dir} was captured with a different setup; start a new capture")
            return 1
        for v in session.views:
            selector.accepted.append(dict(zip(v.ids.tolist(), v.corners)))
            angles.append(_virtual_angles(v, target, intr))
        print(f"  resuming {raw_dir.relative_to(PROJECT_ROOT)} with {len(session.views)} poses")
    else:
        raw_dir = captures / time.strftime("%Y%m%d-%H%M%S")
        session = mirror.MirrorSession(mirror.intrinsics_fingerprint(intr), [px, py],
                                       asdict(placement), board=spec.name)
    target_poses = args.poses

    print("[C4] mirror calibration of camera L against the screen")
    print(f"  screen pitch: {pitch_source}; raw data: {raw_dir.relative_to(PROJECT_ROOT)}")
    print("  Set the screen to full brightness. Prop the mirror 25-35 cm in front of the lid,")
    print("  facing it, so camera L sees the screen in it. A click = a pose was captured; then")
    print("  re-prop the mirror at a different tilt. Tilt it left/right AND up/down, and vary")
    print("  the distance. Q (in the board window) finishes early.")

    name = "eyehand mirror (Q: finish and solve, U: undo)"
    boards.open_fullscreen(name, board_bgr)
    try:
        with Camera(config) as cam:
            while True:
                f = cam.read()
                corners, ids = intrinsics.detect(detector, f.image)
                got = selector.offer(corners, ids, f.t_soe_ms)
                if got is not None:
                    got_ids, mean, t_mid, _ = got
                    num = max([int(v.name) for v in session.views], default=-1) + 1
                    view = mirror.MirrorView(f"{num:03d}", got_ids, mean, t_mid)
                    a = _virtual_angles(view, target, intr)
                    if a is None:
                        selector.undo()
                    else:
                        session.views.append(view)
                        angles.append(a)
                        raw_dir.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(raw_dir / f"{view.name}.png"), f.image)
                        session.save(raw_dir)
                        sound.play("camera-shutter")
                        print(f"  pose {view.name}: {len(got_ids)} corners")
                n = len(session.views)
                spread = (np.std(np.array(angles), axis=0) / 2) if n >= 2 else np.zeros(2)
                n_seen = 0 if ids is None else len(ids)
                status = selector.status
                if n_seen == 0:
                    status = "no board in view: aim the mirror so the camera sees the screen"
                colour = (0, 170, 0) if status == "captured" else (0, 140, 255) if n_seen else (0, 0, 220)
                lines = [f"Mirror poses: {n} of {target_poses}",
                         f"Tilt variety: sideways {spread[0]:.1f} deg, up/down {spread[1]:.1f} deg",
                         f"   (each needs {SPREAD_DEG:.0f} deg: tilt left/right AND up/down)",
                         f"Board corners seen now: {n_seen}",
                         status,
                         "Q: finish and solve    U: undo last pose"]
                cv2.imshow(name, compose(board_bgr, placement, f.image, corners, ids, lines, colour))
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                if key in (ord("u"), ord("U")) and session.views:
                    last = session.views.pop()
                    angles.pop()
                    selector.undo()
                    (raw_dir / f"{last.name}.png").unlink(missing_ok=True)
                    session.save(raw_dir)
                if (n >= target_poses and min(spread) >= SPREAD_DEG) or n >= MAX_POSES:
                    sound.play("complete")
                    break
    except KeyboardInterrupt:
        print("\n  stopped; solving what was captured")
    finally:
        cv2.destroyAllWindows()
    return _solve(session, raw_dir)


def _solve(session: mirror.MirrorSession, raw_dir: Path) -> int:
    if len(session.views) < 3:
        print(f"only {len(session.views)} poses; need at least 3 (20 recommended). Nothing solved.")
        return 1
    config = load_camera_configs()["L"]
    intr = intrinsics.load_intrinsics(config)
    if session.intrinsics_L != mirror.intrinsics_fingerprint(intr):
        print("this capture was made with different camera L intrinsics; capture again")
        return 1
    placement = boards.ScreenPlacement(**session.placement)
    target = mirror.ScreenTarget(boards.SCREEN_BOARDS[session.board], placement,
                                 tuple(session.pitch))
    print(f"\nsolving {len(session.views)} mirror poses ...")
    r = mirror.solve(session.views, target, intr)
    try:
        raw = str(raw_dir.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        raw = str(raw_dir)
    result = {"intrinsics_L": session.intrinsics_L, "captured": session.created,
              "solved": time.strftime("%Y-%m-%dT%H:%M:%S"), "raw_dir": raw,
              "placement": session.placement, **r}
    (raw_dir / "result.json").write_text(json.dumps(result, indent=1))
    print(mirror.format_report(r))
    if r["passed"]:
        mirror.install(result)
        print(f"installed: {mirror.RESULT_PATH.relative_to(PROJECT_ROOT)}")
        return 0
    print(f"not installed (kept in {(raw_dir / 'result.json').relative_to(PROJECT_ROOT)})")
    return 1


def cmd_solve(args) -> int:
    raw_dir = Path(args.raw_dir).resolve()
    session = mirror.MirrorSession.load(raw_dir)
    if args.drop:
        session.views = [v for v in session.views if v.name not in set(args.drop)]
    return _solve(session, raw_dir)


def add_parsers(sub) -> None:
    p = sub.add_parser("mirror", help="C4: camera L to screen, through a mirror")
    msub = p.add_subparsers(dest="mirror_cmd", required=True)
    s = msub.add_parser("capture", help="show the flipped board and capture mirror poses")
    s.add_argument("--novel-px", type=float, default=30.0)
    s.add_argument("--resume", action="store_true", help="add poses to the latest mirror capture")
    s.add_argument("--poses", type=int, default=TARGET_POSES,
                   help=f"finish at this many poses in total (default {TARGET_POSES})")
    s.set_defaults(func=cmd_capture)
    s = msub.add_parser("solve", help="re-solve a saved mirror capture")
    s.add_argument("raw_dir")
    s.add_argument("--drop", nargs="*")
    s.set_defaults(func=cmd_solve)
