"""Eye-Hand app: both cameras live, placement aids, calibration status, and a guided
recalibration of the camera positions (C5), all without the command line.

Start it from the desktop launcher (`eyehand install-launcher`) or with `eyehand app`.

- **Placement:** each camera's view shows the right eye and the right index fingertip as
  MediaPipe finds them, a margin box to keep them inside, and the board corners. When the
  printed board is visible to both cameras, the panel shows the angle between the cameras
  at the board (aim for 30–60°) and the distances.
- **Status:** every calibration's state (docs/RECALIBRATION.md rules, `eyehand.status`).
- **Recalibrate camera positions:** a guided C5 capture with sounds, automatic solve,
  gates in plain words, install, then a face baseline for the health monitor.
- **Health monitor (C7):** with a valid C5, the epipolar error of face landmarks seen by
  both cameras. A sustained jump means something moved.

One worker thread per camera does capture, board detection and (optionally) MediaPipe.
The runtime cursor pipeline will use one process per camera instead (docs/ARCHITECTURE.md);
for placement and calibration, threads are enough.
"""

from __future__ import annotations

import json
import queue
import threading
import webbrowser
import time
import tkinter as tk
from collections import deque
from dataclasses import asdict
from http.server import ThreadingHTTPServer
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from eyehand import boards, browser_cursor, intrinsics, sound, status, stereo
from eyehand.capture import Camera
from eyehand.config import PROJECT_ROOT, CameraConfig, load_camera_configs
from eyehand.intrinsics import Intrinsics, TargetScale, undistort_normalized
from eyehand.landmarks import Landmarks, Tracker
from eyehand.steady import SteadyMeter
from eyehand.timebase import FrameClock

PANEL_WIDTH = 640
MARGIN = 0.08                 # keep eye and fingertip inside this inset from the image edge
TARGET_VIEWS = 25
NEAR_FAR_MM = 450             # board distance from camera L separating "near" and "far"
BASELINE_SAMPLES = 60
ALARM_SECONDS = 2.0
STEADY_SECONDS = 20           # hands-off check before a camera-position capture
STEADY_LIMIT_DEG = 0.05       # ~0.5 mm at the cursor for the Feeltek
ROOM_INTERVAL_S = 5.0         # room check (scenery vs. snapshot after calibration)
ROOM_LIMIT_DEG = 0.1

COLOURS = {status.OK: "#2e7d32", status.CHECK: "#ef6c00", status.STALE: "#c62828",
           status.MISSING: "#c62828", status.FAILED: "#c62828", status.NOT_BUILT: "#9e9e9e"}

GATE_ADVICE = {
    "movement": "A camera moved during the capture (its estimated position drifted between the"
                " first and last views). Check the Feeltek's clip hinge and tripod head, keep"
                " the USB cable from pulling on it, then try again.",
    "reprojection": "Board corners were not sharp. Hold the board stiller, or add light.",
    "held-out": "The views disagree with each other. A camera may have moved during the"
                " capture, or some views were blurred.",
    "known-length": "3D lengths came out wrong. A lens calibration may be off, or a camera"
                    " moved during the capture.",
    "views used": "Not enough views that both cameras saw. Keep the board in both views.",
}


def nominal_intrinsics(config: CameraConfig) -> Intrinsics:
    """Rough camera model for placement before C1/C2 exist (f ~ 1000 px at 1280 wide)."""
    f = 1000.0 * config.width / 1280
    K = np.array([[f, 0, config.width / 2], [0, f, config.height / 2], [0, 0, 1]])
    return Intrinsics(config.name, (config.width, config.height), K, np.zeros(5))


def print_board_scale() -> TargetScale:
    """The printed board's measured scale, as recorded by C1 (falls back to nominal)."""
    path = status.CALIB / "intrinsics_L.json"
    if path.exists():
        r = json.loads(path.read_text())
        if r.get("board", {}).get("name") == boards.PRINT_BOARD.name:
            s = r["scale"]
            return TargetScale(s["x"], s["y"], s["source"])
    return TargetScale(1.0, 1.0, "nominal 24 mm squares")


class Snapshot:
    __slots__ = ("frame", "corners", "ids", "landmarks", "fps")

    def __init__(self, frame, corners, ids, landmarks, fps):
        self.frame, self.corners, self.ids, self.landmarks, self.fps = (
            frame, corners, ids, landmarks, fps)


class CameraWorker(threading.Thread):
    def __init__(self, config: CameraConfig, on_detection):
        super().__init__(name=f"worker-{config.name}", daemon=True)
        self.config = config
        self.on_detection = on_detection
        self.track = True
        self.want_profile = "placement"     # the UI asks; the worker thread switches
        self.profile = None                 # what the camera is actually using
        self.latest: Snapshot | None = None
        self.error: str | None = None
        self._halt = threading.Event()

    def run(self) -> None:
        detector = boards.PRINT_BOARD.detector()
        tracker = None
        clock = FrameClock(self.config.frame_period_ms)
        try:
            with Camera(self.config, profile="placement") as cam:
                self.profile = cam.profile
                while not self._halt.is_set():
                    if self.want_profile != cam.profile:
                        self.profile = "switching"
                        cam.set_profile(self.want_profile)
                        self.profile = cam.profile
                    f = cam.read()
                    clock.add(f.t_soe_ms)
                    corners, ids = intrinsics.detect(detector, f.image)
                    self.on_detection(self.config.name, f, corners, ids)
                    lm = None
                    if self.track:
                        if tracker is None:
                            tracker = Tracker()
                        lm = tracker.process(f.image, f.t_soe_ms)
                    fps = 1000 / clock.period_ms if clock.count > 10 else 0.0
                    self.latest = Snapshot(f, corners, ids, lm, fps)
        except Exception as e:     # shown in the UI; the app keeps running
            self.error = f"{type(e).__name__}: {e}"
        finally:
            if tracker is not None:
                tracker.close()

    def stop(self) -> None:
        self._halt.set()


class HealthMonitor:
    """C7: median epipolar error of face landmarks seen by both cameras."""

    def __init__(self):
        self.T_F_L = None
        self.intr: dict[str, Intrinsics] = {}
        self.baseline: float | None = None
        self.history: deque[tuple[float, float]] = deque(maxlen=300)
        self.prev_face: dict[str, np.ndarray] = {}
        self.alarm_logged = False

    def configure(self, intr_L, intr_F, T_L_F, baseline):
        self.intr = {"L": intr_L, "F": intr_F}
        self.T_F_L = np.linalg.inv(T_L_F)
        self.baseline = baseline
        self.history.clear()
        self.alarm_logged = False

    def measure(self, lm_L: Landmarks | None, lm_F: Landmarks | None) -> float | None:
        if self.T_F_L is None or lm_L is None or lm_F is None:
            return None
        if lm_L.face is None or lm_F.face is None or abs(lm_L.t_ms - lm_F.t_ms) > 25:
            return None
        still = True
        for cam, lm in (("L", lm_L), ("F", lm_F)):       # skip while the head moves
            prev = self.prev_face.get(cam)
            if prev is not None and np.median(np.linalg.norm(lm.face - prev, axis=1)) > 1.5:
                still = False
            self.prev_face[cam] = lm.face
        if not still:
            return None
        iL, iF = self.intr["L"], self.intr["F"]
        e = stereo.epipolar_px(self.T_F_L, undistort_normalized(lm_L.face, iL.K, iL.dist),
                               undistort_normalized(lm_F.face, iF.K, iF.dist),
                               iL.K[0, 0], iF.K[0, 0])
        value = float(np.median(e))
        self.history.append((time.monotonic(), value))
        return value

    def limit(self) -> float | None:
        return None if self.baseline is None else max(2 * self.baseline, self.baseline + 3.0)

    def alarming(self) -> bool:
        limit = self.limit()
        if limit is None or not self.history:
            return False
        now = time.monotonic()
        recent = [v for t, v in self.history if now - t < ALARM_SECONDS]
        span = now - min(t for t, _ in self.history if now - t < ALARM_SECONDS) if recent else 0
        return len(recent) >= 10 and span > 0.8 * ALARM_SECONDS and min(recent) > limit


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Eye-Hand")
        self.configs = load_camera_configs()
        self.mode = "place"     # place | steadycheck | capture | solving | result | baseline
        self.lock = threading.Lock()
        self.selector: stereo.StereoSelector | None = None
        self.session: stereo.StereoSession | None = None
        self.raw_dir = None
        self.last_frames: dict = {}
        self.captured: queue.Queue = queue.Queue()
        self.solve_result: dict | None = None
        self.baseline_samples: list[float] = []
        self.target_views = TARGET_VIEWS
        self.health = HealthMonitor()
        self.monitor_beeps = 0
        self.cursor_server = None
        self.cursor_samples = browser_cursor.Samples()
        self.cursor_geometry = None
        self.cursor_thread = None
        self.cursor_stop = threading.Event()
        self.cursor_error = None
        self.intr: dict[str, Intrinsics | None] = {"L": None, "F": None}
        self.steady: dict = {}
        self.room_meters: dict[str, SteadyMeter] | None = None
        self.room: dict = {"turn": {}, "over": 0, "busy": False, "next": 0.0}

        self._build_ui()
        self.workers = {n: CameraWorker(self.configs[n], self._on_detection) for n in ("L", "F")}
        for w in self.workers.values():
            w.start()
        self.refresh_status()
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.after(200, self.tick)

    # ----------------------------------------------------------------------------- UI layout

    def _build_ui(self) -> None:
        big = ("TkDefaultFont", 15, "bold")
        self.header = ttk.Label(self, text="", font=big, wraplength=1250, justify="left")
        self.header.grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(8, 4))
        self.progress = ttk.Progressbar(self, maximum=TARGET_VIEWS, length=600)

        videos = ttk.Frame(self)
        videos.grid(row=2, column=0, sticky="nw", padx=6)
        self.video, self.caption = {}, {}
        for col, name in enumerate(("L", "F")):
            box = ttk.LabelFrame(videos, text=f"Camera {name}: {self.configs[name].description}")
            box.grid(row=0, column=col, padx=4, sticky="n")
            self.video[name] = ttk.Label(box)
            self.video[name].pack()
            self.caption[name] = ttk.Label(box, text="starting…", font=("TkFixedFont", 10))
            self.caption[name].pack(anchor="w")

        side = ttk.Frame(self, width=430)
        side.grid(row=2, column=1, sticky="nsew", padx=8)
        cursor_link = ttk.Label(side, text="Open cursor: http://127.0.0.1:8765",
                                foreground="#1565c0", cursor="hand2", takefocus=True)
        cursor_link.pack(anchor="w", pady=(0, 10))
        self.cursor_link = cursor_link
        for event in ("<Button-1>", "<Return>", "<space>"):
            cursor_link.bind(event, lambda _: webbrowser.open("http://127.0.0.1:8765"))
        ttk.Label(side, text="Placement", font=("TkDefaultFont", 12, "bold")).pack(anchor="w")
        self.guide = ttk.Label(side, text="", wraplength=420, justify="left")
        self.guide.pack(anchor="w", pady=(0, 10))
        ttk.Label(side, text="Camera positions check (C7)",
                  font=("TkDefaultFont", 12, "bold")).pack(anchor="w")
        self.room_label = tk.Label(side, text="", wraplength=420, justify="left", anchor="w")
        self.room_label.pack(anchor="w", fill="x")
        self.health_label = tk.Label(side, text="", wraplength=420, justify="left", anchor="w")
        self.health_label.pack(anchor="w", fill="x", pady=(0, 10))
        ttk.Label(side, text="Calibrations", font=("TkDefaultFont", 12, "bold")).pack(anchor="w")
        self.status_frame = ttk.Frame(side)
        self.status_frame.pack(anchor="w", fill="x")
        self.next_label = ttk.Label(side, text="", wraplength=420, justify="left")
        self.next_label.pack(anchor="w", pady=(6, 10))

        buttons = ttk.Frame(side)
        buttons.pack(anchor="w", fill="x")
        self.btn_cursor = ttk.Button(buttons, text="Cursor with videos", command=self.start_cursor)
        self.btn_direct_cursor = ttk.Button(buttons, text="Cursor only", command=self.start_direct_cursor)
        self.btn_stop_cursor = ttk.Button(buttons, text="Stop cursor", command=self.stop_cursor)
        self.btn_recal = ttk.Button(buttons, text="Recalibrate camera positions",
                                    command=self.start_recalibration)
        self.btn_emergency = ttk.Button(buttons, text="Emergency camera position recalibration",
                                        command=self.start_emergency_recalibration)
        self.btn_moved = ttk.Button(buttons, text="I moved a camera", command=self.mark_moved)
        self.btn_undo = ttk.Button(buttons, text="Undo last view", command=self.undo_view)
        self.btn_finish = ttk.Button(buttons, text="Finish now", command=self.finish_capture)
        self.btn_cancel = ttk.Button(buttons, text="Cancel", command=self.cancel)
        self.btn_retry = ttk.Button(buttons, text="Try again", command=lambda:
                                   (self.start_capture if self.target_views == 3
                                    else self.start_recalibration)())
        self.btn_done = ttk.Button(buttons, text="Done", command=self.back_to_place)
        self.track_var = tk.BooleanVar(value=True)
        self.chk_track = ttk.Checkbutton(buttons, text="Show eye and finger tracking",
                                         variable=self.track_var, command=self._set_tracking)
        self._show_buttons()

    def _show_buttons(self) -> None:
        for b in (self.btn_recal, self.btn_moved, self.btn_undo, self.btn_finish,
                  self.btn_cancel, self.btn_retry, self.btn_done, self.chk_track,
                  self.btn_cursor, self.btn_direct_cursor, self.btn_stop_cursor, self.btn_emergency):
            b.pack_forget()
        layout = {
            "place": (self.btn_direct_cursor,
                      self.btn_stop_cursor if self.cursor_server else self.btn_cursor,
                      self.btn_emergency, self.btn_recal, self.btn_moved, self.chk_track),
            "cursor": (self.btn_stop_cursor,),
            "steadycheck": (self.btn_cancel,),
            "capture": (self.btn_undo, self.btn_finish, self.btn_cancel),
            "solving": (),
            "result": (self.btn_retry, self.btn_done),
            "baseline": (self.btn_cancel,),
        }[self.mode]
        for b in layout:
            b.pack(anchor="w", fill="x", pady=2)
        if self.mode == "capture":
            self.progress.grid(row=1, column=0, sticky="w", padx=10)
        else:
            self.progress.grid_forget()

    def _set_tracking(self) -> None:
        on = self.track_var.get() and self.mode not in ("steadycheck", "capture")
        for w in self.workers.values():
            w.track = on

    # ----------------------------------------------------------------------------- status

    def refresh_status(self, *, reload_calibration: bool = True) -> None:
        statuses = status.calibration_status(self.configs)
        for child in self.status_frame.winfo_children():
            child.destroy()
        for s in statuses:
            row = ttk.Frame(self.status_frame)
            row.pack(anchor="w", fill="x")
            tk.Label(row, text="●", fg=COLOURS[s.state]).pack(side="left")
            ttk.Label(row, text=f"{s.id} {s.name}: {s.state}" + (f" ({s.detail})" if s.detail else ""),
                      wraplength=390, justify="left").pack(side="left", anchor="w")
        nxt = status.next_action(statuses)
        self.next_label.config(text=f"Next: {nxt.id}, {nxt.action}" if nxt else "")

        if not reload_calibration:
            return

        # Load what the views need: real intrinsics if valid, the stereo pose if valid.
        for cid, name in (("C1", "L"), ("C2", "F")):
            ok = next(s for s in statuses if s.id == cid).usable
            self.intr[name] = intrinsics.load_intrinsics(self.configs[name]) if ok else None
        c5 = next(s for s in statuses if s.id == "C5")
        self.c5_usable = c5.usable and self.intr["L"] is not None and self.intr["F"] is not None
        if self.c5_usable:
            T_L_F = stereo.load_stereo(self.intr["L"], self.intr["F"])
            baseline = json.loads(stereo.STEREO_PATH.read_text()).get("face_epipolar_baseline_px")
            self.health.configure(self.intr["L"], self.intr["F"], T_L_F, baseline)
            self._load_room_meters()
        else:
            self.health.T_F_L = None
            self.room_meters = None
        can = self.intr["L"] is not None and self.intr["F"] is not None
        self.btn_recal.state(["!disabled"] if can else ["disabled"])
        self.btn_emergency.state(["!disabled"] if can else ["disabled"])
        self.can_recal = can

    def mark_moved(self) -> None:
        status.log_event("moved", "pressed in the app")
        self.refresh_status()

    # ----------------------------------------------------------------------------- capture

    def _on_detection(self, camera, frame, corners, ids) -> None:
        """Called from worker threads for every frame."""
        if self.mode != "capture" or any(w.profile != "measurement" for w in self.workers.values()):
            return
        with self.lock:
            if self.target_views == 3 and len(self.selector.accepted) >= 3:
                return
            self.last_frames[camera] = frame
            view = self.selector.offer(camera, corners, ids, frame.t_soe_ms)
            if view is not None:
                self.captured.put((view, dict(self.last_frames)))

    def start_recalibration(self) -> None:
        """Step 1: hands-off check that both cameras hold still, then the capture."""
        if not self.can_recal:
            return
        self.stop_cursor()
        self.target_views = TARGET_VIEWS
        for w in self.workers.values():
            w.want_profile = "measurement"
        self.mode = "steadycheck"
        self.steady = {"elapsed": 0.0, "turn": {}, "done": False}
        self._set_tracking()
        self._show_buttons()
        threading.Thread(target=self._steady_check, daemon=True).start()

    def start_emergency_recalibration(self) -> None:
        if self.mode != "place" or not self.can_recal:
            return
        self.stop_cursor()
        # Emergency results are provisional; skip the separate hands-off wait.
        self.target_views = 3
        self.start_capture()

    def _steady_check(self) -> None:
        """Runs in a thread: compares each camera's view with its view at the start."""
        st = self.steady
        while any(w.profile != "measurement" for w in self.workers.values()):
            if self.mode != "steadycheck":
                return
            time.sleep(0.2)
        time.sleep(1.0)                                    # let the locked exposure settle
        meters = {}
        for n, w in self.workers.items():
            f = (self.intr[n] or nominal_intrinsics(self.configs[n])).K[0, 0]
            meters[n] = SteadyMeter(w.latest.frame.image, f)
        t0 = time.monotonic()
        while self.mode == "steadycheck" and time.monotonic() - t0 < STEADY_SECONDS:
            time.sleep(1.0)
            for n, w in self.workers.items():
                m = meters[n].measure(w.latest.frame.image)
                prev = st["turn"].get(n, 0.0)
                st["turn"][n] = None if m is None or prev is None else max(prev, m.tilt_deg)
            st["elapsed"] = time.monotonic() - t0
        st["done"] = True

    def _steady_result(self) -> None:
        turn = self.steady["turn"]
        names = {"L": "laptop camera", "F": "Feeltek"}
        moved = {n: t for n, t in turn.items() if t is not None and t > STEADY_LIMIT_DEG}
        blind = [n for n, t in turn.items() if t is None]
        if not moved and not blind:
            self.start_capture()
            return
        self.mode = "result"
        sound.play("dialog-warning")
        parts = [f"The {names[n]} turned {t:.2f}° during {STEADY_SECONDS} s hands-off"
                 f" (limit {STEADY_LIMIT_DEG}°)." for n, t in moved.items()]
        parts += [f"The {names[n]} couldn't see enough room detail to check; add light." for n in blind]
        advice = ("Something is still settling or loose: the tripod head or the Feeltek's clip"
                  " hinge, a pulling cable, or the laptop lid. Fix it, wait a minute, then try"
                  " again." if moved else "")
        self.header.config(text="Not starting the capture. " + " ".join(parts) + " " + advice)
        for w in self.workers.values():
            w.want_profile = "placement"
        self._show_buttons()

    def start_capture(self) -> None:
        if not self.can_recal:
            return
        self.progress.config(maximum=self.target_views)
        spec = boards.PRINT_BOARD
        self.scale = print_board_scale()
        self.session = stereo.StereoSession(
            {"L": stereo.intrinsics_fingerprint(self.intr["L"]),
             "F": stereo.intrinsics_fingerprint(self.intr["F"])},
            asdict(spec), asdict(self.scale))
        self.raw_dir = PROJECT_ROOT / "calib" / "raw" / "stereo" / time.strftime("%Y%m%d-%H%M%S")
        with self.lock:
            self.selector = stereo.StereoSelector()
        self.cells = np.zeros((2, 3, 2), int)     # rows x cols (in L's image) x near/far
        for w in self.workers.values():
            w.want_profile = "measurement"
        self.mode = "capture"
        self._set_tracking()
        self._show_buttons()

    def _record_view(self, view: stereo.StereoView, frames: dict) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        for cam, f in frames.items():
            cv2.imwrite(str(self.raw_dir / f"{view.name}_{cam}.png"), f.image)
        self.session.views.append(view)
        self.session.save(self.raw_dir)
        self._add_cell(view, +1)
        sound.play("camera-shutter")

    def _view_cell(self, view):
        w, h = self.configs["L"].width, self.configs["L"].height
        c = view.corners_L.mean(0)
        obj = intrinsics.object_points(boards.PRINT_BOARD, view.ids_L, self.scale)
        T = stereo.board_pose(obj, view.corners_L, self.intr["L"])
        depth = 0 if T is None or T[2, 3] < NEAR_FAR_MM else 1
        return min(1, int(2 * c[1] / h)), min(2, int(3 * c[0] / w)), depth

    def _add_cell(self, view, sign):
        r, c, d = self._view_cell(view)
        self.cells[r, c, d] += sign

    def capture_complete(self) -> bool:
        if self.target_views == 3:
            return len(self.session.views) >= 3
        covered = (self.cells.sum(axis=2) > 0).sum()
        both_depths = (self.cells.sum(axis=(0, 1)) > 0).all()
        return len(self.session.views) >= self.target_views and covered >= 5 and both_depths

    def undo_view(self) -> None:
        if self.session and self.session.views:
            view = self.session.views.pop()
            with self.lock:
                self.selector.undo()
            self._add_cell(view, -1)
            for cam in ("L", "F"):
                (self.raw_dir / f"{view.name}_{cam}.png").unlink(missing_ok=True)
            self.session.save(self.raw_dir)

    def finish_capture(self) -> None:
        minimum = 3 if self.target_views == 3 else 6
        if len(self.session.views) < minimum:
            self.header.config(text=f"Need at least {minimum} views before solving.")
            return
        self.mode = "solving"
        for w in self.workers.values():
            w.want_profile = "placement"
        self._show_buttons()
        sound.play("complete")
        self.header.config(text="Solving the camera positions…")

        def work():
            try:
                obj_of = lambda ids: intrinsics.object_points(boards.PRINT_BOARD, ids, self.scale)
                solve = stereo.solve_emergency if self.target_views == 3 else stereo.solve
                report = solve(self.session.views[:3] if self.target_views == 3 else self.session.views,
                                      obj_of, self.intr["L"], self.intr["F"],
                                      boards.PRINT_BOARD.squares_x)
                result = stereo.build_result(self.session, report, self.raw_dir)
                (self.raw_dir / "result.json").write_text(json.dumps(result, indent=1))
                self.solve_result = result
            except Exception as e:
                self.solve_result = {"error": f"{type(e).__name__}: {e}"}

        threading.Thread(target=work, daemon=True).start()

    def show_result(self) -> None:
        r = self.solve_result
        self.solve_result = None
        if "error" in r:
            self.mode = "result"
            self.header.config(text=f"The solve failed: {r['error']}")
        elif r["passed"]:
            stereo.install(r)
            status.log_event("stereo_installed", r["raw_dir"])
            if r.get("emergency"):
                self._save_room_refs()
                self.back_to_place()
                self.header.config(text="Emergency camera positions applied (3 poses, provisional). "
                                   "Choose a cursor mode. Run full calibration when convenient.")
                return
            self.refresh_status()
            self.mode = "baseline"
            self.baseline_samples = []
            self._set_tracking()
            lt = r["length_test"]
            self.passed_text = (f"Camera positions calibrated: lengths correct to"
                                f" {lt['p95_mm']:.2f} mm, cameras {r['baseline_mm'] / 10:.0f} cm"
                                f" apart, {r['angle_between_cameras_deg']['median']:.0f}° between them.")
        else:
            self.mode = "result"
            sound.play("dialog-warning")
            advice = [GATE_ADVICE[k] for g in r["gates"] if not g["pass"]
                      for k in GATE_ADVICE if k in g["name"].lower()]
            self.header.config(text="Not accurate enough, nothing was changed. "
                               + " ".join(dict.fromkeys(advice)))
            print(r["gates"] if r.get("emergency") else stereo.format_report(r))
        self._show_buttons()

    def _save_room_refs(self) -> None:
        """Snapshot of each camera's view (positioning settings) right after a successful
        calibration: the room check compares against these from then on."""
        r = json.loads(stereo.STEREO_PATH.read_text())
        raw = PROJECT_ROOT / r["raw_dir"]
        refs = {}
        for n, w in self.workers.items():
            if w.latest is not None:
                path = raw / f"room_{n}.png"
                cv2.imwrite(str(path), w.latest.frame.image)
                refs[n] = str(path.relative_to(PROJECT_ROOT))
        r["room_refs"] = refs
        stereo.STEREO_PATH.write_text(json.dumps(r, indent=1))

    def _load_room_meters(self) -> None:
        refs = json.loads(stereo.STEREO_PATH.read_text()).get("room_refs", {})
        meters = {}
        for n, rel in refs.items():
            img = cv2.imread(str(PROJECT_ROOT / rel))
            if img is not None:
                meters[n] = SteadyMeter(img, self.intr[n].K[0, 0])
        self.room_meters = meters or None
        self.room = {"turn": {}, "over": 0, "busy": False, "next": 0.0}

    def finish_baseline(self) -> None:
        value = float(np.median(self.baseline_samples))
        r = json.loads(stereo.STEREO_PATH.read_text())
        r["face_epipolar_baseline_px"] = value
        stereo.STEREO_PATH.write_text(json.dumps(r, indent=1))
        self._save_room_refs()
        sound.play("complete")
        self.refresh_status()
        self.mode = "result"
        self.header.config(text=self.passed_text + f" Face check baseline: {value:.1f} px. All done.")
        self._show_buttons()

    def cancel(self) -> None:
        if self.mode == "baseline":
            self._save_room_refs()
            self.refresh_status()
        self.back_to_place()

    def back_to_place(self) -> None:
        for w in self.workers.values():
            w.want_profile = "measurement" if self.cursor_server else "placement"
        self.mode = "place"
        self._set_tracking()
        self._show_buttons()
        self.refresh_status()

    # ----------------------------------------------------------------------------- frame loop

    def tick(self) -> None:
        try:
            self._tick()
        finally:
            self.after(40, self.tick)

    def _tick(self) -> None:
        if self.mode == "cursor":
            if not self.cursor_thread.is_alive():
                error = self.cursor_error
                self.stop_cursor()
                if error:
                    self.header.config(text=f"Cursor stopped: {error}")
            return
        while not self.captured.empty():
            self._record_view(*self.captured.get())
        if self.mode == "capture" and self.capture_complete():
            self.finish_capture()
        if self.mode == "solving" and self.solve_result is not None:
            self.show_result()
        if self.mode == "steadycheck":
            st = self.steady
            if st.get("done"):
                self._steady_result()
            else:
                readings = ", ".join(
                    f"{'laptop camera' if n == 'L' else 'Feeltek'} "
                    + ("no reading" if t is None else f"{t:.3f}°")
                    for n, t in st.get("turn", {}).items())
                self.header.config(text=(
                    "Step 1 of 2, HANDS OFF: checking that both cameras hold still"
                    f" ({st.get('elapsed', 0):.0f}/{STEADY_SECONDS} s). Nothing to hold up. Sit"
                    " still or step away, and don't touch the laptop, tripod or cables."
                    + (f" Largest turn so far: {readings} (limit {STEADY_LIMIT_DEG}°)."
                       if readings else " Switching to measurement settings…")))

        snaps = {n: w.latest for n, w in self.workers.items()}
        for name, w in self.workers.items():
            if w.error:
                self.caption[name].config(text=f"ERROR: {w.error}")
                continue
            s = snaps[name]
            if s is None:
                continue
            self.video[name].image = ImageTk.PhotoImage(self._draw(name, s))
            self.video[name].config(image=self.video[name].image)
            n_corners = 0 if s.ids is None else len(s.ids)
            self.caption[name].config(
                text=f"[{w.profile}] {s.fps:5.2f} fps  brightness {s.frame.image.mean():3.0f}  "
                     f"board corners {n_corners:2d}")

        if self.mode == "capture":
            self._capture_text()
        elif self.mode == "place":
            self.header.config(text="POSITIONING MODE (automatic camera settings, for a clear"
                                    " picture only). Position the cameras: both should see your right eye and"
                                    " right index fingertip inside the box, with 30–60° between"
                                    " them. Hold the printed board where your hand will be to"
                                    " measure the angle.")
        self.guide.config(text=self._placement_text(snaps))
        self._health(snaps)
        self._publish_cursor(snaps)

    def _capture_text(self) -> None:
        n = len(self.session.views)
        self.progress["value"] = min(n, self.target_views)
        missing = []
        names_r, names_c = ("upper", "lower"), ("right", "centre", "left")   # as shown (mirrored)
        for r in range(2):
            for c in range(3):
                if self.cells[r, c].sum() == 0:
                    missing.append(f"{names_r[r]} {names_c[c]}")
        depth = self.cells.sum(axis=(0, 1))
        if depth[0] == 0:
            missing.append("near the screen")
        if depth[1] == 0:
            missing.append("farther from the screen")
        if any(w.profile != "measurement" for w in self.workers.values()):
            self.header.config(text="Switching both cameras to measurement settings (exposure"
                                    " settling)…")
            return
        with self.lock:
            st = dict(self.selector.status)
        if self.target_views == 3:
            self.header.config(text=f"Emergency recalibration {n}/3: keep cameras fixed. "
                               "Show the board to BOTH cameras, hold for the click, then tilt or "
                               f"move it for the next capture. L: {st['L']}; F: {st['F']}.")
            return
        self.header.config(text=(
            ("Emergency recalibration: keep both cameras fixed. "
             if self.target_views < TARGET_VIEWS else "")
            + f"Views {n}/{self.target_views}. Hold the printed board between you and the screen, where"
            " your face and hand will be, so both cameras see it. Hold it still until the click,"
            " then move or tilt it. "
            + (f"Still needed: {', '.join(missing)}. " if missing else "")
            + f"[L: {st['L']}, F: {st['F']}]"))

    def _draw(self, name: str, s: Snapshot) -> Image.Image:
        """Panel image, mirrored like a mirror so it follows your movements. Only the
        display is mirrored; detection, tracking and calibration use the real image."""
        img = s.frame.image
        h, w = img.shape[:2]
        k = PANEL_WIDTH / w
        out = cv2.flip(cv2.resize(img, None, fx=k, fy=k), 1)
        W = out.shape[1]

        def at(p):            # image pixel -> mirrored panel pixel
            return int(W - 1 - p[0] * k), int(p[1] * k)

        def text(t, p, colour, dx=8, dy=4):
            cv2.putText(out, t, (p[0] + dx, p[1] + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1,
                        cv2.LINE_AA)

        mx, my = int(MARGIN * w * k), int(MARGIN * h * k)
        cv2.rectangle(out, (mx, my), (W - mx, out.shape[0] - my), (200, 200, 200), 1)
        if s.ids is not None:
            for c in s.corners:
                cv2.circle(out, at(c), 2, (0, 220, 0), -1)
        lm = s.landmarks
        if lm is not None:
            if lm.face is not None:
                for idx, label, colour in ((468, "R eye", (0, 0, 255)), (473, "L eye", (160, 160, 160))):
                    p = at(lm.face[idx])
                    cv2.circle(out, p, 5, colour, 2)
                    text(label, p, colour, 7, -7)
            right = lm.right_hand()
            for label, score, pts in lm.hands:
                chosen = right is not None and pts is right
                p = at(pts[8])
                colour = (0, 255, 255) if chosen else (160, 160, 160)
                cv2.circle(out, p, 6 if chosen else 4, colour, 2)
                text(("R index" if chosen else "other hand") + f" [{label}]", p, colour)
        if self.mode == "capture" and name == "L":
            H = out.shape[0]
            cv2.line(out, (0, H // 2), (W, H // 2), (255, 255, 0), 1)
            for c in (1, 2):
                cv2.line(out, (int(c * W / 3), 0), (int(c * W / 3), H), (255, 255, 0), 1)
            for r in range(2):
                for c in range(3):                 # c = column in the real image
                    x0, y0 = int((2 - c) * W / 3), int(r * H / 2)
                    n_near, n_far = self.cells[r, c]
                    text(f"near {n_near} far {n_far}", (x0, y0), (255, 255, 0), 6, 18)
        return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

    def _placement_text(self, snaps) -> str:
        lines = []
        for what, get in (("Right eye", lambda lm: lm.right_eye()),
                          ("Right index fingertip", lambda lm: lm.fingertip())):
            parts = []
            for name in ("L", "F"):
                s = snaps[name]
                if s is None or s.landmarks is None:
                    parts.append(f"{name} –")
                    continue
                p = get(s.landmarks)
                if p is None:
                    parts.append(f"{name} ✘ not seen")
                else:
                    w, h = self.configs[name].width, self.configs[name].height
                    inside = (MARGIN * w < p[0] < (1 - MARGIN) * w and MARGIN * h < p[1] < (1 - MARGIN) * h)
                    parts.append(f"{name} ✔" if inside else f"{name} ⚠ near the edge")
            lines.append(f"{what}: " + ",  ".join(parts))
        if not self.track_var.get():
            lines = ["Eye and finger tracking is off."]

        sL, sF = snaps["L"], snaps["F"]
        if sL and sF and sL.ids is not None and sF.ids is not None \
                and len(sL.ids) >= 6 and len(sF.ids) >= 6:
            geo = self._board_geometry(sL, sF)
            if geo:
                angle, dL, dF, approx = geo
                verdict = ("good" if 30 <= angle <= 60 else
                           "too small: move F farther to the side" if angle < 30 else
                           "too large: move F closer to the laptop's line of sight")
                lines.append(f"Angle between cameras at the board: {angle:.0f}° ({verdict});"
                             f" board {dL / 10:.0f} cm from L, {dF / 10:.0f} cm from F"
                             + (" (approximate: lens calibrations not done yet)" if approx else ""))
        else:
            lines.append("Hold the printed board where your hand will be, visible to both"
                         " cameras, to measure the angle between them.")
        return "\n".join(lines)

    def _board_geometry(self, sL, sF):
        approx = self.intr["L"] is None or self.intr["F"] is None
        poses = []
        for name, s in (("L", sL), ("F", sF)):
            intr = self.intr[name] or nominal_intrinsics(self.configs[name])
            obj = intrinsics.object_points(boards.PRINT_BOARD, s.ids, print_board_scale())
            T = stereo.board_pose(obj, s.corners, intr)
            if T is None:
                return None
            centre = obj.mean(0)
            poses.append((T, centre))
        # Camera centres in the board frame, then the angle they subtend at the board centre.
        cams = [(-T[:3, :3].T @ T[:3, 3]) for T, _ in poses]
        centre = poses[0][1]
        a, b = cams[0] - centre, cams[1] - centre
        angle = float(np.degrees(np.arccos(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))))
        return angle, float(np.linalg.norm(a)), float(np.linalg.norm(b)), approx

    def _room_check(self, snaps) -> str:
        """C7 without a face: compare each camera's view with the snapshot taken right after
        the last successful calibration. Runs every few seconds in a thread."""
        if self.room_meters is None or self.mode != "place":
            return ""
        room = self.room
        if not room["busy"] and time.monotonic() > room["next"] \
                and all(snaps.get(n) for n in self.room_meters):
            room["busy"] = True
            frames = {n: snaps[n].frame.image for n in self.room_meters}

            def work():
                turns = {}
                for n, img in frames.items():
                    m = self.room_meters[n].measure(img)
                    turns[n] = None if m is None else m.tilt_deg
                room["turn"] = turns
                over = any(t is not None and t > ROOM_LIMIT_DEG for t in turns.values())
                room["over"] = room["over"] + 1 if over else 0
                room["next"] = time.monotonic() + ROOM_INTERVAL_S
                room["busy"] = False

            threading.Thread(target=work, daemon=True).start()
        if not room["turn"]:
            return "Room check: measuring…"
        names = {"L": "laptop camera", "F": "Feeltek"}
        text = "Room check (since calibration): " + ", ".join(
            f"{names[n]} " + ("no reading" if t is None else f"{t:.2f}°")
            for n, t in room["turn"].items())
        if room["over"] >= 2:
            if not room.get("logged"):
                worst = max((t for t in room["turn"].values() if t is not None), default=0)
                status.log_event("c7_alarm", f"room check: a camera turned {worst:.2f}°")
                room["logged"] = True
                self._monitor_warning()
            return text + f": MOVED (limit {ROOM_LIMIT_DEG}°). Recalibrate camera positions."
        return text + f": OK (limit {ROOM_LIMIT_DEG}°)"

    def _health(self, snaps) -> None:
        if self.mode == "baseline":
            lmL = snaps["L"].landmarks if snaps["L"] else None
            lmF = snaps["F"].landmarks if snaps["F"] else None
            v = self.health.measure(lmL, lmF)
            if v is not None:
                self.baseline_samples.append(v)
            self.header.config(text=self.passed_text + " Last step: sit as you will when"
                               " pointing, look at the screen and keep your head still for a few"
                               f" seconds ({len(self.baseline_samples)}/{BASELINE_SAMPLES}).")
            if len(self.baseline_samples) >= BASELINE_SAMPLES:
                self.finish_baseline()
            return
        if self.mode == "place":
            room = self._room_check(snaps)
            colour = "#c62828" if "MOVED" in room else "#2e7d32" if "OK" in room else "#616161"
            self.room_label.config(text=room or "Room check: needs a camera-position calibration.",
                                   fg=colour)
        if self.health.T_F_L is None:
            self.health_label.config(text="Needs a valid camera-position calibration (C5).",
                                     fg="#616161")
            return
        if not self.track_var.get():
            self.health_label.config(text="Turn on eye and finger tracking to check.", fg="#616161")
            return
        v = self.health.measure(snaps["L"].landmarks if snaps["L"] else None,
                                snaps["F"].landmarks if snaps["F"] else None)
        limit = self.health.limit()
        recent = [x for _, x in list(self.health.history)[-15:]]
        if not recent:
            self.health_label.config(text="Waiting for your face in both cameras (hold still).",
                                     fg="#616161")
            return
        value = float(np.median(recent))
        if self.health.alarming():
            self.health_label.config(
                text=f"Face check {value:.1f} px, limit {limit:.1f} px: a camera has probably"
                     " moved. Recalibrate camera positions.", fg="#c62828")
            if not self.health.alarm_logged:
                status.log_event("c7_alarm", f"face epipolar {value:.1f} px > {limit:.1f} px")
                self.health.alarm_logged = True
                self._monitor_warning()
        else:
            text = f"Face check {value:.1f} px" + (
                f" (baseline {self.health.baseline:.1f}, limit {limit:.1f}): OK" if limit else
                " (no baseline yet; recalibrating records one)")
            self.health_label.config(text=text, fg="#2e7d32")

    def start_cursor(self) -> None:
        if self.mode != "place":
            return
        if self.cursor_server is None:
            try:
                geometry = browser_cursor.load_geometry(self.configs)
                server = ThreadingHTTPServer(("127.0.0.1", 8765),
                                             browser_cursor._handler(self.cursor_samples))
            except (OSError, RuntimeError, ValueError) as exc:
                self.header.config(text=f"Cursor unavailable: {exc}")
                return
            self.cursor_geometry = geometry
            self.cursor_samples.publish({"valid": False, "reason": "Starting camera tracking…"})
            self.cursor_server = server
            threading.Thread(target=server.serve_forever, name="cursor-http", daemon=True).start()
            for worker in self.workers.values():
                worker.want_profile = "measurement"
            self.track_var.set(True)
            self._set_tracking()
            self._show_buttons()
        webbrowser.open("http://127.0.0.1:8765")

    def start_direct_cursor(self) -> None:
        if self.mode != "place":
            return
        self.stop_cursor()
        for worker in self.workers.values():
            worker.stop()
        for worker in self.workers.values():
            worker.join(timeout=3)
        if any(worker.is_alive() for worker in self.workers.values()):
            self.header.config(text="Waiting for preview cameras to stop. Try Cursor only again.")
            return
        self.mode = "cursor"
        self.cursor_stop.clear()
        self.cursor_error = None
        self.cursor_samples.publish({"valid": False, "reason": "Starting cursor-only tracking…"})
        self._show_buttons()
        self.header.config(text="Cursor only: previews and calibration checks paused. Stop cursor to return.")

        def run():
            try:
                browser_cursor.run(stop_event=self.cursor_stop, open_browser=True,
                                   samples=self.cursor_samples)
            except Exception as exc:
                self.cursor_error = str(exc)

        self.cursor_thread = threading.Thread(target=run, name="cursor-runtime", daemon=True)
        self.cursor_thread.start()

    def stop_cursor(self) -> None:
        if self.cursor_thread is not None:
            self.cursor_stop.set()
            self.cursor_thread.join(timeout=0.1)
            if self.cursor_thread.is_alive():
                self.header.config(text="Stopping cursor cameras…")
                return
            self.cursor_thread = None
            self.workers = {n: CameraWorker(self.configs[n], self._on_detection) for n in ("L", "F")}
            for worker in self.workers.values():
                worker.start()
            self.back_to_place()
            return
        if self.cursor_server is None:
            return
        self.cursor_server.shutdown()
        self.cursor_server.server_close()
        self.cursor_server = None
        self.cursor_geometry = None
        for worker in self.workers.values():
            worker.want_profile = "placement"
        self._show_buttons()

    def _publish_cursor(self, snaps) -> None:
        if self.cursor_server is None:
            return
        reason = None
        if self.mode != "place" or not self.track_var.get():
            reason = "Enable eye and finger tracking in positioning mode"
        elif not self.c5_usable:
            reason = "Camera-position calibration needs updating"
        elif any(w.profile != "measurement" for w in self.workers.values()):
            reason = "Waiting for camera settings to settle"
        elif any(w.error for w in self.workers.values()):
            reason = "Camera error; check the Eye-Hand window"
        a, b = (snaps[n].landmarks if snaps[n] else None for n in ("L", "F"))
        if reason is None:
            if a is None or b is None:
                reason = "Waiting for both cameras"
            elif abs(a.t_ms - b.t_ms) > browser_cursor.MAX_PAIR_GAP_MS:
                reason = "Waiting for aligned frames"
            elif time.monotonic() * 1000 - min(a.t_ms, b.t_ms) > 300:
                reason = "Waiting for fresh camera frames"
        if reason:
            self.cursor_samples.publish({"valid": False, "reason": reason})
            return
        settings = self.cursor_samples.settings()
        pointers = browser_cursor._pointer_samples(a, b, settings["eye"], *self.cursor_geometry)
        self.cursor_samples.publish({"pointers": pointers, **settings})

    def _monitor_warning(self) -> None:
        if self.monitor_beeps < 3:
            sound.play("dialog-warning")
            self.monitor_beeps += 1
        # Updating the warning must not reset the monitor's alarm latch.
        self.refresh_status(reload_calibration=False)

    def close(self) -> None:
        self.cursor_stop.set()
        if self.cursor_thread is not None:
            self.cursor_thread.join(timeout=8)
        if self.cursor_server is not None:
            self.cursor_server.shutdown()
            self.cursor_server.server_close()
        for w in self.workers.values():
            w.stop()
        for w in self.workers.values():
            w.join(timeout=3)
        self.destroy()


def main() -> int:
    App().mainloop()
    return 0
