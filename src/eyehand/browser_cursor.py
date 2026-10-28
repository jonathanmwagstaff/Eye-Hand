"""Development cursor: live eye/hand geometry rendered only inside a localhost page."""

from __future__ import annotations

import json
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from eyehand import cursor, intrinsics, mirror, status, stereo
from eyehand.capture import Camera
from eyehand.config import load_camera_configs
from eyehand.landmarks import Landmarks, Tracker
from eyehand.screen_calib import screen_pitch_mm
from eyehand.timebase import FrameClock

MAX_PAIR_GAP_MS = 20.0

PAGE = """<!doctype html>
<meta charset="utf-8">
<title>Eye-Hand browser cursor</title>
<style>
  html, body { height: 100%; margin: 0; overflow: hidden; font: 16px system-ui; color: #dce7f3;
               background: #111923; }
  body { background-image: linear-gradient(#ffffff12 1px, transparent 1px),
                           linear-gradient(90deg, #ffffff12 1px, transparent 1px);
         background-size: 10vw 10vh; }
  #message { position: fixed; left: 24px; top: 20px; padding: 12px 16px; border-radius: 8px;
             background: #07101bd9; padding-right: 52px; }
  #dismiss { position: absolute; right: 4px; top: 4px; width: 40px; height: 40px;
             border: 0; background: transparent; color: inherit; font-size: 24px; cursor: pointer; }
  #dismiss:focus-visible { outline: 2px solid white; }
  .dot { position: fixed; left: 0; top: 0; width: 28px; height: 28px; margin: -14px;
         border: 3px solid white; border-radius: 50%; background: #ff3b30b8;
         box-shadow: 0 0 0 5px #ff3b3038; pointer-events: none; display: none; }
  .dot.secondary { background: #24cce3b8; box-shadow: 0 0 0 5px #24cce338; }
  .dot span { position: absolute; left: 38px; top: 0; white-space: nowrap; font-size: 13px;
              padding: 3px 6px; border-radius: 4px; background: #07101be6; }
  #controls { position: fixed; right: 24px; top: 20px; z-index: 2; max-width: 300px;
              background: #07101bf2; border-radius: 8px; padding: 12px 16px; }
  summary { cursor: pointer; padding: 4px; }
  fieldset { border: 0; margin: 12px 0 0; padding: 0; }
  label { display: flex; justify-content: space-between; align-items: center; gap: 18px; margin: 10px 0; }
  select { font: inherit; min-width: 72px; padding: 6px; }
  #settings-error { color: #ffb4ab; }
  small { color: #9bb0c3; }
</style>
<div id="message"><span id="tracking-status">Waiting for both cameras…</span><br><small>Movement only · C6/C8 not calibrated</small>
<button id="dismiss" type="button" aria-label="Close status message" onclick="document.querySelector('#message').hidden = true">×</button></div>
<details id="controls">
<summary>Cursor controls</summary>
<fieldset id="settings" disabled>
<label>Aiming eye <select id="eye"><option value="L">L</option><option value="R" selected>R</option></select></label>
<label>Dominant clicker hand <select id="primary-hand"><option value="L">L</option><option value="R" selected>R</option></select></label>
</fieldset>
<p><small>Red: primary · Blue: secondary<br>Both hands aim using the selected eye.</small></p>
<p><small id="secondary-role">L hand: secondary cursor; future right-click.</small></p>
<small>Movement only. Clicking is not implemented.</small>
<p id="settings-error" role="status"></p>
</details>
<div id="dot-L" class="dot"><span></span></div>
<div id="dot-R" class="dot"><span></span></div>
<script>
const message = document.querySelector('#tracking-status');
const eye = document.querySelector('#eye'), primaryHand = document.querySelector('#primary-hand');
const settings = document.querySelector('#settings'), settingsError = document.querySelector('#settings-error');
function applySettings(value) {
  eye.value = value.eye;
  primaryHand.value = value.primary_hand;
  const secondary = value.primary_hand === 'R' ? 'L' : 'R';
  document.querySelector('#secondary-role').textContent = `${secondary} hand: secondary cursor; future right-click.`;
}
async function loadSettings() {
  const response = await fetch('/settings', {cache: 'no-store'});
  if (!response.ok) throw new Error('Could not load cursor settings. Reload to retry.');
  applySettings(await response.json());
}
loadSettings().then(() => { settings.disabled = false; }).catch(error => { settingsError.textContent = error.message; });
settings.addEventListener('change', async () => {
  settings.disabled = true;
  settingsError.textContent = '';
  try {
    const response = await fetch('/settings', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({eye: eye.value, primary_hand: primaryHand.value})});
    if (!response.ok) throw new Error('Could not save cursor settings.');
    await loadSettings();
    settings.disabled = false;
  } catch (error) {
    settingsError.textContent = `${error.message} Reload to retry.`;
  }
});
let sample, received = 0;
new EventSource('/events').onmessage = event => { sample = JSON.parse(event.data); received = performance.now(); };
function draw() {
  const fresh = performance.now() - received < 300;
  const statuses = [];
  for (const hand of ['L', 'R']) {
    const dot = document.querySelector(`#dot-${hand}`), pointer = sample?.pointers?.[hand];
    const primary = hand === sample?.primary_hand;
    dot.classList.toggle('secondary', !primary);
    dot.querySelector('span').textContent = `${hand} · ${primary ? 'Primary' : 'Secondary'}`;
    if (fresh && pointer?.valid) {
      dot.style.display = 'block';
      dot.style.transform = `translate(${pointer.x * innerWidth}px, ${pointer.y * innerHeight}px)`;
      statuses.push(`${hand}: tracking`);
    } else {
      dot.style.display = 'none';
      statuses.push(`${hand}: ${fresh ? (pointer?.reason || sample?.reason || 'waiting') : 'waiting for cameras'}`);
    }
  }
  message.textContent = statuses.join(' · ');
  requestAnimationFrame(draw);
}
requestAnimationFrame(draw);
</script>
""".encode()


class Samples:
    def __init__(self):
        self._condition = threading.Condition()
        self._seq = 0
        self._sample = {"valid": False, "reason": "Starting cameras…"}
        self._settings = {"eye": "R", "primary_hand": "R"}

    def settings(self) -> dict:
        with self._condition:
            return self._settings.copy()

    def set_settings(self, settings: dict) -> None:
        if not isinstance(settings, dict) or set(settings) != {"eye", "primary_hand"} or any(
                value not in ("L", "R") for value in settings.values()):
            raise ValueError("eye and primary_hand must each be L or R")
        with self._condition:
            self._settings = settings.copy()

    def publish(self, sample: dict) -> None:
        with self._condition:
            self._seq += 1
            self._sample = {"seq": self._seq, **sample}
            self._condition.notify_all()

    def after(self, seq: int, timeout: float = 15.0) -> tuple[int, dict] | None:
        with self._condition:
            self._condition.wait_for(lambda: self._seq > seq, timeout)
            return (self._seq, self._sample) if self._seq > seq else None


def _handler(samples: Samples):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/settings":
                self.send_error(404)
                return
            origin = self.headers.get("Origin")
            if origin and origin not in (f"http://127.0.0.1:{self.server.server_port}",
                                         f"http://localhost:{self.server.server_port}"):
                self.send_error(403)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 128:
                    raise ValueError("invalid settings length")
                samples.set_settings(json.loads(self.rfile.read(length)))
            except (ValueError, UnicodeError):
                self.send_error(400)
                return
            self.send_response(204)
            self.end_headers()

        def do_GET(self):
            if self.path == "/settings":
                body = json.dumps(samples.settings()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(PAGE)))
                self.end_headers()
                self.wfile.write(PAGE)
                return
            if self.path != "/events":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seq = -1
            try:
                while True:
                    item = samples.after(seq)
                    if item is None:
                        self.wfile.write(b": keepalive\n\n")
                    else:
                        seq, sample = item
                        self.wfile.write(b"data: " + json.dumps(sample).encode() + b"\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    return Handler


class CameraTracker(threading.Thread):
    def __init__(self, config):
        super().__init__(name=f"cursor-{config.name}", daemon=True)
        self.config = config
        self.history: deque[Landmarks] = deque(maxlen=4)
        self.lock = threading.Lock()
        self.stop_requested = threading.Event()
        self.error: Exception | None = None

    def run(self) -> None:
        tracker = None
        try:
            clock = FrameClock(self.config.frame_period_ms)
            tracker = Tracker()
            with Camera(self.config) as camera:
                while not self.stop_requested.is_set():
                    frame = camera.read()
                    t_ms = clock.add(frame.t_soe_ms).t_fit_ms
                    landmarks = tracker.process(frame.image, t_ms)
                    with self.lock:
                        self.history.append(landmarks)
        except Exception as exc:
            self.error = exc
        finally:
            if tracker is not None:
                tracker.close()

    def snapshot(self) -> list[Landmarks]:
        with self.lock:
            return list(self.history)


def _landmark_pixels(a: Landmarks, b: Landmarks, eye: str = "R",
                     hand: str = "R") -> tuple[np.ndarray, np.ndarray]:
    if eye not in ("L", "R"):
        raise ValueError("eye must be L or R")
    eye_a, eye_b = (a.left_eye(), b.left_eye()) if eye == "L" else (a.right_eye(), b.right_eye())
    tip_a, tip_b = a.pinch_points(hand), b.pinch_points(hand)
    if any(value is None for value in (eye_a, tip_a, eye_b, tip_b)):
        raise cursor.InvalidPointer(f"Show {eye} eye and {hand} hand's thumb and index tips to both cameras")
    return np.vstack((eye_a, tip_a)), np.vstack((eye_b, tip_b))


def _pointer_samples(a, b, eye, intr, T_F_L, T_S_L, screen_mm) -> dict:
    pointers = {}
    for hand in ("L", "R"):
        try:
            px_L, px_F = _landmark_pixels(a, b, eye, hand)
            point = cursor.solve_pointer(px_L, px_F, intr["L"], intr["F"], T_F_L, T_S_L, screen_mm)
            pointers[hand] = {"valid": True, "x": point.x_norm, "y": point.y_norm,
                              "reprojection_px": point.reprojection_px,
                              "ray_angle_deg": point.ray_angle_deg}
        except cursor.InvalidPointer as exc:
            pointers[hand] = {"valid": False, "reason": str(exc)}
    return pointers


def _best_pair(history_L: list[Landmarks], history_F: list[Landmarks]):
    pairs = [(a, b) for a in history_L for b in history_F]
    aligned = [pair for pair in pairs if abs(pair[0].t_ms - pair[1].t_ms) <= MAX_PAIR_GAP_MS]
    return max(aligned, key=lambda pair: pair[0].t_ms + pair[1].t_ms) if aligned else min(
        pairs, key=lambda pair: abs(pair[0].t_ms - pair[1].t_ms))


def load_geometry(configs):
    statuses = status.calibration_status(configs)
    missing = [s.id for s in statuses if s.id in ("C1", "C2", "C4", "C5") and not s.usable]
    if missing:
        raise RuntimeError(f"calibration not usable: {', '.join(missing)}; run `eyehand status`")
    intr = {name: intrinsics.load_intrinsics(configs[name]) for name in ("L", "F")}
    T_F_L = np.linalg.inv(stereo.load_stereo(intr["L"], intr["F"]))
    T_S_L = mirror.load_T_S_L(intr["L"])
    pitch_x, pitch_y, _ = screen_pitch_mm()
    screen_mm = (1920 * pitch_x, 1080 * pitch_y)
    return intr, T_F_L, T_S_L, screen_mm


def run(port: int = 8765, *, stop_event: threading.Event | None = None,
        open_browser: bool = False, samples: Samples | None = None) -> int:
    stop_event = stop_event or threading.Event()
    configs = load_camera_configs()
    intr, T_F_L, T_S_L, screen_mm = load_geometry(configs)

    samples = samples if samples is not None else Samples()
    server = ThreadingHTTPServer(("127.0.0.1", port), _handler(samples))
    server_thread = threading.Thread(target=server.serve_forever, name="cursor-http", daemon=True)
    workers = {name: CameraTracker(configs[name]) for name in ("L", "F")}
    print(f"Open http://127.0.0.1:{port} in your browser. Ctrl-C stops the cameras.")
    print("Development mode: movement only; C6 timing and C8 aiming correction are not complete.")
    server_thread.start()
    last_pair = None
    started = []
    try:
        for worker in workers.values():
            worker.start()
            started.append(worker)
        if open_browser:
            webbrowser.open(f"http://127.0.0.1:{port}")
        while not stop_event.is_set():
            for name, worker in workers.items():
                if worker.error is not None:
                    raise RuntimeError(f"camera {name}: {worker.error}") from worker.error
            histories = {name: worker.snapshot() for name, worker in workers.items()}
            if not all(histories.values()):
                time.sleep(0.01)
                continue
            a, b = _best_pair(histories["L"], histories["F"])
            settings = samples.settings()
            pair = (a.t_ms, b.t_ms, settings["eye"], settings["primary_hand"])
            if pair == last_pair:
                time.sleep(0.005)
                continue
            last_pair = pair
            gap = abs(a.t_ms - b.t_ms)
            if gap > MAX_PAIR_GAP_MS:
                samples.publish({"valid": False, "reason": f"Waiting for aligned frames ({gap:.0f} ms gap)"})
                continue
            pointers = _pointer_samples(a, b, settings["eye"], intr, T_F_L, T_S_L, screen_mm)
            samples.publish({"pointers": pointers, **settings})
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0
    finally:
        for worker in started:
            worker.stop_requested.set()
        for worker in started:
            worker.join()
        server.shutdown()
        server.server_close()
    return 0
