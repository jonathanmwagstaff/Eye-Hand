"""Camera capture with kernel start-of-exposure timestamps.

OpenCV's V4L2 backend reports the driver's buffer timestamp through CAP_PROP_POS_MSEC.
For uvcvideo that is CLOCK_MONOTONIC at the start of exposure (flags ts-monotonic,
ts-src-soe; checked with `v4l2-ctl --stream-mmap --verbose`), so it is directly
comparable with time.clock_gettime(CLOCK_MONOTONIC) and between cameras.

A reader thread dequeues every frame as soon as it arrives, so the driver's queue never
backs up (no stale frames) and the driver always has a free buffer (no drops). With only
one V4L2 buffer the driver drops every frame that ends while Python still holds the
buffer, which in practice halved the frame rate.

Camera settings come in two profiles:
- "measurement": the controls in config/cameras.toml (fixed exposure, white balance, focus).
  Used for every calibration and for tracking.
- "placement": the driver's own defaults (auto exposure, auto white balance, autofocus). A
  clear picture for a person positioning the cameras. Never used for measurements.

The camera's settings are shared with every other program, so `open()` starts from the
driver defaults (whatever an earlier run or crash left behind), and `close()` puts the
defaults back. Camera L's measurement settings, left behind, made it nearly black for other
programs: its manual mode lacks the hidden gain of auto mode.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from eyehand import v4l2
from eyehand.config import CameraConfig

# Frames discarded after controls are applied, so the sensor settles on the new exposure.
SETTLE_FRAMES = 10
V4L2_BUFFERS = 4
# Auto-lock: auto exposure must have stopped changing before it is frozen. Brightness is
# "settled" when the mean of the last 15 frames spans less than this many grey levels.
SETTLE_SPAN = 1.5
SETTLE_WINDOW = 15
SETTLE_TIMEOUT_S = 8.0
PROFILES = ("measurement", "placement")


def monotonic_ms() -> float:
    return time.clock_gettime(time.CLOCK_MONOTONIC) * 1000.0


@dataclass
class Frame:
    camera: str
    image: np.ndarray
    t_soe_ms: float      # kernel timestamp: start of exposure, CLOCK_MONOTONIC
    t_arrival_ms: float  # when the reader thread received the frame, CLOCK_MONOTONIC
    index: int           # frames received since open (driver drops show as timestamp gaps)
    skipped: int         # frames received but never returned by read() (consumer too slow)


class CameraError(RuntimeError):
    pass


class Camera:
    """One configured camera. Use as a context manager.

    Opening applies the configured mode, starts streaming, then applies and verifies
    the controls (the Feeltek only accepts focus changes while streaming).
    `read()` returns the newest frame not yet returned, waiting for one if needed.
    """

    def __init__(self, config: CameraConfig, profile: str = "measurement"):
        if profile not in PROFILES:
            raise ValueError(f"unknown profile {profile!r}")
        self.config = config
        self.profile = profile
        self.cap: cv2.VideoCapture | None = None
        self.controls: dict[str, int] = {}
        self.lock_brightness: float | None = None
        self._cond = threading.Condition()
        self._latest: Frame | None = None
        self._returned_index = -1
        self._error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def touched_controls(self) -> list[str]:
        """Every control this class may change, in configuration order."""
        names = list(self.config.controls)
        if self.config.exposure_mode == "auto-lock" and "auto_exposure" not in names:
            names.insert(0, "auto_exposure")
        return names

    def open(self) -> "Camera":
        c = self.config
        cap = cv2.VideoCapture(c.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise CameraError(f"{c.name}: cannot open {c.device}")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*c.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, c.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c.height)
        cap.set(cv2.CAP_PROP_FPS, c.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, V4L2_BUFFERS)
        self.cap = cap

        # The first read starts the stream; controls such as focus only stick after that.
        ok, image = cap.read()
        if not ok:
            self.close()
            raise CameraError(f"{c.name}: no frame from {c.device}")
        h, w = image.shape[:2]
        if (w, h) != (c.width, c.height):
            self.close()
            raise CameraError(f"{c.name}: asked for {c.width}x{c.height}, camera gives {w}x{h}")

        self._stop.clear()
        self._thread = threading.Thread(target=self._reader, name=f"camera-{c.name}", daemon=True)
        self._thread.start()
        try:
            # Start from the driver defaults, whatever an earlier run left behind.
            v4l2.restore_defaults(c.device, self.touched_controls)
            self.set_profile(self.profile)
        except (v4l2.ControlError, CameraError):
            self.close()
            raise
        return self

    def set_profile(self, profile: str) -> None:
        """Switch settings while streaming. Blocks until the image has settled."""
        c = self.config
        if profile == "placement":
            v4l2.restore_defaults(c.device, self.touched_controls)
            self.controls = {}
            self.lock_brightness = None
        elif profile == "measurement":
            if c.exposure_mode == "auto-lock":
                # Let auto exposure settle (frame rate held fixed by the controls), then
                # switch to manual WITHOUT writing a value: the camera keeps what auto chose.
                self.controls = v4l2.apply_and_verify(c.device, {**c.controls, "auto_exposure": 3})
                self.lock_brightness = self._wait_until_settled()
                self.controls |= v4l2.apply_and_verify(c.device, {"auto_exposure": 1})
            else:
                self.controls = v4l2.apply_and_verify(c.device, c.controls)
        else:
            raise ValueError(f"unknown profile {profile!r}")
        self.profile = profile
        for _ in range(SETTLE_FRAMES):
            self.read()

    def _wait_until_settled(self) -> float:
        """Read frames until mean brightness stops changing; returns the settled level."""
        means: deque[float] = deque(maxlen=SETTLE_WINDOW)
        t_end = time.monotonic() + SETTLE_TIMEOUT_S
        n = 0
        while time.monotonic() < t_end:
            means.append(float(self.read().image[::4, ::4].mean()))
            n += 1
            if (n >= self.config.lock_after_frames and len(means) == SETTLE_WINDOW
                    and max(means) - min(means) < SETTLE_SPAN):
                break
        return means[-1]

    def _reader(self) -> None:
        cap = self.cap
        index = 0
        while not self._stop.is_set():
            ok, image = cap.read()
            t_arrival = monotonic_ms()
            if not ok:
                with self._cond:
                    self._error = f"{self.config.name}: read failed"
                    self._cond.notify_all()
                return
            t_soe = cap.get(cv2.CAP_PROP_POS_MSEC)
            with self._cond:
                skipped = index - self._returned_index - 1
                self._latest = Frame(self.config.name, image, t_soe, t_arrival, index, skipped)
                self._cond.notify_all()
            index += 1

    def read(self, timeout_s: float = 2.0) -> Frame:
        with self._cond:
            ok = self._cond.wait_for(
                lambda: self._error is not None
                or (self._latest is not None and self._latest.index > self._returned_index),
                timeout=timeout_s,
            )
            if self._error is not None:
                raise CameraError(self._error)
            if not ok:
                raise CameraError(f"{self.config.name}: no frame for {timeout_s} s")
            frame = self._latest
            self._returned_index = frame.index
            return frame

    def close(self) -> None:
        """Stop streaming and give the camera back with its driver-default settings."""
        if self.cap is not None:
            try:
                v4l2.restore_defaults(self.config.device, self.touched_controls)
            except v4l2.ControlError:
                pass
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> "Camera":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()
