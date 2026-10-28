"""MediaPipe hand and face landmarks, in pixels, stamped with the frame's exposure time.

Models are pinned to version 1 and checked by SHA-256, so a silent model update cannot
change results. `ensure_models()` downloads them into models/ if missing.

Which landmark is which (verified in the placement view, which labels them on the image):
- Right iris centre: face landmark 468. Left iris centre: 473.
- Index fingertip: hand landmark 8.
- MediaPipe returns coordinates normalised to the image edges (0 = left edge, 1 = right
  edge). OpenCV puts pixel centres at integers, so pixel = x * width - 0.5.
- Handedness: on our unmirrored images MediaPipe labels the person's right hand "Right".
  Checked by the user in the placement view on 2026-10-27, in both cameras (the opposite
  assumption, from MediaPipe's selfie-mirroring note, was wrong). The right iris (468) was
  confirmed the same way.
"""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from eyehand.config import PROJECT_ROOT

MODELS_DIR = PROJECT_ROOT / "models"
MODELS = {
    "hand_landmarker": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/"
        "float16/1/hand_landmarker.task",
        "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
    ),
    "face_landmarker": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/"
        "float16/1/face_landmarker.task",
        "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    ),
}

RIGHT_IRIS = 468
LEFT_IRIS = 473
INDEX_TIP = 8
THUMB_TIP = 4
RIGHT_HAND_LABEL = "Right"   # see module docstring


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_models() -> dict[str, Path]:
    paths = {}
    MODELS_DIR.mkdir(exist_ok=True)
    for name, (url, sha) in MODELS.items():
        path = MODELS_DIR / f"{name}.task"
        if not path.exists():
            urllib.request.urlretrieve(url, path)
        if _sha256(path) != sha:
            raise RuntimeError(f"{path} does not match its pinned SHA-256; delete it to re-download")
        paths[name] = path
    return paths


@dataclass
class Landmarks:
    t_ms: float                         # frame start-of-exposure timestamp
    face: np.ndarray | None             # (478, 2) px, or None
    hands: list[tuple[str, float, np.ndarray]]   # (label, score, (21, 2) px)

    def right_eye(self) -> np.ndarray | None:
        return None if self.face is None else self.face[RIGHT_IRIS]

    def left_eye(self) -> np.ndarray | None:
        return None if self.face is None else self.face[LEFT_IRIS]

    def right_hand(self) -> np.ndarray | None:
        return self.hand("R")

    def hand(self, side: str) -> np.ndarray | None:
        if side not in ("L", "R"):
            raise ValueError("hand must be L or R")
        best = [h for h in self.hands if h[0] == ("Left" if side == "L" else RIGHT_HAND_LABEL)]
        return max(best, key=lambda h: h[1])[2] if best else None

    def fingertip(self) -> np.ndarray | None:
        hand = self.right_hand()
        return None if hand is None else hand[INDEX_TIP]

    def pinch_points(self, side: str = "R") -> np.ndarray | None:
        """Index and thumb tips from the same tracked hand."""
        hand = self.hand(side)
        return None if hand is None else hand[[INDEX_TIP, THUMB_TIP]]


class Tracker:
    """Hand + face landmarkers for one camera, in VIDEO mode with real timestamps."""

    def __init__(self, hands: bool = True, face: bool = True):
        from mediapipe.tasks.python import BaseOptions, vision

        paths = ensure_models()
        mode = vision.RunningMode.VIDEO
        self._hand = self._face = None
        if hands:
            self._hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(paths["hand_landmarker"])),
                running_mode=mode, num_hands=2))
        if face:
            self._face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(paths["face_landmarker"])),
                running_mode=mode, num_faces=1))
        self._last_ts = -1

    def process(self, image_bgr: np.ndarray, t_ms: float) -> Landmarks:
        import mediapipe as mp

        # MediaPipe needs strictly increasing integer timestamps.
        ts = max(int(round(t_ms)), self._last_ts + 1)
        self._last_ts = ts
        h, w = image_bgr.shape[:2]
        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        face, hands = None, []
        if self._face is not None:
            r = self._face.detect_for_video(img, ts)
            if r.face_landmarks:
                face = np.array([(p.x * w - 0.5, p.y * h - 0.5) for p in r.face_landmarks[0]])
        if self._hand is not None:
            r = self._hand.detect_for_video(img, ts)
            for lms, hd in zip(r.hand_landmarks, r.handedness):
                hands.append((hd[0].category_name, hd[0].score,
                              np.array([(p.x * w - 0.5, p.y * h - 0.5) for p in lms])))
        return Landmarks(t_ms, face, hands)

    def close(self) -> None:
        for lm in (self._hand, self._face):
            if lm is not None:
                lm.close()
        self._hand = self._face = None
