"""Load the camera configuration (calibration C0) from config/cameras.toml."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAMERA_CONFIG = PROJECT_ROOT / "config" / "cameras.toml"


@dataclass(frozen=True)
class CameraConfig:
    name: str
    description: str
    device: str
    usb_id: str
    width: int
    height: int
    fourcc: str
    fps: int
    # Insertion order matters: auto modes must be switched off before their
    # manual value is written (e.g. auto_exposure before exposure_time_absolute).
    controls: dict[str, int] = field(default_factory=dict)
    exposure_mode: str = "manual"   # "manual" or "auto-lock" (see config/cameras.toml)
    lock_after_frames: int = 60

    @property
    def frame_period_ms(self) -> float:
        return 1000.0 / self.fps


def load_camera_configs(path: Path = DEFAULT_CAMERA_CONFIG) -> dict[str, CameraConfig]:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    configs = {}
    for name, c in raw["cameras"].items():
        configs[name] = CameraConfig(
            name=name,
            description=c.get("description", ""),
            device=c["device"],
            usb_id=c["usb_id"],
            width=int(c["width"]),
            height=int(c["height"]),
            fourcc=c["fourcc"],
            fps=int(c["fps"]),
            controls={k: int(v) for k, v in c.get("controls", {}).items()},
            exposure_mode=c.get("exposure_mode", "manual"),
            lock_after_frames=int(c.get("lock_after_frames", 60)),
        )
        if configs[name].exposure_mode not in ("manual", "auto-lock"):
            raise ValueError(f"camera {name}: unknown exposure_mode {configs[name].exposure_mode!r}")
    return configs
