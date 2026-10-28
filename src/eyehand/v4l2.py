"""Camera controls through v4l2-ctl: list devices, apply controls, verify read-back."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


class ControlError(RuntimeError):
    pass


def _run(args: list[str]) -> str:
    result = subprocess.run(["v4l2-ctl", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise ControlError(f"v4l2-ctl {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def set_controls(device: str, controls: dict[str, int]) -> None:
    """Apply controls one at a time, in order, so auto modes are off before manual values."""
    for name, value in controls.items():
        _run(["-d", device, "-c", f"{name}={value}"])


def get_controls(device: str, names: list[str]) -> dict[str, int]:
    out = _run(["-d", device, "-C", ",".join(names)])
    values = {}
    for line in out.splitlines():
        m = re.match(r"\s*(\w+):\s*(-?\d+)", line)
        if m:
            values[m.group(1)] = int(m.group(2))
    return values


def apply_and_verify(device: str, controls: dict[str, int]) -> dict[str, int]:
    """Apply controls, read them back, and raise if any differs from what was set."""
    set_controls(device, controls)
    actual = get_controls(device, list(controls))
    mismatched = {
        name: (want, actual.get(name))
        for name, want in controls.items()
        if actual.get(name) != want
    }
    if mismatched:
        detail = ", ".join(f"{n}: wanted {w}, got {a}" for n, (w, a) in mismatched.items())
        raise ControlError(f"{device}: control read-back mismatch ({detail})")
    return actual


def get_defaults(device: str) -> dict[str, int]:
    """The driver's default value of every control (`v4l2-ctl -l`, "default=")."""
    defaults = {}
    for line in list_controls(device).splitlines():
        m = re.match(r"\s*(\w+)\s+0x[0-9a-f]+\s+\(\w+\)\s*:.*\bdefault=(-?\d+)", line)
        if m:
            defaults[m.group(1)] = int(m.group(2))
    return defaults


def restore_defaults(device: str, names: list[str]) -> list[str]:
    """Put the named controls back to the driver defaults. Best effort: returns the names that
    could not be written instead of raising, because this runs on close and cleanup.

    `names` is in configuration order (auto modes off first, then manual values). Restoring
    goes in reverse, so manual values are written while their auto mode is still off, and
    the auto modes come back last.
    """
    defaults = get_defaults(device)
    failed = []
    for name in reversed(names):
        if name not in defaults:
            continue
        try:
            set_controls(device, {name: defaults[name]})
        except ControlError:
            failed.append(name)
    return failed


def list_controls(device: str) -> str:
    return _run(["-d", device, "-l"])


def list_formats(device: str) -> str:
    return _run(["-d", device, "--list-formats-ext"])


@dataclass(frozen=True)
class VideoDevice:
    by_id: str
    node: str
    usb_id: str | None


def _usb_id_for_node(node: str) -> str | None:
    # /sys/class/video4linux/videoN/device -> USB interface; idVendor/idProduct live one level up.
    iface = Path("/sys/class/video4linux") / Path(node).name / "device"
    try:
        usb_dev = iface.resolve().parent
        vendor = (usb_dev / "idVendor").read_text().strip()
        product = (usb_dev / "idProduct").read_text().strip()
        return f"{vendor}:{product}"
    except OSError:
        return None


def list_video_devices() -> list[VideoDevice]:
    """Capture nodes (index0) under /dev/v4l/by-id, which survive replugging."""
    by_id_dir = Path("/dev/v4l/by-id")
    if not by_id_dir.exists():
        return []
    devices = []
    for link in sorted(by_id_dir.iterdir()):
        if not link.name.endswith("video-index0"):
            continue
        node = os.path.realpath(link)
        devices.append(VideoDevice(by_id=str(link), node=node, usb_id=_usb_id_for_node(node)))
    return devices
