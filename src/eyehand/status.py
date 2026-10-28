"""Which calibrations are valid, which are stale and why, and what to do next.

The rules come from docs/RECALIBRATION.md. Every result file records what it depended on
(camera mode, focus, the intrinsics it was solved with), and `calib/events.jsonl` logs
events that invalidate results without changing any file, such as "a camera was moved"
(pressed in the app) or a health-monitor (C7) alarm. Status is always recomputed from those
files, never stored, so it cannot drift out of date.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from eyehand import intrinsics, stereo
from eyehand.config import PROJECT_ROOT, CameraConfig

CALIB = PROJECT_ROOT / "calib"
EVENTS = CALIB / "events.jsonl"

OK, MISSING, STALE, FAILED, CHECK, NOT_BUILT = "ok", "missing", "stale", "failed", "check", "not built"


@dataclass
class Status:
    id: str
    name: str
    state: str
    detail: str
    action: str = ""

    @property
    def usable(self) -> bool:
        return self.state in (OK, CHECK)


def log_event(event: str, detail: str = "", path: Path = EVENTS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event,
                            "detail": detail}) + "\n")


def read_events(path: Path = EVENTS) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _intrinsics_status(cid: str, config: CameraConfig, calib: Path) -> tuple[Status, object]:
    name = f"Lens calibration, camera {config.name}"
    path = calib / f"intrinsics_{config.name}.json"
    action = f"eyehand intrinsics capture {config.name} --square-mm <measured>"
    if not path.exists():
        return Status(cid, name, MISSING, "not done yet", action), None
    result = json.loads(path.read_text())
    try:
        intrinsics.check_matches(result, config)
    except intrinsics.StaleCalibration as e:
        return Status(cid, name, STALE, str(e), action), None
    if not result["passed"]:
        return Status(cid, name, FAILED, "result did not pass its gates", action), None
    intr = intrinsics.load_intrinsics(config, path)
    return Status(cid, name, OK, f"RMS {result['rms_px']:.2f} px, {result['views_used']} views,"
                  f" solved {result['solved'][:10]}"), intr


def _stereo_status(intr_L, intr_F, calib: Path, events: list[dict]) -> Status:
    name = "Camera positions (Feeltek relative to laptop)"
    action = "app: Recalibrate camera positions"
    path = calib / "stereo.json"
    if intr_L is None or intr_F is None:
        return Status("C5", name, MISSING if not path.exists() else STALE,
                      "needs both lens calibrations (C1, C2) first", "do C1 and C2 first")
    if not path.exists():
        return Status("C5", name, MISSING, "not done yet", action)
    result = json.loads(path.read_text())
    want = {"L": stereo.intrinsics_fingerprint(intr_L), "F": stereo.intrinsics_fingerprint(intr_F)}
    if result["intrinsics"] != want:
        return Status("C5", name, STALE, "a lens calibration was redone since", action)
    if not result["passed"]:
        return Status("C5", name, FAILED, "result did not pass its gates", action)
    later = [e for e in events if e["time"] > result["solved"]]
    moved = [e for e in later if e["event"] == "moved"]
    if moved:
        return Status("C5", name, STALE, f"camera marked as moved at {moved[-1]['time']}", action)
    alarms = [e for e in later if e["event"] == "c7_alarm"]
    if alarms:
        return Status("C5", name, CHECK, f"health monitor alarm at {alarms[-1]['time']}:"
                      f" {alarms[-1]['detail']}", action)
    if result.get("emergency"):
        return Status("C5", name, CHECK, "emergency 3-pose estimate (provisional)",
                      "run full camera-position calibration when convenient")
    lt = result["length_test"]
    return Status("C5", name, OK, f"known lengths within {lt['p95_mm']:.2f} mm (95 %),"
                  f" solved {result['solved'][:16].replace('T', ' ')}")


def _mirror_status(intr_L, calib: Path) -> Status:
    from eyehand import mirror

    name = "Laptop camera to screen (mirror)"
    action = "eyehand mirror capture"
    path = calib / "screen_from_L.json"
    if intr_L is None:
        return Status("C4", name, MISSING, "needs the laptop camera's lens calibration (C1)", action)
    if not path.exists():
        return Status("C4", name, MISSING, "not done yet", action)
    r = json.loads(path.read_text())
    if r["intrinsics_L"] != mirror.intrinsics_fingerprint(intr_L):
        return Status("C4", name, STALE, "the laptop camera's lens calibration was redone since",
                      action)
    if not r["passed"]:
        return Status("C4", name, FAILED, "result did not pass its gates", action)
    return Status("C4", name, OK, f"cursor uncertainty {r['cursor_uncertainty_mm']:.2f} mm,"
                  f" {r['views_used']} poses, solved {r['solved'][:10]}")


def calibration_status(configs: dict[str, CameraConfig], calib: Path = CALIB) -> list[Status]:
    events = read_events(calib / "events.jsonl")
    out = []
    s1, intr_L = _intrinsics_status("C1", configs["L"], calib)
    s2, intr_F = _intrinsics_status("C2", configs["F"], calib)
    out += [s1, s2]
    screen = calib / "screen.json"
    if screen.exists():
        sc = json.loads(screen.read_text())
        out.append(Status("C3", "Screen size (calipers)", OK,
                          f"{sc['active_area_mm'][0]:.1f} x {sc['active_area_mm'][1]:.1f} mm"))
    else:
        # Optional: the EDID size (344 x 194 mm, rounded to 1 mm) moves the cursor by at most
        # ~0.4 mm at the screen corner farthest from camera L (docs/CALIBRATION.md, C3).
        out.append(Status("C3", "Screen size (calipers, optional)", OK,
                          "not measured; using the EDID size, worst-case cursor effect ~0.4 mm"))
    out.append(_mirror_status(intr_L, calib))
    out.append(_stereo_status(intr_L, intr_F, calib, events))
    out.append(Status("C6", "Camera timing (LED)", NOT_BUILT, "tool not built yet"))
    return out


def next_action(statuses: list[Status]) -> Status | None:
    for s in statuses:
        if s.state in (MISSING, STALE, FAILED, CHECK):
            return s
    return None


def format_status(statuses: list[Status]) -> str:
    marks = {OK: "✔", MISSING: "✘", STALE: "!", FAILED: "✘", CHECK: "?", NOT_BUILT: "·"}
    lines = [f" {marks[s.state]} {s.id} {s.name}: {s.state}" + (f" ({s.detail})" if s.detail else "")
             for s in statuses]
    nxt = next_action(statuses)
    if nxt:
        lines.append(f"\nnext: {nxt.id} {nxt.name}: {nxt.action}")
    return "\n".join(lines)
