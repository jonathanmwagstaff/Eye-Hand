"""C3: the screen's pixel pitch, measured with calipers on a full-screen pattern.

Two vertical lines whose left edges are exactly SPAN_PX pixels apart, and two horizontal lines
whose top edges are SPAN_PX apart. Measuring same-edge to same-edge cancels the line width.
800 px is about 143 mm, inside a 150 mm caliper's range. Each span is measured at three places,
and the spread shows how well the calipers were read.

Frame S (docs/ARCHITECTURE.md) is defined on the pixel grid itself, with its origin at the
outer top-left edge of the active area. So only the pitch is needed, not the physical
position of the active area.
"""

from __future__ import annotations

import json
import time
import tkinter as tk
from pathlib import Path

from eyehand.config import PROJECT_ROOT

SPAN_PX = 800
LINE_PX = 3
SCREEN_PATH = PROJECT_ROOT / "calib" / "screen.json"


def screen_pitch_mm(size=(1920, 1080)) -> tuple[float, float, str]:
    """Pixel pitch (x, y) in mm: calipers (C3) if measured, else the EDID size."""
    if SCREEN_PATH.exists():
        r = json.loads(SCREEN_PATH.read_text())
        return r["pitch_x_mm"], r["pitch_y_mm"], "calipers (C3)"
    return EDID_MM[0] / size[0], EDID_MM[1] / size[1], "EDID 344 x 194 mm"
EDID_MM = (344.0, 194.0)
GATES = {"spread_mm": 0.3, "edid_percent": 1.0, "aspect_percent": 0.5}


def line_positions(size=(1920, 1080)) -> dict[str, tuple[int, int]]:
    """Left edges of the vertical lines and top edges of the horizontal lines, in pixels."""
    w, h = size
    x0, y0 = (w - SPAN_PX) // 2, (h - SPAN_PX) // 2
    return {"x": (x0, x0 + SPAN_PX), "y": (y0, y0 + SPAN_PX)}


def evaluate(horizontal_mm: list[float], vertical_mm: list[float], size=(1920, 1080)) -> dict:
    px = sum(horizontal_mm) / len(horizontal_mm) / SPAN_PX
    py = sum(vertical_mm) / len(vertical_mm) / SPAN_PX
    spread = max(max(horizontal_mm) - min(horizontal_mm), max(vertical_mm) - min(vertical_mm))
    edid_x, edid_y = EDID_MM[0] / size[0], EDID_MM[1] / size[1]
    edid_dev = 100 * max(abs(px / edid_x - 1), abs(py / edid_y - 1))
    aspect = 100 * abs(px / py - 1)
    gates = [
        {"name": "spread of repeated measurements (mm)", "value": spread,
         "limit": GATES["spread_mm"], "pass": spread <= GATES["spread_mm"]},
        {"name": "difference from EDID size (%), which is rounded to 1 mm", "value": edid_dev,
         "limit": GATES["edid_percent"], "pass": edid_dev <= GATES["edid_percent"]},
        {"name": "pixel aspect, x pitch vs y pitch (%)", "value": aspect,
         "limit": GATES["aspect_percent"], "pass": aspect <= GATES["aspect_percent"]},
    ]
    return {
        "pitch_x_mm": px, "pitch_y_mm": py,
        "active_area_mm": [px * size[0], py * size[1]],
        "resolution_px": list(size),
        "span_px": SPAN_PX,
        "measurements_mm": {"horizontal": horizontal_mm, "vertical": vertical_mm},
        "measured": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gates": gates,
        "passed": all(g["pass"] for g in gates),
    }


def save(result: dict, path: Path = SCREEN_PATH) -> None:
    if path.exists():
        history = path.parent / "history"
        history.mkdir(exist_ok=True)
        path.rename(history / f"{path.stem}.{time.strftime('%Y%m%d-%H%M%S')}.json")
    path.write_text(json.dumps(result, indent=1))


class ScreenCalibApp(tk.Tk):
    """Full-screen pattern with the entry fields in the middle, clear of the lines."""

    def __init__(self):
        super().__init__()
        self.attributes("-fullscreen", True)
        self.configure(bg="white", cursor="crosshair")
        self.update_idletasks()
        size = (self.winfo_screenwidth(), self.winfo_screenheight())
        self.size = size
        canvas = tk.Canvas(self, width=size[0], height=size[1], bg="white", highlightthickness=0)
        canvas.place(x=0, y=0)
        pos = line_positions(size)
        for x in pos["x"]:     # rectangle fill covers [x, x + LINE_PX)
            canvas.create_rectangle(x, 0, x + LINE_PX, size[1], fill="black", width=0)
        for y in pos["y"]:
            canvas.create_rectangle(0, y, size[0], y + LINE_PX, fill="black", width=0)

        panel = tk.Frame(self, bg="#eeeeee", padx=16, pady=12)
        panel.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(panel, bg="#eeeeee", justify="left", wraplength=560, font=("TkDefaultFont", 11),
                 text=(f"Screen size (C3). The black lines' edges are exactly {SPAN_PX} pixels apart."
                       " Measure with calipers from the LEFT edge of the left line to the LEFT edge"
                       " of the right line, and from the TOP edge of the top line to the TOP edge of"
                       " the bottom line. Measure each three times at different places: near the"
                       " top, middle and bottom (or left, middle and right). Touch the glass gently"
                       " or hold the jaws just off it.")).grid(row=0, column=0, columnspan=4,
                                                                 pady=(0, 10))
        self.entries = {}
        for row, (key, label, places) in enumerate((
                ("horizontal", "Left line → right line (mm)", ("top", "middle", "bottom")),
                ("vertical", "Top line → bottom line (mm)", ("left", "middle", "right"))), start=1):
            tk.Label(panel, text=label, bg="#eeeeee").grid(row=row * 2 - 1, column=0, columnspan=4,
                                                            sticky="w")
            self.entries[key] = []
            for col, place in enumerate(places):
                cell = tk.Frame(panel, bg="#eeeeee")
                cell.grid(row=row * 2, column=col, padx=6, pady=4, sticky="w")
                tk.Label(cell, text=place, bg="#eeeeee").pack(anchor="w")
                e = tk.Entry(cell, width=10)
                e.pack(anchor="w")
                self.entries[key].append(e)
        self.message = tk.Label(panel, text="", bg="#eeeeee", fg="#c62828", justify="left",
                                wraplength=560)
        self.message.grid(row=5, column=0, columnspan=4, sticky="w", pady=(8, 4))
        buttons = tk.Frame(panel, bg="#eeeeee")
        buttons.grid(row=6, column=0, columnspan=4, sticky="e")
        tk.Button(buttons, text="Cancel (Esc)", command=self.destroy).pack(side="left", padx=4)
        tk.Button(buttons, text="Save", command=self.submit).pack(side="left")
        self.bind("<Escape>", lambda e: self.destroy())
        self.entries["horizontal"][0].focus_set()
        self.result: dict | None = None

    def submit(self) -> None:
        try:
            values = {k: [float(e.get().replace(",", ".")) for e in es]
                      for k, es in self.entries.items()}
        except ValueError:
            self.message.config(text="Enter all six measurements in mm, e.g. 143.35")
            return
        if not all(100 < v < 200 for vs in values.values() for v in vs):
            self.message.config(text=f"{SPAN_PX} px should measure about 143 mm; check the numbers.")
            return
        r = evaluate(values["horizontal"], values["vertical"], self.size)
        if not r["passed"]:
            failed = "; ".join(f"{g['name']}: {g['value']:.3f} (limit {g['limit']})"
                               for g in r["gates"] if not g["pass"])
            self.message.config(text=f"Not saved: {failed}. Measure again.")
            return
        save(r)
        self.result = r
        self.destroy()


def main() -> int:
    app = ScreenCalibApp()
    app.mainloop()
    r = app.result
    if r is None:
        print("cancelled; nothing saved")
        return 1
    print(f"pixel pitch x {r['pitch_x_mm']:.5f} mm, y {r['pitch_y_mm']:.5f} mm;"
          f" active area {r['active_area_mm'][0]:.2f} x {r['active_area_mm'][1]:.2f} mm")
    for g in r["gates"]:
        print(f"  PASS  {g['name']}: {g['value']:.3f} (limit {g['limit']})")
    print(f"saved: {SCREEN_PATH.relative_to(PROJECT_ROOT)}")
    return 0
