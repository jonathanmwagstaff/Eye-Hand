"""ChArUco calibration boards: one to print (C1, C5) and one for the laptop screen (C2, C4).

Board object coordinates follow OpenCV: origin at the board's outer top-left corner,
x to the right, y down, in mm. The two boards use different ArUco dictionaries so a
detector never mixes them up.

The screen board is drawn at exact integer pixel sizes. Its spec records where the board
sits on the screen, so board coordinates convert to screen pixels and, after C3, to mm in
the screen frame S. The *mirrored* screen board (for C4) is the same image flipped
left-right: in a mirror it reads correctly, and the ArUco detector rejects mirror-image
markers otherwise.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

MM_PER_INCH = 25.4


@dataclass(frozen=True)
class BoardSpec:
    name: str
    squares_x: int
    squares_y: int
    square: float          # mm for the print board, px for the screen board
    marker: float
    dictionary: str        # name of a cv2.aruco.DICT_* constant
    unit: str              # "mm" or "px"

    def aruco_dictionary(self) -> cv2.aruco.Dictionary:
        return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary))

    def board(self) -> cv2.aruco.CharucoBoard:
        return cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y), self.square, self.marker, self.aruco_dictionary()
        )

    def detector(self) -> cv2.aruco.CharucoDetector:
        return cv2.aruco.CharucoDetector(self.board())


# A4 landscape: 11 x 8 squares of 24 mm = 264 x 192 mm. The printed size must be measured
# with calipers afterwards; printers rescale.
PRINT_BOARD = BoardSpec("print_a4", 11, 8, 24.0, 18.0, "DICT_5X5_250", "mm")

# 1920x1080 screen: 18 x 10 squares of 100 px = 1800 x 1000 px, centred.
SCREEN_BOARD = BoardSpec("screen_1080p", 18, 10, 100, 75, "DICT_4X4_250", "px")

# Mirror calibration (C4): 18 x 7 squares at the top of the screen, leaving the bottom ~320 px
# for a live view of camera L, so the person propping the mirror can see what it sees.
MIRROR_BOARD = BoardSpec("screen_mirror", 18, 7, 100, 75, "DICT_4X4_250", "px")
SCREEN_BOARDS = {b.name: b for b in (SCREEN_BOARD, MIRROR_BOARD)}


def open_fullscreen(name: str, image: np.ndarray) -> None:
    """Open `image` pixel-for-pixel fullscreen, or stop before calibration."""
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    # GTK ignores an early fullscreen request until the window has displayed a frame.
    cv2.imshow(name, image)
    cv2.waitKey(1)
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    expected = (image.shape[1], image.shape[0])
    for _ in range(20):
        cv2.waitKey(10)
        _, _, width, height = cv2.getWindowImageRect(name)
        if (width, height) == expected:
            return
    cv2.destroyWindow(name)
    raise RuntimeError(
        f"fullscreen image area is {width}x{height}, expected {expected[0]}x{expected[1]}; "
        "set the display to 1920x1080 at 100% scaling"
    )


def render_print(spec: BoardSpec = PRINT_BOARD, dpi: int = 600) -> tuple[np.ndarray, float]:
    """Render the print board at `dpi`. Returns the image and the true square size in mm.

    The square is rounded to whole pixels, so the returned size can differ slightly from
    `spec.square`; it is the value to compare against the calipers.
    """
    square_px = round(spec.square / MM_PER_INCH * dpi)
    size = (spec.squares_x * square_px, spec.squares_y * square_px)
    image = spec.board().generateImage(size, marginSize=0, borderBits=1)
    return image, square_px * MM_PER_INCH / dpi


@dataclass(frozen=True)
class ScreenPlacement:
    screen_width: int
    screen_height: int
    offset_x: int          # screen pixel of the board's outer top-left corner (unmirrored)
    offset_y: int
    mirrored: bool

    def board_to_screen_px(self, xy_board_px: np.ndarray) -> np.ndarray:
        """Board coordinates (px, origin at board corner) -> screen pixel-edge coordinates.

        Screen coordinates here measure from the outer top-left edge of the active area,
        in pixel units, so pixel (u, v) spans [u, u+1) x [v, v+1). Mirroring maps
        x -> screen_width - x.
        """
        xy = np.asarray(xy_board_px, float).copy()
        xy[..., 0] += self.offset_x
        xy[..., 1] += self.offset_y
        if self.mirrored:
            xy[..., 0] = self.screen_width - xy[..., 0]
        return xy


def render_screen(
    spec: BoardSpec = SCREEN_BOARD,
    screen_size: tuple[int, int] = (1920, 1080),
    mirrored: bool = False,
    top: int | None = None,
) -> tuple[np.ndarray, ScreenPlacement]:
    """Board on a white screen-sized canvas: centred, or `top` pixels from the top edge."""
    w, h = screen_size
    bw, bh = int(spec.squares_x * spec.square), int(spec.squares_y * spec.square)
    if bw > w or bh > h:
        raise ValueError(f"board {bw}x{bh} px does not fit on a {w}x{h} screen")
    board_img = spec.board().generateImage((bw, bh), marginSize=0, borderBits=1)
    canvas = np.full((h, w), 255, np.uint8)
    ox, oy = (w - bw) // 2, (h - bh) // 2 if top is None else top
    canvas[oy : oy + bh, ox : ox + bw] = board_img
    if mirrored:
        canvas = np.ascontiguousarray(canvas[:, ::-1])
    return canvas, ScreenPlacement(w, h, ox, oy, mirrored)


PAGES_MM = {"a4": (297.0, 210.0), "letter": (279.4, 215.9)}   # landscape


def save_print_pdfs(out_dir: Path, spec: BoardSpec = PRINT_BOARD, dpi: int = 600) -> list[Path]:
    """The board centred on full A4 and US Letter landscape pages, as PDFs.

    A bare PNG lets the printing program place and crop it (one print lost the bottom 4 mm at
    the printer's margin). A full-page PDF with the board centred leaves 9-12 mm on every
    side. Print at "Actual size" / 100 %.
    """
    from PIL import Image

    image, _ = render_print(spec, dpi)
    paths = []
    for name, (w_mm, h_mm) in PAGES_MM.items():
        page = Image.new("L", (round(w_mm / MM_PER_INCH * dpi), round(h_mm / MM_PER_INCH * dpi)), 255)
        board = Image.fromarray(image)
        page.paste(board, ((page.width - board.width) // 2, (page.height - board.height) // 2))
        pdf = out_dir / f"{spec.name}_{name}_landscape.pdf"
        page.save(pdf, resolution=dpi)
        paths.append(pdf)
    return paths


def save_print(out_dir: Path, spec: BoardSpec = PRINT_BOARD, dpi: int = 600) -> Path:
    from PIL import Image  # only needed to embed the DPI so printing at 100 % is exact

    image, square_mm = render_print(spec, dpi)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{spec.name}_{dpi}dpi.png"
    Image.fromarray(image).save(png, dpi=(dpi, dpi))
    meta = {**asdict(spec), "dpi": dpi, "square_mm_rendered": square_mm,
            "note": "Print at 100 % (no fit-to-page). Measure the square size with calipers."}
    png.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return png


def save_screen(out_dir: Path, spec: BoardSpec = SCREEN_BOARD,
                screen_size: tuple[int, int] = (1920, 1080)) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for mirrored in (False, True):
        image, placement = render_screen(spec, screen_size, mirrored)
        suffix = "_mirrored" if mirrored else ""
        png = out_dir / f"{spec.name}{suffix}.png"
        cv2.imwrite(str(png), image)
        meta = {**asdict(spec), "placement": asdict(placement)}
        png.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        paths.append(png)
    return paths
