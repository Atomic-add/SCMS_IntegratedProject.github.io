"""A4 sheets to print for the real robot: the calibration checkerboard and the ArUco markers, true size.

    python experiments/make_prints.py           # writes results/print/track_a_A4.pdf

Page 1 (landscape): the W3 board, 9 x 6 internal corners of SQUARESIZE (25 mm) -> 250 x 175 mm.
Pages 2-4 (portrait): DICT_4X4_50 IDs 0 (dock), 1 and 2, black square MARKER_SIDE (100 mm) with a
one-cell white quiet zone inside the cut line, and a cross at the marker centre (config/arena_lab.json
gives the centre height, 0.10 m above the floor).
Every page has a 100 mm scale bar: print at 100 % / "Actual size" (not "Fit to page") and measure it.
The squares are drawn as vector rectangles, so the PDF is exact at any printer resolution. Sizes come
from src/perception, so the print always matches what calibrate_camera.py and aruco.measure assume.

Adapted from 41014 Week 3 E (marker_image) and B (the checkerboard).
Additions: printable pages with scale bars.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root, so "src" imports work

import cv2                                                      # noqa: E402
import matplotlib                                               # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages            # noqa: E402
from matplotlib.patches import Rectangle                        # noqa: E402

from src.perception.aruco import ARUCO_DICT, MARKER_SIDE        # noqa: E402
from src.perception.camera import GRIDSHAPE, SQUARESIZE         # noqa: E402
from src.robot.sim_robot import PROJECT_ROOT                    # noqa: E402

A4 = (210.0, 297.0)                 # mm, portrait (width, height)
MM = 1 / 25.4                       # inch per mm
OUT = PROJECT_ROOT / "results" / "print" / "track_a_A4.pdf"
MARKERS = {0: "dock", 1: "landmark", 2: "landmark"}


def page(landscape=False):
    """A4 figure whose axes are the whole sheet in mm, origin bottom-left."""
    w, h = (A4[1], A4[0]) if landscape else A4
    fig = plt.figure(figsize=(w * MM, h * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax, w, h


def black(ax, x, y, w, h):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="black", edgecolor="none", antialiased=False))


def scale_bar(ax, x, y, length=100.0):
    """A length mm bar with end ticks, to check the printer did not scale the page."""
    ax.plot([x, x + length], [y, y], color="black", lw=0.8)
    for xe in (x, x + length):
        ax.plot([xe, xe], [y - 2, y + 2], color="black", lw=0.8)
    ax.text(x + length + 3, y, f"{length:.0f} mm: measure this; print at 100 % (Actual size)",
            fontsize=7, va="center")


def checkerboard_page(pdf):
    fig, ax, w, h = page(landscape=True)
    s = SQUARESIZE * 1000
    nx, ny = GRIDSHAPE[0] + 1, GRIDSHAPE[1] + 1            # squares = internal corners + 1
    x0, y0 = (w - nx * s) / 2, (h - ny * s) / 2 + 6
    for j in range(ny):
        for i in range(nx):
            if (i + j) % 2 == 0:
                black(ax, x0 + i * s, y0 + (ny - 1 - j) * s, s, s)
    ax.text(x0, y0 + ny * s + 4, f"Calibration board: {GRIDSHAPE[0]} x {GRIDSHAPE[1]} internal corners, "
            f"{s:.0f} mm squares ({nx * s:.0f} x {ny * s:.0f} mm). Glue flat on card, matte paper.", fontsize=8)
    scale_bar(ax, x0, y0 - 9)
    pdf.savefig(fig)
    plt.close(fig)


def marker_page(pdf, marker_id, role):
    fig, ax, w, h = page()
    bits = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(ARUCO_DICT), marker_id, 6)
    side = MARKER_SIDE * 1000
    cell = side / bits.shape[0]                            # 4 x 4 data cells + 1-cell black border
    x0, y0 = (w - side) / 2, (h - side) / 2 + 20
    for r in range(bits.shape[0]):
        for c in range(bits.shape[1]):
            if bits[r, c] == 0:
                black(ax, x0 + c * cell, y0 + (bits.shape[0] - 1 - r) * cell, cell, cell)
    q = cell                                               # quiet zone: one cell of white all round
    ax.add_patch(Rectangle((x0 - q, y0 - q), side + 2 * q, side + 2 * q, fill=False, edgecolor="0.6",
                           lw=0.4, ls=(0, (3, 3))))
    cx, cy = x0 + side / 2, y0 + side / 2                  # centre ticks outside the marker, for mounting
    for d in (-1, 1):
        ax.plot([cx, cx], [cy + d * (side / 2 + q + 2), cy + d * (side / 2 + q + 8)], color="0.4", lw=0.5)
        ax.plot([cx + d * (side / 2 + q + 2), cx + d * (side / 2 + q + 8)], [cy, cy], color="0.4", lw=0.5)
    ax.text(w / 2, y0 + side + q + 14, f"ArUco DICT_4X4_50  ID {marker_id}  ({role})", fontsize=12,
            ha="center", weight="bold")
    ax.text(w / 2, y0 - q - 12, f"black square {side:.0f} mm. Cut on the dashed line (keeps the white border). "
            "Ticks mark the centre.", fontsize=7.5, ha="center")
    scale_bar(ax, 20, 25)
    pdf.savefig(fig)
    plt.close(fig)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT) as pdf:
        checkerboard_page(pdf)
        for mid, role in MARKERS.items():
            marker_page(pdf, mid, role)
    print(f"wrote {OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
