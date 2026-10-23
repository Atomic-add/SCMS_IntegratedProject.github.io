"""Setup sheet for laying the lab track out on the floor, every position given from the nearest tile lines.

    python experiments/plot_setup.py                     # config/arena_lab.json -> results/setup_arena_lab.png

The plan numbers the 0.5 m tiles c1..c10 (along x) and h1..h4 (along y, h1 = the row with the furniture),
draws the walls, markers, start pose and obstacle boxes, and puts a dimension arrow from the nearest
tile line to everything that is not on a line. Below it, numbered instructions (in Vietnamese, for the
person laying it out) say the same in words. Both are generated from the task file, so after a marker
moves in config/arena_lab.json the sheet follows.
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root, so "src" imports work

import matplotlib                                               # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402
import numpy as np                                              # noqa: E402

from src.robot.sim_robot import PROJECT_ROOT                    # noqa: E402

TILE = 0.5            # m
PANEL = 0.57          # m, one wall panel
BOX = 0.20            # m, printed obstacle box
EPS = 1e-6


def x_line(k, n_cols):
    """Name of the vertical tile line x = k * TILE."""
    if k == 0:
        return "mép trái (tấm chắn trái)"
    if k == n_cols:
        return "mép phải (tấm chắn phải)"
    return f"đường kẻ giữa c{k} và c{k + 1}"


def y_line(k, n_rows):
    """Name of the horizontal tile line y = k * TILE."""
    if k == 0:
        return "mép dưới"
    if k == n_rows:
        return "mép trên"
    return f"đường kẻ giữa h{k} và h{k + 1}"


def where(x, y, n_cols, n_rows):
    """A point in words: on a tile line, or how far from the nearest one, and which tile it is in."""
    parts = []
    for v, line, plus, minus, n in ((x, x_line, "về phía c10", "về phía c1", n_cols),
                                    (y, y_line, "lên phía h4", "xuống phía h1", n_rows)):
        k = int(round(v / TILE))
        off = v - k * TILE
        if abs(off) < EPS:
            parts.append(f"nằm trên {line(k, n)}")
        else:
            parts.append(f"cách {line(k, n)} {100 * abs(off):.0f} cm {plus if off > 0 else minus}")
    c = min(int(x // TILE) + 1, n_cols)
    r = min(int(y // TILE) + 1, n_rows)
    return "; ".join(parts) + f" (thuộc ô c{c} h{r})"


def heading_words(th):
    """Which way a heading points, in tile words."""
    deg = int(round(math.degrees(math.atan2(math.sin(th), math.cos(th)))))
    return {0: "về phía c10", 90: "lên phía h4", 180: "về phía c1", -180: "về phía c1", -90: "xuống phía h1"}.get(deg, f"{deg} độ")


def dim(ax, p0, p1, text, color="#444444"):
    """Dimension arrow between two points with its length text."""
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="<->", color=color, lw=1.2, shrinkA=0, shrinkB=0))
    mid = (np.asarray(p0) + np.asarray(p1)) / 2
    ax.text(*mid, text, fontsize=8, color=color, ha="center", va="center",
            bbox=dict(fc="white", ec="none", alpha=0.9, pad=0.5))


def dims_from_lines(ax, x, y, shift=(0.0, 0.0)):
    """Arrows from the nearest vertical and horizontal tile lines to (x, y), where it is not on them,
    drawn shifted by shift [m] (to keep them clear of a wall the point sits on)."""
    kx, ky = round(x / TILE) * TILE, round(y / TILE) * TILE
    sx, sy = shift
    if abs(x - kx) > EPS:
        dim(ax, (kx + sx, y + sy), (x + sx, y + sy), f"{100 * abs(x - kx):.0f} cm")
    if abs(y - ky) > EPS:
        dim(ax, (x + sx, ky + sy), (x + sx, y + sy), f"{100 * abs(y - ky):.0f} cm")



def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "config" / "arena_lab.json"
    cfg = json.loads(Path(path).read_text())
    out = PROJECT_ROOT / "results" / f"setup_{Path(path).stem}.png"
    outline = np.array(cfg["outline"], float)
    xmax, ymax = outline.max(axis=0)
    n_cols, n_rows = int(round(xmax / TILE)), int(round(ymax / TILE))

    fig = plt.figure(figsize=(16, 14))
    ax = fig.add_axes([0.06, 0.44, 0.90, 0.53])
    tx = fig.add_axes([0.04, 0.01, 0.92, 0.42])
    tx.axis("off")

    # tiles: grey outside the track, numbered c1..c10 and h1..h4
    ax.add_patch(plt.Rectangle((0, 0), xmax, ymax, color="0.78", zorder=0))
    ax.fill(*outline.T, color="#f6f6f2", zorder=0)
    for k in range(n_cols + 1):
        ax.plot([k * TILE] * 2, [0, ymax], color="0.5", lw=0.8, zorder=1)
    for k in range(n_rows + 1):
        ax.plot([0, xmax], [k * TILE] * 2, color="0.5", lw=0.8, zorder=1)
    for c in range(n_cols):
        ax.text((c + 0.5) * TILE, ymax + 0.08, f"c{c + 1}", ha="center", fontsize=11, weight="bold", color="0.3")
    for r in range(n_rows):
        ax.text(xmax + 0.12, (r + 0.5) * TILE, f"h{r + 1}", ha="left", va="center", fontsize=11, weight="bold", color="0.3")

    n_panels = 0
    for a, b in zip(outline, np.roll(outline, -1, axis=0)):          # walls (panels)
        n = math.ceil(float(np.hypot(*(b - a))) / PANEL - 0.05)
        n_panels += n
        ax.plot(*np.c_[a, b], color="#333333", lw=6, solid_capstyle="projecting", zorder=3)

    steps = [f"1. Tấm chắn ({n_panels} tấm, 57 × 20 cm): dựng dọc theo các đường viền đậm, mặt trong sát đường kẻ ô. "
             f"Ô xám là chỗ để trống / có đồ đạc."]

    # markers
    for mid, m in cfg["markers"].items():
        if mid.startswith("_"):
            continue
        x, y, z, th = m
        dock = int(mid) == cfg["dock"]["marker_id"]
        ax.plot(x, y, "s", color="black", ms=12, zorder=5)
        ax.annotate("", xy=(x + 0.22 * math.cos(th), y + 0.22 * math.sin(th)), xytext=(x, y),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5), zorder=5)
        ox, oy = -math.cos(th), -math.sin(th)                     # outside the wall, behind the marker
        ax.text(x + 0.1 * ox, y + 0.1 * oy, f"Marker {mid}{' (dock)' if dock else ''}\ntâm cao {100 * z:.0f} cm",
                fontsize=10, weight="bold", zorder=6, va="center" if abs(ox) > 0.5 else ("top" if oy < 0 else "bottom"),
                ha="center" if abs(ox) < 0.5 else ("right" if ox < 0 else "left"),
                bbox=dict(fc="white", ec="black", pad=2))
        dims_from_lines(ax, x, y, shift=(0.07 * math.cos(th), 0.07 * math.sin(th)))
        steps.append(f"{len(steps) + 1}. Marker {mid}{' (dock)' if dock else ''}: tâm marker {where(x, y, n_cols, n_rows)}. "
                     f"Tâm cao {100 * z:.0f} cm so với sàn, dán trên tấm chắn, mặt quay {heading_words(th)}.")

    # start pose
    sx, sy, sth = cfg["start"]
    ax.add_patch(plt.Circle((sx, sy), 0.153, fc="#2ca02c", alpha=0.35, ec="#2ca02c", lw=2, zorder=4))
    ax.annotate("", xy=(sx + 0.3 * math.cos(sth), sy + 0.3 * math.sin(sth)), xytext=(sx, sy),
                arrowprops=dict(arrowstyle="-|>", color="#1a7a1a", lw=2.5), zorder=5)
    ax.plot(sx, sy, "o", color="#1a7a1a", ms=6, zorder=6)
    ax.text(sx + 0.05, sy + 0.2, "XUẤT PHÁT", fontsize=10, weight="bold", color="#1a7a1a", zorder=6)
    dims_from_lines(ax, sx, sy)
    steps.append(f"{len(steps) + 1}. Robot xuất phát: điểm giữa hai bánh (trục bánh) {where(sx, sy, n_cols, n_rows)}. "
                 f"Mặt robot quay {heading_words(sth)}, thẳng theo hàng.")

    # obstacle boxes
    for name, c in cfg["obstacles"].items():
        if name.startswith("_") or c is None:
            continue
        main_case = name == "centre"
        ax.add_patch(plt.Rectangle((c[0] - BOX / 2, c[1] - BOX / 2), BOX, BOX, fc="#ff7f0e" if main_case else "none",
                                   ec="#ff7f0e", ls="-" if main_case else "--", lw=2, alpha=0.8 if main_case else 1, zorder=4))
        ax.text(c[0], c[1] + BOX / 2 + 0.03, {"centre": "hộp (chính)", "left": "hộp lệch trái", "right": "hộp lệch phải"}.get(name, name),
                fontsize=9, ha="center", va="bottom", color="#c55a00", zorder=6,
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.5))
        x0, y0 = math.floor(c[0] / TILE) * TILE, math.floor(c[1] / TILE) * TILE   # tile the box sits in
        gap_x, gap_y = c[0] - BOX / 2 - x0, c[1] - BOX / 2 - y0
        if main_case:
            dim(ax, (x0, c[1] - 0.04), (c[0] - BOX / 2, c[1] - 0.04), f"{100 * gap_x:.0f} cm", "#c55a00")
            dim(ax, (c[0] - 0.04, y0), (c[0] - 0.04, c[1] - BOX / 2), f"{100 * gap_y:.0f} cm", "#c55a00")
        tile = f"c{int(c[0] // TILE) + 1} h{int(c[1] // TILE) + 1}"
        centred = abs(gap_x - (TILE - BOX) / 2) < EPS and abs(gap_y - (TILE - BOX) / 2) < EPS
        text = (f"giữa ô {tile}: mỗi mặt hộp cách đường kẻ ô {100 * gap_x:.0f} cm" if centred else
                f"trong ô {tile}: mặt hộp cách đường kẻ trái {100 * gap_x:.0f} cm, đường kẻ dưới {100 * gap_y:.0f} cm")
        label = {"centre": "Hộp 20 cm (trường hợp chính)", "left": "Hộp, trường hợp lệch trái",
                 "right": "Hộp, trường hợp lệch phải"}.get(name, f"Hộp {name}")
        steps.append(f"{len(steps) + 1}. {label}: {text}.")

    # route and dock stop, for reference
    pts = np.array([[sx, sy]] + cfg["waypoints"], float)
    ax.plot(*pts.T, "--", color="#1f77b4", lw=1.5, zorder=2)
    gx, gy, _ = cfg["dock"]["goal_pose"]
    ax.plot(gx, gy, "*", color="#d62728", ms=16, zorder=6)
    ax.text(gx + 0.08, gy + 0.02, "robot dừng ở đây", fontsize=9, color="#d62728", zorder=6)
    steps.append(f"{len(steps) + 1}. Đường nét đứt xanh là đường robot sẽ đi; ngôi sao đỏ là chỗ robot tự dừng khi dock "
                 f"(không cần đánh dấu). Bày xong, đo lại vị trí thật của các marker và báo để sửa config.")

    ax.set_aspect("equal")
    ax.set_xlim(-0.75, xmax + 0.45)
    ax.set_ylim(-0.45, ymax + 0.3)
    ax.axis("off")
    ax.set_title(f"Sơ đồ bày sân ({Path(path).name}): ô 50 × 50 cm, c = cột, h = hàng (h1 = hàng có đồ đạc)",
                 fontsize=13, pad=22)

    y = 0.98
    for s in steps:
        lines = _wrap(s, 150)
        for i, line in enumerate(lines):
            tx.text(0.0 if i == 0 else 0.02, y, line, fontsize=11, va="top", transform=tx.transAxes)
            y -= 0.058
        y -= 0.016
    fig.savefig(out, dpi=110)
    print(f"wrote {out.relative_to(PROJECT_ROOT)}")


def _wrap(text, width):
    """Split text into lines of at most width characters at spaces."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    return lines + [cur]


if __name__ == "__main__":
    main()
