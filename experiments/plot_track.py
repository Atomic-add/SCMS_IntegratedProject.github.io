"""Top-view plan of a task file, to lay the track out on the floor.

    python experiments/plot_track.py                          # config/arena_lab.json -> results/track_arena_lab.png
    python experiments/plot_track.py config/corridor.json

Draws the 0.5 m tiles, the walls on the outline (with the number of 57 cm panels per edge), the start
pose, the waypoint path, the obstacle positions, the dock goal and every marker with the direction it
faces, all labelled with their coordinates in {W}.
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

PANEL = 0.57          # m, length of one wall panel
BOX = 0.20            # m, side of the printed obstacle box


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "config" / "arena_lab.json"
    cfg = json.loads(Path(path).read_text())
    out = PROJECT_ROOT / "results" / f"track_{Path(path).stem}.png"
    outline = np.array(cfg["outline"], float)
    xmin, ymin = outline.min(axis=0)
    xmax, ymax = outline.max(axis=0)

    fig, ax = plt.subplots(figsize=(2.6 * (xmax - xmin) + 2, 2.6 * (ymax - ymin) + 2))
    ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color="0.8", zorder=0))
    ax.fill(*outline.T, color="#f4f4f0", zorder=0)               # free floor; grey = tiles outside the track
    for gx in np.arange(xmin, xmax + 1e-9, 0.5):                # 0.5 m tiles, numbered from the origin
        ax.plot([gx, gx], [ymin, ymax], color="0.55", lw=0.8, zorder=1)
    for gy in np.arange(ymin, ymax + 1e-9, 0.5):
        ax.plot([xmin, xmax], [gy, gy], color="0.55", lw=0.8, zorder=1)
    for i, cx in enumerate(np.arange(xmin + 0.25, xmax, 0.5)):
        ax.text(cx, ymax + 0.3, f"col {i + 1}", ha="center", fontsize=8, color="0.35")
    for j, cy in enumerate(np.arange(ymin + 0.25, ymax, 0.5)):
        ax.text(xmax + 0.85, cy, f"row {j + 1}", ha="left", va="center", fontsize=8, color="0.35")

    n_total = 0
    for a, b in zip(outline, np.roll(outline, -1, axis=0)):    # walls and their panel counts
        length = float(np.hypot(*(b - a)))
        n = math.ceil(length / PANEL - 0.05)
        n_total += n
        ax.plot(*np.c_[a, b], color="#555555", lw=4, solid_capstyle="butt")
        mid, d = (a + b) / 2, (b - a) / length
        off = np.array([d[1], -d[0]]) * 0.12                    # outside (outline runs counter-clockwise)
        ax.text(*(mid + off), f"{length:.2f} m\n{n} panels", ha="center", va="center", fontsize=8, color="#555555")

    same_wall = {}
    for mid, m in cfg["markers"].items():                      # markers and where they face
        if mid.startswith("_"):
            continue
        x, y, z, th = m
        ax.plot(x, y, "s", color="black", ms=9, zorder=5)
        ax.annotate("", xy=(x + 0.25 * math.cos(th), y + 0.25 * math.sin(th)), xytext=(x, y),
                    arrowprops=dict(arrowstyle="->", color="black"))
        dock = int(mid) == cfg["dock"]["marker_id"]
        k = same_wall[round(th, 2)] = same_wall.get(round(th, 2), -1) + 1   # stagger labels on one wall
        o = -np.array([math.cos(th), math.sin(th)])            # label outside the wall, behind the marker
        lx, ly = np.array([x, y]) + o * (0.3 + 0.2 * k)
        ax.plot([x, lx], [y, ly], color="0.6", lw=0.6)
        ax.text(lx, ly, f"ArUco {mid}{' (dock)' if dock else ''}: ({x:.2f}, {y:.2f}), centre {100 * z:.0f} cm up",
                fontsize=8, va="center", ha="center" if abs(o[0]) < 0.5 else ("left" if o[0] > 0 else "right"),
                bbox=dict(fc="white", ec="none", alpha=0.85))

    sx, sy, sth = cfg["start"]                                 # path
    pts = np.array([[sx, sy]] + cfg["waypoints"], float)
    ax.plot(*pts.T, "--", color="#1f77b4", lw=1.5)
    names = cfg.get("waypoint_names", [f"W{i + 1}" for i in range(len(cfg["waypoints"]))])
    for (x, y), name in zip(cfg["waypoints"], names):
        ax.plot(x, y, "o", color="#1f77b4", ms=7)
        ax.text(x + 0.05, y + 0.07, f"{name}\n({x:.2f}, {y:.2f})", fontsize=8, color="#1f77b4")
    ax.plot(sx, sy, "o", color="#2ca02c", ms=12)
    ax.annotate("", xy=(sx + 0.3 * math.cos(sth), sy + 0.3 * math.sin(sth)), xytext=(sx, sy),
                arrowprops=dict(arrowstyle="-|>", color="#2ca02c", lw=2))
    ax.text(sx, sy - 0.17, f"START ({sx:.2f}, {sy:.2f})", fontsize=8, ha="center", color="#2ca02c")

    gx, gy, gth = cfg["dock"]["goal_pose"]
    ax.plot(gx, gy, "*", color="#d62728", ms=14)
    ax.text(gx - 0.08, gy, f"dock stop\n({gx:.2f}, {gy:.3f})", fontsize=8, ha="right", va="center", color="#d62728")

    for name, c in cfg["obstacles"].items():                   # obstacle cases
        if name.startswith("_") or c is None:
            continue
        ax.add_patch(plt.Rectangle((c[0] - BOX / 2, c[1] - BOX / 2), BOX, BOX, fill=False, ec="#ff7f0e",
                                   ls="-" if name == "centre" else ":", lw=1.5))
        ax.text(c[0] + BOX / 2 + 0.03, c[1], f"box {name} ({c[0]:.2f}, {c[1]:.2f})", fontsize=7,
                va="center", color="#ff7f0e")

    ax.set_aspect("equal")
    ax.set_xlim(xmin - 1.6, xmax + 1.4)
    ax.set_ylim(ymin - 0.85, ymax + 0.5)
    ax.set_xlabel("x [m]  (origin = outer corner of the bottom-left tile)")
    ax.set_ylabel("y [m]")
    ax.set_title(f"{Path(path).name}: {n_total} wall panels of {100 * PANEL:.0f} cm, 0.5 m tiles, "
                 f"box {100 * BOX:.0f} cm")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out.relative_to(PROJECT_ROOT)} ({n_total} panels)")


if __name__ == "__main__":
    main()
