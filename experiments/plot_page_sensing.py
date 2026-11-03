"""Figures for the project page, sections 3 (Sensing) and 7 (Real robot).

    MUJOCO_GL=egl python experiments/plot_page_sensing.py

Writes to results/page/:
  frame_chain.png        3.4  the frame chain from a pixel / a LiDAR beam to what the controller uses
  measurement_flow.png   3.3  how a camera frame and a LiDAR scan are accepted, rejected or waited on
  real_nis.png           3.3, 7.2  NIS of every marker measurement on two real runs (docked 09:28, failed 09:41)
  lidar_sim_vs_real.png  3.5  one LiDAR scan in the sim and on the lab robot: the black panels return nothing
  real_paths.png         7.2  the EKF path of the nine filmed real runs of 30 Oct, by result
  camera_calibration_real.png  3.1  the real camera calibration, a frame of the second session shown
  aruco_real_simple.png  3.2  real ArUco range before / after the correction

Reads only logged data (results/hardware) and one sim scan; nothing here feeds the pipeline.
Adapted from the Assessment 3 provided plots (style: bold titled panels, dashed grid, black / red) and
41014 Week 2 C / Week 4 2.2 (frame chains T_WB T_BS p_S, T_BC T_CM).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root, so "src" imports work

import matplotlib                                               # noqa: E402
import numpy as np                                              # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from evaluation.run_demo import CFG, draw_map                   # noqa: E402
from src.robot.sim_robot import PROJECT_ROOT                    # noqa: E402

OUT = PROJECT_ROOT / "results" / "page"
HW = PROJECT_ROOT / "results" / "hardware"
GRID = dict(ls="--", alpha=0.5)
INK, MUTED, BLUE, GREEN, ORANGE, RED = "#2b2b2b", "#777777", "#2a78d6", "#3d7a4f", "#eb6834", "#c83a2e"
RUNS = [("092814", "middle"), ("093249", "middle"), ("093635", "middle"), ("094016", "right"), ("094141", "right"),
        ("094354", "left"), ("104852", "right"), ("105431", "right"), ("105928", "left")]
BOX = {"middle": (2.75, 1.25), "left": (2.75, 1.10), "right": (2.75, 1.40)}


# ---------------------------------------------------------------- diagram helpers
def box(ax, x, y, w, h, title, sub="", fc="#ffffff", ec=INK, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h * (0.62 if sub else 0.5), title, ha="center", va="center", fontsize=11, fontweight="bold", color=tc)
    if sub:
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=8.5, color=tc)


def arrow(ax, p, q, label="", above=True, color=INK):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, lw=1.3, color=color))
    if label:
        mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
        ax.text(mx, my + (0.12 if above else -0.12), label, ha="center", va="bottom" if above else "top", fontsize=8,
                color=MUTED, style="italic")


def blank(ax, w, h):
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")


# ---------------------------------------------------------------- 3.4 frame chain
def frame_chain(path):
    fig, ax = plt.subplots(figsize=(17.4, 5.4))
    blank(ax, 17.4, 5.4)
    W, H, STEP = 1.9, 1.0, 3.0
    cam = [("pixels", "4 corners (u, v)"), ("{C} camera", "T_CM by PnP (IPPE)"), ("{B} base", "z = [r, β]"),
           ("{W} lab map", "x̂ = (x, y, θ), P"), ("path segment", "e_y, e_θ"), ("command", "v, ω")]
    labels = ["K, dist\n(calibration)", "T_BC\n(pitch 6.3°)", "EKF update\nz vs h(x̂)", "segment\nframe", "LQR /\nIBVS"]
    y1 = 3.5
    for k, (t, sub) in enumerate(cam):
        dark = t == "{W} lab map"
        box(ax, 0.2 + k * STEP, y1, W, H, t, sub, fc="#2b2b2b" if dark else "#eef4fb", tc="#ffffff" if dark else INK)
        if k < len(labels):
            x0 = 0.2 + k * STEP + W
            arrow(ax, (x0, y1 + H / 2), (x0 + STEP - W, y1 + H / 2))
            ax.text(x0 + (STEP - W) / 2, y1 + H / 2 - 0.12, labels[k], ha="center", va="top", fontsize=8, color=MUTED, style="italic")
    lid = [("{S} LiDAR", "360 ranges, 1° apart"), ("{B} base", "p_B = T_BS p_S"), ("path frame", "uses x̂ from the EKF"),
           ("AVOID", "box edges → offset d")]
    llab = ["T_BS\n(x −0.064 m)", "valid\nbeams only", "free ahead,\nside clear"]
    y2 = 1.25
    for k, (t, sub) in enumerate(lid):
        box(ax, 0.2 + k * STEP, y2, W, H, t, sub, fc="#fbf1ea")
        if k < len(llab):
            x0 = 0.2 + k * STEP + W
            arrow(ax, (x0, y2 + H / 2), (x0 + STEP - W, y2 + H / 2))
            ax.text(x0 + (STEP - W) / 2, y2 + H / 2 - 0.12, llab[k], ha="center", va="top", fontsize=8, color=MUTED, style="italic")
    xa = 0.2 + 3 * STEP + W
    arrow(ax, (xa, y2 + H / 2), (0.2 + 4 * STEP + W / 2, y1), color=ORANGE)
    ax.text(xa + 1.15, 2.25, "d shifts the path", fontsize=8.5, color=ORANGE, style="italic")
    ax.text(0.2, 4.95, "Camera (ArUco)", fontsize=12, fontweight="bold", color=BLUE)
    ax.text(0.2, 2.7, "LiDAR (LDS-01)", fontsize=12, fontweight="bold", color=ORANGE)
    ax.text(0.2, 0.45, "h(x̂) = [ √(Δx² + Δy²),  atan2(Δy, Δx) − θ ],   (Δx, Δy) = marker − robot in {W};   "
                       "every angle difference goes through wrap()", fontsize=10.5, color=INK)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 3.3 measurement handling
def measurement_flow(path):
    fig, ax = plt.subplots(figsize=(15, 5.6))
    blank(ax, 15, 5.6)
    W, H = 2.15, 0.95
    steps = [(0.2, "camera frame", "30 Hz"), (2.75, "fresh?", "≤ 0.2 s old"), (5.3, "marker found?", "ArUco ID 0, 1, 2"),
             (7.85, "ambiguous pose?", "IPPE: 2 solutions"), (10.4, "NIS ≤ 13.8?", "χ²₂ 99.9 % gate")]
    y = 4.1
    for x, t, s in steps:
        box(ax, x, y, W, H, t, s, fc="#eef4fb")
    for (x0, *_), (x1, *_) in zip(steps[:-1], steps[1:]):
        arrow(ax, (x0 + W, y + H / 2), (x1, y + H / 2), "yes" if x0 > 0.2 else "")
    box(ax, 12.95, y, 1.85, H, "EKF update", "", fc="#d9ead3")
    arrow(ax, (10.4 + W, y + H / 2), (12.95, y + H / 2), "yes")
    outs = [(2.75, "drop the frame", "never fuse late data"), (5.3, "predict only", "P grows (no marker)"),
            (7.85, "keep r, β", "yaw not used"), (10.4, "reject + log", "outlier")]
    for x, t, s in outs:
        box(ax, x, 2.45, W, 0.9, t, s, fc="#f3f3f3")
        arrow(ax, (x + W / 2, y), (x + W / 2, 2.45 + 0.9))
        ax.text(x + W / 2 + 0.08, (y + 3.35) / 2, "yes" if t == "keep r, β" else "no", fontsize=8, color=MUTED, style="italic", va="center")
    ax.text(0.2, 1.75, "Marker lost while docking", fontsize=11, fontweight="bold", color=BLUE)
    ax.text(0.2, 1.35, "ID 0 older than marker_max_age (0.2 s sim, 0.5 s real) → stop and wait;  still lost after 3 s → "
                       "stop_reason marker_lost (never drive in blind).", fontsize=9.5, color=INK)
    ax.text(0.2, 0.8, "LiDAR scan", fontsize=11, fontweight="bold", color=ORANGE)
    ax.text(0.2, 0.4, "forward coverage < min_coverage (0.8 sim, 0.2 real) or scan > 0.5 s old → hold v = w = 0;  "
                      "lost beams (0 / inf) never count as free space;  /scan silent > 3 s (real) → stop the run.",
            fontsize=9.5, color=INK)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 3.3 / 7.2 real NIS
def real_nis(path):
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.4), sharey=True)
    col = {0: GREEN, 1: BLUE, 2: ORANGE}
    for ax, (run, title) in zip(axs, [("092814", "09:28, docked"), ("094141", "09:41, marker_lost")]):
        d = np.load(HW / f"run_20261030_{run}.npz", allow_pickle=True)
        t0, nis = d["t"][0], d["nis"]
        for mid in (1, 2, 0):
            m = nis[nis[:, 1] == mid]
            if len(m):
                ax.semilogy(m[:, 0] - t0, np.maximum(m[:, 2], 1e-3), ".", ms=3, color=col[mid],
                            label=f"ID {mid}" + (" (dock)" if mid == 0 else "") + f": {len(m)} frames, "
                                  f"{100 * np.mean(m[:, 3] > 0):.0f} % accepted")
        ax.axhline(13.8, color="k", ls="--", lw=1, label="gate 13.8 (χ²₂ 99.9 %)")
        ax.axhline(2.0, color="tab:green", ls=":", lw=1, label="expected value, 2")
        ax.set_title(f"Real robot {title}", fontweight="bold")
        ax.set_xlabel("time [s]")
        ax.grid(True, which="both", **GRID)
        ax.legend(fontsize=8, loc="lower left")
    axs[0].set_ylabel("NIS of each marker update")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- 3.5 LiDAR sim vs real
def lidar_sim_vs_real(path):
    from src.perception import lidar
    from src.robot.task import make_robot
    r = make_robot("centre", seed=0)
    with r:
        r.reset(*CFG["start"])
        r.step(0.3)
        _, rho_sim = r.read_scan()
        rmin, rmax = r.lidar.range_min, r.lidar.range_max
    s = np.load(HW / "20261030_090500" / "scan.npz")
    rho_real = s["ranges"]
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, rho, title in [(axs[0], np.asarray(rho_sim, float), "Simulation, start mark"),
                           (axs[1], rho_real.astype(float), "Lab robot, 30 Oct 09:05 (check_robot.py)")]:
        alpha = lidar.beam_angles(len(rho))
        ok = lidar.valid_beams(rho, rmin, rmax)
        P = np.asarray(lidar.to_base(lidar.scan_to_points(rho, alpha, ok)), float).reshape(2, -1)
        fwd = np.abs(np.degrees(np.angle(np.exp(1j * alpha)))) <= 90
        cov = ok[fwd].mean()
        a = np.radians(np.linspace(-90, 90, 60))
        ax.fill(np.r_[0, 3.4 * np.cos(a)], np.r_[0, 3.4 * np.sin(a)], color=MUTED, alpha=0.07, lw=0)
        ax.plot(P[0], P[1], ".", ms=4, color=INK, label=f"returns: {ok.sum()} of {len(rho)}")
        ax.add_patch(plt.Rectangle((-0.2, -0.153), 0.28, 0.306, fc="none", ec=RED, lw=1.5))
        ax.set_title(f"{title}\nforward coverage {100 * cov:.0f} %", fontweight="bold")
        ax.set_aspect("equal")
        ax.set_xlim(-3.6, 3.6)
        ax.set_ylim(-2.6, 2.6)
        ax.set_xlabel("x in {B} [m] (forward)")
        ax.grid(True, **GRID)
        ax.legend(loc="lower right", fontsize=9)
    axs[0].set_ylabel("y in {B} [m]")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- 7.2 real paths
def real_paths(path):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    draw_map(ax, "none")
    for side, (bx, by) in BOX.items():
        ax.add_patch(plt.Rectangle((bx - 0.1, by - 0.1), 0.2, 0.2, fc="#dddddd", ec="#888888", lw=0.8, ls="--"))
        ax.text(bx + 0.13, by - 0.03, side, fontsize=8, color=MUTED)
    for run, side in RUNS:
        d = np.load(HW / f"run_20261030_{run}.npz", allow_pickle=True)
        ok = str(d["result"]) == "docked"
        ax.plot(d["xh"][:, 0], d["xh"][:, 1], "-", lw=1.1, color=BLUE if ok else RED, alpha=0.85)
        ax.plot(d["xh"][-1, 0], d["xh"][-1, 1], "o" if ok else "x", ms=6, color=BLUE if ok else RED)
    ax.plot([], [], "-o", color=BLUE, label="docked (6)")
    ax.plot([], [], "-x", color=RED, label="stopped: marker_lost (3)")
    ax.set_title("Nine filmed real runs, 30 Oct: EKF path (no truth on the robot)", fontweight="bold")
    ax.grid(True, **GRID)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- 3.1 real camera calibration
def real_calibration(path, seed=0):
    """experiments/calibrate_camera.plot on the real frames, showing a random frame of the second session.

    The calibration is the one in config/camera_real.json (all 51 frames, 40 usable; recomputed here, nothing is
    written to the config). The frames were taken in two sessions on 23 Oct: files 000-024 (11:12-11:13) and
    025-050 (11:15-11:17); the shown frame is drawn at random (fixed seed) from the usable frames of the second.
    """
    import cv2
    from experiments.calibrate_camera import folder_views, plot
    from src.perception import camera
    paths = sorted(p for ext in ("*.png", "*.jpg", "*.jpeg") for p in (HW / "calib").glob(ext))
    gray, frames = folder_views(HW / "calib")
    calib = camera.calibrate(gray)
    second = [u for u in calib["used"] if int(paths[u].stem.split("_")[-1]) >= 25]
    shown = int(np.random.default_rng(seed).choice(second))
    plot(dict(calib, used=[shown] + [u for u in calib["used"] if u != shown]), frames, path)
    return paths[shown].name


# ---------------------------------------------------------------- 3.2 real ArUco, simplified
def real_aruco_simple(path):
    """The real-camera ArUco check (experiments/aruco_real_noise.py -> results/aruco_noise_real.csv) as one question:
    how far off is the range before and after the correction stored in config/camera_real.json. Only the sets whose
    tape reading passes the Week 3 apparent-size check (size_consistent) are drawn, the ones the correction was fitted on;
    the two others are listed in the csv and named in the page caption."""
    import csv
    import json
    rows = list(csv.DictReader(open(PROJECT_ROOT / "results" / "aruco_noise_real.csv")))
    x = np.array([float(r["x_true"]) for r in rows])
    raw = np.array([float(r["x_bias_mm"]) for r in rows])
    cor = np.array([float(r["corr_bias_mm"]) for r in rows])
    good = np.array([r["size_consistent"] == "True" for r in rows])
    rc = json.loads((PROJECT_ROOT / "config" / "camera_real.json").read_text())["range_correction"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.axhspan(-3, 3, color=GREEN, alpha=0.12, lw=0, label="±3 mm")
    ax.axhline(0, color="k", lw=0.8)
    ax.plot(x[good], raw[good], "o-", color=RED, lw=1.5, label="before correction (raw ArUco)")
    ax.plot(x[good], cor[good], "s-", color=GREEN, lw=1.5, label="after correction (what the EKF gets)")
    ax.set_xlabel("true distance, robot to marker [m]")
    ax.set_ylabel("ArUco range − tape [mm]")
    ax.set_title(f"How far off is the real camera?\nreads {100 * rc['bias_per_m']:.1f} % too long; "
                 "after correction within ±3 mm", fontweight="bold", fontsize=10.5)
    ax.grid(True, **GRID)
    ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    frame_chain(OUT / "frame_chain.png")
    measurement_flow(OUT / "measurement_flow.png")
    real_nis(OUT / "real_nis.png")
    lidar_sim_vs_real(OUT / "lidar_sim_vs_real.png")
    real_paths(OUT / "real_paths.png")
    print("calibration frame shown:", real_calibration(OUT / "camera_calibration_real.png"))
    real_aruco_simple(OUT / "aruco_real_simple.png")
    print("saved", sorted(p.name for p in OUT.glob("*.png")))
