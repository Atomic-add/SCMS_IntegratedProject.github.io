"""The EKF figures and clips for the project page and the video, in the style of the Assessment 3 plots.

    python -m evaluation.plot_like_a3                    # record the nominal sim run, then all figures and clips
    python -m evaluation.plot_like_a3 --from-record      # re-plot from results/a3style_centre.npz (no sim)
    python -m evaluation.plot_like_a3 --real run_20261030_092814   # the real-robot clip of another run

Outputs in results/:
  ekf_bands_centre.png     estimation error of x, y, heading with the +-2 sigma band from P; grey where no marker
                           update was accepted (Assessment 3 cell 43, "EKF watching the LQI run")
  ekf_summary_centre.png   2 x 2: true vs estimated path, cross-track error true vs estimated, position error vs
                           2 sigma, NEES against its chi-square bound (Assessment 3 cell 53, "LQG path following")
  ekf_anim_centre.mp4      the run drawn as it happens: true (black) and estimated (red) robot, the 2-sigma
                           position ellipse (drawn 10x enlarged), a ray to every marker the camera measures
                           (Assessment 3 cell 56, animate_final_lap); 2x real time
  real_anim_<run>.mp4      the same for a real run from results/hardware: estimate, ellipse and marker rays only
                           (no truth on the real robot); 2x real time, to sit next to the phone video

The sim run is the nominal case "centre", seed 0, on the EKF (what evaluation/run_demo.py runs). Truth comes from
evaluation/ground_truth.py and is used only for the error, the NEES and the black robot. A marker "in view" at time t
= a measurement of it reached the EKF within MARKER_WINDOW s (green ray: accepted, orange ray: rejected by the NIS gate).

Adapted from 41014 Assessment 3 provided plotting code (cells 43, 53 and 56: error with 2-sigma bounds and grey
measurement bands; the 2 x 2 LQG results figure; animate_final_lap with the enlarged ellipse and anchor rays).
Additions: the lab map and box, marker rays from the EKF's own log, the TRACK / AVOID / DOCK state in the clips,
the real-robot replay from the run logs.
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root, so "src" imports work

import matplotlib                                               # noqa: E402
import numpy as np                                              # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402
from matplotlib import animation                                # noqa: E402
from scipy.stats import chi2                                    # noqa: E402

from evaluation.ground_truth import true_pose                   # noqa: E402
from evaluation.run_demo import CFG, draw_map, reference_path   # noqa: E402
from src.control.behaviour import run_mission                   # noqa: E402
from src.estimation.ekf import EKF                              # noqa: E402
from src.estimation.odometry import wrap                        # noqa: E402
from src.robot.sim_robot import PROJECT_ROOT                    # noqa: E402

OUT = PROJECT_ROOT / "results"
RECORD = OUT / "a3style_centre.npz"
TAPE = OUT / "hardware" / "final_pose_tape_30oct.csv"
CASE, SEED = "centre", 0
MARKER_WINDOW = 0.15            # s, a marker counts as in view if measured this recently (sim camera ~20 Hz)
REAL_MARKER_WINDOW = 0.5        # s, the same on the real robot (camera frames arrive over wifi, a few Hz)
ELLIPSE_SCALE = 10.0            # the 2-sigma ellipse is ~1 cm: drawn this much bigger so it can be seen
FRAME_DT, FPS = 0.1, 20         # one frame per 0.1 s of the run, played at 20 fps -> 2x real time
CHI2_3_95 = chi2.ppf(0.95, 3)   # one-sided 95 % bound of NEES with 3 dof (as Assessment 3)
GRID = dict(ls="--", alpha=0.5)


# ---------------------------------------------------------------- data
def record(case=CASE, seed=SEED):
    """Run the nominal mission on the EKF; truth, estimate, P, state and the EKF's marker log (t, id, NIS, used)."""
    rows = []
    ekf = EKF(CFG["start"], marker_map=CFG["markers"])

    def on_step(robot, row):
        rows.append((row["t"], np.asarray(true_pose(robot), float), row["xh"], row["P"], row["state"]))

    info, t_end = run_mission(case, seed=seed, verbose=False, estimator=ekf, on_step=on_step)
    log = dict(t=np.array([r[0] for r in rows]), truth=np.array([r[1] for r in rows]),
               xh=np.array([r[2] for r in rows]), P=np.array([r[3] for r in rows]),
               state=np.array([r[4] for r in rows]), markers=np.array(ekf.log, float).reshape(-1, 4))
    np.savez(RECORD, **log)
    print(f"recorded {case} seed {seed}: {info['stop_reason']} after {t_end:.1f} s -> {RECORD.relative_to(PROJECT_ROOT)}")
    return log


def in_view(t, markers, used_only=False, window=MARKER_WINDOW):
    """Marker ids measured within window [s] before t, as {id: accepted?}."""
    m = markers[(markers[:, 0] <= t) & (markers[:, 0] > t - window)]
    if used_only:
        m = m[m[:, 3] > 0]
    return {int(r[1]): bool(r[3]) for r in m}


def signed_cross_track(xy, path):
    """Signed distance to the nearest segment of the reference path (+ = left of its direction)."""
    out = np.empty(len(xy))
    for k, p in enumerate(xy):
        best = (np.inf, 0.0)
        for a, b in zip(path[:-1], path[1:]):
            ab = b - a
            s = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0)
            d = p - (a + s * ab)
            n = np.linalg.norm(d)
            if n < best[0]:
                best = (n, np.sign(ab[0] * d[1] - ab[1] * d[0]) * n)
        out[k] = best[1]
    return out


def errors(log):
    e = log["truth"] - log["xh"]
    e[:, 2] = wrap(e[:, 2])
    sig = np.sqrt(np.einsum("kii->ki", log["P"]))
    nees = np.einsum("ki,ki->k", e, np.linalg.solve(log["P"], e[..., None])[..., 0])
    return e, sig, nees


def no_marker_mask(log):
    return np.array([not in_view(t, log["markers"], used_only=True) for t in log["t"]])


# ---------------------------------------------------------------- figures (Assessment 3 cells 43 and 53)
def plot_bands(log, path):
    e, sig, _ = errors(log)
    t, blind = log["t"], no_marker_mask(log)
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for i, (lab, scale) in enumerate((("x error [cm]", 100), ("y error [cm]", 100), ("heading error [deg]", 180 / np.pi))):
        ax = axs[i]
        ax.fill_between(t, -2 * scale * sig[:, i], 2 * scale * sig[:, i], color="tab:blue", alpha=0.2, label="$\\pm 2\\sigma$")
        ax.plot(t, scale * e[:, i], color="tab:blue", lw=1, label="error")
        lim = 1.15 * max(np.abs(scale * e[:, i]).max(), 2 * scale * sig[5:, i].max())
        ax.set_ylim(-lim, lim)
        ax.fill_between(t, -lim, lim, where=blind, color="0.5", alpha=0.25, step="mid", label="no marker accepted")
        ax.set_ylabel(lab)
        ax.grid(True, **GRID)
    axs[0].legend(loc="upper right", ncol=3)
    axs[-1].set_xlabel("time [s]")
    inside = np.mean(np.all(np.abs(e) <= 2 * sig, axis=1))
    axs[0].set_title(f"EKF on the lab track, case '{CASE}' (error inside $\\pm 2\\sigma$ on all three states in "
                     f"{100 * inside:.0f} % of the steps)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)


def plot_summary(log, path):
    e, sig, nees = errors(log)
    t, ref = log["t"], reference_path()
    avoiding = np.isin(log["state"], ["AVOID", "REJOIN"])
    ey_true = signed_cross_track(log["truth"][:, :2], ref)
    ey_est = signed_cross_track(log["xh"][:, :2], ref)
    pos_err = np.hypot(e[:, 0], e[:, 1])
    sig_pos = np.sqrt(log["P"][:, 0, 0] + log["P"][:, 1, 1])
    track = ~avoiding & ~np.isin(log["state"], ["DOCK", "IBVS", "STOP"])
    anees, inside = nees.mean(), np.mean(nees <= CHI2_3_95)

    fig, axs = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Lab track on the range + bearing ArUco EKF", fontsize=16, fontweight="bold", y=0.98)
    ax = axs[0, 0]
    draw_map(ax, CASE)
    ax.plot(*log["truth"][:, :2].T, "k-", lw=1.2, label="true path")
    ax.plot(*log["xh"][:, :2].T, "r-", lw=0.8, alpha=0.7, label="EKF estimate")
    ax.set_title(f"Trajectory, case '{CASE}' (docked in {t[-1]:.1f} s)", fontweight="bold")
    ax.legend(loc="upper center", ncol=3, fontsize=8)

    ax = axs[0, 1]
    ax.plot(t, 100 * ey_true, "k-", lw=1, label="true $e_y$")
    ax.plot(t, 100 * ey_est, "r-", lw=0.8, alpha=0.7, label="estimated $e_y$ (what the controller sees)")
    ax.fill_between(t, -50, 50, where=avoiding, color="0.5", alpha=0.2, step="mid", label="AVOID / REJOIN (off the path on purpose)")
    ax.set_ylim(-50, 50)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("cross-track error [cm]")
    ax.set_title(f"Cross-track error (RMS outside avoidance {100 * np.sqrt(np.mean(ey_true[track] ** 2)):.1f} cm)",
                 fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, **GRID)

    ax = axs[1, 0]
    ax.plot(t, 100 * pos_err, "tab:blue", lw=1, label="position error")
    ax.plot(t, 200 * sig_pos, "tab:blue", ls="--", lw=1, label="$2\\sigma$ (from $P$)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("[cm]")
    ax.set_ylim(bottom=0)
    ax.set_title(f"Position estimation error (RMSE {100 * np.sqrt(np.mean(pos_err ** 2)):.2f} cm, "
                 f"max {100 * pos_err.max():.1f} cm)", fontweight="bold")
    ax.legend()
    ax.grid(True, **GRID)

    ax = axs[1, 1]
    ax.semilogy(t, nees, "tab:purple", lw=0.8, label="NEES")
    ax.axhline(CHI2_3_95, color="k", ls="--", label="95% $\\chi^2_3$")
    ax.axhline(3, color="tab:green", ls=":", label="expected value, 3")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("NEES")
    ax.set_title(f"Consistency (ANEES {anees:.2f}, {100 * inside:.0f}% inside bound)", fontweight="bold")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- clips (Assessment 3 cell 56)
def ellipse_xy(xh, P, k=2.0, scale=ELLIPSE_SCALE):
    ang = np.linspace(0, 2 * np.pi, 60)
    w, V = np.linalg.eigh(P[:2, :2])
    pts = V @ (k * np.sqrt(np.maximum(w, 0))[:, None] * np.vstack([np.cos(ang), np.sin(ang)]))
    return xh[0] + scale * pts[0], xh[1] + scale * pts[1]


def animate(log, path, title, box=None, truth=True, window=MARKER_WINDOW):
    """Clip of one run: estimate (red) [+ truth (black)], 2-sigma ellipse, rays to the markers in view, state."""
    t = log["t"]
    idx = np.minimum(np.searchsorted(t, np.arange(t[0], t[-1], FRAME_DT)), len(t) - 1)   # by time: the real loop is not exactly Ts
    marker_xy = {int(k): np.array(v[:2], float) for k, v in CFG["markers"].items() if not str(k).startswith("_")}
    fig, ax = plt.subplots(figsize=(10, 4.8))
    draw_map(ax, "none" if box is not None else CASE)
    if box is not None:
        ax.add_patch(plt.Rectangle((box[0] - 0.1, box[1] - 0.1), 0.2, 0.2, fc="#bbbbbb", ec="#555555", lw=0.8))
    ax.set_title(title, fontweight="bold")
    ax.grid(True, **GRID)
    true_tr, = ax.plot([], [], "k-", lw=1, label="true path")
    est_tr, = ax.plot([], [], "r-", lw=1, alpha=0.7, label="EKF estimate")
    rob, = ax.plot([], [], "k", marker=(3, 0, 0), ms=11, ls="")
    rob_h, = ax.plot([], [], "r", marker=(3, 0, 0), ms=9, ls="")
    ell, = ax.plot([], [], "r-", lw=1.2, label=f"2$\\sigma$ ellipse ({ELLIPSE_SCALE:.0f}x enlarged)")
    rays = {m: ax.plot([], [], lw=0.9, alpha=0.9)[0] for m in marker_xy}
    ax.plot([], [], color="tab:green", lw=0.9, label="marker measured, accepted")
    ax.plot([], [], color="tab:orange", lw=0.9, label="marker measured, NIS-gated")
    if not truth:
        true_tr.set_label("_")
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    txt = ax.text(0.01, 0.97, "", transform=ax.transAxes, fontsize=9, va="top", family="monospace")

    def frame(j):
        k = idx[j]
        xe, Pk = log["xh"][k], log["P"][k]
        est_tr.set_data(log["xh"][:k + 1, 0], log["xh"][:k + 1, 1])
        rob_h.set_data([xe[0]], [xe[1]])
        rob_h.set_marker((3, 0, np.degrees(xe[2]) - 90))
        if truth:
            xt = log["truth"][k]
            true_tr.set_data(log["truth"][:k + 1, 0], log["truth"][:k + 1, 1])
            rob.set_data([xt[0]], [xt[1]])
            rob.set_marker((3, 0, np.degrees(xt[2]) - 90))
        ell.set_data(*ellipse_xy(xe, Pk))
        seen = in_view(log["t"][k], log["markers"], window=window)
        src = log["truth"][k] if truth else xe
        for m, r in rays.items():
            if m in seen:
                r.set_data([src[0], marker_xy[m][0]], [src[1], marker_xy[m][1]])
                r.set_color("tab:green" if seen[m] else "tab:orange")
            else:
                r.set_data([], [])
        sd = np.sqrt(np.diag(Pk))
        txt.set_text(f"t = {log['t'][k] - log['t'][0]:5.1f} s  (2x speed)   state: {str(log['state'][k]):6s}\n"
                     f"markers in view: {sorted(seen) or '-'}   sd x {100 * sd[0]:.1f} cm, y {100 * sd[1]:.1f} cm, "
                     f"heading {np.degrees(sd[2]):.1f} deg")
        return true_tr, est_tr, rob, rob_h, ell, txt, *rays.values()

    anim = animation.FuncAnimation(fig, frame, frames=len(idx), interval=1000 / FPS, blit=True)
    anim.save(path, writer=animation.FFMpegWriter(fps=FPS, bitrate=2400), dpi=150)
    plt.close(fig)


def load_real(run):
    """A real run log as the same dict (no truth); markers = the EKF's (t, id, NIS, used) rows."""
    d = np.load(OUT / "hardware" / f"{run}.npz", allow_pickle=True)
    return dict(t=d["t"], xh=d["xh"], P=d["P"], state=d["state"], markers=d["nis"]), str(d["result"])


def real_box(run):
    """Box centre of a taped real run (final_pose_tape_30oct.csv), else the task's centre case."""
    if TAPE.exists():
        for r in csv.DictReader(open(TAPE, newline="")):
            if r["run_log"] == run:
                return [float(r["box_x"]), float(r["box_y"])]
    return CFG["obstacles"]["centre"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from-record", action="store_true", help=f"re-plot from {RECORD.name} instead of simulating")
    ap.add_argument("--real", default="run_20261030_092814", help="real run log in results/hardware (default 9:28, 30 Oct)")
    ap.add_argument("--no-clips", action="store_true", help="figures only (the clips take a minute or two)")
    args = ap.parse_args()
    log = dict(np.load(RECORD, allow_pickle=True)) if args.from_record else record()
    plot_bands(log, OUT / "ekf_bands_centre.png")
    plot_summary(log, OUT / "ekf_summary_centre.png")
    print("saved results/ekf_bands_centre.png and results/ekf_summary_centre.png")
    if not args.no_clips:
        animate(log, OUT / "ekf_anim_centre.mp4", f"Simulation, case '{CASE}': truth (black) vs EKF estimate (red)")
        rl, result = load_real(args.real)
        animate(rl, OUT / f"real_anim_{args.real.split('run_')[-1]}.mp4",
                f"Real robot {args.real.split('run_')[-1]} ({result}): EKF estimate, no truth on hardware",
                box=real_box(args.real), truth=False, window=REAL_MARKER_WINDOW)
        print(f"saved results/ekf_anim_centre.mp4 and results/real_anim_{args.real.split('run_')[-1]}.mp4")
