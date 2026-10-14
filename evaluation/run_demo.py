"""Evidence for the controller: run the four obstacle cases, record truth vs estimate, plot and score.

EVERYTHING AT ONCE: press Run in VS Code, or  python -m evaluation.run_demo  with no options. It runs,
in order: the 4 cases on odometry, the 4 cases on the EKF, ALL_TRIALS random trials of none + centre
on each, and the odometry vs EKF comparison. Every figure and json below ends up in results/.

Single parts (options):      python -m evaluation.run_demo --odometry   (4 cases on odometry only)
                             python -m evaluation.run_demo centre       (one case only)

Lives in evaluation/ because it reads the simulator's ground truth (evaluation.ground_truth.true_pose),
which src/control must never do. The controller runs exactly as in behaviour.py; run_mission's on_step
hook hands us each step's log row, and we read the true pose next to it.

Repeated trials:            python -m evaluation.run_demo --trials 20   (20 random trials per case)
  Trial k uses seed k for: the LiDAR range noise and camera pixel noise already in the simulator, a
  random TRUE start pose (the controller still believes the config start, so it starts with an
  unknown error), and a random scale error on each wheel encoder (wheel radius / slip mismatch, so the
  odometry drifts differently every run). Same N -> same numbers, so the trials are reproducible too.
  Outputs: demo_trials_odometry.json (per case: success rate, mean +- std, worst case; plus every
  trial) and demo_trials_odometry.png. The single run (no --trials) is the nominal seed-0 run.

Odometry or EKF:            add --ekf to any command above
  Without it the controller drives on odometry (DeadReckoning in behaviour.py, the baseline); with it,
  on src/estimation/ekf.py. Outputs get "_ekf" in their names, so both sets are kept side by side.
  After both trial sets exist:  python -m evaluation.run_demo --compare  -> demo_compare.png / .json

Outputs of the single run (in results/):
  demo_paths.png       all four cases on the map + a zoom on the dock pad (final poses)
  demo_<case>.png      none and centre only (left / right look the same as none), in detail: truth vs estimate, cross-track error, commands, estimate error
  demo_metrics.json    the numbers quoted in the report (success, time, errors, clearance, collisions)
  (with --ekf: demo_paths_ekf.png, demo_<case>_ekf.png, demo_metrics_ekf.json; NEES / NIS added)
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                                          # files only, no window needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import chi2  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root, for the VS Code Run button

from evaluation.ground_truth import true_pose  # noqa: E402
from src.control.behaviour import run_mission  # noqa: E402
from src.estimation.ekf import EKF  # noqa: E402
from src.estimation.odometry import wrap  # noqa: E402
from src.robot.corridor import load_config  # noqa: E402
from src.robot.sim_robot import PROJECT_ROOT  # noqa: E402

CASES = ["none", "left", "centre", "right"]
ALL_TRIALS = 10                     # random trials per case when everything runs at once
DETAIL_CASES = ["none", "centre"]   # detail plot only for these: left / right drive the same path as none
COLOR = dict(zip(CASES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))   # fixed order, one hue per case
WIDTH = dict(zip(CASES, [5.0, 3.2, 1.4, 1.4]))                         # thick to thin, see plot_overview
ESTS = ["odometry", "ekf"]
EST_COLOR = dict(odometry="#2a78d6", ekf="#eb6834")


def bounds(dof, n=1):
    """Two-sided 95 % chi-square bounds. n=1: one sample (NEES dof 3 -> [0.22, 9.35], NIS dof 2 -> [0.05, 7.38]).
    n>1: the mean of n samples (ANEES / ANIS), which is chi-square with dof*n, divided by n."""
    lo, hi = chi2.ppf([0.025, 0.975], dof * n) / n
    return [round(float(lo), 3), round(float(hi), 3)]
TRUE_C, EST_C = "#2a78d6", "#eb6834"                                     # truth blue, estimate orange
BOX_HALF, ROBOT_HALF_WIDTH, ROBOT_LENGTH = 0.15, 0.153, 0.28
OUT = PROJECT_ROOT / "results"
CFG = load_config()


# ---------------------------------------------------------------- geometry helpers
def reference_path():
    """The path the controller is asked to follow: start -> W1..W4 -> dock goal."""
    pts = [CFG["start"][:2]] + CFG["waypoints"] + [CFG["dock"]["goal_pose"][:2]]
    return np.array(pts, float)


def dist_to_polyline(p, poly, closed=False):
    """Shortest distance from point p to a polyline (or polygon when closed)."""
    segs = list(zip(poly[:-1], poly[1:])) + ([(poly[-1], poly[0])] if closed else [])
    best = np.inf
    for a, b in segs:
        ab = b - a
        s = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0)
        best = min(best, np.linalg.norm(p - (a + s * ab)))
    return best


def dist_to_box(p, c):
    """Distance from point p to the 0.3 x 0.3 m box centred at c (0 inside)."""
    d = np.maximum(np.abs(p - c) - BOX_HALF, 0.0)
    return float(np.hypot(*d))


def wall_contacts(robot):
    """Names of wall / box / dock geoms the robot is touching right now (floor contacts ignored)."""
    hit = set()
    for i in range(robot.d.ncon):
        c = robot.d.contact[i]
        names = [robot.m.geom(g).name for g in (c.geom1, c.geom2)]
        world = [n for n in names if n.startswith("wall") or n in ("obstacle", "dock_body")]
        if world and "floor" not in names:
            hit.update(world)
    return hit


# ---------------------------------------------------------------- run one case
# Trial randomness (1 sigma): how carefully the robot is placed at the start, and how well the wheel
# radii are calibrated. The controller drives on odometry until the EKF exists, and odometry cannot see
# these errors, so they are kept at "calibrated robot, careful placement" level. (Tried 2 deg / 0.5 %:
# a 1 % left/right mismatch turns into ~8 deg of heading drift by the obstacle; the robot then stops
# safely with too_close instead of docking. That drift is exactly what the EKF's marker updates fix.)
START_XY_SD, START_YAW_SD_DEG, ENC_SCALE_SD = 0.01, 0.5, 0.001


def trial_setup(seed):
    """setup(robot) for run_mission: random true start pose and encoder scale errors for this seed.
    Only the SIMULATED robot changes; the controller is untouched and starts from the config prior."""
    rng = np.random.default_rng(1000 + seed)
    dx, dy = rng.normal(0.0, START_XY_SD, 2)
    dth = np.deg2rad(rng.normal(0.0, START_YAW_SD_DEG))
    scale = 1.0 + rng.normal(0.0, ENC_SCALE_SD, 2)             # left, right wheel

    def setup(robot):
        x, y, th = CFG["start"]
        robot.reset(x + dx, y + dy, th + dth)
        read = robot.read_encoders

        def read_scaled():                                     # encoders that over/under-count a little
            t, phi_l, phi_r = read()
            return t, phi_l * scale[0], phi_r * scale[1]
        robot.read_encoders = read_scaled
    return setup


def sfx(est):
    """File-name suffix: nothing for the odometry baseline (the original names), "_ekf" for the EKF."""
    return "" if est == "odometry" else f"_{est}"


def run_case(case, seed=0, randomise=False, save=True, est="odometry"):
    rows = []

    def on_step(robot, row):                                   # called by run_mission every Ts
        row["truth"] = np.asarray(true_pose(robot), float)
        row["contacts"] = wall_contacts(robot)
        rows.append(row)

    setup = trial_setup(seed) if randomise else None
    # The EKF starts from the config start and P0, like odometry: it is never told the true start.
    estimator = EKF(CFG["start"], marker_map=CFG["markers"]) if est == "ekf" else None
    info, t_end = run_mission(case, seed=seed, verbose=False, on_step=on_step, setup=setup, estimator=estimator)
    log = dict(
        t=np.array([r["t"] for r in rows]),
        xh=np.array([r["xh"] for r in rows]),
        truth=np.array([r["truth"] for r in rows]),
        v=np.array([r["v"] for r in rows]),
        w=np.array([r["w"] for r in rows]),
        state=np.array([r["state"] for r in rows]),
        P=np.array([r["P"] for r in rows]),
    )
    if estimator is not None:                                  # NIS of every marker the EKF looked at
        log["nis"] = np.array([n for _, _, n, _ in estimator.log], float)
        log["nis_t"] = np.array([tt for tt, _, _, _ in estimator.log], float)
    OUT.mkdir(exist_ok=True)
    if save:
        np.savez(OUT / f"demo_{case}{sfx(est)}.npz", **log)
    contacts = sorted(set().union(*[r["contacts"] for r in rows]))
    return info, t_end, log, contacts


def nees(log):
    """NEES per step: e^T P^-1 e with e = true - estimate (heading wrapped). Simulation only."""
    e = log["truth"] - log["xh"]
    e[:, 2] = wrap(e[:, 2])
    return np.array([ei @ np.linalg.solve(Pi, ei) for ei, Pi in zip(e, log["P"])])


def metrics(case, info, t_end, log, contacts):
    ref, outline = reference_path(), np.array(CFG["outline"], float)
    xy = log["truth"][:, :2]
    avoiding = np.isin(log["state"], ["AVOID", "REJOIN"])
    # cross-track error of the TRUE pose to the reference path; while avoiding the robot is off the path
    # on purpose, so those steps are left out of the tracking score.
    cte = np.array([dist_to_polyline(p, ref) for p in xy])
    track = cte[~avoiding]
    gx, gy, gth = CFG["dock"]["goal_pose"]
    x, y, th = log["truth"][-1]
    odo = np.hypot(*(log["xh"][:, :2] - xy).T)
    box = CFG["obstacles"][case]
    m = dict(
        success=bool(info["dock_complete"]), stop_reason=info["stop_reason"], time_s=round(float(t_end), 1),
        cross_track_rms_cm=round(100 * float(np.sqrt(np.mean(track**2))), 2),
        cross_track_max_cm=round(100 * float(track.max()), 2),
        dock_lateral_err_cm=round(100 * (x - gx), 2),          # sideways off the approach line
        dock_along_err_cm=round(100 * (y - gy), 2),            # + = past the goal, towards the marker
        dock_heading_err_deg=round(float(np.rad2deg(np.angle(np.exp(1j * (th - gth))))), 2),
        estimate_err_final_cm=round(100 * float(odo[-1]), 2),
        estimate_err_max_cm=round(100 * float(odo.max()), 2),
        estimate_rmse_cm=round(100 * float(np.sqrt(np.mean(odo**2))), 2),
        min_centre_to_wall_m=round(float(min(dist_to_polyline(p, outline, closed=True) for p in xy)), 3),
        collisions=contacts,
    )
    if box is not None:
        m["min_centre_to_box_m"] = round(float(min(dist_to_box(p, np.array(box)) for p in xy)), 3)
    if "nis" in log:                                           # EKF only: is P honest? (Assessment 3 C.2)
        # If P is honest, about 95 % of the per-step values land inside the two-sided bounds,
        # and the averages (ANEES, ANIS) land inside their own, much tighter, bounds.
        ne, ni = nees(log), log["nis"]
        nb, ib = bounds(3), bounds(2)
        m.update(anees=round(float(ne.mean()), 2), anees_bounds=bounds(3, ne.size),
                 nees_bounds=nb, nees_inside_95=round(float(np.mean((ne >= nb[0]) & (ne <= nb[1]))), 3),
                 marker_updates=int(ni.size))
        if ni.size:
            m.update(anis=round(float(ni.mean()), 2), anis_bounds=bounds(2, ni.size),
                     nis_bounds=ib, nis_inside_95=round(float(np.mean((ni >= ib[0]) & (ni <= ib[1]))), 3))
        m["mean_nis"] = m.get("anis")                          # old name, kept for older json readers
    return m


# ---------------------------------------------------------------- drawing
def draw_map(ax, case=None):
    """Walls, the box (if any), the dock pad, waypoints and the reference path."""
    outline = np.array(CFG["outline"] + [CFG["outline"][0]], float)
    ax.plot(*outline.T, color="#555555", lw=1.2)
    ax.add_patch(plt.Rectangle((3.4, 4.15), 0.4, 0.4, color="#9fd3a8", alpha=0.6, lw=0, label="dock pad"))
    boxes = [CFG["obstacles"][case]] if case else [CFG["obstacles"][c] for c in CASES[1:]]
    for c in boxes:
        if c is not None:
            ax.add_patch(plt.Rectangle((c[0] - BOX_HALF, c[1] - BOX_HALF), 0.3, 0.3, fc="#bbbbbb",
                                       ec="#555555", lw=0.8))
    ref = reference_path()
    ax.plot(*ref.T, color="#999999", lw=1, ls=":", label="reference path")
    ax.plot(*np.array(CFG["waypoints"]).T, "o", color="#555555", ms=4)
    for mid, (mx, my, *_) in CFG["markers"].items():
        if mid.startswith("_"):
            continue
        ax.plot(mx, my, "s", color="#222222", ms=5)
        ax.annotate(f"ID {mid}", (mx, my), textcoords="offset points", xytext=(4, -10), fontsize=7)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def robot_box(ax, pose, color):
    """Outline of the robot footprint at a pose (0.28 x 0.306 m, origin near the axle)."""
    x, y, th = pose
    corners = np.array([[0.08, 0.153], [0.08, -0.153], [-0.20, -0.153], [-0.20, 0.153], [0.08, 0.153]])
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    ax.plot(*(corners @ R.T + [x, y]).T, color=color, lw=1.2)
    ax.plot(x, y, "o", color=color, ms=3)


def plot_overview(logs, est="odometry"):
    fig, (ax, az) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw=dict(width_ratios=[1.6, 1]))
    draw_map(ax)
    # none / left / right drive the same path (the box is not in the lane, avoidance is not needed), so
    # they would hide each other: draw them thick to thin so every case stays visible.
    for case, log in logs.items():
        ax.plot(*log["truth"][:, :2].T, color=COLOR[case], lw=WIDTH[case], label=f"{case} (true path)")
    ax.set_title(f"True paths, all four obstacle cases (driving on {est})")
    ax.legend(fontsize=8, loc="upper left")

    draw_map(az)
    for case, log in logs.items():
        robot_box(az, log["truth"][-1], COLOR[case])
        az.plot(*log["truth"][-120:, :2].T, color=COLOR[case], lw=WIDTH[case], label=case)
    gx, gy, _ = CFG["dock"]["goal_pose"]
    az.plot(gx, gy, "+", color="#222222", ms=12, mew=1.5, label="goal pose")
    az.set_xlim(3.25, 3.95)
    az.set_ylim(3.75, 4.75)
    az.set_title("Docking: final true poses")
    az.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / f"demo_paths{sfx(est)}.png", dpi=150)
    plt.close(fig)


def shade_states(ax, log, label=False):
    """Grey bands while avoiding, green while docking (same on every time plot, named on the first)."""
    t, s = log["t"], log["state"]
    for name, color in [("AVOID", "#dddddd"), ("REJOIN", "#eeeeee"), ("DOCK", "#d9f0dc"), ("IBVS", "#bfe3c6")]:
        on = s == name
        if np.any(on):
            ax.axvspan(t[on][0], t[on][-1], color=color, lw=0, label=name if label else None)


def plot_case(case, log, m, est="odometry"):
    rows = 4 if "nees_bounds" in m else 3                      # EKF gets a 4th panel: NEES / NIS vs 95 % bounds
    fig = plt.figure(figsize=(13, 2.7 * rows))
    gs = fig.add_gridspec(rows, 2, width_ratios=[1, 1.3])
    ax = fig.add_subplot(gs[:, 0])
    draw_map(ax, case)
    ax.plot(*log["truth"][:, :2].T, color=TRUE_C, lw=1.4, label="true path (sim)")
    ax.plot(*log["xh"][:, :2].T, color=EST_C, lw=1.2, ls="--", label=f"estimate x̂ ({est})")
    ax.set_title(f"Case '{case}': {'docked' if m['success'] else 'NOT docked'} in {m['time_s']} s")
    ax.legend(fontsize=8, loc="upper left")

    t = log["t"]
    ref = reference_path()
    cte = np.array([dist_to_polyline(p, ref) for p in log["truth"][:, :2]])
    a1 = fig.add_subplot(gs[0, 1])
    shade_states(a1, log, label=True)
    a1.plot(t, 100 * cte, color=TRUE_C, lw=1.2, label="|cross-track error|, true pose")
    a1.set_ylabel("[cm]")
    a1.set_title(f"Distance from the reference path (RMS {m['cross_track_rms_cm']} cm outside avoidance)",
                 fontsize=10)
    a1.legend(fontsize=8, loc="upper left")

    a2 = fig.add_subplot(gs[1, 1], sharex=a1)
    shade_states(a2, log)
    a2.plot(t, log["v"], color=TRUE_C, lw=1.2, label="v command [m/s]")
    a2.plot(t, log["w"], color=EST_C, lw=1.0, label="ω command [rad/s]")
    a2.axhline(0, color="#999999", lw=0.6)
    a2.set_title("Commands sent to the robot (limits 0.26 m/s, 1.82 rad/s)", fontsize=10)
    a2.legend(fontsize=8, loc="lower left", ncol=2)

    a3 = fig.add_subplot(gs[2, 1], sharex=a1)
    shade_states(a3, log)
    odo = np.hypot(*(log["xh"][:, :2] - log["truth"][:, :2]).T)
    a3.plot(t, 100 * odo, color=EST_C, lw=1.2, label="|x̂ − true| position")
    if est == "ekf":                                           # the filter's own 2-sigma claim, for honesty
        sd = np.sqrt(log["P"][:, 0, 0] + log["P"][:, 1, 1])
        a3.plot(t, 200 * sd, color="#555555", lw=1.0, ls="--", label="2σ from P (what the EKF claims)")
    a3.set_ylabel("[cm]")
    a3.set_title(f"Estimate error, {est}", fontsize=10)
    a3.legend(fontsize=8, loc="upper left")
    if rows == 4:                                              # consistency: NEES every step, NIS every marker
        a4 = fig.add_subplot(gs[3, 1], sharex=a1)
        a4.semilogy(t, np.maximum(nees(log), 1e-4), color=EST_C, lw=1.0, label=f"NEES ({100 * m['nees_inside_95']:.0f} % inside)")
        for b in m["nees_bounds"]:
            a4.axhline(b, color=EST_C, lw=0.8, ls="--")
        if "nis_bounds" in m:
            a4.semilogy(log["nis_t"], np.maximum(log["nis"], 1e-4), ".", color=TRUE_C, ms=3,
                        label=f"NIS ({100 * m['nis_inside_95']:.0f} % inside)")
            for b in m["nis_bounds"]:
                a4.axhline(b, color=TRUE_C, lw=0.8, ls=":")
        a4.set_title(f"95 % chi-square bounds (dashed NEES, dotted NIS). ANEES {m['anees']} in {m['anees_bounds']}"
                     + (f", ANIS {m['anis']} in {m['anis_bounds']}" if "anis" in m else ""), fontsize=9)
        a4.legend(fontsize=8, loc="best", ncol=2)
    fig.axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(OUT / f"demo_{case}{sfx(est)}.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- repeated trials
def one_trial(job):
    """One (case, seed, estimator) trial -> its metrics. Runs in a worker process."""
    case, seed, est = job
    try:
        info, t_end, log, contacts = run_case(case, seed=seed, randomise=True, save=False, est=est)
        m = metrics(case, info, t_end, log, contacts)
    except Exception as e:                                     # a crash counts as a failed trial
        m = dict(success=False, stop_reason=f"error: {e}")
    return case, dict(seed=seed, **m)


def summarise(trials):
    """Per case: success rate, mean +- std and worst case of the numbers in the report.
    Docking errors are over the trials that docked (a failed dock has no meaningful final error)."""
    def stats(key, worst=max, docked_only=False):
        x = np.array([t[key] for t in trials if key in t and (t["success"] or not docked_only)], float)
        if x.size == 0:
            return None
        return dict(mean=round(float(x.mean()), 2), std=round(float(x.std(ddof=1)) if x.size > 1 else 0.0, 2),
                    worst=round(float(worst(x, key=abs) if worst is max else worst(x)), 2))
    out = dict(n=len(trials), success_rate=sum(t["success"] for t in trials) / len(trials),
               collisions=sum(bool(t.get("collisions")) for t in trials),
               failures=[(t["seed"], t["stop_reason"]) for t in trials if not t["success"]],
               time_s=stats("time_s"), cross_track_rms_cm=stats("cross_track_rms_cm"),
               dock_lateral_err_cm=stats("dock_lateral_err_cm", docked_only=True), dock_along_err_cm=stats("dock_along_err_cm", docked_only=True),
               dock_heading_err_deg=stats("dock_heading_err_deg", docked_only=True), estimate_err_final_cm=stats("estimate_err_final_cm"),
               estimate_rmse_cm=stats("estimate_rmse_cm"),
               min_centre_to_wall_m=stats("min_centre_to_wall_m", worst=min))
    if any("min_centre_to_box_m" in t for t in trials):
        out["min_centre_to_box_m"] = stats("min_centre_to_box_m", worst=min)
    if any("anees" in t for t in trials):
        out["anees"] = stats("anees")
        out["mean_nis"] = stats("mean_nis")
        out["nees_inside_95"] = stats("nees_inside_95", worst=min)
        out["nis_inside_95"] = stats("nis_inside_95", worst=min)
        out["anis"] = stats("anis")
        out["anees_inside_bounds"] = sum(t["anees_bounds"][0] <= t["anees"] <= t["anees_bounds"][1]
                                         for t in trials if "anees_bounds" in t)
    return out


def plot_trials(results, est="odometry"):
    """Small multiples: one dot per trial, per case; the black bar is the mean."""
    panels = [("dock_lateral_err_cm", "Docking: sideways error [cm]"),
              ("dock_heading_err_deg", "Docking: heading error [deg]"),
              ("time_s", "Mission time [s]"),
              ("min_centre_to_box_m", "Closest robot centre to the box [m]")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    rng = np.random.default_rng(0)                             # jitter only, so dots don't stack
    cases = [c for c in CASES if results.get(c)]                # only the cases that were run
    for ax, (key, title) in zip(axes.ravel(), panels):
        for i, case in enumerate(cases):
            trials = [t for t in results.get(case, []) if key in t]
            if not trials:
                continue
            x = np.array([t[key] for t in trials], float)
            ok = np.array([t["success"] for t in trials])
            jit = i + rng.uniform(-0.15, 0.15, x.size)
            ax.plot(jit[ok], x[ok], "o", ms=5, color=COLOR[case], alpha=0.8)
            if key.startswith("dock_"):                        # a failed dock has no final error to show
                x = x[ok]
            elif np.any(~ok):                                  # failed runs: hollow, so they stand out
                ax.plot(jit[~ok], x[~ok], "o", ms=7, mfc="none", mec=COLOR[case], mew=1.5)
            if x.size:
                ax.plot([i - 0.25, i + 0.25], [x.mean()] * 2, color="#222222", lw=1.5)
        ax.set_xticks(range(len(cases)), cases)
        ax.set_title(title, fontsize=10)
        if key.startswith("dock_"):
            ax.axhline(0, color="#999999", lw=0.6)
        ax.grid(axis="y", color="#eeeeee")
    n = max(len(v) for v in results.values())
    rates = ", ".join(f"{c} {100 * np.mean([t['success'] for t in results[c]]):.0f}%" for c in results)
    fig.suptitle(f"Repeated trials on {est}: {n} random runs per case (start pose, encoder scale, sensor noise)\n"
                 f"docked: {rates}   (filled = docked, hollow = failed run, bar = mean)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / f"demo_trials_{est}.png", dpi=140)
    plt.close(fig)


def run_trials(cases, n, est="odometry"):
    from multiprocessing import Pool
    jobs = [(c, s, est) for c in cases for s in range(1, n + 1)]
    results = {c: [] for c in cases}
    with Pool() as pool:                                       # one trial per CPU core at a time
        for case, m in pool.imap_unordered(one_trial, jobs):
            results[case].append(m)
            print(f"  {case:7s} seed {m['seed']:3d}: {'docked' if m['success'] else 'FAILED ' + str(m['stop_reason'])}")
    for c in cases:
        results[c].sort(key=lambda t: t["seed"])
    summary = {c: summarise(results[c]) for c in cases}
    OUT.mkdir(exist_ok=True)
    (OUT / f"demo_trials_{est}.json").write_text(json.dumps(dict(summary=summary, trials=results), indent=2))
    plot_trials(results, est)
    print()
    for c, s in summary.items():
        lat, hd, t = s["dock_lateral_err_cm"], s["dock_heading_err_deg"], s["time_s"]
        print(f"{c:7s} success {100 * s['success_rate']:.0f}% ({s['n']} trials), collisions {s['collisions']}, "
              f"time {t['mean']} +- {t['std']} s, dock lateral {lat['mean']} +- {lat['std']} cm (worst {lat['worst']}), "
              f"heading {hd['mean']} +- {hd['std']} deg (worst {hd['worst']})")
        if s.get("nees_inside_95"):                            # EKF: chi-square 95 % consistency over the trials
            ne, ni = s["nees_inside_95"], s["nis_inside_95"] or dict(mean=0, worst=0)
            print(f"        chi2 95%: NEES inside {100 * ne['mean']:.0f}% (worst {100 * ne['worst']:.0f}%), "
                  f"NIS inside {100 * ni['mean']:.0f}% (worst {100 * ni['worst']:.0f}%), "
                  f"ANEES {s['anees']['mean']} and ANIS {(s['anis'] or {}).get('mean')} (target about 3 and 2), "
                  f"ANEES inside its band in {s['anees_inside_bounds']}/{s['n']} trials")
    print(f"\nsaved results/demo_trials_{est}.json, demo_trials_{est}.png")
    if all((OUT / f"demo_trials_{e}.json").exists() for e in ESTS):
        plot_compare()


# ---------------------------------------------------------------- odometry vs EKF
def plot_compare():
    """The same random trials driven on odometry and on the EKF, side by side (demo_compare.png)."""
    data = {e: json.loads((OUT / f"demo_trials_{e}.json").read_text()) for e in ESTS}
    panels = [("success", "Docked [% of trials]"),
              ("estimate_err_final_cm", "Estimate error at the end [cm]"),
              ("dock_lateral_err_cm", "Docking: sideways error [cm]"),
              ("dock_heading_err_deg", "Docking: heading error [deg]")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    rng = np.random.default_rng(0)
    table = {}
    cases = [c for c in CASES if data["odometry"]["trials"].get(c) and data["ekf"]["trials"].get(c)]
    for ax, (key, title) in zip(axes.ravel(), panels):
        for j, est in enumerate(ESTS):
            for i, case in enumerate(cases):
                trials = data[est]["trials"].get(case, [])
                if not trials:
                    continue
                xpos = i + (j - 0.5) * 0.36                   # odometry left, EKF right of each case
                if key == "success":
                    rate = 100 * np.mean([t["success"] for t in trials])
                    table.setdefault(case, {})[est] = rate
                    ax.bar(xpos, rate, width=0.32, color=EST_COLOR[est], label=est if i == 0 else None)
                    continue
                vals = [t.get(key, t.get("odometry_err_final_cm") if key == "estimate_err_final_cm" else None)
                        for t in trials if t["success"] or not key.startswith("dock_")]
                x = np.array([v for v in vals if v is not None], float)
                if x.size:
                    ax.plot(xpos + rng.uniform(-0.1, 0.1, x.size), x, "o", ms=4, color=EST_COLOR[est], alpha=0.8,
                            label=est if i == 0 else None)
                    ax.plot([xpos - 0.14, xpos + 0.14], [np.mean(np.abs(x)) if key.startswith("dock_") else x.mean()] * 2,
                            color="#222222", lw=1.2)
        ax.set_xticks(range(len(cases)), cases)
        ax.set_title(title, fontsize=10)
        if key.startswith("dock_"):
            ax.axhline(0, color="#999999", lw=0.6)
        if key == "success":
            ax.set_ylim(0, 105)
        ax.grid(axis="y", color="#eeeeee")
        if key == "success":                                   # one legend for the whole figure
            ax.legend(fontsize=8, loc="lower right")
    n = max(len(v) for v in data["odometry"]["trials"].values())
    fig.suptitle(f"Odometry vs EKF: the same {n} random trials per case (dots = trials; bar = mean, or mean |error| for docking)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "demo_compare.png", dpi=140)
    plt.close(fig)
    summary = {e: data[e]["summary"] for e in ESTS}
    (OUT / "demo_compare.json").write_text(json.dumps(summary, indent=2))
    print("docked [%]:", {c: {e: round(r) for e, r in v.items()} for c, v in table.items()})
    print("saved results/demo_compare.png, demo_compare.json")


def run_single(cases, est):
    """Each case once (seed 0): detail plots, the paths overview and demo_metrics[_ekf].json."""
    logs, all_m = {}, {}
    for case in cases:
        info, t_end, log, contacts = run_case(case, est=est)
        m = metrics(case, info, t_end, log, contacts)
        logs[case], all_m[case] = log, m
        if case in DETAIL_CASES:
            plot_case(case, log, m, est)
        print(f"{case:7s} docked {m['success']}  {m['time_s']:5.1f} s  cross-track RMS {m['cross_track_rms_cm']} cm  "
              f"dock lateral {m['dock_lateral_err_cm']:+.1f} cm, heading {m['dock_heading_err_deg']:+.1f} deg  "
              f"collisions {m['collisions'] or 'none'}" + (f"  ANEES {m['anees']} (95 % {m['anees_bounds']}), NEES inside {100 * m['nees_inside_95']:.0f} %,"
                 f" ANIS {m.get('anis')}, NIS inside {100 * m.get('nis_inside_95', 0):.0f} %" if "anees" in m else ""))
    plot_overview(logs, est)
    (OUT / f"demo_metrics{sfx(est)}.json").write_text(json.dumps(all_m, indent=2))
    print(f"\nsaved results/demo_paths{sfx(est)}.png, demo_<case>{sfx(est)}.png, demo_metrics{sfx(est)}.json")


def run_all():
    """Everything the report needs, in one go (the Run button)."""
    steps = [("4 cases on odometry", lambda: run_single(CASES, "odometry")),
             ("4 cases on the EKF", lambda: run_single(CASES, "ekf")),
             (f"{ALL_TRIALS} random trials of none + centre on odometry", lambda: run_trials(DETAIL_CASES, ALL_TRIALS, "odometry")),
             (f"{ALL_TRIALS} random trials of none + centre on the EKF, then the comparison",
              lambda: run_trials(DETAIL_CASES, ALL_TRIALS, "ekf"))]
    for k, (name, step) in enumerate(steps, 1):
        print(f"\n===== step {k}/{len(steps)}: {name}")
        step()
    print("\nall done: see results/ (demo_compare.png is the odometry vs EKF summary)")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:                                               # Run button: everything
        run_all()
        sys.exit(0)
    est = "ekf" if "--ekf" in args else "odometry"
    if "--compare" in args:
        plot_compare()
        sys.exit(0)
    if "--trials" in args:
        n = int(args[args.index("--trials") + 1])
        run_trials([a for a in args if a in CASES] or CASES, n, est)
        sys.exit(0)
    run_single([a for a in args if a in CASES] or CASES, est)
