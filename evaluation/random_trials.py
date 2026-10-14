"""Repeated randomised trials: N scenarios from src/robot/scenario.py, judged against simulator truth.

    python -m evaluation.random_trials                 # 20 scenarios on the EKF (seeds 1..20)
    python -m evaluation.random_trials --n 20 --odometry   # the same 20 on wheel odometry only
    python -m evaluation.random_trials --seeds 3 7 11  # just these seeds
    python -m evaluation.random_trials --leg1-box      # stress test: a second box in the narrow leg 1

Each trial: a different route and start spot (told to the robot), and a different placement error,
obstacle layout, sensor noise and wheel error (not told). Success = dock_complete AND no contact with a
wall, a box or the dock body AND the true final pose within the dock tolerances below. Truth is read
here only. Writes results/random_trials[_odometry].json and .png (every true path on one map).
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root, so "src" imports work

import matplotlib                                               # noqa: E402
import numpy as np                                              # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402

from evaluation.ground_truth import true_pose                   # noqa: E402
from src.control.behaviour import run_mission                   # noqa: E402
from src.estimation.ekf import EKF                              # noqa: E402
from src.estimation.odometry import wrap                        # noqa: E402
from src.robot.corridor import load_config                      # noqa: E402
from src.robot.scenario import apply_world, random_scenario     # noqa: E402
from src.robot.sim_robot import PROJECT_ROOT                    # noqa: E402

OUT = PROJECT_ROOT / "results"
CFG = load_config()
DOCK_POS_TOL, DOCK_YAW_TOL_DEG = 0.03, 5.0     # true final pose vs goal_pose: same tolerances as docking.py


def contacts(robot):
    """Names of wall / box / dock geoms the robot touches right now (floor ignored)."""
    hit = set()
    for i in range(robot.d.ncon):
        c = robot.d.contact[i]
        names = [robot.m.geom(g).name for g in (c.geom1, c.geom2)]
        world = [n for n in names if n.startswith(("wall", "obstacle")) or n == "dock_body"]
        if world and "floor" not in names:
            hit.update(world)
    return hit


def one_trial(job):
    seed, use_ekf, leg1_box = job
    sc = random_scenario(seed, leg1_box=leg1_box)
    task, world = sc["task"], sc["world"]
    est = EKF(task["start"], marker_map=task["markers"]) if use_ekf else None
    rows = []

    def on_step(robot, row):
        rows.append((np.asarray(true_pose(robot), float), row["xh"], row["P"], contacts(robot)))

    info, t_end = run_mission("none", seed=seed, verbose=False, estimator=est, on_step=on_step,
                              setup=lambda robot: apply_world(robot, world), task=task)
    truth = np.array([r[0] for r in rows])
    xh = np.array([r[1] for r in rows])
    hits = sorted(set().union(*[r[3] for r in rows]))
    e = truth - xh
    e[:, 2] = wrap(e[:, 2])
    nees = np.array([ei @ np.linalg.solve(r[2], ei) for ei, r in zip(e, rows)])
    gx, gy, gth = CFG["dock"]["goal_pose"]
    x, y, th = truth[-1]
    lat, along, yaw = 100 * (x - gx), 100 * (y - gy), float(np.degrees(wrap(th - gth)))
    in_tol = np.hypot(lat, along) <= 100 * DOCK_POS_TOL and abs(yaw) <= DOCK_YAW_TOL_DEG
    return dict(seed=seed, success=bool(info["dock_complete"] and not hits and in_tol),
                dock_complete=bool(info["dock_complete"]), stop_reason=info["stop_reason"], time_s=round(float(t_end), 1),
                collisions=hits, dock_lateral_cm=round(lat, 2), dock_along_cm=round(along, 2), dock_yaw_deg=round(yaw, 2),
                est_err_max_cm=round(100 * float(np.hypot(*e[:, :2].T).max()), 2),
                est_err_final_cm=round(100 * float(np.hypot(*e[-1, :2])), 2),
                anees=round(float(nees.mean()), 2) if use_ekf else None,
                n_boxes=len(world["obstacles"]), placement_cm=round(100 * float(np.hypot(*world["placement_error"][:2])), 1),
                path=truth[::10, :2].round(3).tolist(), boxes=world["obstacles"],
                waypoints=np.vstack([task["start"][:2], task["waypoints"]]).round(3).tolist())


def stats(trials, key):
    """mean, std and worst of one metric. Dock errors only over the runs that docked: a run that stopped
    safely half way has no dock error, and its distance to the goal would swamp the statistic."""
    if key.startswith("dock_"):
        trials = [t for t in trials if t["dock_complete"]]
    v = np.array([t[key] for t in trials if t[key] is not None], float)
    return dict(mean=round(float(v.mean()), 2), std=round(float(v.std()), 2), worst=round(float(v[np.argmax(np.abs(v))]), 2)) if v.size else None


def plot(trials, path, title):
    fig, ax = plt.subplots(figsize=(6.2, 6.4))
    outline = np.array(CFG["outline"] + [CFG["outline"][0]], float)
    ax.plot(*outline.T, color="#555555", lw=1.2)
    for t in trials:
        for bx, by in t["boxes"]:
            ax.add_patch(plt.Rectangle((bx - 0.15, by - 0.15), 0.3, 0.3, fc="#bbbbbb", ec="none", alpha=0.25))
    for t in trials:
        p = np.array(t["path"])
        ax.plot(*p.T, lw=1.0, color="#2a78d6" if t["success"] else "#d64545", alpha=0.75)
    ax.plot([], [], color="#2a78d6", label="true path, success")
    ax.plot([], [], color="#d64545", label="true path, failure")
    ax.add_patch(plt.Rectangle((3.4, 4.15), 0.4, 0.4, color="#9fd3a8", alpha=0.6, lw=0, label="dock pad"))
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=20, help="number of scenarios, seeds 1..n (default 20)")
    ap.add_argument("--seeds", type=int, nargs="*", help="run exactly these seeds instead")
    ap.add_argument("--odometry", action="store_true", help="steer on wheel odometry only (baseline)")
    ap.add_argument("--leg1-box", action="store_true", help="stress test: also a box in the 1.0 m wide leg 1")
    ap.add_argument("--from-json", help="re-summarise and re-plot a saved results/random_trials*.json")
    args = ap.parse_args()
    seeds = args.seeds or list(range(1, args.n + 1))
    trials = []
    if args.from_json:                                         # re-summarise saved trials, no simulation
        trials = json.loads(Path(args.from_json).read_text())["trials"]
        seeds = []
    with Pool() as pool:
        for t in pool.imap_unordered(one_trial, [(s, not args.odometry, args.leg1_box) for s in seeds]):
            trials.append(t)
            print(f"  seed {t['seed']:3d}: {'OK    ' if t['success'] else 'FAILED'} {t['stop_reason']!s:28s} "
                  f"{t['time_s']:5.1f} s  dock {t['dock_lateral_cm']:+.1f} / {t['dock_along_cm']:+.1f} cm "
                  f"{t['dock_yaw_deg']:+.1f} deg  boxes {t['n_boxes']}  contacts {t['collisions'] or 'none'}  "
                  f"est err max {t['est_err_max_cm']:.1f} cm" + (f"  ANEES {t['anees']}" if t["anees"] is not None else ""))
    trials.sort(key=lambda t: t["seed"])
    est = "odometry" if args.odometry else "ekf"
    summary = dict(estimator=est, n=len(trials), success_rate=round(float(np.mean([t["success"] for t in trials])), 3),
                   collisions=sum(bool(t["collisions"]) for t in trials),
                   failures=[(t["seed"], t["stop_reason"], t["collisions"]) for t in trials if not t["success"]],
                   **{k: stats(trials, k) for k in ("time_s", "dock_lateral_cm", "dock_along_cm", "dock_yaw_deg",
                                                    "est_err_max_cm", "est_err_final_cm", "anees")})
    sfx = ("" if est == "ekf" else "_odometry") + ("_leg1box" if args.leg1_box else "")
    OUT.mkdir(exist_ok=True)
    (OUT / f"random_trials{sfx}.json").write_text(json.dumps(dict(summary=summary, trials=trials), indent=1))
    plot(trials, OUT / f"random_trials{sfx}.png",
         f"{len(trials)} random scenarios on the {est.upper() if est == 'ekf' else est}: "
         f"{100 * summary['success_rate']:.0f} % success, {summary['collisions']} with contact")
    print(f"\n{est}: success {100 * summary['success_rate']:.0f} % of {len(trials)}, contacts in {summary['collisions']}, "
          f"time {summary['time_s']['mean']} ± {summary['time_s']['std']} s, dock lateral "
          f"{summary['dock_lateral_cm']['mean']} ± {summary['dock_lateral_cm']['std']} cm, yaw "
          f"{summary['dock_yaw_deg']['mean']} ± {summary['dock_yaw_deg']['std']} deg")
    print(f"saved results/random_trials{sfx}.json and .png")
