"""Real robot vs its simulation twin: the final docked pose, measured by tape on the lab floor and in the sim.

    python -m evaluation.real_vs_sim                 # one sim run per real run (middle 3, the others 1)
    python -m evaluation.real_vs_sim --n 5           # 5 seeds per condition, to see the sim spread
    python -m evaluation.real_vs_sim --view middle --seed 2   # watch one twin in the MuJoCo viewer (no files written)

The real runs (30 Oct, lab track, 20 cm box) are in results/hardware/final_pose_tape_30oct.csv, one row per
docked run: the box centre, how far the robot was turned off the start mark, and four tape readings at the end
(the two wheels, P = right, T = left; the robot faces the dock marker, i.e. -y):
    dP, dT  wheel to the marker wall (the plane y = 0 of marker 0)   [cm]
    sP, sT  wheel to the dock centre line x = 0.5, each on its own side [cm]
They give the pose of the wheel axle centre, whatever point of the tyre the tape touched (same on both sides):
    along  = (dP + dT) / 2                      distance to the marker wall (goal: stop_range_base 0.35 m)
    e_lat  = (sT - sP) / 2                      + = towards +x (the robot's left when it faces -y)
    e_yaw  = atan2(dT - dP, sP + sT)            + = turned to the robot's left (counter-clockwise)
The sim twin drives the same task file (config/arena_lab.json) with the box at the same centre, the true start
turned by the same angle (the controller is NOT told: it starts from the task prior, like the real robot), and
the real robot's control overrides (config/robot_real.json "control", merged the way main.py --real does). The
same three numbers come from the simulator truth at the end of the run. Every seed also has the small placement
and encoder scale errors of evaluation/run_demo.trial_setup, so the sim spread is not zero.

Truth is read only here (evaluation/ground_truth.py). Writes results/real_vs_sim.json and results/real_vs_sim.png.

Adapted from the team coding guide (hardware: "end pose by tape, compare with the sim twin on the same task
file") and evaluation/run_demo.py (trial_setup, docking error convention).
Additions: tape readings of both wheels turned into the axle pose, the twin with the real control overrides
and the hidden start rotation, sim vs real table per condition.
"""
import argparse
import copy
import csv
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root, so "src" imports work

import matplotlib                                               # noqa: E402
import numpy as np                                              # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402

from evaluation import DEFAULT_WORKERS                          # noqa: E402
from evaluation.ground_truth import true_pose                   # noqa: E402
from evaluation.random_trials import contacts                   # noqa: E402
from evaluation.run_demo import ENC_SCALE_SD, START_XY_SD, START_YAW_SD_DEG   # noqa: E402
from src.control.behaviour import run_mission                   # noqa: E402
from src.control.models import load_control_config              # noqa: E402
from src.estimation.ekf import EKF                              # noqa: E402
from src.estimation.odometry import wrap                        # noqa: E402
from src.robot.sim_robot import PROJECT_ROOT                    # noqa: E402
from src.robot.task import apply_control_overrides, load_task, place_obstacles   # noqa: E402

OUT = PROJECT_ROOT / "results"
TAPE = OUT / "hardware" / "final_pose_tape_30oct.csv"
REAL_CFG = PROJECT_ROOT / "config" / "robot_real.json"
COND_COLOR = dict(middle="#2a78d6", left="#1baf7a", right="#eda100", random="#eb6834")


def tape_pose(dP, dT, sP, sT):
    """Axle pose from the four tape readings [cm] -> (along [cm], e_lat [cm], e_yaw [deg])."""
    return (dP + dT) / 2, (sT - sP) / 2, float(np.degrees(np.arctan2(dT - dP, sP + sT)))


def read_tape(path=TAPE):
    """The real runs: one dict per row, with the axle pose added."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            num = {k: float(v) for k, v in r.items() if k not in ("name", "condition")}
            along, lat, yaw = tape_pose(num["dP_cm"], num["dT_cm"], num["sP_cm"], num["sT_cm"])
            rows.append(dict(name=r["name"], condition=r["condition"], box=[num["box_x"], num["box_y"]],
                             start_yaw_offset_deg=num["start_yaw_offset_deg"],
                             along_cm=round(along, 2), lat_cm=round(lat, 2), yaw_deg=round(yaw, 2)))
    return rows


def real_control_cfg():
    """config/control.json with the real robot's overrides on top (what main.py --real runs)."""
    real = json.loads(REAL_CFG.read_text())
    return apply_control_overrides(copy.deepcopy(load_control_config()), {"control": real["control"]})


def one_run(job, view=False):
    """One sim twin run: box, hidden start rotation, small placement / encoder errors; returns the end pose."""
    cond, box, yaw_off_deg, seed = job
    task = load_task()
    rng = np.random.default_rng(2000 + seed)
    dx, dy = rng.normal(0.0, START_XY_SD, 2)
    dth = np.deg2rad(yaw_off_deg + rng.normal(0.0, START_YAW_SD_DEG))
    scale = 1.0 + rng.normal(0.0, ENC_SCALE_SD, 2)
    rows = []

    def setup(robot):
        place_obstacles(robot, [box])
        x, y, th = task["start"]
        robot.reset(x + dx, y + dy, th + dth)                  # the controller still starts from task["start"]
        read = robot.read_encoders

        def read_scaled():
            t, phi_l, phi_r = read()
            return t, phi_l * scale[0], phi_r * scale[1]
        robot.read_encoders = read_scaled

    def on_step(robot, row):
        rows.append((np.asarray(true_pose(robot), float), contacts(robot)))

    est = EKF(task["start"], marker_map=task["markers"])
    info, t_end = run_mission("none", seed=seed, verbose=view, estimator=est, on_step=on_step, setup=setup,
                              task=task, cfg=real_control_cfg(), view=view, label=f"{cond} twin, seed {seed}")
    truth = np.array([r[0] for r in rows])
    gx, gy, gth = task["dock"]["goal_pose"]
    x, y, th = truth[-1]
    passing = (truth[:, 0] > box[0] - 0.4) & (truth[:, 0] < box[0] + 0.4)    # beside the box
    side_y = truth[passing, 1] if passing.any() else np.array([np.nan])
    return dict(condition=cond, seed=seed, docked=bool(info["dock_complete"]), stop_reason=info["stop_reason"],
                time_s=round(float(t_end), 1), collisions=sorted(set().union(*[r[1] for r in rows])),
                along_cm=round(100 * float(y), 2), lat_cm=round(100 * float(x - gx), 2),
                yaw_deg=round(float(np.degrees(wrap(th - gth))), 2),
                pass_y_m=round(float(side_y[np.argmax(np.abs(side_y - box[1]))]), 3))


def stats(v):
    v = np.asarray(v, float)
    return dict(mean=round(float(v.mean()), 2), std=round(float(v.std()), 2)) if v.size else None


def summarise(real, sims):
    out = {}
    for cond in dict.fromkeys(r["condition"] for r in real):
        rr = [r for r in real if r["condition"] == cond]
        ss = [s for s in sims if s["condition"] == cond]
        dk = [s for s in ss if s["docked"]]
        out[cond] = dict(
            box=rr[0]["box"], start_yaw_offset_deg=rr[0]["start_yaw_offset_deg"],
            real=dict(n=len(rr), **{k: stats([r[k] for r in rr]) for k in ("along_cm", "lat_cm", "yaw_deg")}),
            sim=dict(n=len(ss), docked=len(dk), stops={s["stop_reason"]: sum(t["stop_reason"] == s["stop_reason"] for t in ss) for s in ss},
                     **{k: stats([s[k] for s in dk]) for k in ("along_cm", "lat_cm", "yaw_deg", "time_s")}))
    return out


def plot(real, sims, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for cond, col in COND_COLOR.items():
        dk = [s for s in sims if s["condition"] == cond and s["docked"]]
        rr = [r for r in real if r["condition"] == cond]
        if not rr:
            continue
        ax[0].scatter([s["lat_cm"] for s in dk], [s["along_cm"] for s in dk], s=14, color=col, alpha=0.35)
        ax[0].scatter([r["lat_cm"] for r in rr], [r["along_cm"] for r in rr], s=70, color=col, marker="*",
                      edgecolor="k", lw=0.5, label=f"{cond}: real (star), sim (dots)")
        ax[1].scatter([s["lat_cm"] for s in dk], [s["yaw_deg"] for s in dk], s=14, color=col, alpha=0.35)
        ax[1].scatter([r["lat_cm"] for r in rr], [r["yaw_deg"] for r in rr], s=70, color=col, marker="*",
                      edgecolor="k", lw=0.5)
    ax[0].axhline(35.0, color="#555555", ls="--", lw=1)
    ax[0].set(xlabel="lateral error e_lat [cm] (+ = robot's left)", ylabel="axle to marker wall [cm]",
              title="Final position (goal: 0 cm, 35 cm)")
    ax[1].axhline(0.0, color="#555555", ls="--", lw=1)
    ax[1].set(xlabel="lateral error e_lat [cm]", ylabel="heading error [deg] (+ = turned left)",
              title="Final heading")
    ax[0].legend(fontsize=7, loc="best")
    fig.suptitle("Docked pose: real robot (tape, 30 Oct) vs sim twin (truth), same task, box and controller settings",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=None,
                    help="sim seeds per condition (default: as many as real runs of that condition)")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help=f"sim runs in parallel (default {DEFAULT_WORKERS}; each renders the camera)")
    ap.add_argument("--view", metavar="CONDITION", help="watch one twin (middle / left / right) in the viewer")
    ap.add_argument("--seed", type=int, default=1, help="seed for --view (default 1)")
    args = ap.parse_args()
    real = read_tape()
    if args.view:
        r = next(r for r in real if r["condition"] == args.view)
        s = one_run((args.view, r["box"], r["start_yaw_offset_deg"], args.seed), view=True)
        print(f"{s['stop_reason']} after {s['time_s']} s: along {s['along_cm']} cm, lat {s['lat_cm']:+} cm, "
              f"yaw {s['yaw_deg']:+} deg (real {args.view}: along {r['along_cm']} cm, lat {r['lat_cm']:+} cm, "
              f"yaw {r['yaw_deg']:+} deg)")
        sys.exit(0)
    conds = {r["condition"]: (r["box"], r["start_yaw_offset_deg"]) for r in real}
    n_real = {c: sum(r["condition"] == c for r in real) for c in conds}
    jobs = [(c, box, yaw, seed) for c, (box, yaw) in conds.items() for seed in range(1, (args.n or n_real[c]) + 1)]
    sims = []
    with Pool(args.workers) as pool:
        for s in pool.imap_unordered(one_run, jobs):
            sims.append(s)
            print(f"  {s['condition']:7s} seed {s['seed']:3d}: {s['stop_reason']:12s} {s['time_s']:5.1f} s  "
                  f"along {s['along_cm']:6.1f} cm  lat {s['lat_cm']:+5.1f} cm  yaw {s['yaw_deg']:+5.1f} deg  "
                  f"passed the box at y {s['pass_y_m']:.2f} m  contacts {s['collisions'] or 'none'}")
    sims.sort(key=lambda s: (s["condition"], s["seed"]))
    summary = summarise(real, sims)
    OUT.mkdir(exist_ok=True)
    (OUT / "real_vs_sim.json").write_text(json.dumps(dict(tape=str(TAPE.relative_to(PROJECT_ROOT)), seeds={c: list(range(1, (args.n or n_real[c]) + 1)) for c in conds},
                                                          summary=summary, real=real, sim=sims), indent=1))
    plot(real, sims, OUT / "real_vs_sim.png")
    print()
    f = lambda s: "-" if s is None else f"{s['mean']:+.1f} ± {s['std']:.1f}"   # noqa: E731
    for cond, s in summary.items():
        r, m = s["real"], s["sim"]
        print(f"{cond:7s} real n {r['n']}: along {f(r['along_cm'])} cm, lat {f(r['lat_cm'])} cm, yaw {f(r['yaw_deg'])} deg | "
              f"sim docked {m['docked']}/{m['n']}: along {f(m['along_cm'])} cm, lat {f(m['lat_cm'])} cm, yaw {f(m['yaw_deg'])} deg")
    print("\nsaved results/real_vs_sim.json and results/real_vs_sim.png")
