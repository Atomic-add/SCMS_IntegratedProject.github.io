"""IBVS versus PBVS docking in the simulator (Week 9 Parts B and D, on the Waffle Pi).

    python -m evaluation.check_ibvs              # 10 seeds per docking mode
    python -m evaluation.check_ibvs --n 20

Each trial starts 0.35 m before the pre-dock point W4, on the approach line as far as the robot knows,
but with a random TRUE placement error it is not told about: sideways N(0, 2 cm), heading N(0, 3 deg).
It drives to W4 on the EKF and docks. The same seeds run with docking servo "pbvs" (the LQR on the
camera position all the way) and "ibvs" (the LQR onto the marker normal, then image-based visual
servoing on the four corners). Truth is read here only, for the final dock error.

What the robot's IBVS sees is re-measured here from the same camera frames (corners -> undistorted
pixels -> error to p*), so the figure shows the image-plane behaviour Week 9 B1 plots: corners moving
to p*, the error decaying, and the commands. Writes results/ibvs_docking.png and results/ibvs_docking.json.
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
from src.control import visual_servo as vs                      # noqa: E402
from src.control.behaviour import run_mission                   # noqa: E402
from src.control.models import load_control_config              # noqa: E402
from src.estimation.ekf import EKF                              # noqa: E402
from src.estimation.odometry import wrap                        # noqa: E402
from src.perception import aruco, camera                        # noqa: E402
from src.robot.corridor import load_task                        # noqa: E402
from src.robot.sim_robot import PROJECT_ROOT                    # noqa: E402

OUT = PROJECT_ROOT / "results"
START = np.array([3.6, 3.55, np.pi / 2])        # 0.35 m before W4 (3.6, 3.9), on the approach line
LATERAL_SD, HEADING_SD_DEG = 0.02, 3.0          # true placement error, not told to the robot
MODES = ("pbvs", "ibvs")
COL = dict(pbvs="#2a78d6", ibvs="#eb6834")


def one_trial(job):
    mode, seed = job
    rng = np.random.default_rng(2000 + seed)
    dx, dth = rng.normal(0.0, LATERAL_SD), np.deg2rad(rng.normal(0.0, HEADING_SD_DEG))
    task = load_task()
    task["start"], task["waypoints"] = START.copy(), np.array([task["waypoints"][-1]])
    cfg = load_control_config()
    cfg["docking"]["servo"] = mode
    K, dist, _ = camera.load_intrinsics(PROJECT_ROOT / cfg["docking"]["camera_intrinsics"])
    p_star = vs.desired_features(K, task["dock"]["stop_range_base"], task["markers"][0][2])
    rows, img = [], {"t": None}

    def on_step(robot, row):
        rec = dict(t=row["t"], state=row["state"], v=row["v"], w=row["w"], truth=np.asarray(true_pose(robot), float))
        t_frame, rgb = robot.read_image()                    # the frame the robot used (no new render)
        if row["state"] == "IBVS" and t_frame != img["t"]:
            img["t"] = t_frame
            c = aruco.detect(rgb, marker_id=0)
            if c is not None:
                rec["uv"] = vs.undistort(c, K, dist)
        rows.append(rec)

    def setup(robot):
        robot.reset(START[0] + dx, START[1], START[2] + dth)  # where it really is (sim side only)

    est = EKF(task["start"], marker_map=task["markers"])
    info, t_end = run_mission("none", seed=seed, verbose=False, estimator=est, on_step=on_step, setup=setup,
                              task=task, cfg=cfg, t_max=60.0)
    gx, gy, gth = task["dock"]["goal_pose"]
    x, y, th = rows[-1]["truth"]
    uv = [r["uv"] for r in rows if "uv" in r]
    t_uv = [r["t"] for r in rows if "uv" in r]
    t_dock = next((r["t"] for r in rows if r["state"] in ("DOCK", "IBVS")), np.nan)
    return dict(mode=mode, seed=seed, docked=bool(info["dock_complete"]), stop_reason=info["stop_reason"],
                start_lateral_cm=round(100 * dx, 2), start_heading_deg=round(float(np.degrees(dth)), 2),
                dock_lateral_cm=round(100 * (x - gx), 2), dock_along_cm=round(100 * (y - gy), 2),
                dock_yaw_deg=round(float(np.degrees(wrap(th - gth))), 2),
                docking_time_s=round(float(t_end - t_dock), 2),
                ibvs_steps=len(uv),
                e_rms=[float(np.sqrt(np.mean((p_star - u) ** 2))) for u in uv], t_e=t_uv,
                uv=[u.round(2).tolist() for u in uv], p_star=p_star.round(2).tolist(),
                t=[r["t"] for r in rows], v=[r["v"] for r in rows], w=[r["w"] for r in rows],
                state=[r["state"] for r in rows])


def summary(trials):
    out = {}
    for mode in MODES:
        ts = [t for t in trials if t["mode"] == mode]
        s = dict(n=len(ts), success_rate=round(float(np.mean([t["docked"] for t in ts])), 3))
        for k in ("dock_lateral_cm", "dock_along_cm", "dock_yaw_deg", "docking_time_s"):
            v = np.array([t[k] for t in ts], float)
            s[k] = dict(mean=round(float(v.mean()), 2), std=round(float(v.std()), 2),
                        worst=round(float(v[np.argmax(np.abs(v))]), 2))
        out[mode] = s
    return out


def plot(trials, path):
    ex = next(t for t in sorted(trials, key=lambda t: t["seed"]) if t["mode"] == "ibvs" and t["ibvs_steps"] > 5)
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    ps, uv = np.array(ex["p_star"]), np.array(ex["uv"])
    a = ax[0, 0]                                                       # (a) image plane, as Week 9 B1
    for i in range(4):
        a.plot(uv[:, 0, i], uv[:, 1, i], "-", color=f"C{i}", lw=1.5)
        a.plot(*uv[0, :, i], "o", color=f"C{i}", mfc="none", ms=8)
        a.plot(*ps[:, i], "*", color=f"C{i}", ms=14)
    a.plot(*np.c_[ps, ps[:, :1]], "k:", lw=1, label="desired p*")
    a.set_xlim(0, 640); a.set_ylim(480, 0); a.set_aspect("equal")
    a.set_xlabel("u [px]"); a.set_ylabel("v [px]"); a.legend(loc="lower right", fontsize=8)
    a.set_title(f"(a) marker corners during IBVS, seed {ex['seed']} (o start, * p*)", fontsize=10)
    a = ax[0, 1]                                                       # (b) feature error
    t0 = ex["t_e"][0]
    a.semilogy(np.array(ex["t_e"]) - t0, ex["e_rms"], color=COL["ibvs"])
    a.set_xlabel("time since IBVS started [s]"); a.set_ylabel("feature error [px rms]")
    a.set_title("(b) image error decays roughly as exp(-lambda t)", fontsize=10); a.grid(alpha=0.3)
    a = ax[1, 0]                                                       # (c) commands while docking
    t, st = np.array(ex["t"]), np.array(ex["state"])
    on = np.isin(st, ["DOCK", "IBVS"])
    a.plot(t[on] - t[on][0], np.array(ex["v"])[on], label="v [m/s]", color="#2a78d6")
    a.plot(t[on] - t[on][0], np.array(ex["w"])[on], label="omega [rad/s]", color="#eb6834")
    ib = st[on] == "IBVS"
    if np.any(ib):
        a.axvspan((t[on] - t[on][0])[ib][0], (t[on] - t[on][0])[ib][-1], color="#bfe3c6", lw=0, label="IBVS")
    a.set_xlabel("time since docking started [s]"); a.legend(fontsize=8)
    a.set_title("(c) commands: LQR onto the normal, then IBVS (saturated at v_dock)", fontsize=10)
    a = ax[1, 1]                                                       # (d) final dock errors, both modes
    a.add_patch(plt.Rectangle((-3, -5), 6, 10, fc="#eeeeee", ec="none", label="dock tolerance"))
    for mode in MODES:
        ts = [t for t in trials if t["mode"] == mode]
        a.plot([t["dock_lateral_cm"] for t in ts], [t["dock_yaw_deg"] for t in ts], "o", color=COL[mode],
               label=f"{mode.upper()} ({sum(t['docked'] for t in ts)}/{len(ts)} docked)", alpha=0.8)
    a.set_xlabel("true lateral error at the dock [cm]"); a.set_ylabel("true heading error [deg]")
    a.set_title("(d) final pose, same seeds and start errors", fontsize=10); a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=10, help="seeds per docking mode (default 10)")
    args = ap.parse_args()
    jobs = [(m, s) for s in range(1, args.n + 1) for m in MODES]
    trials = []
    with Pool() as pool:
        for r in pool.imap_unordered(one_trial, jobs):
            trials.append(r)
            print(f"  {r['mode']} seed {r['seed']:3d}: {'docked' if r['docked'] else 'FAILED ' + str(r['stop_reason'])}  "
                  f"start {r['start_lateral_cm']:+.1f} cm {r['start_heading_deg']:+.1f} deg -> dock "
                  f"{r['dock_lateral_cm']:+.2f} / {r['dock_along_cm']:+.2f} cm {r['dock_yaw_deg']:+.2f} deg, "
                  f"{r['docking_time_s']:.1f} s" + (f", IBVS {r['ibvs_steps']} frames, error {r['e_rms'][0]:.1f} -> "
                                                    f"{r['e_rms'][-1]:.2f} px" if r["ibvs_steps"] else ""))
    trials.sort(key=lambda t: (t["mode"], t["seed"]))
    s = summary(trials)
    OUT.mkdir(exist_ok=True)
    (OUT / "ibvs_docking.json").write_text(json.dumps(dict(summary=s, trials=trials)))
    plot(trials, OUT / "ibvs_docking.png")
    print()
    for mode, m in s.items():
        print(f"{mode}: docked {100 * m['success_rate']:.0f} % of {m['n']}, lateral {m['dock_lateral_cm']['mean']:+.2f} ± "
              f"{m['dock_lateral_cm']['std']:.2f} cm, along {m['dock_along_cm']['mean']:+.2f} ± {m['dock_along_cm']['std']:.2f} cm, "
              f"heading {m['dock_yaw_deg']['mean']:+.2f} ± {m['dock_yaw_deg']['std']:.2f} deg, "
              f"time {m['docking_time_s']['mean']:.1f} ± {m['docking_time_s']['std']:.1f} s")
    print("saved results/ibvs_docking.png and results/ibvs_docking.json")
