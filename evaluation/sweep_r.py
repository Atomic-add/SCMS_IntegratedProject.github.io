"""Consistency vs responsiveness: scale the EKF's measurement noise R and watch NEES, NIS and the error.

    python -m evaluation.sweep_r                     # R x10, x1, /10 and the measured bearing sd, 20 seeds each
    python -m evaluation.sweep_r --n 5 --case none   # quicker

Every run is the whole mission on the EKF (case centre by default), steering on the estimate. Only this
file (in evaluation/) reads the simulator truth, to compute NEES. Writes results/sweep_r.json and
results/sweep_r.png.

What to expect (Week 7 consistency): R too small -> P too small -> NEES above its band, and the NIS gate
starts rejecting good measurements; R too large -> NEES and NIS below their bands, the filter leans on
odometry. Note that the simulated odometry uses the same wheel radius as the simulator, so leaning on it
costs little here; on the real robot it will cost more.
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

from evaluation import DEFAULT_WORKERS                          # noqa: E402
from evaluation.ground_truth import true_pose                   # noqa: E402
from src.control.behaviour import run_mission                   # noqa: E402
from src.estimation.ekf import EKF                              # noqa: E402
from src.estimation.models import load_estimation_config        # noqa: E402
from src.estimation.odometry import wrap                        # noqa: E402
from src.robot.task import load_task                        # noqa: E402
from src.robot.sim_robot import PROJECT_ROOT                    # noqa: E402

OUT = PROJECT_ROOT / "results"
# (label, scale on the R variances, bearing sd override [deg] or None)
CONDITIONS = [("R x10", 10.0, None), ("R (config)", 1.0, None), ("R, measured bearing sd", 1.0, 0.02),
              ("R / 10", 0.1, None)]


def one_run(job):
    """One mission on the EKF with R scaled; returns the run's consistency and error numbers."""
    label, scale, bearing, case, seed = job
    task = load_task()
    cfg = load_estimation_config()
    cfg["range_sd"] = [a * np.sqrt(scale) for a in cfg["range_sd"]]            # sd scales with sqrt(variance)
    cfg["bearing_sd_deg"] = bearing if bearing is not None else cfg["bearing_sd_deg"] * np.sqrt(scale)
    ekf = EKF(task["start"], marker_map=task["markers"], params=cfg)
    err, nees = [], []

    def on_step(robot, row):
        e = np.asarray(true_pose(robot), float) - row["xh"]
        e[2] = wrap(e[2])
        err.append(np.hypot(e[0], e[1]))
        nees.append(float(e @ np.linalg.solve(row["P"], e)))

    info, t_end = run_mission(case, seed=seed, verbose=False, estimator=ekf, on_step=on_step)
    nis = np.array([n for _, _, n, _ in ekf.log])
    used = np.array([u for *_, u in ekf.log], bool)
    return dict(label=label, seed=seed, docked=bool(info["dock_complete"]), time_s=round(float(t_end), 1),
                err_max_cm=100 * float(np.max(err)), err_rms_cm=100 * float(np.sqrt(np.mean(np.square(err)))),
                anees=float(np.mean(nees)), anis=float(nis.mean()) if nis.size else float("nan"),
                gated=int((~used).sum()), updates=int(nis.size))


def summarise(runs):
    """mean +- std per condition, plus the success rate."""
    out = {}
    for label, *_ in CONDITIONS:
        rs = [r for r in runs if r["label"] == label]
        if not rs:
            continue
        s = dict(n=len(rs), success_rate=float(np.mean([r["docked"] for r in rs])))
        for k in ("err_max_cm", "err_rms_cm", "anees", "anis", "gated"):
            v = np.array([r[k] for r in rs], float)
            s[k] = dict(mean=round(float(v.mean()), 3), std=round(float(v.std()), 3))
        out[label] = s
    return out


def plot(summary, path):
    labels = list(summary)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for a, key, title, ideal in [(ax[0], "anees", "ANEES (ideal 3)", 3.0), (ax[1], "anis", "mean NIS (ideal 2)", 2.0),
                                 (ax[2], "err_max_cm", "worst |x̂ − true| [cm]", None)]:
        m = [summary[l][key]["mean"] for l in labels]
        s = [summary[l][key]["std"] for l in labels]
        a.errorbar(range(len(labels)), m, yerr=s, fmt="o", color="#2a78d6", capsize=4)
        if ideal is not None:
            a.axhline(ideal, color="#555555", ls="--", lw=1)
        a.set_xticks(range(len(labels)), labels, rotation=20, ha="right", fontsize=8)
        a.set_title(title, fontsize=10)
    fig.suptitle(f"EKF measurement noise R scaled (mean ± std over {summary[labels[0]]['n']} seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=20, help="seeds per condition (default 20)")
    ap.add_argument("--case", default="centre", choices=("none", "left", "centre", "right"))
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help=f"sim runs in parallel (default {DEFAULT_WORKERS}; each renders the camera)")
    args = ap.parse_args()
    jobs = [(lab, sc, b, args.case, seed) for lab, sc, b in CONDITIONS for seed in range(1, args.n + 1)]
    with Pool(args.workers) as pool:
        runs = []
        for r in pool.imap_unordered(one_run, jobs):
            runs.append(r)
            print(f"  {r['label']:24s} seed {r['seed']:3d}: ANEES {r['anees']:6.2f}  NIS {r['anis']:5.2f}  "
                  f"gated {r['gated']:3d}  worst error {r['err_max_cm']:.2f} cm  {'docked' if r['docked'] else 'NOT docked'}")
    summary = summarise(runs)
    OUT.mkdir(exist_ok=True)
    (OUT / "sweep_r.json").write_text(json.dumps(dict(case=args.case, summary=summary, runs=runs), indent=2))
    plot(summary, OUT / "sweep_r.png")
    print()
    for label, s in summary.items():
        print(f"{label:24s} n {s['n']}  docked {100 * s['success_rate']:.0f} %  ANEES {s['anees']['mean']:.2f} ± "
              f"{s['anees']['std']:.2f}  NIS {s['anis']['mean']:.2f}  gated {s['gated']['mean']:.1f}  "
              f"worst error {s['err_max_cm']['mean']:.2f} ± {s['err_max_cm']['std']:.2f} cm")
    print("\nsaved results/sweep_r.json and results/sweep_r.png")
