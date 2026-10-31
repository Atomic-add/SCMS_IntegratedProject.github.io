"""Control effort vs tracking error: scale the LQR's control weight R and run the mission on the EKF.

    python -m evaluation.sweep_lqr --n 20            # R x10, R (config), R / 10; 20 seeds each, case centre
    python -m evaluation.sweep_lqr --n 3             # quick check that it runs

Bryson's rule (Week 6 C.2) sets R = 1 / d_omega_max^2 with d_omega_max = 0.5 rad/s (config/control.json
"lqr"). The sweep keeps Q fixed and scales R over a factor of 100 in three steps:

    "R x10"     d_omega_max / sqrt(10)   gentler steering, slower correction
    "R (config)" d_omega_max             the design in the repo
    "R / 10"    d_omega_max * sqrt(10)   harder steering, faster correction

The weight is passed at run time (copy.deepcopy(load_control_config()) -> run_mission(cfg=...)); no value
in config/ changes. Every LQR in the mission is built from cfg["lqr"], so the sweep changes all of them:
waypoint tracking (waypoints.py, PathLQR at v_nom), avoidance (avoidance.py, PathLQR at v_nom) and the
docking approach (docking.py, design_lqr at v_dock). Turns on the spot (waypoints.py k_turn) and IBVS do
not use the LQR and do not change.

Each run: EKF from the config prior (task start, P0), case centre unless --case, seeds 1..n with
run_demo.trial_setup(seed) (random true start and encoder scale error, so there is an error to correct).
Metrics, from the TRUE pose (on_step), only in LQR-driven tracking steps (state TRACK and v > 0: the
TRACK state also holds the waypoint tracker's turns on the spot, where v = 0 and w comes from k_turn):
  cross-track RMS / max to run_demo.reference_path();  RMS w, peak |w|;  jitter = RMS of (w[k]-w[k-1])/Ts
  (a high gain amplifies the EKF's estimate noise into the command);  share of steps with |w| >= OMEGA_MAX.
Plus docked, stop_reason, time and contacts (random_trials.contacts).

Linear prediction per condition (no simulator, src/control/lqr.py): K, |closed-loop eig| at v_nom and at
0.05 m/s, and the response from e_y = 0.20 m (lqr._linear_response): 2 % settling distance, peak
d_omega, overshoot. src/control/lqr_weights_test.py checks these K against control.dlqr.

Why ground truth is read here: the tracking error must be measured against where the robot really is;
the controller still drives on the EKF estimate (run_mission), and the truth goes only into the metrics.

Adapted from 41014 Week 6 A.5 / C.2 (Bryson's rule, the trade-off curve over R, saturation) and Week 8
C.3 (steering on the estimate, RMS cross-track).
Additions: the sweep run on the whole mission with the EKF, effort and jitter from the commanded w, the
linear prediction next to the simulated result.
"""
import argparse
import copy
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
from evaluation.run_demo import dist_to_polyline, reference_path, trial_setup   # noqa: E402
from src.control.behaviour import TRACK, run_mission           # noqa: E402
from src.control.lqr import _linear_response, design_lqr, design_weights   # noqa: E402
from src.control.models import OMEGA_MAX, TS, V_NOM, error_model, load_control_config   # noqa: E402
from src.estimation.ekf import EKF                              # noqa: E402
from src.robot.task import load_task                            # noqa: E402
from src.robot.sim_robot import PROJECT_ROOT                    # noqa: E402

OUT = PROJECT_ROOT / "results"
CONDITIONS = [("R x10", 10.0), ("R (config)", 1.0), ("R / 10", 0.1)]   # (label, scale on R)
E0 = 0.20           # m, initial cross-track error of the linear response (as lqr.py's design report)
V_SLOW = 0.05       # m/s, a slow speed where the fixed gain is weakest (lqr.py's design report)


def scaled_cfg(scale):
    """Controller config with R = scale * R_config, i.e. d_omega_max / sqrt(scale); Q unchanged."""
    cfg = copy.deepcopy(load_control_config())
    cfg["lqr"]["d_omega_max"] = cfg["lqr"]["d_omega_max"] / np.sqrt(scale)
    return cfg


def linear_prediction(scale):
    """K, |eig| and the linear response from e_y = E0 for this weight (no simulator)."""
    cfg = scaled_cfg(scale)
    K, eig = design_lqr(cfg=cfg)
    Ad, Bd = error_model(V_SLOW, 0.0)
    t, E, U = _linear_response(K, V_NOM)
    outside = np.nonzero(np.abs(E[:, 0]) > 0.02 * E0)[0]
    t_settle = float(t[min(outside[-1] + 1, len(t) - 1)]) if len(outside) else 0.0
    Q, R = design_weights(cfg)
    return dict(d_omega_max=round(cfg["lqr"]["d_omega_max"], 4), R=round(float(R[0, 0]), 4),
                K=np.round(K.ravel(), 4).tolist(), eig_abs_vnom=np.round(np.abs(eig), 4).tolist(),
                eig_abs_slow=np.round(np.abs(np.linalg.eigvals(Ad - Bd @ K)), 4).tolist(),
                settle_s=round(t_settle, 2), settle_dist_m=round(V_NOM * t_settle, 2),
                peak_d_omega=round(float(np.abs(U).max()), 3),
                peak_e_theta_deg=round(float(np.rad2deg(np.abs(E[:, 1]).max())), 1),
                overshoot_cm=round(100 * max(0.0, -float(E[:, 0].min())), 2),
                _t=t.tolist(), _e_y=E[:, 0].tolist())


def one_run(job):
    """One mission on the EKF with the scaled R; returns tracking, effort and outcome numbers."""
    label, scale, case, seed = job
    task = load_task()
    ekf = EKF(task["start"], marker_map=task["markers"])
    ref = reference_path()
    rows = []

    def on_step(robot, row):
        rows.append((np.asarray(true_pose(robot), float)[:2], row["state"], row["v"], row["w"], contacts(robot)))

    info, t_end = run_mission(case, seed=seed, verbose=False, estimator=ekf, on_step=on_step,
                              setup=trial_setup(seed), cfg=scaled_cfg(scale))
    xy = np.array([r[0] for r in rows])
    state = np.array([r[1] for r in rows])
    v = np.array([r[2] for r in rows])
    w = np.array([r[3] for r in rows])
    lqr = (state == TRACK) & (v > 0.0)                       # LQR-driven tracking steps only
    cte = np.array([dist_to_polyline(p, ref) for p in xy[lqr]])
    w_l = w[lqr]
    dw = np.diff(w)[lqr[1:] & lqr[:-1]] / TS                 # consecutive tracking steps only
    hits = sorted(set().union(*[r[4] for r in rows]))
    return dict(label=label, scale=scale, seed=seed, docked=bool(info["dock_complete"]),
                stop_reason=info["stop_reason"], time_s=round(float(t_end), 1), collisions=hits,
                track_steps=int(lqr.sum()),
                cte_rms_cm=round(100 * float(np.sqrt(np.mean(cte**2))), 3),
                cte_max_cm=round(100 * float(cte.max()), 3),
                w_rms=round(float(np.sqrt(np.mean(w_l**2))), 4), w_peak=round(float(np.abs(w_l).max()), 4),
                jitter_rms=round(float(np.sqrt(np.mean(dw**2))), 4) if dw.size else None,
                sat_share=round(float(np.mean(np.abs(w_l) >= OMEGA_MAX - 1e-6)), 4))


def summarise(runs):
    """mean +- std per condition, success rate, and the failures with their stop reasons."""
    out = {}
    for label, scale in CONDITIONS:
        rs = [r for r in runs if r["label"] == label]
        if not rs:
            continue
        s = dict(n=len(rs), scale=scale, success_rate=round(float(np.mean([r["docked"] for r in rs])), 3),
                 contacts=sum(bool(r["collisions"]) for r in rs),
                 failures=[(r["seed"], r["stop_reason"]) for r in rs if not r["docked"]])
        for k in ("cte_rms_cm", "cte_max_cm", "w_rms", "w_peak", "jitter_rms", "sat_share", "time_s"):
            vals = np.array([r[k] for r in rs if r[k] is not None], float)
            s[k] = dict(mean=round(float(vals.mean()), 4), std=round(float(vals.std()), 4))
        out[label] = s
    return out


def plot(summary, pred, case, path):
    labels = [l for l, _ in CONDITIONS if l in summary]
    colors = dict(zip([l for l, _ in CONDITIONS], ["#2a78d6", "#555555", "#eb6834"]))
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.0))
    a = ax[0]
    for l in labels:
        a.plot(pred[l]["_t"], 100 * np.array(pred[l]["_e_y"]), color=colors[l],
               label=f"{l}: K = [{pred[l]['K'][0]:.2f}, {pred[l]['K'][1]:.2f}]")
    a.axhline(0, color="#999999", lw=0.8)
    a.set(xlim=(0, 15), xlabel="time [s]", ylabel="e_y [cm]")
    a.set_title(f"(a) linear model, e_y from {100 * E0:.0f} cm at {V_NOM} m/s", loc="left", fontsize=10)
    a.legend(fontsize=8)
    a = ax[1]
    for l in labels:
        s = summary[l]
        a.errorbar(s["w_rms"]["mean"], s["cte_rms_cm"]["mean"], xerr=s["w_rms"]["std"], yerr=s["cte_rms_cm"]["std"],
                   fmt="o", color=colors[l], capsize=4, label=l)
    a.plot([summary[l]["w_rms"]["mean"] for l in labels], [summary[l]["cte_rms_cm"]["mean"] for l in labels],
           color="#bbbbbb", lw=1, zorder=0)
    a.set(xlabel="RMS turn-rate command w while tracking [rad/s]", ylabel="RMS cross-track error, true pose [cm]")
    a.set_title("(b) trade-off: tracking error vs control effort", loc="left", fontsize=10)
    a.legend(fontsize=8)
    a = ax[2]
    x = np.arange(len(labels))
    a.bar(x - 0.2, [100 * summary[l]["success_rate"] for l in labels], width=0.4, color="#1baf7a", label="docked [%]")
    a.set_ylim(0, 110); a.set_ylabel("docked [% of seeds]")
    a2 = a.twinx()
    a2.bar(x + 0.2, [summary[l]["jitter_rms"]["mean"] for l in labels], width=0.4, color="#eda100",
           yerr=[summary[l]["jitter_rms"]["std"] for l in labels], capsize=3, label="jitter")
    a2.set_ylabel("RMS of (w[k] - w[k-1]) / Ts [rad/s^2]")
    a.set_xticks(x, labels)
    a.set_title("(c) success rate (green) and command jitter (amber)", loc="left", fontsize=10)
    n = summary[labels[0]]["n"]
    fig.suptitle(f"LQR control weight R scaled, Q fixed: EKF, case {case}, {n} seeds per weight "
                 f"(evaluation/sweep_lqr.py)", fontsize=11)
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
    seeds = list(range(1, args.n + 1))
    pred = {label: linear_prediction(scale) for label, scale in CONDITIONS}
    for label, p in pred.items():
        print(f"{label:11s} d_omega_max {p['d_omega_max']:.3f} rad/s  K {p['K']}  |eig| v_nom {p['eig_abs_vnom']}  "
              f"|eig| {V_SLOW} m/s {p['eig_abs_slow']}  settles in {p['settle_dist_m']} m, peak d_omega "
              f"{p['peak_d_omega']} rad/s, overshoot {p['overshoot_cm']} cm")
    jobs = [(label, scale, args.case, seed) for label, scale in CONDITIONS for seed in seeds]
    with Pool(args.workers) as pool:
        runs = []
        for r in pool.imap_unordered(one_run, jobs):
            runs.append(r)
            print(f"  {r['label']:11s} seed {r['seed']:3d}: {'docked' if r['docked'] else 'FAILED ' + str(r['stop_reason']):22s} "
                  f"cte rms {r['cte_rms_cm']:.2f} cm  w rms {r['w_rms']:.3f}  jitter {r['jitter_rms']}  "
                  f"sat {100 * r['sat_share']:.1f} %  contacts {r['collisions'] or 'none'}")
    runs.sort(key=lambda r: ([l for l, _ in CONDITIONS].index(r["label"]), r["seed"]))
    summary = summarise(runs)
    OUT.mkdir(exist_ok=True)
    plot(summary, pred, args.case, OUT / "sweep_lqr.png")
    for p in pred.values():
        p.pop("_t"), p.pop("_e_y")
    (OUT / "sweep_lqr.json").write_text(json.dumps(dict(
        script="evaluation/sweep_lqr.py", case=args.case, seeds=seeds, conditions=CONDITIONS,
        linear_prediction=pred, summary=summary, runs=runs), indent=1))
    print()
    for label, s in summary.items():
        print(f"{label:11s} n {s['n']}  docked {100 * s['success_rate']:.0f} %  cte rms {s['cte_rms_cm']['mean']:.2f} ± "
              f"{s['cte_rms_cm']['std']:.2f} cm (max {s['cte_max_cm']['mean']:.2f})  w rms {s['w_rms']['mean']:.3f}  "
              f"peak {s['w_peak']['mean']:.2f}  jitter {s['jitter_rms']['mean']:.2f}  sat {100 * s['sat_share']['mean']:.2f} %  "
              f"failures {s['failures']}")
    print("\nsaved results/sweep_lqr.json and results/sweep_lqr.png")
