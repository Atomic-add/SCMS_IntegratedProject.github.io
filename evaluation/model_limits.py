"""Where the error model breaks down: closed-loop eigenvalues vs speed, linear vs nonlinear, and the mission run faster.

    python -m evaluation.model_limits --n 20         # linear analysis + the mission at 0.10, 0.20, 0.26 m/s
    python -m evaluation.model_limits --n 0          # linear analysis only (no simulator, a few seconds)

The LQR gain is designed once, at v_nom = 0.2 m/s on a straight (src/control/lqr.design_lqr). The robot
uses it at every speed it drives. This script asks how far that one design can be trusted:

(a) Closed-loop eigenvalues vs speed (Assessment 3 D.4, Week 5 D2). |eig(Ad - Bd K)| with the FIXED K
    on the discrete error model (src/control/models.error_model) at v = 0.02 ... 0.26 m/s, on a straight
    (kappa = 0) and on an arc (kappa = 2, R = 0.5 m). The continuous characteristic equation is
        lambda^2 + k_theta lambda + k_y v = 0,
    so the eigenvalues move with v. For each speed: |z|, the continuous s = ln(z) / Ts, the slow time
    constant tau = -1 / Re(s_slow) and the distance the robot covers in it, v * tau.

(b) Validity of the linearised model (Week 5 lecture "Validity of a linearised model", Week 5 B2). From a
    lateral offset e_y0, the linear ZOH model against the nonlinear error dynamics
        de_y/dt = v sin(e_theta),   de_theta/dt = d_omega      (straight, kappa = 0)
    with the angle wrapped and omega clipped to the Waffle Pi's 1.82 rad/s, both with the same K. Reported:
    peak d_omega, peak e_theta, max |linear - nonlinear| and the 2 % settling time of each.

(c) The mission run faster, without redesigning anything (Assessment 3 D.4). The whole mission on the
    EKF, case centre, seeds 1..n with run_demo.trial_setup(seed) (random true start, encoder scale error),
    with cfg["v_nom"] = 0.10, 0.20 and 0.26 m/s (MAX_V) passed at run time; no value in config/ changes and K
    stays the 0.2 m/s design. Controller side: cross-track RMS / max / mean (true pose, LQR tracking steps),
    RMS w and saturation share. Estimator side: position RMSE, ANEES and the share of steps with
    NEES <= 7.81 (chi-square(3), 95 %). Plus docked, stop reason, time and contacts. This says which part
    degrades first when the robot drives faster: the controller or the estimator.

Why ground truth is read here: (c) measures tracking and estimation errors against the true pose; the
controller still drives on the EKF estimate (run_mission). (a) and (b) use no simulator.

Adapted from 41014 Assessment 3 D.4 (closed-loop eigenvalues at higher speeds, straight and turn, against
the simulation), Week 5 B2 / D2 (linear vs nonlinear from the same initial condition; eigenvalues that
depend on v) and Week 6 A.8 (saturation).
Additions: the distance constant v * tau (the fixed gain corrects over a distance, not a time), the
validity check with the actuator clip, and the speed runs on the full mission with the EKF in the loop.
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
from evaluation.sweep_lqr import estimate_metrics, signed_cte   # noqa: E402
from src.control.behaviour import TRACK, run_mission           # noqa: E402
from src.control.lqr import design_lqr                          # noqa: E402
from src.control.models import OMEGA_MAX, TS, V_NOM, error_model, load_control_config   # noqa: E402
from src.estimation.ekf import EKF                              # noqa: E402
from src.robot.sim_robot import MAX_V, PROJECT_ROOT             # noqa: E402
from src.robot.task import load_task                            # noqa: E402

OUT = PROJECT_ROOT / "results"
EIG_SPEEDS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.26]      # m/s, (a): from the docking floor up to MAX_V
ARC_KAPPA = 2.0                                       # 1/m, (a): an arc of R = 0.5 m
OFFSETS = [0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.80]  # m, (b): starting lateral offsets
RUN_SPEEDS = [0.10, V_NOM, MAX_V]                     # m/s, (c): half the design speed, the design, the limit
T_RESP = 15.0                                         # s, (b): response length


def eig_row(K, v, kappa):
    """|z|, s = ln z / Ts, slow time constant and its distance for the fixed K at (v, kappa)."""
    Ad, Bd = error_model(v, kappa)
    z = np.linalg.eigvals(Ad - Bd @ K)
    s = np.log(z.astype(complex)) / TS
    slow = float(np.max(s.real))
    tau = -1.0 / slow if slow < -1e-9 else float("inf")
    order = np.argsort(-np.abs(z))                    # slow (largest |z|) first
    return dict(v=v, kappa=kappa, z_abs=np.round(np.abs(z)[order], 4).tolist(),
                s=[[round(float(x.real), 4), round(float(x.imag), 4)] for x in s[order]],
                tau_s=round(tau, 3) if np.isfinite(tau) else None,
                dist_m=round(v * tau, 3) if np.isfinite(tau) else None)


def responses(K, e_y0, v=V_NOM):
    """Linear ZOH and nonlinear (sin, wrap, clip) responses from (e_y0, 0); returns t, E_lin, E_nl, U_lin, U_nl."""
    K = np.atleast_2d(K)
    Ad, Bd = error_model(v, 0.0)
    n = int(round(T_RESP / TS))
    el, en = np.array([e_y0, 0.0]), np.array([e_y0, 0.0])
    El, En, Ul, Un = [el.copy()], [en.copy()], [], []
    for _ in range(n):
        ul = float(-(K @ el)[0])
        el = Ad @ el + Bd.ravel() * ul
        un = float(np.clip(-(K @ en)[0], -OMEGA_MAX, OMEGA_MAX))
        for _ in range(10):                           # held command, integrated in 10 sub-steps (ZOH)
            en = en + TS / 10 * np.array([v * np.sin(en[1]), un])
        en[1] = (en[1] + np.pi) % (2 * np.pi) - np.pi
        El.append(el.copy()); En.append(en.copy()); Ul.append(ul); Un.append(un)
    return np.arange(n + 1) * TS, np.array(El), np.array(En), np.array(Ul), np.array(Un)


def settle_time(t, e, e0):
    """2 % settling time of e from e0 [s]."""
    out = np.nonzero(np.abs(e) > 0.02 * e0)[0]
    return float(t[min(out[-1] + 1, len(t) - 1)]) if len(out) else 0.0


def validity_row(K, e_y0):
    t, El, En, Ul, Un = responses(K, e_y0)
    d = float(np.abs(El[:, 0] - En[:, 0]).max())
    return dict(e_y0_m=e_y0, peak_d_omega_lin=round(float(np.abs(Ul).max()), 3),
                clipped_share=round(float(np.mean(np.abs(Un) >= OMEGA_MAX - 1e-9)), 3),
                peak_e_theta_deg=round(float(np.rad2deg(np.abs(En[:, 1]).max())), 1),
                max_diff_cm=round(100 * d, 3), max_diff_pct=round(100 * d / e_y0, 2),
                settle_lin_s=round(settle_time(t, El[:, 0], e_y0), 2),
                settle_nl_s=round(settle_time(t, En[:, 0], e_y0), 2))


def speed_cfg(v):
    """Controller config with the cruise speed v [m/s]; K is NOT redesigned (design_lqr uses V_NOM)."""
    cfg = copy.deepcopy(load_control_config())
    cfg["v_nom"] = float(v)
    return cfg


def one_run(job):
    """One mission on the EKF at cruise speed v; tracking (controller) and estimate (EKF) numbers."""
    v_cruise, case, seed = job
    task = load_task()
    ekf = EKF(task["start"], marker_map=task["markers"])
    ref = reference_path()
    rows = []

    def on_step(robot, row):
        tp = np.asarray(true_pose(robot), float)
        rows.append((tp, row["state"], row["v"], row["w"], contacts(robot), row["xh"], row["P"]))

    info, t_end = run_mission(case, seed=seed, verbose=False, estimator=ekf, on_step=on_step,
                              setup=trial_setup(seed), cfg=speed_cfg(v_cruise))
    truth = np.array([r[0] for r in rows])
    state = np.array([r[1] for r in rows])
    v = np.array([r[2] for r in rows])
    w = np.array([r[3] for r in rows])
    lqr = (state == TRACK) & (v > 0.0)                       # LQR-driven tracking steps only
    cte = np.array([dist_to_polyline(p, ref) for p in truth[lqr, :2]])
    cte_s = np.array([signed_cte(p, ref) for p in truth[lqr, :2]])
    est = estimate_metrics(truth, np.array([r[5] for r in rows]), np.array([r[6] for r in rows]))
    hits = sorted(set().union(*[r[4] for r in rows]))
    return dict(v_cruise=v_cruise, seed=seed, docked=bool(info["dock_complete"]), stop_reason=info["stop_reason"],
                time_s=round(float(t_end), 1), collisions=hits, track_steps=int(lqr.sum()),
                cte_rms_cm=round(100 * float(np.sqrt(np.mean(cte**2))), 3),
                cte_max_cm=round(100 * float(cte.max()), 3), cte_mean_cm=round(100 * float(cte_s.mean()), 3),
                w_rms=round(float(np.sqrt(np.mean(w[lqr] ** 2))), 4),
                sat_share=round(float(np.mean(np.abs(w[lqr]) >= OMEGA_MAX - 1e-6)), 4), **est)


def summarise(runs):
    out = {}
    for v in RUN_SPEEDS:
        rs = [r for r in runs if r["v_cruise"] == v]
        if not rs:
            continue
        s = dict(n=len(rs), success_rate=round(float(np.mean([r["docked"] for r in rs])), 3),
                 contacts=sum(bool(r["collisions"]) for r in rs),
                 failures=[(r["seed"], r["stop_reason"]) for r in rs if not r["docked"]])
        for k in ("cte_rms_cm", "cte_max_cm", "cte_mean_cm", "w_rms", "sat_share", "pos_rmse_cm", "anees",
                  "nees_le_781", "time_s"):
            vals = np.array([r[k] for r in rs], float)
            s[k] = dict(mean=round(float(vals.mean()), 4), std=round(float(vals.std()), 4))
        out[f"{v:.2f}"] = s
    return out


def plot(K, eig_s, eig_a, path, summary=None):
    fig, ax = plt.subplots(1, 3 if summary else 2, figsize=(15 if summary else 10, 4.2))
    a = ax[0]
    vs = [r["v"] for r in eig_s]
    a.plot(vs, [r["z_abs"][0] for r in eig_s], "o-", color="#2a78d6", label="slow pole, straight")
    a.plot(vs, [r["z_abs"][1] for r in eig_s], "o--", color="#2a78d6", alpha=0.6, label="fast pole, straight")
    a.plot(vs, [r["z_abs"][0] for r in eig_a], "s-", color="#eb6834", label=f"slow pole, arc R = {1 / ARC_KAPPA:.1f} m")
    a.plot(vs, [r["z_abs"][1] for r in eig_a], "s--", color="#eb6834", alpha=0.6, label="fast pole, arc")
    a.axhline(1.0, color="#999999", lw=0.8)
    a.axvline(V_NOM, color="#555555", lw=0.8, ls=":")
    a.set(xlabel="speed v [m/s]", ylabel="|z| of Ad - Bd K (fixed K)")
    a.set_title(f"(a) closed-loop |eig| vs speed, K = [{K[0]:.2f}, {K[1]:.2f}] designed at {V_NOM} m/s",
                loc="left", fontsize=9)
    a.legend(fontsize=7)
    a = ax[1]
    for e0, c in zip([0.10, 0.30, 0.50, 0.80], ["#1baf7a", "#2a78d6", "#eda100", "#eb6834"]):
        t, El, En, _, _ = responses(K, e0)
        a.plot(t, 100 * En[:, 0], color=c, label=f"nonlinear, e_y0 = {100 * e0:.0f} cm")
        a.plot(t, 100 * El[:, 0], color=c, ls="--", lw=1)
    a.axhline(0, color="#999999", lw=0.8)
    a.set(xlim=(0, 10), xlabel="time [s]", ylabel="e_y [cm]")
    a.set_title(f"(b) linear (dashed) vs nonlinear (sin, clip at {OMEGA_MAX} rad/s), {V_NOM} m/s", loc="left",
                fontsize=9)
    a.legend(fontsize=7)
    if summary:
        a = ax[2]
        keys = list(summary)
        x = np.array([float(k) for k in keys])
        a.errorbar(x, [summary[k]["cte_rms_cm"]["mean"] for k in keys],
                   yerr=[summary[k]["cte_rms_cm"]["std"] for k in keys], fmt="o-", color="#2a78d6", capsize=3,
                   label="controller: cross-track RMS [cm]")
        a.errorbar(x, [summary[k]["pos_rmse_cm"]["mean"] for k in keys],
                   yerr=[summary[k]["pos_rmse_cm"]["std"] for k in keys], fmt="s-", color="#eb6834", capsize=3,
                   label="estimator: position RMSE [cm]")
        a.set(xlabel="cruise speed v_nom [m/s] (K not redesigned)", ylabel="cm")
        a2 = a.twinx()
        a2.plot(x, [100 * summary[k]["nees_le_781"]["mean"] for k in keys], "^:", color="#555555",
                label="NEES <= 7.81 [% of steps]")
        a2.set(ylabel="NEES <= 7.81 [% of steps]", ylim=(0, 105))
        h1, l1 = a.get_legend_handles_labels()
        h2, l2 = a2.get_legend_handles_labels()
        a.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower left")
        n = summary[keys[0]]["n"]
        a.set_title(f"(c) mission on the EKF, case centre, {n} seeds per speed", loc="left", fontsize=9)
    fig.suptitle("Model limits of the path-following error model (evaluation/model_limits.py)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=20, help="seeds per speed for (c) (default 20; 0 = no simulator)")
    ap.add_argument("--case", default="centre", choices=("none", "left", "centre", "right"))
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help=f"sim runs in parallel (default {DEFAULT_WORKERS}; each renders the camera)")
    args = ap.parse_args()

    K = design_lqr()[0].ravel()
    eig_s = [eig_row(K[None, :], v, 0.0) for v in EIG_SPEEDS]
    eig_a = [eig_row(K[None, :], v, ARC_KAPPA) for v in EIG_SPEEDS]
    eig_0 = eig_row(K[None, :], 0.0, 0.0)
    print(f"K = {np.round(K, 4)} (designed at v = {V_NOM} m/s, straight)\n\n(a) closed-loop eigenvalues, fixed K")
    for rs, ra in zip(eig_s, eig_a):
        print(f"  v {rs['v']:.2f} m/s  straight |z| {rs['z_abs']}  tau {rs['tau_s']} s  v*tau {rs['dist_m']} m   "
              f"arc |z| {ra['z_abs']}")
    print(f"  v 0.00 m/s  straight |z| {eig_0['z_abs']}  (one pole on z = 1: the e_y mode is uncontrollable)")
    val = [validity_row(K[None, :], e0) for e0 in OFFSETS]
    print("\n(b) linear vs nonlinear from a lateral offset, v = 0.2 m/s")
    for r in val:
        print(f"  e_y0 {100 * r['e_y0_m']:4.0f} cm: peak d_omega (linear) {r['peak_d_omega_lin']:.2f} rad/s, clipped "
              f"{100 * r['clipped_share']:.0f} % of steps, peak e_theta {r['peak_e_theta_deg']:.0f} deg, max |lin - nl| "
              f"{r['max_diff_cm']:.2f} cm ({r['max_diff_pct']:.1f} %), settle {r['settle_lin_s']} / {r['settle_nl_s']} s")

    summary, runs = None, []
    if args.n > 0:
        seeds = list(range(1, args.n + 1))
        jobs = [(v, args.case, s) for v in RUN_SPEEDS for s in seeds]
        print(f"\n(c) mission on the EKF at {RUN_SPEEDS} m/s, {args.n} seeds each, K not redesigned")
        with Pool(args.workers) as pool:
            for r in pool.imap_unordered(one_run, jobs):
                runs.append(r)
                print(f"  v {r['v_cruise']:.2f} seed {r['seed']:3d}: {'docked' if r['docked'] else 'FAILED ' + str(r['stop_reason']):22s} "
                      f"cte rms {r['cte_rms_cm']:.2f} cm (mean {r['cte_mean_cm']:+.2f})  pos rmse {r['pos_rmse_cm']:.2f} cm  "
                      f"NEES<=7.81 {100 * r['nees_le_781']:.0f} %  sat {100 * r['sat_share']:.1f} %  {r['time_s']} s")
        runs.sort(key=lambda r: (r["v_cruise"], r["seed"]))
        summary = summarise(runs)
        print()
        for k, s in summary.items():
            print(f"  v {k} m/s: docked {100 * s['success_rate']:.0f} %, cte rms {s['cte_rms_cm']['mean']:.2f} ± "
                  f"{s['cte_rms_cm']['std']:.2f} cm (max {s['cte_max_cm']['mean']:.2f}, mean {s['cte_mean_cm']['mean']:+.2f}), "
                  f"pos rmse {s['pos_rmse_cm']['mean']:.2f} cm, ANEES {s['anees']['mean']:.2f}, NEES<=7.81 "
                  f"{100 * s['nees_le_781']['mean']:.1f} %, sat {100 * s['sat_share']['mean']:.2f} %, "
                  f"time {s['time_s']['mean']:.1f} s, failures {s['failures']}")

    OUT.mkdir(exist_ok=True)
    plot(K, eig_s, eig_a, OUT / "model_limits.png", summary)
    out = dict(script="evaluation/model_limits.py", K=np.round(K, 4).tolist(), v_design=V_NOM, Ts=TS,
               eig_straight=eig_s, eig_arc=eig_a, eig_v0=eig_0, validity=val)
    if summary:
        out.update(case=args.case, speeds=RUN_SPEEDS, seeds=list(range(1, args.n + 1)), summary=summary, runs=runs)
    (OUT / "model_limits.json").write_text(json.dumps(out, indent=1))
    print("\nsaved results/model_limits.json and results/model_limits.png")
