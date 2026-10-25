"""EKF consistency: sweep the wheel noise Qd and the marker noise Rn on recorded sim data, score NEES and NIS.

    python -m evaluation.consistency_sweep                       # lab track, lab camera: record 20 seeds x 2 cases, sweep
    python -m evaluation.consistency_sweep --task config/corridor.json --sim-camera urdf
    python -m evaluation.consistency_sweep --from-record results/consistency_record_arena_lab_lab.npz
    python -m evaluation.consistency_sweep --from-record ... --write-noise   # store the fitted noise in the camera file

Step 1 records what the EKF is handed in N_SEEDS sim runs per case (the encoder angles and every ArUco
range-bearing measurement, rendered and detected as in a normal run) next to the true pose. Step 2 replays
src/estimation/ekf.py on that record for every pair of scale factors on the Qd variance (k_wheel) and the
Rn variance (range_sd, bearing_sd_deg), and scores each pair the Week 7 way (B.6, D.3, D.4):
    ANEES      mean over all steps and runs of e^T P^-1 e, e = truth - xh    -> 3 when P is honest
    ANIS       mean NIS of the fused markers, nu^T S^-1 nu                    -> 2 when S is honest
    coverage   share of steps with NEES <= chi2_3(0.95) = 7.81               -> 0.95
    rms        position error, the price of the tuning
The replay keeps the recorded path, so the controller does not react to the new estimate (open loop):
the scores measure the filter, not the closed loop. Truth is read here only (on_step hook).

Step 3 (fit_noise) measures the camera's marker noise on the same record, against the true range and
bearing: the range bias c r^2, the residual sd a + b r^2 per range bin, the rms of the bias model's own
residual (a floor added to a), the bearing sd, and the integrated autocorrelation time tau of both errors
(frames). Consecutive frames repeat nearly the same error (tau ~ 5 frames for range, ~ 19 for bearing on
23 Oct), so only one frame in tau is new information: the variances are multiplied by tau (Week 7 B.6:
the Kalman filter assumes white measurement noise). --write-noise stores the result as "ekf_noise" in
the camera file, which models.load_estimation_config(camera=...) applies. With correlated errors the NIS
stays below 2 even when P is honest (the estimate already holds the repeated error), so in the sim the
NEES against the truth is the test that counts.
Writes results/consistency_<task>_<camera>.json and .png.
Adapted from 41014 Week 7 B.6 (is the filter telling the truth), D.3 (consistency) and D.4 (Qd scale
sweep). Additions: recording real sim measurements once and replaying the EKF on them for a 2-D sweep.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root, so "src" imports work
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")                  # offscreen rendering, before mujoco loads

import matplotlib                                               # noqa: E402
import numpy as np                                              # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402

from src.estimation.ekf import EKF                              # noqa: E402
from src.estimation.models import load_estimation_config        # noqa: E402
from src.estimation.odometry import wrap                        # noqa: E402
from src.robot.sim_robot import PROJECT_ROOT, SIM_CAMERAS       # noqa: E402

OUT = PROJECT_ROOT / "results"
CASES = ("none", "centre")
N_SEEDS = 20                    # runs per case (the guide's N >= 20)
SCALES = (0.1, 0.3, 1.0, 3.0, 10.0)   # factors on the noise VARIANCES (Week 7 D.4 sweeps x0.1 ... x10)
NEES_95 = 7.815                 # chi2 with 3 dof, 95 % point (scipy.stats.chi2.ppf(0.95, 3))


class Recorder(EKF):
    """The normal EKF, which also keeps every input it is given (for the offline replay)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.inputs = []

    def step(self, encoders, marker_frame=None):
        frame = None if marker_frame is None else (marker_frame["t"], {int(k): np.asarray(m["z"], float).tolist()
                                                                        for k, m in marker_frame["markers"].items()})
        self.inputs.append((tuple(map(float, encoders)), frame))
        return super().step(encoders, marker_frame)


def record(task_path, camera_name):
    """Run N_SEEDS x CASES missions; return a list of dict(case, seed, inputs, truth, start, markers)."""
    from evaluation.ground_truth import true_pose
    from src.control.behaviour import run_mission
    from src.control.models import load_control_config
    from src.robot.corridor import apply_control_overrides, load_task, make_robot

    cam_spec, cam_file = SIM_CAMERAS[camera_name]
    task = load_task(task_path)
    params = load_estimation_config(camera=PROJECT_ROOT / cam_file)
    runs = []
    for case in CASES:
        for seed in range(N_SEEDS):
            est, truth = Recorder(task["start"], marker_map=task["markers"], params=params), []
            info, _ = run_mission(case, seed=seed, verbose=False, task=task, estimator=est,
                                  robot=make_robot(case, seed=seed, path=task_path, camera=cam_spec),
                                  cfg=apply_control_overrides(load_control_config(), task),
                                  intrinsics=PROJECT_ROOT / cam_file,
                                  on_step=lambda robot, row: truth.append(np.asarray(true_pose(robot), float).tolist()))
            runs.append(dict(case=case, seed=seed, inputs=est.inputs, truth=truth, result=info["stop_reason"]))
            print(f"  {case} seed {seed}: {info['stop_reason']}, {len(truth)} steps")
    return dict(task=str(Path(task_path).resolve().relative_to(PROJECT_ROOT)), camera=camera_name,
                start=np.asarray(task["start"], float).tolist(),
                markers={int(k): np.asarray(v, float).tolist() for k, v in task["markers"].items()},
                params=params, runs=runs)


def replay(rec, q_scale, r_scale):
    """Scores of the EKF on the record with Qd variance x q_scale and Rn variance x r_scale."""
    p = dict(rec["params"])
    p["k_wheel"] = rec["params"]["k_wheel"] * np.sqrt(q_scale)        # Qd sd = k_wheel |d|
    p["range_sd"] = [v * np.sqrt(r_scale) for v in rec["params"]["range_sd"]]
    p["bearing_sd_deg"] = rec["params"]["bearing_sd_deg"] * np.sqrt(r_scale)
    nees, nis, d = [], [], []
    for run in rec["runs"]:
        ekf = EKF(rec["start"], marker_map=rec["markers"], params=p)
        for (enc, frame), x in zip(run["inputs"], run["truth"]):
            mf = None if frame is None else {"t": frame[0], "markers": {k: {"z": z} for k, z in frame[1].items()}}
            xh, P = ekf.step(enc, mf)
            e = np.asarray(x) - xh
            e[2] = wrap(e[2])
            nees.append(float(e @ np.linalg.solve(P, e)))
            d.append(float(np.hypot(e[0], e[1])))
        nis += [n for _, _, n, used in ekf.log if used]
    nees, d = np.array(nees), np.array(d)
    return dict(q_scale=q_scale, r_scale=r_scale, anees=round(float(nees.mean()), 3),
                anis=round(float(np.mean(nis)), 3) if nis else None, coverage=round(float(np.mean(nees <= NEES_95)), 3),
                rms_cm=round(100 * float(np.sqrt(np.mean(d ** 2))), 3), n_updates=len(nis))


def fit_noise(rec):
    """The camera's marker noise on the record, measured against the truth; dict for "ekf_noise"."""
    from src.estimation.models import h_rb
    mk = {int(k): np.asarray(v, float) for k, v in rec["markers"].items()}
    tracks, dr, db, rr = [], [], [], []
    for run in rec["runs"]:
        seqs = {}
        for (enc, frame), x in zip(run["inputs"], run["truth"]):
            for k, z in (frame[1].items() if frame else ()):
                zt = h_rb(np.asarray(x), mk[k])
                seqs.setdefault(k, []).append((z[0] - zt[0], wrap(z[1] - zt[1]), zt[0]))
        tracks += [np.array(v) for v in seqs.values()]
    allr = np.vstack(tracks)
    dr, db, rr = allr[:, 0], allr[:, 1], allr[:, 2]
    c = float(np.sum(dr * rr ** 2) / np.sum(rr ** 4))                 # bias c r^2, least squares
    res = dr - c * rr ** 2
    edges = np.linspace(rr.min(), rr.max(), 12)
    mid, sd, mean = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (rr >= lo) & (rr < hi)
        if s.sum() > 50:
            mid.append(rr[s].mean()), sd.append(res[s].std()), mean.append(res[s].mean())
    a, b = np.linalg.lstsq(np.c_[np.ones(len(mid)), np.square(mid)], np.array(sd), rcond=None)[0]
    floor = float(np.sqrt(np.mean(np.square(mean))))                  # what c r^2 leaves, per range bin
    taus = {0: [], 1: []}
    for tr in tracks:
        if len(tr) < 60:
            continue
        for j in (0, 1):
            e = tr[:, j] - (c * tr[:, 2] ** 2 if j == 0 else 0.0)
            e = e - e.mean()
            if e.std() > 0:
                rho = [np.corrcoef(e[:-L], e[L:])[0, 1] for L in range(1, 40)]
                taus[j].append(1 + 2 * sum(r for r in rho if r > 0.05))   # integrated autocorrelation time
    tau_r, tau_b = float(np.median(taus[0])), float(np.median(taus[1]))
    bsd = float(np.degrees(db.std()))
    return dict(range_bias_r2=round(c, 5),
                range_sd=[round((max(a, 0.0) + floor) * np.sqrt(tau_r), 5), round(b * np.sqrt(tau_r), 5)],
                bearing_sd_deg=round(bsd * np.sqrt(tau_b), 4),
                _doc=f"EKF marker noise of this camera, fitted by evaluation/consistency_sweep.py on {len(rec['runs'])} "
                     f"recorded runs of {rec['task']} against the truth: range bias {1000 * c:.2f} mm * r^2; residual "
                     f"range sd {1000 * max(a, 0.0):.2f} mm + {1000 * b:.2f} mm * r^2 plus {1000 * floor:.1f} mm left by "
                     f"the bias model; bearing sd {bsd:.4f} deg. Errors repeat over tau = {tau_r:.1f} frames (range) "
                     f"and {tau_b:.1f} frames (bearing), so the variances are multiplied by tau (sd by sqrt(tau)).")


def badness(s):
    """Distance from an honest filter: relative ANEES and ANIS errors (log scale) and the coverage gap."""
    return abs(np.log(s["anees"] / 3.0)) + abs(np.log((s["anis"] or 1e-9) / 2.0)) + 5 * abs(s["coverage"] - 0.95)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", type=Path, default=PROJECT_ROOT / "config" / "arena_lab.json")
    ap.add_argument("--sim-camera", default=None, choices=list(SIM_CAMERAS),
                    help="default: the task file's sim_camera")
    ap.add_argument("--from-record", type=Path, help="replay a saved record instead of simulating")
    ap.add_argument("--write-noise", action="store_true",
                    help="store the fitted marker noise as ekf_noise in the record camera's calibration file")
    args = ap.parse_args()

    if args.from_record:
        data = np.load(args.from_record, allow_pickle=True)
        rec = data["record"].item()
    else:
        from src.robot.corridor import load_config
        cam = args.sim_camera or load_config(args.task).get("sim_camera", "urdf")
        print(f"recording {N_SEEDS} seeds x {len(CASES)} cases on {args.task.name}, camera {cam}")
        rec = record(args.task, cam)
        path = OUT / f"consistency_record_{Path(rec['task']).stem}_{cam}.npz"
        np.savez_compressed(path, record=np.array(rec, dtype=object))
        print(f"wrote {path.relative_to(PROJECT_ROOT)}")

    name = f"consistency_{Path(rec['task']).stem}_{rec['camera']}"
    grid = [replay(rec, q, r) for q in SCALES for r in SCALES]
    best = min(grid, key=badness)
    now = next(s for s in grid if s["q_scale"] == 1.0 and s["r_scale"] == 1.0)
    print(f"\n{'Qd x':>6} {'Rn x':>6} {'ANEES':>7} {'ANIS':>6} {'cover':>6} {'rms cm':>7}")
    for s in grid:
        mark = "  <- current" if s is now else ("  <- best" if s is best else "")
        print(f"{s['q_scale']:6.1f} {s['r_scale']:6.1f} {s['anees']:7.2f} {s['anis']:6.2f} {s['coverage']:6.3f} "
              f"{s['rms_cm']:7.2f}{mark}")
    p = rec["params"]
    suggestion = dict(k_wheel=round(p["k_wheel"] * np.sqrt(best["q_scale"]), 4),
                      range_sd=[round(v * np.sqrt(best["r_scale"]), 5) for v in p["range_sd"]],
                      bearing_sd_deg=round(p["bearing_sd_deg"] * np.sqrt(best["r_scale"]), 4))
    print(f"\ncurrent: k_wheel {p['k_wheel']}, range_sd {p['range_sd']}, bearing_sd_deg {p['bearing_sd_deg']}")
    print(f"best by scaling the current values alone: Qd x{best['q_scale']}, Rn x{best['r_scale']} -> {suggestion} "
          "(a scale only hides a wrong noise model; the fit below measures it)")
    noise = fit_noise(rec)
    fitted = dict(rec)
    fitted["params"] = {**p, **{k: v for k, v in noise.items() if not k.startswith("_")}}
    scored = replay(fitted, 1.0, 1.0)
    print(f"\nfitted marker noise: {noise['_doc']}")
    print(f"  -> range_bias_r2 {noise['range_bias_r2']}, range_sd {noise['range_sd']}, bearing_sd_deg "
          f"{noise['bearing_sd_deg']}: ANEES {scored['anees']:.2f}, ANIS {scored['anis']:.2f}, coverage "
          f"{scored['coverage']:.3f}, rms {scored['rms_cm']:.2f} cm (current: ANEES {now['anees']:.2f}, coverage "
          f"{now['coverage']:.3f}, rms {now['rms_cm']:.2f} cm)")
    if args.write_noise:
        cam_file = PROJECT_ROOT / SIM_CAMERAS[rec["camera"]][1]
        data = json.loads(cam_file.read_text())
        data["ekf_noise"] = noise
        cam_file.write_text(json.dumps(data, indent=2))
        print(f"wrote ekf_noise into {cam_file.relative_to(PROJECT_ROOT)}")
    summary = dict(task=rec["task"], camera=rec["camera"], cases=list(CASES), seeds=list(range(N_SEEDS)),
                   fitted_noise=noise, fitted_scores=scored,
                   scales=list(SCALES), current_params={k: p[k] for k in ("k_wheel", "range_sd", "bearing_sd_deg")},
                   grid=grid, current=now, best=best, suggestion=suggestion,
                   results=[r["result"] for r in rec["runs"]], script="evaluation/consistency_sweep.py")
    (OUT / f"{name}.json").write_text(json.dumps(summary, indent=1))
    plot(grid, OUT / f"{name}.png", f"{rec['task']}, {rec['camera']} camera, {len(rec['runs'])} runs")
    print(f"wrote results/{name}.json and .png")


def plot(grid, path, title):
    """Heat maps of ANEES, ANIS and coverage over the Qd x Rn grid (log axes)."""
    n = len(SCALES)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, key, ideal, label in ((axes[0], "anees", 3.0, "ANEES (ideal 3)"), (axes[1], "anis", 2.0, "ANIS (ideal 2)"),
                                  (axes[2], "coverage", 0.95, "NEES <= 7.81 share (ideal 0.95)")):
        M = np.array([s[key] for s in grid], float).reshape(n, n)
        im = ax.imshow(np.log(M / ideal) if key != "coverage" else M - ideal, cmap="RdBu_r", origin="lower",
                       vmin=-2 if key != "coverage" else -0.5, vmax=2 if key != "coverage" else 0.5)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7)
        ax.set_xticks(range(n), [f"{s:g}" for s in SCALES])
        ax.set_yticks(range(n), [f"{s:g}" for s in SCALES])
        ax.set(xlabel="Rn variance scale [-]", ylabel="Qd variance scale [-]", title=label)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
