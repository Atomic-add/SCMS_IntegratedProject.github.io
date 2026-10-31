"""Which pose states the camera's markers make observable, and how fast P grows when they are lost.

    python -m evaluation.observability               # -> results/observability.json and .png (a few seconds)

Three parts, all on the EKF's own linearisation (src/estimation/observability.py: F_jac, H_rb):

  1. Cases at a pose on the lab lane (1.5, 1.25, facing -x), lab markers 1 (0, 1.75) and 2 (0, 0.75):
     (a) no marker, (b) one marker still, (c) one marker while driving 1 s at v_nom (straight and on an
     arc), (d) range only (the Assessment 3 filter), still and driving, (e) markers 1 and 2 together,
     and marker 2 then marker 1 one at a time. For each: rank of O over 1 frame and over 1 s of frames
     (20 at Ts), smallest singular value of O (Week 8 A.7), the hidden direction, and the best sd in the
     weakest direction from the Rn-weighted Gramian, 1 / sqrt(lambda_min(W)).
  2. Marker lost: predict only (P <- F P F^T + Qd) at v_nom for T_LOST s, from the config prior P0 and
     from the P the EKF converges to with markers 1 and 2 in view. Marks when the 3-sigma bound passes
     the dock tolerance (3 cm, 5 deg: docking.py dock_lateral_tol / dock_heading_tol_deg) and when the
     1-sigma passes the docking gate on P (dock_pos_sigma_max, dock_heading_sigma_max_deg).
  3. Along the lab path: rank and weakest sd of a 1 s window vs distance travelled, coloured by the
     markers in view. Poses and visibility come from evaluation/marker_layout.py: its recorded sim paths
     (results/marker_layout_paths.json, cases none and centre) and its visibility() test for the lab
     camera (corners in the image, size, incidence, occlusion by panels and the box). Those paths were
     recorded on 25 Oct, before the 28 Oct dock-stop change; run python -m evaluation.marker_layout
     to record them again. Each window is linearised along the EKF's own motion model from its first pose.

Why ground truth is read here: part 3 linearises at the TRUE path recorded by marker_layout.py (the
pose the camera really looked from). Parts 1 and 2 use no simulator at all.

Adapted from 41014 Week 5 C4 (ct.obsv rank), Week 7 A.2 (SVD null space) and D.5 (range only), and
Week 8 A.7 - A.8 (Gramian, PBH).
Additions: the time-varying rank on the EKF's own Jacobians, the marker-lost P growth against the
docking tolerances, the rank along the lab path with the lab camera's visibility.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root, so "src" imports work

import matplotlib                                               # noqa: E402
import numpy as np                                              # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402

from evaluation import marker_layout as ml                      # noqa: E402
from src.control.models import TS, V_NOM, load_control_config   # noqa: E402
from src.estimation.models import load_estimation_config        # noqa: E402
from src.estimation.odometry import move                        # noqa: E402
from src.estimation.observability import (converged_P, obsv_tv, pbh_detectable, H_stack,  # noqa: E402
                                          predict_only, rank_and_sv, steps_along, weakest_sd)
from src.perception import aruco, camera                        # noqa: E402
from src.robot.sim_robot import PROJECT_ROOT                    # noqa: E402
from src.robot.task import load_config                          # noqa: E402

OUT = PROJECT_ROOT / "results"
TASK = load_config(ml.TASK)
MARKERS = {int(k): np.array(v, float) for k, v in TASK["markers"].items() if not k.startswith("_")}
M1, M2 = MARKERS[1][:2], MARKERS[2][:2]
POSE = np.array([1.5, 1.25, np.pi])     # on the lane 1.5 m from the left end panel, facing it: both markers in view
N_WIN = 20                              # frames in the window = 1 s at Ts = 0.05 s
W_ARC = 0.5                             # rad/s: turn rate of the arc case (a gentle curve, within the 1.82 rad/s limit)
T_LOST = 10.0                           # s of predict only


def case_rows(cfg):
    """Part 1: rank / singular value / weakest sd per case, over 1 frame and over N_WIN frames."""
    cases = [("(a) no marker", [], True, V_NOM, 0.0),
             ("(b) marker 2, still", [M2], True, 0.0, 0.0),
             ("(c) marker 2, driving straight", [M2], True, V_NOM, 0.0),
             ("(c) marker 2, driving an arc", [M2], True, V_NOM, W_ARC),
             ("(d) marker 2 range only, still", [M2], False, 0.0, 0.0),
             ("(d) marker 2 range only, driving", [M2], False, V_NOM, 0.0),
             ("(e) markers 1 + 2, still", [M1, M2], True, 0.0, 0.0),
             ("marker 2, then marker 1, driving", None, True, V_NOM, 0.0)]
    rows = []
    for name, ms, bearing, v, w in cases:
        row = dict(case=name, v=v, w=w, bearing=bearing)
        for n in (1, N_WIN):
            poses, inc = steps_along(POSE, v, w, TS, n)
            seen = [[M2]] * (n // 2) + [[M1]] * (n - n // 2) if ms is None else [ms] * n
            if ms is None and n == 1:
                seen = [[M2]]
            O, W = obsv_tv(poses, inc, seen, bearing, cfg)
            r, sv = rank_and_sv(O)
            hidden = np.linalg.svd(O)[2][-1].round(3).tolist() if 0 < r < 3 else None
            row[f"N{n}"] = dict(rank=r, sv_min=float(sv), weakest_sd=weakest_sd(W) if O.size else float("inf"),
                                hidden_direction=hidden)
        F = np.eye(3)
        Hs = H_stack(POSE, [] if ms is None else ms, bearing)
        row["detectable_still"] = bool(pbh_detectable(F, Hs)[0])
        rows.append(row)
    return rows


def first_time(t, y, limit):
    """First time y exceeds limit (None if it never does within the run)."""
    k = np.nonzero(y > limit)[0]
    return round(float(t[k[0]]), 2) if k.size else None


def lost_rows(cfg, dock):
    """Part 2: predict-only growth from the prior P0 and from the converged two-marker P."""
    sx, sy, sth = cfg["P0_sd"]
    P0 = np.diag([sx**2, sy**2, np.deg2rad(sth)**2])
    starts = {"config prior P0": P0,
              "converged with markers 1 + 2": converged_P(P0, POSE, [M1, M2], cfg, TS)}
    out = {}
    for name, Ps0 in starts.items():
        t, Ps = predict_only(Ps0, POSE, V_NOM, 0.0, TS, T_LOST, cfg["k_wheel"], cfg["q_floor"])
        sd_x, sd_y, sd_th = (np.sqrt(Ps[:, i, i]) for i in range(3))
        sd_pos = np.sqrt(np.linalg.eigvalsh(Ps[:, :2, :2])[:, -1])        # sd along the worst horizontal axis
        out[name] = dict(
            t=t[::10].round(2).tolist(), sd_x_cm=(100 * sd_x[::10]).round(3).tolist(),
            sd_y_cm=(100 * sd_y[::10]).round(3).tolist(), sd_theta_deg=np.rad2deg(sd_th[::10]).round(3).tolist(),
            start_sd=[round(100 * float(sd_x[0]), 3), round(100 * float(sd_y[0]), 3), round(float(np.rad2deg(sd_th[0])), 3)],
            end_sd=[round(100 * float(sd_x[-1]), 3), round(100 * float(sd_y[-1]), 3), round(float(np.rad2deg(sd_th[-1])), 3)],
            t_3sigma_pos_over_tol=first_time(t, 3 * sd_pos, dock["dock_lateral_tol"]),
            t_3sigma_heading_over_tol=first_time(t, 3 * np.rad2deg(sd_th), dock["dock_heading_tol_deg"]),
            t_sd_pos_over_gate=first_time(t, np.sqrt(Ps[:, 0, 0] + Ps[:, 1, 1]), dock["dock_pos_sigma_max"]),
            t_sd_heading_over_gate=first_time(t, np.rad2deg(sd_th), dock["dock_heading_sigma_max_deg"]),
            _series=(t, sd_pos, sd_th))
    return out


def step_increment(a, b):
    """(ds, dth) of the motion model between two poses: dth wrapped, ds along the midpoint heading [m, rad]."""
    dth = float(np.angle(np.exp(1j * (b[2] - a[2]))))
    m = a[2] + dth / 2
    return float((b[0] - a[0]) * np.cos(m) + (b[1] - a[1]) * np.sin(m)), dth


def path_rows(cfg):
    """Part 3: rank and weakest sd of the 1 s window starting at each frame along the recorded lab paths."""
    paths = json.loads((OUT / "marker_layout_paths.json").read_text())["paths"]
    K, _, _ = camera.load_intrinsics(ml.CAMERA)
    T_BC = aruco.load_T_BC(ml.CAMERA)
    spots = {i: dict(xy=list(m[:2]), normal=[float(np.cos(m[3])), float(np.sin(m[3]))], z=float(m[2]),
                     edge=ml.edge_of(TASK["outline"], m[:2])) for i, m in MARKERS.items()}
    out = {}
    for case, p in paths.items():
        pose = np.array(p["pose"], float)
        cams = ml.camera_poses(pose, T_BC)
        boxes = [] if TASK["obstacles"][case] is None else [(TASK["obstacles"][case], 0.10)]   # 0.20 m box, as marker_layout
        vis = {i: ml.visibility(s, cams, K, TASK["outline"], boxes, ml.MIN_SIDE_PX)[0] for i, s in spots.items()}
        ids = [[i for i in sorted(vis) if vis[i][k]] for k in range(len(pose))]
        dist = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pose[:, :2], axis=0).T))])
        rank1, rankw, sdw = [], [], []
        for k in range(len(pose)):
            win = pose[k:k + N_WIN]
            inc = [step_increment(a, b) for a, b in zip(win[:-1], win[1:])]
            # linearise along the EKF's own motion model from the window's first pose (as the EKF does), so
            # F and the poses agree; the recorded poses would add the sim's small slip as fake information
            model = [win[0]]
            for ds, dth in inc:
                model.append(move(model[-1], ds, dth))
            win = np.array(model)
            seen = [[MARKERS[i][:2] for i in ids[j]] for j in range(k, k + len(win))]
            O1, _ = obsv_tv(win[:1], [], seen[:1], True, cfg)
            O, W = obsv_tv(win, inc, seen, True, cfg)
            rank1.append(rank_and_sv(O1)[0])
            rankw.append(rank_and_sv(O)[0])
            sdw.append(weakest_sd(W) if O.size else float("inf"))
        state = np.array(p["state"])
        n_in = np.array([len(i) for i in ids])
        out[case] = dict(dist_m=dist.round(3).tolist(), ids=ids, rank_instant=rank1, rank_window=rankw,
                         weakest_sd_window=[None if not np.isfinite(s) else round(s, 5) for s in sdw],
                         share_rank3_instant=round(float(np.mean(np.array(rank1) == 3)), 3),
                         share_rank3_window=round(float(np.mean(np.array(rankw) == 3)), 3),
                         share_no_marker=round(float(np.mean(n_in == 0)), 3),
                         share_one_marker=round(float(np.mean(n_in == 1)), 3),
                         share_two_plus=round(float(np.mean(n_in >= 2)), 3),
                         longest_no_marker_m=round(float(longest(dist, n_in == 0)), 2),
                         rank_window_at_dock_start=int(rankw[int(np.argmax(state == "DOCK"))]) if np.any(state == "DOCK") else None)
    return out


def longest(dist, mask):
    """Longest stretch of travel [m] over which mask stays True."""
    best, start = 0.0, None
    for k, m in enumerate(mask):
        if m and start is None:
            start = k
        if (not m or k == len(mask) - 1) and start is not None:
            best = max(best, dist[k] - dist[start])
            start = None
    return best


def plot(cases, lost, along, dock, path):
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.4))
    # (a) weakest sd per case
    a = ax[0, 0]
    names = [c["case"] for c in cases]
    for j, (n, col) in enumerate([(1, "#2a78d6"), (N_WIN, "#eb6834")]):
        sd = [c[f"N{n}"]["weakest_sd"] for c in cases]
        y = [s if np.isfinite(s) else 1e3 for s in sd]
        a.barh(np.arange(len(names)) + (j - 0.5) * 0.38, y, height=0.36, color=col,
               label=f"{'1 frame' if n == 1 else f'{N_WIN} frames (1 s)'}")
        for i, (s, c) in enumerate(zip(sd, cases)):
            a.text(min(y[i], 1e3) * 1.15, i + (j - 0.5) * 0.38, f"rank {c[f'N{n}']['rank']}" + ("" if np.isfinite(s) else ", unobservable"),
                   va="center", fontsize=7)
    a.set_xscale("log"); a.set_xlim(1e-3, 1e5)
    a.set_yticks(range(len(names)), names, fontsize=8); a.invert_yaxis()
    a.set_xlabel("best sd in the weakest direction, 1/sqrt(lambda_min(W)) [m or rad]")
    a.set_title("(a) observability per case (lab markers 1, 2)", loc="left", fontsize=10)
    a.legend(fontsize=7, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False)
    # (b) marker lost
    a = ax[0, 1]
    for (name, d), col in zip(lost.items(), ["#2a78d6", "#eb6834"]):
        t, sd_pos, sd_th = d["_series"]
        a.plot(t, 300 * sd_pos, color=col, label=f"3 sd position, from {name}")
        a.plot(t, 3 * np.rad2deg(sd_th), color=col, ls="--", label=f"3 sd heading [deg], from {name}")
    a.axhline(100 * dock["dock_lateral_tol"], color="#555555", lw=1)
    a.axhline(dock["dock_heading_tol_deg"], color="#555555", lw=1, ls="--")
    a.text(0.1, 100 * dock["dock_lateral_tol"] * 1.05, "dock tolerance 3 cm", fontsize=7)
    a.text(0.1, dock["dock_heading_tol_deg"] * 1.05, "dock tolerance 5 deg", fontsize=7)
    a.set_yscale("log")
    a.set(xlabel="time with no marker [s]", ylabel="3 sd: position [cm], heading [deg]")
    a.set_title(f"(b) marker lost: predict only at v = {V_NOM} m/s", loc="left", fontsize=10)
    a.legend(fontsize=7)
    # (c), (d) along the path
    colors = {0: "#bbbbbb", 1: "#eda100", 2: "#1baf7a"}
    for case, d in along.items():
        a = ax[1, 0] if case == "none" else ax[1, 1]
        dist = np.array(d["dist_m"])
        sd = np.array([np.nan if s is None else s for s in d["weakest_sd_window"]], float)
        n_in = np.array([len(i) for i in d["ids"]])
        for k in (0, 1, 2):
            m = np.minimum(n_in, 2) == k
            a.scatter(dist[m], np.where(np.isfinite(sd[m]), 100 * sd[m], 1e4), s=4, color=colors[k],
                      label={0: "no marker in view", 1: "one marker", 2: "two or more"}[k])
        a2 = a.twinx()
        a2.step(dist, d["rank_window"], where="post", color="#2a78d6", lw=1)
        a2.set_ylim(-0.2, 3.5); a2.set_yticks([0, 1, 2, 3]); a2.set_ylabel("rank of O over the next 1 s", color="#2a78d6")
        a.set_yscale("log"); a.set_ylim(1e-2, 2e4)
        a.set(xlabel="distance travelled [m]", ylabel="best sd, weakest direction, next 1 s [cm] (top = none)")
        a.set_title(f"({'c' if case == 'none' else 'd'}) along the lab path, case {case}: rank 3 in "
                    f"{100 * d['share_rank3_window']:.0f} % of windows", loc="left", fontsize=10)
        a.legend(fontsize=7, loc="upper left", markerscale=3)
    fig.suptitle("Observability of the pose from the ArUco markers (evaluation/observability.py)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args()
    cfg = load_estimation_config()
    dock = load_control_config()["docking"]
    cases = case_rows(cfg)
    lost = lost_rows(cfg, dock)
    along = path_rows(cfg)
    print(f"{'case':36s} {'rank 1 frame':>12s} {'rank 1 s':>9s} {'weakest sd, 1 s':>16s}  hidden direction (1 s)")
    for c in cases:
        sd = c[f"N{N_WIN}"]["weakest_sd"]
        print(f"{c['case']:36s} {c['N1']['rank']:12d} {c[f'N{N_WIN}']['rank']:9d} "
              f"{(f'{100 * sd:.2f} cm' if np.isfinite(sd) else 'unobservable'):>16s}  {c[f'N{N_WIN}']['hidden_direction']}")
    print()
    for name, d in lost.items():
        print(f"marker lost from {name}: sd x, y, theta {d['start_sd']} -> {d['end_sd']} (cm, cm, deg) in {T_LOST:.0f} s; "
              f"3 sd over 3 cm after {d['t_3sigma_pos_over_tol']} s, over 5 deg after {d['t_3sigma_heading_over_tol']} s; "
              f"docking gate on sd (3 cm, 3 deg) passed after {d['t_sd_pos_over_gate']} s / {d['t_sd_heading_over_gate']} s")
    print()
    for case, d in along.items():
        print(f"path {case}: rank 3 in {100 * d['share_rank3_instant']:.0f} % of frames (one instant) and "
              f"{100 * d['share_rank3_window']:.0f} % of 1 s windows; markers in view: none {100 * d['share_no_marker']:.0f} %, "
              f"one {100 * d['share_one_marker']:.0f} %, two+ {100 * d['share_two_plus']:.0f} %; longest stretch with none "
              f"{d['longest_no_marker_m']} m; rank at the start of DOCK {d['rank_window_at_dock_start']}")
    OUT.mkdir(exist_ok=True)
    plot(cases, lost, along, dock, OUT / "observability.png")
    for d in lost.values():
        d.pop("_series")
    (OUT / "observability.json").write_text(json.dumps(dict(
        script="evaluation/observability.py", pose=POSE.tolist(), markers={k: v.tolist() for k, v in MARKERS.items()},
        Ts=TS, v=V_NOM, window_frames=N_WIN, seeds="none (deterministic); part 3 uses the paths of marker_layout.py, seed 0",
        cases=cases, marker_lost=lost, along_path=along), indent=1))
    print("\nsaved results/observability.json and results/observability.png")
