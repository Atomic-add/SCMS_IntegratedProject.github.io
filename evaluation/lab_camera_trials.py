"""The lab track with each sim camera: docking, true dock error, EKF marker fusion, dock marker in view.

    python -m evaluation.lab_camera_trials                         # urdf and lab cameras, 4 cases x 10 seeds
    python -m evaluation.lab_camera_trials --cameras lab --seeds 0 1
    python -m evaluation.lab_camera_trials --cameras lab --rate 15 # the lab geometry at the old 15 Hz
    python -m evaluation.lab_camera_trials --from-json results/lab_camera_trials.json   # summary only
    python -m evaluation.lab_camera_trials --resume                # carry on after an interrupted run

Each trial is the run of
    python main.py --task config/arena_lab.json --case CASE --seed SEED --headless --sim-camera CAMERA
(SIM_CAMERAS in src/robot/sim_robot.py; the camera file's K and T_BC go to ArUco and IBVS). Truth is read
here only: the true final pose (robot.d.qpos) against the task's goal_pose, contacts with the world, the
true range to each marker when the EKF first fuses it (ekf.log), when it last fuses it, the longest
stretch with no marker fused, and whether marker 0 is still detected, and how far its corners are from
the image border, in the last second of the dock.
Trials run one after another (one EGL renderer at a time). Writes results/lab_camera_trials.json.
"""
import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root, so "src" imports work
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")                  # offscreen rendering, before mujoco loads

import numpy as np                                              # noqa: E402

from evaluation.ground_truth import true_pose                   # noqa: E402
from src.control.behaviour import run_mission                   # noqa: E402
from src.control.models import load_control_config              # noqa: E402
from src.estimation.ekf import EKF                              # noqa: E402
from src.estimation.odometry import wrap                        # noqa: E402
from src.perception import aruco                                # noqa: E402
from src.robot.corridor import apply_control_overrides, load_task, make_robot  # noqa: E402
from src.robot.sim_robot import PROJECT_ROOT, SIM_CAMERAS       # noqa: E402

TASK = PROJECT_ROOT / "config" / "arena_lab.json"
OUT = PROJECT_ROOT / "results" / "lab_camera_trials.json"
CASES = ("none", "left", "centre", "right")
DOCK_POS_TOL, DOCK_YAW_TOL_DEG = 0.03, 5.0     # true final pose vs goal_pose: same tolerances as docking.py
DOCK_STATES = ("DOCK", "IBVS", "STOP")
LAST_S = 1.0                                   # s: "the dock stop" = the frames of the last second


def contacts(robot):
    """Names of world geoms (not the floor) the robot touches right now."""
    m, d, root = robot.m, robot.d, robot.m.body("base").id
    hit = set()
    for g in zip(d.contact.geom1, d.contact.geom2):
        mine = [m.body_rootid[m.geom_bodyid[i]] == root for i in g]
        if mine[0] != mine[1]:
            other = m.geom(g[1] if mine[0] else g[0]).name
            if other != "floor":
                hit.add(other)
    return hit


def one_trial(cam, case, seed, rate=None):
    spec, intrinsics = SIM_CAMERAS[cam]
    if rate is not None:
        spec = replace(spec, rate_hz=rate)
    task = load_task(TASK)
    robot = make_robot(case, seed=seed, path=TASK, camera=spec)
    est = EKF(task["start"], marker_map=task["markers"])
    cfg = apply_control_overrides(load_control_config(), task)
    cfg["docking"]["camera_intrinsics"] = intrinsics           # as main.py: IBVS uses the same camera file
    dock_id = task["dock"]["marker_id"]
    truth, hits, dock_frames = [], set(), []
    last = {"t": None}

    def on_step(robot, row):
        truth.append((row["t"], *true_pose(robot)))
        hits.update(contacts(robot))
        if row["state"] in DOCK_STATES or dock_frames:
            t_frame, rgb = robot.read_image()                  # the frame run_mission just used (cached)
            if t_frame != last["t"]:
                last["t"] = t_frame
                dock_frames.append((t_frame, aruco.detect(rgb, dock_id)))

    info, t_end = run_mission(case, seed=seed, verbose=False, estimator=est, task=task, cfg=cfg, robot=robot,
                              intrinsics=PROJECT_ROOT / intrinsics, on_step=on_step)
    truth = np.array(truth)
    gx, gy, gth = task["dock"]["goal_pose"]
    x, y, th = true_pose(robot)
    lat, along, yaw = 100 * (x - gx), 100 * (y - gy), float(np.degrees(wrap(th - gth)))
    in_tol = np.hypot(lat, along) <= 100 * DOCK_POS_TOL and abs(yaw) <= DOCK_YAW_TOL_DEG

    fused = {}                                                 # marker id -> first fusion and counts
    for mid in sorted(task["markers"]):
        rows = [r for r in est.log if r[1] == mid]
        used = [r[0] for r in rows if r[3]]
        first = dict(t=None, range_m=None)
        if used:
            t0 = used[0]
            px, py = np.interp(t0, truth[:, 0], truth[:, 1]), np.interp(t0, truth[:, 0], truth[:, 2])
            first = dict(t=round(float(t0), 2), range_m=round(float(np.hypot(*(task["markers"][mid][:2] - [px, py]))), 2))
        fused[str(mid)] = dict(first_t=first["t"], first_range_m=first["range_m"],
                               last_t=round(float(used[-1]), 2) if used else None, n_fused=len(used),
                               n_gated=len(rows) - len(used))
    # longest stretch with no marker fused at all (the EKF on odometry only), from the start to the stop
    t_used = np.unique([0.0] + [r[0] for r in est.log if r[3]] + [t_end])
    k = int(np.argmax(np.diff(t_used)))
    gap = dict(s=round(float(t_used[k + 1] - t_used[k]), 2), from_t=round(float(t_used[k]), 2),
               at_xy=[round(float(np.interp(t_used[k], truth[:, 0], truth[:, i])), 2) for i in (1, 2)])

    w, h = spec.width, spec.height
    tail = [(t, c) for t, c in dock_frames if t >= t_end - LAST_S]
    seen = [c for _, c in tail if c is not None]
    margins = [dict(top=float(c[:, 1].min()), bottom=float(h - 1 - c[:, 1].max()),
                    left=float(c[:, 0].min()), right=float(w - 1 - c[:, 0].max())) for c in seen]
    stop = dict(frames=len(tail), detected=len(seen),
                last_detected=bool(tail and tail[-1][1] is not None),
                min_margin_px={k: round(min(mg[k] for mg in margins), 1) for k in ("top", "bottom", "left", "right")}
                if margins else None,
                dock_frames=len(dock_frames), dock_lost=sum(c is None for _, c in dock_frames))
    return dict(camera=cam, rate_hz=spec.rate_hz, case=case, seed=seed,
                success=bool(info["dock_complete"] and not hits and in_tol), dock_complete=bool(info["dock_complete"]),
                stop_reason=info["stop_reason"], time_s=round(float(t_end), 1), contacts=sorted(hits),
                dock_lateral_cm=round(lat, 2), dock_along_cm=round(along, 2), dock_yaw_deg=round(yaw, 2),
                dock_err_cm=round(float(np.hypot(lat, along)), 2), fused=fused, blind_gap=gap, marker0_at_stop=stop)


def label(t):
    return t["camera"] if t["rate_hz"] == SIM_CAMERAS[t["camera"]][0].rate_hz else f"{t['camera']}@{t['rate_hz']:g}Hz"


def summary(trials):
    """One block of numbers per camera: what the final report table shows."""
    out = {}
    for name in dict.fromkeys(label(t) for t in trials):
        T = [t for t in trials if label(t) == name]
        D = [t for t in T if t["dock_complete"]]
        err = np.array([t["dock_err_cm"] for t in D]) if D else np.zeros(0)
        yaw = np.array([abs(t["dock_yaw_deg"]) for t in D]) if D else np.zeros(0)
        s = dict(n=len(T), docked=len(D), success=sum(t["success"] for t in T),
                 per_case={c: f"{sum(t['success'] for t in T if t['case'] == c)}/{sum(t['case'] == c for t in T)}"
                           for c in CASES},
                 failures=[f"{t['case']} s{t['seed']}: {t['stop_reason']}" + (f" contact {t['contacts']}" if t["contacts"] else "")
                           for t in T if not t["success"]],
                 dock_err_cm=dict(mean=round(float(err.mean()), 2), max=round(float(err.max()), 2)) if err.size else None,
                 dock_yaw_deg=dict(mean=round(float(yaw.mean()), 2), max=round(float(yaw.max()), 2)) if yaw.size else None,
                 time_s=round(float(np.mean([t["time_s"] for t in T])), 1))
        for mid in T[0]["fused"]:
            ts = [t["fused"][mid]["first_t"] for t in T if t["fused"][mid]["first_t"] is not None]
            rs = [t["fused"][mid]["first_range_m"] for t in T if t["fused"][mid]["first_t"] is not None]
            s[f"marker{mid}_first_fused"] = dict(
                runs=f"{len(ts)}/{len(T)}", t_median=round(float(np.median(ts)), 2) if ts else None,
                t_range=[round(min(ts), 2), round(max(ts), 2)] if ts else None,
                range_m_median=round(float(np.median(rs)), 2) if rs else None,
                last_t_median=round(float(np.median([t["fused"][mid]["last_t"] for t in T
                                                     if t["fused"][mid]["last_t"] is not None])), 2) if ts else None,
                n_fused_median=float(np.median([t["fused"][mid]["n_fused"] for t in T])),
                n_gated_total=sum(t["fused"][mid]["n_gated"] for t in T))
        gaps = [t["blind_gap"]["s"] for t in T]
        worst = T[int(np.argmax(gaps))]
        s["blind_gap_s"] = dict(median=round(float(np.median(gaps)), 2), max=max(gaps),
                                worst=f"{worst['case']} s{worst['seed']} from t {worst['blind_gap']['from_t']} "
                                      f"at {worst['blind_gap']['at_xy']}")
        st = [t["marker0_at_stop"] for t in T]
        tops = [m["min_margin_px"]["top"] for m in st if m["min_margin_px"]]
        bots = [m["min_margin_px"]["bottom"] for m in st if m["min_margin_px"]]
        s["marker0_at_stop"] = dict(
            last_frame_detected=f"{sum(m['last_detected'] for m in st)}/{len(st)}",
            all_last_1s_detected=f"{sum(m['frames'] > 0 and m['detected'] == m['frames'] for m in st)}/{len(st)}",
            min_top_margin_px=min(tops) if tops else None, min_bottom_margin_px=min(bots) if bots else None,
            dock_frames_lost=f"{sum(m['dock_lost'] for m in st)}/{sum(m['dock_frames'] for m in st)}")
        out[name] = s
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cameras", nargs="*", default=list(SIM_CAMERAS), choices=list(SIM_CAMERAS))
    ap.add_argument("--cases", nargs="*", default=list(CASES), choices=CASES)
    ap.add_argument("--seeds", type=int, nargs="*", default=list(range(10)), help="default 0..9")
    ap.add_argument("--rate", type=float, default=None, help="camera frame rate [Hz] instead of the camera's own")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--from-json", type=Path, help="print the summary of a saved file, no simulation")
    ap.add_argument("--resume", action="store_true", help="keep the trials already in --out and run only the rest")
    args = ap.parse_args()
    if args.from_json:
        print(json.dumps(summary(json.loads(args.from_json.read_text())["trials"]), indent=1))
        return
    trials = json.loads(args.out.read_text())["trials"] if args.resume and args.out.exists() else []
    done = {(t["camera"], t["rate_hz"], t["case"], t["seed"]) for t in trials}
    for cam in args.cameras:
        rate = args.rate if args.rate is not None else SIM_CAMERAS[cam][0].rate_hz
        for case in args.cases:
            for seed in args.seeds:
                if (cam, rate, case, seed) in done:
                    continue
                t = one_trial(cam, case, seed, args.rate)
                trials.append(t)
                f = t["fused"]
                print(f"{label(t):10s} {case:6s} seed {seed}: {'ok ' if t['success'] else 'FAIL'} {t['stop_reason']:12s} "
                      f"dock err {t['dock_err_cm']:5.2f} cm yaw {t['dock_yaw_deg']:+5.2f} deg  first fused "
                      + " ".join(f"{k}:{v['first_t']}-{v['last_t']}" for k, v in f.items())
                      + f"  blind {t['blind_gap']['s']} s"
                      + f"  m0 at stop {t['marker0_at_stop']['detected']}/{t['marker0_at_stop']['frames']}", flush=True)
                args.out.parent.mkdir(exist_ok=True)
                args.out.write_text(json.dumps(dict(task=str(TASK.relative_to(PROJECT_ROOT)), trials=trials,
                                                    summary=summary(trials)), indent=1))
    print(json.dumps(summary(trials), indent=1))
    print(f"wrote {args.out.relative_to(PROJECT_ROOT) if args.out.is_relative_to(PROJECT_ROOT) else args.out}")


if __name__ == "__main__":
    main()
