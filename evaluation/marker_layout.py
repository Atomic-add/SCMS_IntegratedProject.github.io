"""Where to put the ArUco landmarks on the lab track so the real camera sees two at once as often as possible.

    python -m evaluation.marker_layout                # record the true paths, rank layouts -> results/marker_layout.*
    python -m evaluation.marker_layout --from-paths   # re-rank from results/marker_layout_paths.json (no simulation)

The EKF fuses range and bearing to known markers (Week 7 C.4). One marker gives 2 constraints on the
3-DOF pose, so with one marker in view the pose is pinned only through the motion; two in view pin it at
once (Week 7 A.8: observability depends on where the robot is). This script follows the robot's TRUE path
through config/arena_lab.json and finds, frame by frame, which candidate marker spots the REAL camera
would see, then ranks layouts (the dock marker 0 is fixed) by the share of the run with >= 2 in view.

Candidate spots: every SPACING along the inner face of the panels (tile centres and tile edges), centre
at the task's marker height, facing into the arena. A marker counts as seen in a frame when
  - all four corners project inside the 640 x 480 image with MARGIN px to spare (pinhole with the K and
    T_BC of config/camera_real.json, i.e. the 6.3 deg tilt; Week 3 Part A projection),
  - its side spans at least MIN_SIDE_PX pixels (detection range),
  - the camera is within MAX_INCIDENCE_DEG of the marker normal (Week 3 Part E: the planar pose degrades
    at grazing views),
  - the sight line from the camera to the marker centre crosses no box and no panel (both 0.20 m tall,
    above the camera and the marker centres).
The paths are sim runs with the lab camera (cases none and centre, seed 0); the true pose is read here
only. The path barely depends on where the landmarks are, so one recording serves every layout.

Counting markers in view ignores geometry (two markers 0.25 m apart pin the pose poorly from 3 m), so the
best layouts by count are then judged by what matters, the EKF's pose error: src/estimation/ekf.py is
replayed on each true path with encoder angles from the true motion (+1.3 % scale, the lab robot's
tape-vs-encoder runs on 23 Oct, quantised to ENCODER_CPR), a prior off by a draw from P0, and a
range-bearing measurement of every marker in view in each frame, noisy by the real camera's range_sd
(camera_real.json) and bearing_sd_deg (estimation.json). N_SEEDS draws per case.
Adapted from 41014 Week 3 Parts A and E (projection, marker corners) and Week 7 C.4 (range-bearing EKF).
Additions: the candidate spots on the lab panels, the occlusion test, the layout search and ranking.
"""
import argparse
import itertools
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

from src.perception import aruco, camera                        # noqa: E402
from src.robot.corridor import load_config                      # noqa: E402
from src.robot.sim_robot import PROJECT_ROOT                    # noqa: E402

TASK = PROJECT_ROOT / "config" / "arena_lab.json"
CAMERA = PROJECT_ROOT / "config" / "camera_real.json"
OUT = PROJECT_ROOT / "results" / "marker_layout"
CASES = ("none", "centre")
SEED = 0
W, H = 640, 480                 # px, the real camera's image (camera_real.json image_size)
MARGIN = 2                      # px kept clear of the image border: a corner on the border is not detected
MIN_SIDE_PX = 20.0              # px: the real camera detected 5/5 frames at 2.08 m (34 px, results/hardware/
                                # aruco_cam2.0m); 20 px (about 3.4 m at f = 692 px) is an ASSUMED limit, so the
                                # ranking is repeated at MIN_SIDE_STRICT
MIN_SIDE_STRICT = 30.0          # px: the conservative limit (about 2.3 m), only for the robustness column
MAX_INCIDENCE_DEG = 60.0        # deg: design budget; beyond it the planar pose ambiguity grows (W3 Part E)
SPACING = 0.25                  # m: tile centres and tile edges of the 0.5 m floor tiles
END_CLEAR = 0.20                # m: no spot this close to a panel corner (sheet half width 0.067 + panel joints)
DOCK_CLEAR = 0.20               # m: no landmark this close to the dock marker (the sheets are 0.133 m wide)
ENCODER_SCALE = 0.013           # the lab robot's encoders count 1.3 % more than the tape (3 x 0.5 m, 23 Oct)
N_SEEDS = 20                    # EKF replays per case and layout (the guide's N >= 20)
N_EKF = 15                      # best layouts by count, per marker count, that go to the EKF replay
FIX_SETTLE = 2.0                # s after the first fused marker before the "after the first fix" errors count


def candidate_spots(outline, z):
    """Marker spots on the inner face of every outline edge: list of dict(xy, normal, z, edge)."""
    pts = np.asarray(outline, float)
    ccw = np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1]) > 0
    spots = []
    for i, (a, b) in enumerate(zip(pts, np.roll(pts, -1, axis=0))):
        L = float(np.hypot(*(b - a)))
        d = (b - a) / L
        n = np.array([-d[1], d[0]]) if ccw else np.array([d[1], -d[0]])     # inward normal
        for s in np.arange(SPACING, L - END_CLEAR + 1e-9, SPACING):
            if s < END_CLEAR:
                continue
            spots.append(dict(xy=(a + s * d).round(3).tolist(), normal=n.round(6).tolist(), z=z, edge=i))
    return spots


def edge_of(outline, xy):
    """Index of the outline edge the point xy lies on."""
    pts = np.asarray(outline, float)
    for i, (a, b) in enumerate(zip(pts, np.roll(pts, -1, axis=0))):
        d, v = b - a, np.asarray(xy, float) - a
        if abs(d[0] * v[1] - d[1] * v[0]) < 1e-6 and -1e-6 <= v @ d <= d @ d + 1e-6:
            return i
    raise ValueError(f"{xy} is not on the outline")


def seg_cross(p, q, a, b):
    """True where segments p->q (arrays N x 2) cross the segment a->b (strictly, endpoints excluded)."""
    def orient(u, v, w):
        return (v[..., 0] - u[..., 0]) * (w[..., 1] - u[..., 1]) - (v[..., 1] - u[..., 1]) * (w[..., 0] - u[..., 0])
    a, b = np.broadcast_to(a, p.shape), np.broadcast_to(b, p.shape)
    d1, d2 = orient(p, q, a), orient(p, q, b)
    d3, d4 = orient(a, b, p), orient(a, b, q)
    return (d1 * d2 < -1e-12) & (d3 * d4 < -1e-12)


def visibility(spot, cams, K, outline, boxes, min_side):
    """Bool per camera pose: is this marker spot seen (corners in image, size, incidence, no occlusion)?

    cams: N x 4 x 4 world-to-camera transforms T_CW (OpenCV camera axes)."""
    c = np.array([*spot["xy"], spot["z"]])
    n = np.array([*spot["normal"], 0.0])
    r = np.array([-n[1], n[0], 0.0])                           # along the panel
    up = np.array([0.0, 0.0, 1.0])
    s = aruco.MARKER_SIDE / 2
    corners = np.array([c + s * (-r + up), c + s * (r + up), c + s * (r - up), c + s * (-r - up)])   # 4 x 3
    P = np.einsum("nij,kj->nki", cams[:, :3, :3], corners) + cams[:, None, :3, 3]                  # N x 4 x 3
    front = np.all(P[..., 2] > 0.05, axis=1)
    uv = np.einsum("ij,nkj->nki", K, P / np.where(P[..., 2:] > 0.05, P[..., 2:], 1.0))[..., :2]   # N x 4 x 2
    inside = np.all((uv[..., 0] >= MARGIN) & (uv[..., 0] <= W - 1 - MARGIN) &
                    (uv[..., 1] >= MARGIN) & (uv[..., 1] <= H - 1 - MARGIN), axis=1)
    side = np.mean(np.linalg.norm(uv - np.roll(uv, 1, axis=1), axis=2), axis=1)
    cam_xyz = -np.einsum("nji,nj->ni", cams[:, :3, :3], cams[:, :3, 3])          # camera centre in {W}
    to_cam = cam_xyz - c
    cos_inc = (to_cam @ n) / np.linalg.norm(to_cam, axis=1)
    seen = front & inside & (side >= min_side) & (cos_inc >= np.cos(np.radians(MAX_INCIDENCE_DEG)))
    p, q = cam_xyz[:, :2], np.broadcast_to(c[:2], cam_xyz[:, :2].shape)
    pts = np.asarray(outline, float)
    for i, (a, b) in enumerate(zip(pts, np.roll(pts, -1, axis=0))):
        if i != spot["edge"]:                                  # the marker's own panel is behind it
            seen &= ~seg_cross(p, q, a, b)
    for (bx, by), half in boxes:
        sq = np.array([[bx - half, by - half], [bx + half, by - half], [bx + half, by + half], [bx - half, by + half]])
        for a, b in zip(sq, np.roll(sq, -1, axis=0)):
            seen &= ~seg_cross(p, q, a, b)
    return seen, side


def camera_poses(path, T_BC):
    """World-to-camera transforms T_CW (N x 4 x 4) for base poses path = N x [x, y, theta]."""
    out = []
    for x, y, th in path:
        T_WB = np.eye(4)
        T_WB[:2, :2] = [[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]
        T_WB[:2, 3] = [x, y]
        out.append(np.linalg.inv(T_WB @ T_BC.A))
    return np.array(out)


def record_paths():
    """True base poses of the sim runs (lab camera), per case: dict(case -> dict(t, pose, state))."""
    from evaluation.ground_truth import true_pose
    from src.control.behaviour import run_mission
    from src.control.models import load_control_config
    from src.estimation.ekf import EKF
    from src.estimation.models import load_estimation_config
    from src.robot.corridor import apply_control_overrides, load_task, make_robot
    from src.robot.sim_robot import SIM_CAMERAS

    cam_spec, cam_file = SIM_CAMERAS["lab"]
    task = load_task(TASK)
    paths = {}
    for case in CASES:
        rows = []
        info, t_end = run_mission(case, seed=SEED, verbose=False, task=task,
                                  estimator=EKF(task["start"], marker_map=task["markers"],
                                                params=load_estimation_config(camera=PROJECT_ROOT / cam_file)),
                                  robot=make_robot(case, seed=SEED, path=TASK, camera=cam_spec),
                                  cfg=apply_control_overrides(load_control_config(), task),
                                  intrinsics=PROJECT_ROOT / cam_file,
                                  on_step=lambda robot, row: rows.append((row["t"], *np.asarray(true_pose(robot), float),
                                                                          row["state"])))
        paths[case] = dict(t=[r[0] for r in rows], pose=[list(r[1:4]) for r in rows], state=[r[4] for r in rows],
                           result=info["stop_reason"], time_s=round(float(t_end), 1))
        print(f"  {case}: {info['stop_reason']} after {t_end:.1f} s, {len(rows)} steps")
    return paths


def run_lengths(mask, dt):
    """Longest run of True in mask, in seconds."""
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return round(best * dt, 2)


def score(count, dts):
    """Layout metrics from the per-frame count of markers in view (concatenated cases, list of (slice, dt))."""
    return dict(share_ge2=round(float(np.mean(count >= 2)), 3), share_ge1=round(float(np.mean(count >= 1)), 3),
                longest_none_s=max(run_lengths(count[sl] == 0, dt) for sl, dt in dts),
                longest_below2_s=max(run_lengths(count[sl] < 2, dt) for sl, dt in dts))


def ekf_replay(t, pose, state, vis, layout, seed, params):
    """Pose error of src/estimation/ekf.py along one true path with synthetic encoders and markers.

    vis: dict marker id -> bool per frame (in view). Errors over the whole run, and from FIX_SETTLE after
    the first fused marker on (the start, where no marker is close enough whatever the layout, is the
    same for every layout), plus the error when docking takes over (the first DOCK frame)."""
    from src.estimation.ekf import EKF
    from src.estimation.models import R_marker, h_rb
    from src.estimation.odometry import wrap
    from src.robot.sim_robot import ENCODER_CPR, WHEEL_RADIUS, WHEEL_SEPARATION

    rng = np.random.default_rng(seed)
    pose = np.asarray(pose, float)
    dth = wrap(np.diff(pose[:, 2]))
    mid = pose[:-1, 2] + dth / 2                                  # midpoint heading (W2 B.0)
    ds = np.diff(pose[:, 0]) * np.cos(mid) + np.diff(pose[:, 1]) * np.sin(mid)
    q = 2 * np.pi / ENCODER_CPR
    phi = np.zeros((len(pose), 2))
    phi[1:, 0] = np.cumsum((ds - dth * WHEEL_SEPARATION / 2) / WHEEL_RADIUS) * (1 + ENCODER_SCALE)
    phi[1:, 1] = np.cumsum((ds + dth * WHEEL_SEPARATION / 2) / WHEEL_RADIUS) * (1 + ENCODER_SCALE)
    phi = np.round(phi / q) * q
    sd0 = np.array([params["P0_sd"][0], params["P0_sd"][1], np.radians(params["P0_sd"][2])])
    x0 = pose[0] + np.clip(rng.normal(0.0, sd0), -2 * sd0, 2 * sd0)       # the prior is off by a P0 draw
    ekf = EKF(x0, marker_map=layout, params=params)
    err, nees, t_fix = [], [], None
    for k in range(len(pose)):
        frame = {"t": t[k], "markers": {}}
        for mid_, seen in vis.items():
            if seen[k]:
                z = h_rb(pose[k], layout[mid_])
                Rn = R_marker(z[0], params)
                frame["markers"][mid_] = {"z": z + rng.normal(0.0, np.sqrt(np.diag(Rn)))}
        xh, P = ekf.step((t[k], phi[k, 0], phi[k, 1]), frame)
        if t_fix is None and ekf.last_update_ids:
            t_fix = t[k]
        e = pose[k] - xh
        e[2] = wrap(e[2])
        err.append(e)
        nees.append(float(e @ np.linalg.solve(P, e)))
    err, t = np.array(err), np.asarray(t, float)
    d = np.hypot(err[:, 0], err[:, 1])
    late = t >= (t_fix if t_fix is not None else t[-1]) + FIX_SETTLE
    k_dock = next(k for k, st in enumerate(state) if st in ("DOCK", "IBVS"))
    return dict(pos_rms=float(np.sqrt(np.mean(d ** 2))), pos_max=float(d.max()),
                head_rms_deg=float(np.degrees(np.sqrt(np.mean(err[:, 2] ** 2)))), anees=float(np.mean(nees)),
                late_pos_rms=float(np.sqrt(np.mean(d[late] ** 2))), late_pos_max=float(d[late].max()),
                late_head_rms_deg=float(np.degrees(np.sqrt(np.mean(err[late, 2] ** 2)))),
                late_anees=float(np.mean(np.array(nees)[late])),
                dock_pos=float(d[k_dock]), dock_head_deg=float(np.degrees(abs(err[k_dock, 2]))))


def judge(layouts, paths, vis_case, vis0_case, markers0, params):
    """Mean EKF errors of each layout (list of spot index lists) over N_SEEDS x CASES replays."""
    out = []
    for idx, spots_xy in layouts:
        layout = {0: np.asarray(markers0, float)}
        layout.update({k + 1: np.array([*xy, 0.0, 0.0]) for k, xy in enumerate(spots_xy)})
        runs = []
        for case in CASES:
            vis = {0: vis0_case[case]}
            vis.update({k + 1: vis_case[case][i] for k, i in enumerate(idx)})
            for seed in range(N_SEEDS):
                runs.append(ekf_replay(paths[case]["t"], paths[case]["pose"], paths[case]["state"], vis, layout,
                                       1000 + seed, params))
        out.append(dict(spots=spots_xy, **{k: round(float(np.mean([r[k] for r in runs])), 4) for k in runs[0]},
                        late_pos_max_worst=round(float(max(r["late_pos_max"] for r in runs)), 4),
                        dock_pos_worst=round(float(max(r["dock_pos"] for r in runs)), 4)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from-paths", action="store_true", help="re-rank from the saved true paths, no simulation")
    ap.add_argument("--top", type=int, default=5, help="layouts listed per marker count (default 5)")
    args = ap.parse_args()

    cfg = load_config(TASK)
    paths_file = OUT.with_name(OUT.name + "_paths.json")
    if args.from_paths:
        paths = json.loads(paths_file.read_text())["paths"]
    else:
        print(f"recording the true paths on {TASK.name} (lab camera, seed {SEED})")
        paths = record_paths()
        paths_file.parent.mkdir(parents=True, exist_ok=True)
        paths_file.write_text(json.dumps(dict(task=TASK.name, seed=SEED, cases=list(CASES), paths=paths,
                                              script="evaluation/marker_layout.py")))

    K, _, _ = camera.load_intrinsics(CAMERA)
    T_BC = aruco.load_T_BC(CAMERA)
    markers = {int(k): v for k, v in cfg["markers"].items() if not k.startswith("_")}
    z = float(markers[cfg["dock"]["marker_id"]][2])
    spots = [s for s in candidate_spots(cfg["outline"], z)
             if np.hypot(*(np.array(s["xy"]) - markers[0][:2])) >= DOCK_CLEAR]
    m0 = markers[cfg["dock"]["marker_id"]]
    dock_spot = dict(xy=list(m0[:2]), normal=[np.cos(m0[3]), np.sin(m0[3])], z=z, edge=edge_of(cfg["outline"], m0[:2]))

    results = {}
    for limit in (MIN_SIDE_PX, MIN_SIDE_STRICT):
        vis, vis0, dts, k0 = [], [], [], 0
        for case in CASES:
            pose = np.array(paths[case]["pose"])
            cams = camera_poses(pose, T_BC)
            boxes = [] if cfg["obstacles"][case] is None else [(cfg["obstacles"][case], 0.10)]
            vis.append(np.array([visibility(s, cams, K, cfg["outline"], boxes, limit)[0] for s in spots]))
            vis0.append(visibility(dock_spot, cams, K, cfg["outline"], boxes, limit)[0])
            dt = float(np.median(np.diff(paths[case]["t"])))
            dts.append((slice(k0, k0 + len(pose)), dt))
            k0 += len(pose)
        if limit == MIN_SIDE_PX:                               # per case, for the EKF replay
            vis_case = {c: v for c, v in zip(CASES, vis)}
            vis0_case = {c: v for c, v in zip(CASES, vis0)}
        V, V0 = np.hstack(vis), np.concatenate(vis0).astype(int)
        index = {tuple(s["xy"]): i for i, s in enumerate(spots)}
        current = [index[tuple(np.round(v[:2], 3))] for k, v in sorted(markers.items()) if k != cfg["dock"]["marker_id"]]
        ranked = {}
        for n_land in (2, 3):
            rows = []
            for combo in itertools.combinations(range(len(spots)), n_land):
                count = V0 + V[list(combo)].sum(axis=0)
                rows.append((dict(spots=[spots[i]["xy"] for i in combo], **score(count, dts))))
            rows.sort(key=lambda r: (-r["share_ge2"], r["longest_none_s"], -r["share_ge1"]))
            ranked[n_land + 1] = rows[:max(args.top, N_EKF)]
        cur = dict(spots=[spots[i]["xy"] for i in current], **score(V0 + V[current].sum(axis=0), dts))
        results[f"min_side_{int(limit)}px"] = dict(current=cur, best=ranked,
                                                   spot_share=[round(float(v.mean()), 3) for v in V],
                                                   dock_share=round(float(V0.mean()), 3))
        print(f"\nmarker counts as seen at >= {limit:.0f} px (incidence <= {MAX_INCIDENCE_DEG:.0f} deg):")
        print(f"  current  3 markers {cur['spots']}: >=2 in view {100 * cur['share_ge2']:.0f} %, >=1 "
              f"{100 * cur['share_ge1']:.0f} %, longest with none {cur['longest_none_s']} s")
        for n_mk, rows in ranked.items():
            for r in rows[:3]:
                print(f"  best {n_mk} markers {r['spots']}: >=2 in view {100 * r['share_ge2']:.0f} %, >=1 "
                      f"{100 * r['share_ge1']:.0f} %, longest with none {r['longest_none_s']} s, "
                      f"longest below 2 {r['longest_below2_s']} s")

    # ---- the EKF replay: current layout and the best by count, judged by pose error
    from src.estimation.models import load_estimation_config
    params = load_estimation_config(camera=CAMERA)
    main_res = results[f"min_side_{int(MIN_SIDE_PX)}px"]
    index = {tuple(s["xy"]): i for i, s in enumerate(spots)}
    candidates = [("current", main_res["current"]["spots"])]
    for n_mk in (3, 4):
        candidates += [(f"{n_mk} markers", r["spots"]) for r in main_res["best"][n_mk][:N_EKF]]
    print(f"\nEKF replay ({N_SEEDS} seeds x {len(CASES)} cases per layout, real camera noise, +{100 * ENCODER_SCALE:.1f} % "
          "encoder scale):")
    judged = judge([([index[tuple(xy)] for xy in sp], sp) for _, sp in candidates], paths, vis_case, vis0_case,
                   markers[cfg["dock"]["marker_id"]], params)
    for (kind, _), j in zip(candidates, judged):
        j["kind"] = kind
    ekf_rank = {}
    for kind in ("current", "3 markers", "4 markers"):
        rows = sorted([j for j in judged if j["kind"] == kind], key=lambda j: j["late_pos_rms"])
        ekf_rank[kind] = rows
        for j in rows[:4]:
            print(f"  {kind:9s} {j['spots']}: after the first fix rms {100 * j['late_pos_rms']:.2f} cm, max "
                  f"{100 * j['late_pos_max']:.2f} cm (worst {100 * j['late_pos_max_worst']:.2f}), heading "
                  f"{j['late_head_rms_deg']:.2f} deg, ANEES {j['late_anees']:.2f} | at DOCK {100 * j['dock_pos']:.2f} cm "
                  f"(worst {100 * j['dock_pos_worst']:.2f}) {j['dock_head_deg']:.2f} deg | whole run rms {100 * j['pos_rms']:.2f} cm")
    summary = dict(task=TASK.name, camera=CAMERA.name, seed=SEED, cases=list(CASES), spacing_m=SPACING,
                   margin_px=MARGIN, max_incidence_deg=MAX_INCIDENCE_DEG, encoder_scale=ENCODER_SCALE,
                   n_seeds=N_SEEDS, ekf_seeds=list(range(1000, 1000 + N_SEEDS)), spots=spots, results=results,
                   ekf=ekf_rank, script="evaluation/marker_layout.py")
    OUT.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    plot(cfg, spots, paths, results[f"min_side_{int(MIN_SIDE_PX)}px"])
    print(f"wrote {OUT.with_suffix('.json').relative_to(PROJECT_ROOT)} and .png")


def plot(cfg, spots, paths, res):
    """Map: candidate spots coloured by how long each is in view; current and best layouts marked."""
    outline = np.array(cfg["outline"] + [cfg["outline"][0]], float)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(*outline.T, color="0.3", lw=2)
    for case, style in zip(CASES, ("-", "--")):
        p = np.array(paths[case]["pose"])
        ax.plot(p[:, 0], p[:, 1], style, color="0.6", lw=1, label=f"true path, box {case}")
    xy = np.array([s["xy"] for s in spots])
    sc = ax.scatter(*xy.T, c=100 * np.array(res["spot_share"]), cmap="viridis", s=36, zorder=3)
    fig.colorbar(sc, ax=ax, label="share of the run in view [%]")
    best3, best4 = np.array(res["best"][3][0]["spots"]), np.array(res["best"][4][0]["spots"])
    cur = np.array(res["current"]["spots"])
    ax.scatter(*cur.T, marker="s", s=140, facecolors="none", edgecolors="tab:red", lw=2, label="current landmarks")
    ax.scatter(*best3.T, marker="o", s=230, facecolors="none", edgecolors="tab:orange", lw=2, label="best, 3 markers")
    ax.scatter(*best4.T, marker="D", s=130, facecolors="none", edgecolors="tab:blue", lw=1.5, label="best, 4 markers")
    m0 = cfg["markers"][str(cfg["dock"]["marker_id"])]
    ax.plot(m0[0], m0[1], "k*", ms=14, label="dock marker 0")
    ax.set_aspect("equal")
    ax.set(xlabel="x [m]", ylabel="y [m]", title=f"Landmark spots on the lab panels (seen at >= {MIN_SIDE_PX:.0f} px, "
           f"<= {MAX_INCIDENCE_DEG:.0f} deg incidence)")
    ax.legend(fontsize=7, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout()
    fig.savefig(OUT.with_suffix(".png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
