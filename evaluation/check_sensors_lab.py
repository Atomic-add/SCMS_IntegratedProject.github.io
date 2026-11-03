"""The LiDAR and camera checks on the lab track itself (config/arena_lab.json, scene_lab.xml).

    MUJOCO_GL=egl python -m evaluation.check_sensors_lab

evaluation/check_lidar.py and evaluation/check_camera.py run in their own test rooms. This script repeats the
same checks on the track the robot drives, at poses on its route (box in the lane, case "centre"), so every
sensor number on the project page comes from the one environment used in the sim and on the real robot.

LiDAR (Week 4 Part 5 raycast / simulate_scan, Week 2 C.2):
  1. noise-free scans against the independent analytic ray caster (evaluation/ground_truth.raycast_2d over
     the panels, the furniture and the box), five route poses;
  2. range noise and lost beams against LidarSpec over 100 scans;
  3. the frame chain p_W = T_WB T_BS p_S: full chain, translation forgotten, 2 deg mounting yaw error;
  4. forward coverage (the fraction of the +-90 deg beams that return) at each pose.
Camera (Week 3 Part E, Week 3 wrap-up law sigma_Z ~ Z^2 sigma_px / (f s), Week 2 C.2):
  5. the three lab markers measured from route poses (approach to the dock marker; markers 1 and 2 from the
     long leg), 20 frames each: detection rate, range / bearing bias and sd against the truth, and the
     EKF's noise model for this camera (config/camera_sim.json "ekf_noise");
  6. planar-pose ambiguity: the two IPPE solutions' range / bearing difference in the flagged frames;
  7. the frame chain for the marker in {W}: full chain, translation forgotten, 2 deg mounting yaw error.
Writes results/lab_sensors.json, results/page/lab_lidar.png, results/page/lab_raycast.png (one scan on the
track, check 1's result in the title), results/page/lab_lidar_ranges.png (check 2 in the Assessment 3 style) and results/page/lab_aruco.png.

Only this file (in evaluation/) reads the simulator truth. Adapted from evaluation/check_lidar.py and
evaluation/check_camera.py (same checks); additions: the lab track, route poses, coverage, the three lab markers
against the EKF's own noise model.
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root, so "src" imports work

import matplotlib                                               # noqa: E402
import numpy as np                                              # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402
from spatialmath import SE2                                     # noqa: E402

from evaluation.ground_truth import (raycast_2d, room_shapes, true_marker_pose, true_pose, true_pose_se3,  # noqa: E402
                                     true_sensor_pose)
from evaluation.run_demo import CFG, draw_map                   # noqa: E402
from src.estimation.odometry import wrap                        # noqa: E402
from src.perception import aruco, camera, lidar                 # noqa: E402
from src.robot.sim_robot import PROJECT_ROOT, SIM_INTRINSICS, CameraSpec, LidarSpec   # noqa: E402
from src.robot.task import make_robot                           # noqa: E402

OUT = PROJECT_ROOT / "results"
SPEC = LidarSpec(**{k: v for k, v in CFG["lidar"].items() if not k.startswith("_")})
ROUTE = {"start": (4.5, 1.25, np.pi), "box ahead": (3.7, 1.25, np.pi), "beside the box": (2.75, 0.86, np.pi),
         "corner, turning": (0.5, 1.25, -np.pi / 2 + 0.6), "pre-dock": (0.5, 0.74, -np.pi / 2)}
MARKERS = {0: "aruco_dock", 1: "aruco_L1", 2: "aruco_L2"}
N_FRAMES = 20
GRID = dict(ls="--", alpha=0.5)
FAILS = []


def check(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILS.append(name)


def forward_coverage(rho, alpha, spec):
    fwd = np.abs(wrap(alpha + lidar.T_BS.theta())) <= np.pi / 2
    return float(lidar.valid_beams(rho, spec.range_min, spec.range_max)[fwd].mean())


# ---------------------------------------------------------------- LiDAR
def lidar_checks():
    res = {}
    exact = make_robot("centre", seed=0, lidar=replace(SPEC, sigma=0.0, dropout=0.0))
    with exact:
        alpha = exact.scan_angles
        worst, cover, scans = 0.0, {}, {}
        for name, pose in ROUTE.items():
            exact.reset(*pose)
            _, rho = exact.read_scan()
            rho_true = raycast_2d(true_sensor_pose(exact), alpha, room_shapes(exact))
            inside = (rho_true >= SPEC.range_min) & (rho_true <= SPEC.range_max)
            worst = max(worst, float(np.max(np.abs(rho[inside] - rho_true[inside]))))
            cover[name] = forward_coverage(rho, alpha, SPEC)
            scans[name] = dict(T_WS=true_sensor_pose(exact), rho=rho, rho_true=rho_true, inside=inside)
        check("LiDAR ranges vs analytic ray caster on the lab track (5 route poses x 360 beams, noise off)",
              worst < 1e-3, f"max |error| {worst * 1000:.3f} mm (beams within {SPEC.range_min}-{SPEC.range_max} m)")
        res.update(raycast_max_err_mm=worst * 1000, forward_coverage=cover)
        res["raycast_scans"] = scans
    noisy = make_robot("centre", seed=1)
    with noisy:
        spec = noisy.lidar
        noisy.reset(*ROUTE["box ahead"])
        r_all, n_lost, n_total = [], 0, 0
        for _ in range(100):
            noisy.step(1.0 / spec.rate_hz)
            _, rho = noisy.read_scan()
            rho_true = raycast_2d(true_sensor_pose(noisy), alpha, room_shapes(noisy))
            inside = (rho_true >= spec.range_min) & (rho_true <= spec.range_max)
            ok = lidar.valid_beams(rho, spec.range_min, spec.range_max) & inside
            r_all.append(rho[ok] - rho_true[ok])
            res["noisy_scan"] = (alpha, rho.copy(), rho_true.copy(), inside)   # the last one, for the figure
            n_lost += int((~lidar.valid_beams(rho, spec.range_min, spec.range_max) & inside).sum())
            n_total += int(inside.sum())
        r_all = np.concatenate(r_all)
        lost = n_lost / n_total
        check("range noise and lost beams match LidarSpec", abs(r_all.std() / spec.sigma - 1) < 0.05 and abs(lost - spec.dropout) < 0.004,
              f"residual mean {r_all.mean() * 1000:+.2f} mm, sd {r_all.std() * 1000:.2f} mm (model {spec.sigma * 1000:.0f} mm), "
              f"{100 * lost:.2f} % lost (model {100 * spec.dropout:.0f} %), {r_all.size} returns")
        res.update(noise_sd_mm=r_all.std() * 1000, noise_mean_mm=r_all.mean() * 1000, lost_pct=100 * lost)
        noisy.reset(*ROUTE["beside the box"])
        _, rho = noisy.read_scan()
        ok = lidar.valid_beams(rho, spec.range_min, spec.range_max)
        P_S = lidar.scan_to_points(rho, alpha, ok)
        T_WS, x_true = true_sensor_pose(noisy), true_pose(noisy)
        hits = T_WS * lidar.scan_to_points(raycast_2d(T_WS, alpha, room_shapes(noisy)), alpha, ok)
        variants = {"full chain": lidar.T_BS, "translation forgotten": SE2(0.0, 0.0, 0.0),
                    "2 deg mounting yaw error": SE2(*lidar.T_BS.t, lidar.T_BS.theta() + np.deg2rad(2))}
        rms = {k: float(np.sqrt(np.mean(np.sum((lidar.to_world(P_S, x_true, T) - hits) ** 2, axis=0)))) * 1000
               for k, T in variants.items()}
        check("LiDAR world points with the full frame chain (beside the box)", rms["full chain"] < 15,
              ", ".join(f"{k} {v:.0f} mm" for k, v in rms.items()))
        res.update(frame_chain_rms_mm=rms)
        res["residuals"] = r_all
        res["scan_beside"] = (x_true, rho, alpha)
    return res


# ---------------------------------------------------------------- camera
def camera_poses():
    """Route poses with each marker in view: (marker id, base pose)."""
    out = [(0, (0.5, y, -np.pi / 2)) for y in (0.35, 0.45, 0.6, 0.75, 0.95, 1.25, 1.6, 1.95)]
    # markers 1 and 2 are 0.5 m to each side of the long leg: closer than x = 1.1 m they leave the camera's
    # +-24.9 deg view (49.7 deg horizontal field), so the poses start at x = 1.6 m
    out += [(m, (x, 1.25, np.pi)) for m in (1, 2) for x in (1.6, 2.0, 2.5, 3.0, 3.6, 4.5)]
    return out


def marker_truth(robot, mid):
    T_BM = true_pose_se3(robot).inv() * true_marker_pose(robot, MARKERS[mid])
    p = T_BM.t
    return np.array([np.hypot(p[0], p[1]), np.arctan2(p[1], p[0])])


def camera_checks():
    K, dist, _ = camera.load_intrinsics(SIM_INTRINSICS)
    noise = json.loads(Path(SIM_INTRINSICS).read_text())["ekf_noise"]
    T_BC = aruco.load_T_BC(SIM_INTRINSICS)
    rows, amb = [], []
    robot = make_robot("none", seed=2)
    with robot:
        for mid, pose in camera_poses():
            robot.reset(*pose)
            z_true = marker_truth(robot, mid)
            got = []
            for _ in range(N_FRAMES):
                _, rgb = robot.next_image()
                m = aruco.measure(rgb, K, dist, T_BC=T_BC, marker_id=mid)
                if m is None:
                    continue
                got.append(m["z"])
                if m["ambiguous"] and m["T_CM_alt"] is not None:
                    p = (T_BC * m["T_CM_alt"]).t
                    amb.append((abs(np.hypot(p[0], p[1]) - m["z"][0]), abs(wrap(np.arctan2(p[1], p[0]) - m["z"][1]))))
            g = np.array(got)
            row = dict(id=mid, r_true=float(z_true[0]), detected=len(g) / N_FRAMES)
            if len(g):
                row.update(r_bias_mm=float(1000 * (g[:, 0].mean() - z_true[0])), r_sd_mm=float(1000 * g[:, 0].std()),
                           b_bias_deg=float(np.degrees(wrap(g[:, 1].mean() - z_true[1]))),
                           b_sd_deg=float(np.degrees(g[:, 1].std())))
            rows.append(row)
        # frame chain for the marker in {W}, from the pre-dock pose and from the long leg
        chain = {"full chain": [], "translation forgotten": [], "2 deg mounting yaw error": []}
        for mid, pose in [(0, (0.5, 0.75, -np.pi / 2)), (1, (2.0, 1.25, np.pi)), (2, (2.0, 1.25, np.pi))]:
            robot.reset(*pose)
            _, rgb = robot.next_image()
            m = aruco.measure(rgb, K, dist, T_BC=T_BC, marker_id=mid)
            T_WB = true_pose_se3(robot)
            p_true = true_marker_pose(robot, MARKERS[mid]).t
            Rz = T_BC.R.copy()
            c, s = np.cos(np.deg2rad(2)), np.sin(np.deg2rad(2))
            T_yaw = aruco.T_BC.__class__.Rt(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ Rz, T_BC.t)
            T_notr = aruco.T_BC.__class__.Rt(Rz, np.zeros(3))
            for k, T in [("full chain", T_BC), ("translation forgotten", T_notr), ("2 deg mounting yaw error", T_yaw)]:
                p = (T_WB * T * m["T_CM"]).t
                chain[k].append(np.hypot(*(p[:2] - p_true[:2])))
    chain = {k: float(1000 * np.sqrt(np.mean(np.square(v)))) for k, v in chain.items()}
    ok_rows = [r for r in rows if r["detected"] > 0]
    det = min(r["detected"] for r in rows)
    model_sd = lambda r: 1000 * np.hypot(noise["range_sd"][0], noise["range_sd"][1] * r ** 2)   # noqa: E731
    within = np.mean([abs(r["r_bias_mm"] - 1000 * noise["range_bias_r2"] * r["r_true"] ** 2) <= 3 * model_sd(r["r_true"]) + 1
                      for r in ok_rows])
    check("every lab marker detected in at least 70 % of the frames from each route pose", det >= 0.7,
          f"lowest detection rate {100 * det:.0f} % over {len(rows)} poses ("
          + ", ".join(f"ID {r['id']} at {r['r_true']:.2f} m: {100 * r['detected']:.0f} %" for r in rows if r['detected'] < 0.95) + ")")
    check("range bias explained by the EKF's model (bias 4.5 mm r^2, within 3 sd)", within >= 0.9,
          f"{100 * within:.0f} % of poses; bias {min(r['r_bias_mm'] for r in ok_rows):+.1f} to {max(r['r_bias_mm'] for r in ok_rows):+.1f} mm "
          f"over {min(r['r_true'] for r in ok_rows):.2f}-{max(r['r_true'] for r in ok_rows):.2f} m")
    bb = max(abs(r["b_bias_deg"]) for r in ok_rows)
    check("bearing bias below 0.5 deg at every pose", bb < 0.5, f"worst {bb:.2f} deg")
    if amb:
        a = np.array(amb)
        check("ambiguous IPPE frames: both solutions agree on range and bearing", a[:, 0].max() < 0.005 and a[:, 1].max() < np.deg2rad(0.5),
              f"{len(a)} flagged frames, differences up to {1000 * a[:, 0].max():.1f} mm and {np.degrees(a[:, 1].max()):.2f} deg")
    check("marker in {W} with the full frame chain", chain["full chain"] < 15, ", ".join(f"{k} {v:.0f} mm" for k, v in chain.items()))
    return dict(rows=rows, frame_chain_rms_mm=chain, ambiguous_frames=len(amb), noise_model=noise)


# ---------------------------------------------------------------- figures
def fig_lidar(res, path):
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4), gridspec_kw=dict(width_ratios=[1.9, 1, 1]))
    ax = axs[0]
    draw_map(ax, "centre")
    x, rho, alpha = res["scan_beside"]
    ok = lidar.valid_beams(rho, SPEC.range_min, SPEC.range_max)
    P = lidar.to_world(lidar.scan_to_points(rho, alpha, ok), x, lidar.T_BS)
    ax.plot(P[0], P[1], ".", ms=3, color="#eb6834", label="one scan beside the box (noise on)")
    ax.plot(x[0], x[1], "k^", ms=9)
    ax.set_title(f"Scan on the lab track; noise-free scans match the ray caster to {res['raycast_max_err_mm']:.3f} mm", fontweight="bold", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, **GRID)
    ax = axs[1]
    r = res["residuals"] * 1000
    ax.hist(r, bins=80, density=True, color="#2a78d6", alpha=0.75)
    g = np.linspace(-40, 40, 200)
    s = SPEC.sigma * 1000
    ax.plot(g, np.exp(-g ** 2 / (2 * s * s)) / (s * np.sqrt(2 * np.pi)), "k--", lw=1, label=f"N(0, {s:.0f} mm)")
    ax.set_title(f"Range noise: sd {res['noise_sd_mm']:.2f} mm", fontweight="bold", fontsize=10)
    ax.set_xlabel("range minus ray-cast truth [mm]")
    ax.legend(fontsize=8)
    ax.grid(True, **GRID)
    ax = axs[2]
    k, v = zip(*res["frame_chain_rms_mm"].items())
    ax.bar(range(3), v, color=["#3d7a4f", "#eb6834", "#eb6834"])
    for i, val in enumerate(v):
        ax.text(i, val, f"{val:.0f} mm", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(3), ["full chain", "translation\nforgotten", "2° yaw\nerror"], fontsize=8)
    ax.set_title("Cost of a wrong T_BS (W2 C.2)", fontweight="bold", fontsize=10)
    ax.set_ylabel("rms distance to the true hits [mm]")
    ax.grid(True, axis="y", **GRID)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_raycast(res, path):
    """Check 1 for the page, one panel: a scan on the lab track and the ray-caster result in the title."""
    fig, ax = plt.subplots(figsize=(9, 4.2))
    draw_map(ax, "centre")
    x, rho, alpha = res["scan_beside"]
    ok = lidar.valid_beams(rho, SPEC.range_min, SPEC.range_max)
    P = lidar.to_world(lidar.scan_to_points(rho, alpha, ok), x, lidar.T_BS)
    ax.plot(P[0], P[1], ".", ms=3, color="#eb6834", label="one scan beside the box (noise on)")
    ax.plot(x[0], x[1], "k^", ms=9)
    ax.set_title(f"Scan on the lab track; noise-free scans match the ray caster to {res['raycast_max_err_mm']:.3f} mm",
                 fontweight="bold", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, **GRID)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_ranges(res, path):
    """Check 2 in the style of the Assessment 3 provided plot (error with +-2 sigma, grey where data is missing):
    one noisy scan (box ahead) against the ray caster, beam by beam."""
    alpha, rho, rho_true, inside = res["noisy_scan"]
    a = np.degrees(alpha)
    ok = lidar.valid_beams(rho, SPEC.range_min, SPEC.range_max) & inside
    lost = inside & ~ok
    sig = SPEC.sigma * 1000
    fig, axs = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
    ax = axs[0]
    ref = np.where(inside, rho_true, np.nan)
    ax.plot(a, ref, "k-", lw=1, label="reference: analytic ray caster")
    ax.plot(a[ok], rho[ok], ".", ms=3.5, color="tab:orange", label="simulated LiDAR (noise on)")
    ax.set_ylabel("range [m]")
    ax.grid(True, **GRID)
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    ax.set_title(f"Simulated LiDAR on the lab track vs the ray caster (one scan, box ahead)\n"
                 f"range noise sd {res['noise_sd_mm']:.2f} mm (model {sig:.0f} mm), {res['lost_pct']:.2f} % beams lost "
                 f"(model 1 %), over 100 scans",
                 fontweight="bold", fontsize=10)
    ax = axs[1]
    ax.fill_between(a, -2 * sig, 2 * sig, color="tab:blue", alpha=0.2, label="$\\pm 2\\sigma$ of the model (20 mm)")
    err = np.where(ok, (rho - rho_true) * 1000, np.nan)
    ax.plot(a, err, ".", ms=3, color="tab:blue", label="error = LiDAR − reference")
    lim = 45
    ax.fill_between(a, -lim, lim, where=lost, color="0.5", alpha=0.35, step="mid", label="beam lost (no return)")
    ax.set_ylim(-lim, lim)
    ax.set_ylabel("range error [mm]")
    ax.set_xlabel("beam angle in {S} [deg]")
    ax.grid(True, **GRID)
    ax.legend(loc="upper right", ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_aruco(cam, path):
    rows = [r for r in cam["rows"] if r["detected"] > 0]
    nz = cam["noise_model"]
    rr = np.linspace(0.3, 4.6, 100)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4))
    col = {0: "#3d7a4f", 1: "#2a78d6", 2: "#eb6834"}
    for mid in (0, 1, 2):
        rs = [r for r in rows if r["id"] == mid]
        lab = f"ID {mid}" + (" (dock)" if mid == 0 else "")
        axs[0].plot([r["r_true"] for r in rs], [r["r_bias_mm"] for r in rs], "o", color=col[mid], label=lab)
        axs[1].semilogy([r["r_true"] for r in rs], [max(r["r_sd_mm"], 1e-3) for r in rs], "o", color=col[mid], label=lab)
        axs[2].plot([r["r_true"] for r in rs], [r["b_bias_deg"] for r in rs], "o", color=col[mid], label=lab)
    axs[0].plot(rr, 1000 * nz["range_bias_r2"] * rr ** 2, "k--", lw=1, label="EKF bias model 4.5 mm·r²")
    axs[0].set_title("Range bias vs truth", fontweight="bold", fontsize=10)
    axs[0].set_ylabel("[mm]")
    axs[1].semilogy(rr, 1000 * np.hypot(nz["range_sd"][0], nz["range_sd"][1] * rr ** 2), "k--", lw=1, label="EKF range sd model")
    axs[1].set_title("Range spread over 20 frames (Week 3: grows ~ r²)", fontweight="bold", fontsize=10)
    axs[1].set_ylabel("[mm]")
    axs[2].axhline(0, color="k", lw=0.8)
    axs[2].set_title("Bearing bias vs truth", fontweight="bold", fontsize=10)
    axs[2].set_ylabel("[deg]")
    for a in axs:
        a.set_xlabel("true range, base origin to marker [m]")
        a.grid(True, which="both", **GRID)
        a.legend(fontsize=8)
    fig.suptitle("ArUco on the lab track: the three markers seen from the robot's route (sim, 20 frames per pose)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    (OUT / "page").mkdir(parents=True, exist_ok=True)
    lr = lidar_checks()
    cam = camera_checks()
    fig_lidar(lr, OUT / "page" / "lab_lidar.png")
    fig_aruco(cam, OUT / "page" / "lab_aruco.png")
    fig_raycast(lr, OUT / "page" / "lab_raycast.png")
    fig_ranges(lr, OUT / "page" / "lab_lidar_ranges.png")
    keep = {k: v for k, v in lr.items() if k not in ("residuals", "scan_beside", "raycast_scans", "noisy_scan")}
    (OUT / "lab_sensors.json").write_text(json.dumps(dict(lidar=keep, camera=cam), indent=1, default=float))
    print(f"\nwrote results/lab_sensors.json, results/page/lab_lidar.png, lab_raycast.png, lab_lidar_ranges.png, lab_aruco.png; "
          f"{len(FAILS)} failed" + (": " + ", ".join(FAILS) if FAILS else ""))
    sys.exit(len(FAILS))
