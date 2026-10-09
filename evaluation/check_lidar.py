"""Check the simulated LiDAR and the LiDAR processing.

    python evaluation/check_lidar.py            # checks + results/lidar_check.png
    python evaluation/check_lidar.py --viewer   # then open the viewer with the hit points drawn

1. Extrinsic: the base_scan site in the model matches T_BS in src/perception/lidar.py.
2. Geometry: noise-free scans match an independent analytic ray caster at three poses
   (one turned 30 deg, which catches a transposed rotation).
3. Noise model: range residuals and the fraction of lost beams match LidarSpec.
4. Frame chain: world points rebuilt with the true pose, and the cost of a wrong extrinsic (W2 C.2).
5. Avoidance helpers: free distance ahead and nearest obstacle against analytic values.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root, so "src" imports work

import matplotlib                                               # noqa: E402
import numpy as np                                              # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle       # noqa: E402
from spatialmath import SE2                                     # noqa: E402

from evaluation.ground_truth import raycast_2d, room_shapes, true_pose, true_sensor_pose  # noqa: E402
from src.perception import lidar                                # noqa: E402
from src.robot.sim_robot import PROJECT_ROOT, LidarSpec, SimRobot  # noqa: E402

POSES = [(0.0, 0.0, 0.0), (0.5, 0.2, np.deg2rad(30)), (-0.4, -0.2, np.deg2rad(-120))]
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]
results = []


def check(name, ok, detail):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def main(open_viewer):
    # ------------------------------------------------------------ 1. extrinsic in the model vs T_BS
    exact = SimRobot(lidar=LidarSpec(sigma=0.0, dropout=0.0))
    exact.reset(*POSES[1])
    T_BS_model = SE2(*true_pose(exact)).inv() * true_sensor_pose(exact)
    dt_mm = np.linalg.norm(T_BS_model.t - lidar.T_BS.t) * 1000
    dth = np.degrees(abs(T_BS_model.theta() - lidar.T_BS.theta()))
    check("extrinsic T_BS matches the model", dt_mm < 0.5 and dth < 0.05,
          f"model site in base: ({T_BS_model.t[0]:+.4f}, {T_BS_model.t[1]:+.4f}) m, {np.degrees(T_BS_model.theta()):+.3f} deg "
          f"| T_BS: ({lidar.T_BS.t[0]:+.4f}, {lidar.T_BS.t[1]:+.4f}) m  (diff {dt_mm:.2f} mm)")

    # ------------------------------------------------------------ 2. geometry vs analytic ray caster
    alpha = exact.scan_angles
    worst = 0.0
    for pose in POSES:
        exact.reset(*pose)
        _, rho = exact.read_scan()
        rho_true = raycast_2d(true_sensor_pose(exact), alpha, room_shapes(exact))
        worst = max(worst, np.max(np.abs(rho - rho_true)))
    check("ranges vs analytic ray caster (3 poses x 360 beams, noise off)", worst < 1e-3,
          f"max |error| {worst * 1000:.3f} mm")

    # ------------------------------------------------------------ 3. noise model
    noisy = SimRobot(seed=1)
    spec = noisy.lidar
    noisy.reset(*POSES[0])
    res, n_lost, n_total, codes_ok = [], 0, 0, True
    for _ in range(100):
        noisy.step(1.0 / spec.rate_hz)
        _, rho = noisy.read_scan()
        rho_true = raycast_2d(true_sensor_pose(noisy), alpha, room_shapes(noisy))
        ok = lidar.valid_beams(rho, spec.range_min, spec.range_max)
        codes_ok &= bool(np.all(rho[~ok] == spec.no_return))
        res.append(rho[ok] - rho_true[ok])
        n_lost += int((~ok).sum()); n_total += len(rho)
    res = np.concatenate(res)
    lost = n_lost / n_total
    check("range noise matches LidarSpec.sigma", abs(res.mean()) < 0.001 and abs(res.std() / spec.sigma - 1) < 0.05,
          f"residual mean {res.mean() * 1000:+.2f} mm, std {res.std() * 1000:.2f} mm (spec {spec.sigma * 1000:.0f} mm), {len(res)} returns")
    check("lost beams match LidarSpec.dropout and are filtered", codes_ok and abs(lost - spec.dropout) < 0.002,
          f"{100 * lost:.2f} % lost (spec {100 * spec.dropout:.0f} %), all reported as {spec.no_return} and rejected by valid_beams")

    # ------------------------------------------------------------ 4. frame chain and wrong extrinsics
    noisy.reset(*POSES[1])
    _, rho = noisy.read_scan()
    ok = lidar.valid_beams(rho, spec.range_min, spec.range_max)
    P_S = lidar.scan_to_points(rho, alpha, ok)
    x_true = true_pose(noisy)
    T_WS = true_sensor_pose(noisy)
    rho_true = raycast_2d(T_WS, alpha, room_shapes(noisy))
    hits_true = T_WS * lidar.scan_to_points(rho_true, alpha, ok)          # where the beams really hit
    variants = {"full chain": lidar.T_BS,
                "translation forgotten": SE2(0.0, 0.0, 0.0),
                "2 deg mounting yaw error": SE2(-0.064, 0.0, np.deg2rad(2))}
    rms = {k: float(np.sqrt(np.mean(np.sum((lidar.to_world(P_S, x_true, T) - hits_true) ** 2, axis=0))))
           for k, T in variants.items()}
    check("world points with the full frame chain", rms["full chain"] < 0.015,
          ", ".join(f"{k} {v * 1000:.0f} mm" for k, v in rms.items()) + " (rms distance to the true hits)")

    # ------------------------------------------------------------ 5. avoidance helpers
    exact.reset(*POSES[0])
    _, rho = exact.read_scan()
    ok = lidar.valid_beams(rho, exact.lidar.range_min, exact.lidar.range_max)
    P_B = lidar.to_base(lidar.scan_to_points(rho, alpha, ok))
    x0 = true_pose(exact)
    free_expected = free_ahead_geometry(exact, x0)
    d_near, b_near = lidar.nearest_obstacle(P_B)
    rho_true = raycast_2d(true_sensor_pose(exact), alpha, room_shapes(exact))
    d_true = np.min(np.hypot(*lidar.to_base(lidar.scan_to_points(rho_true, alpha, np.isfinite(rho_true)))))
    free = lidar.free_distance_ahead(P_B)
    check("free distance ahead and nearest obstacle", abs(free - free_expected) < 1e-3 and abs(d_near - d_true) < 1e-3,
          f"free ahead {free:.3f} m (expected {free_expected:.3f}), nearest {d_near:.3f} m at {np.degrees(b_near):.0f} deg (expected {d_true:.3f})")

    # ------------------------------------------------------------ figure
    out = PROJECT_ROOT / "results" / "lidar_check.png"
    out.parent.mkdir(exist_ok=True)
    make_figure(noisy, alpha, res, spec.sigma, rms, out)
    print(f"\nwrote {out.relative_to(PROJECT_ROOT)}")
    print(f"{sum(results)}/{len(results)} checks passed")

    if open_viewer:
        run_viewer()
    return 0 if all(results) else 1


def free_ahead_geometry(robot, x, margin=0.05, n=201):
    """Free distance ahead from the room geometry, independently of the scan: parallel rays cast
    forward from n points across the front edge of the robot's corridor."""
    shapes = room_shapes(robot)
    c, s = np.cos(x[2]), np.sin(x[2])
    best = np.inf
    for dy in np.linspace(-(lidar.ROBOT_HALF_WIDTH + margin), lidar.ROBOT_HALF_WIDTH + margin, n):
        o = (x[0] + lidar.ROBOT_FRONT * c - dy * s, x[1] + lidar.ROBOT_FRONT * s + dy * c)
        best = min(best, raycast_2d(SE2(o[0], o[1], x[2]), np.array([0.0]), shapes)[0])
    return best


def make_figure(robot, alpha, res, sigma, rms, out):
    spec = robot.lidar
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw=dict(width_ratios=[1.45, 1, 1]))
    for s in room_shapes(robot):
        if s["kind"] == "box":
            (x0, y0), (x1, y1) = s["lo"], s["hi"]
            ax0.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color="#d6d6d0", lw=0, zorder=1))
        else:
            ax0.add_patch(Circle(s["c"], s["r"], color="#d6d6d0", lw=0, zorder=1))
    for pose, col in zip(POSES, COLOURS):
        robot.reset(*pose)
        _, rho = robot.read_scan()
        ok = lidar.valid_beams(rho, spec.range_min, spec.range_max)
        x = true_pose(robot)
        P_W = lidar.to_world(lidar.scan_to_points(rho, alpha, ok), x)
        ax0.scatter(P_W[0], P_W[1], s=5, color=col, zorder=3,
                    label=f"scan from ({pose[0]:+.1f}, {pose[1]:+.1f}, {np.degrees(pose[2]):+.0f} deg)")
        c, s_ = np.cos(x[2]), np.sin(x[2])
        body = np.array([[-0.197, -0.153], [0.086, -0.153], [0.086, 0.153], [-0.197, 0.153]])
        ax0.add_patch(Polygon(body @ np.array([[c, s_], [-s_, c]]) + x[:2], fill=False, ec=col, lw=1.5, zorder=4))
        ax0.annotate("", xy=(x[0] + 0.25 * c, x[1] + 0.25 * s_), xytext=(x[0], x[1]),
                     arrowprops=dict(arrowstyle="-|>", color=col, lw=1.5), zorder=5)
    ax0.set_aspect("equal"); ax0.set_xlim(-2.1, 2.1); ax0.set_ylim(-1.6, 1.6)
    ax0.set_xlabel("x [m]"); ax0.set_ylabel("y [m]")
    ax0.set_title("Scans placed in the world with p_W = T_WB T_BS p_S", loc="left", fontsize=11)
    ax0.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8.5, frameon=False)

    ax1.hist(res * 1000, bins=60, density=True, color="#2a78d6", alpha=0.85)
    g = np.linspace(-4 * sigma, 4 * sigma, 200)
    ax1.plot(g * 1000, np.exp(-0.5 * (g / sigma) ** 2) / (sigma * 1000 * np.sqrt(2 * np.pi)), color="#333333", lw=1.5,
             label=f"N(0, {sigma * 1000:.0f} mm)")
    ax1.set_xlabel("range residual vs analytic [mm]"); ax1.set_ylabel("density")
    ax1.set_title(f"Range noise: std {res.std() * 1000:.2f} mm over {len(res)} returns", loc="left", fontsize=11)
    ax1.legend(fontsize=9, frameon=False)

    names = list(rms)
    vals = [rms[k] * 1000 for k in names]
    bars = ax2.bar(range(len(vals)), vals, color="#2a78d6", width=0.6)
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f} mm", ha="center", fontsize=9.5, color="#333333")
    ax2.set_xticks(range(len(vals)), ["full chain", "translation\nforgotten", "2 deg yaw\nerror"], fontsize=9.5)
    ax2.set_ylabel("rms distance to true hits [mm]")
    ax2.set_title("Cost of a wrong extrinsic T_BS (W2 C.2)", loc="left", fontsize=11)
    for a in (ax0, ax1, ax2):
        a.grid(color="#eeeeee", lw=0.8, zorder=0)
        a.set_axisbelow(True)
        for sp in a.spines.values():
            sp.set_color("#cccccc")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")               # keep the legend below panel 1
    plt.close(fig)


def draw_hits(scn, P_W, cam=None, z=0.132, size=0.015, px=0.003):
    """Red dot at every LiDAR hit point, like RViz. Display only, no physics.

    MuJoCo draws real 3D spheres, so with a fixed radius near dots look big and far ones small.
    Given the viewer camera, the radius grows with distance to the camera (px metres per metre),
    so every dot has the same size on screen, like RViz's "Points" style.
    """
    import mujoco
    if cam is not None:
        az, el = np.deg2rad(cam.azimuth), np.deg2rad(cam.elevation)
        fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        cam_pos = np.asarray(cam.lookat) - cam.distance * fwd
    scn.ngeom = 0
    for p in P_W.T:
        if scn.ngeom >= scn.maxgeom:
            break
        pos = np.array([p[0], p[1], z])
        r = px * np.linalg.norm(pos - cam_pos) if cam is not None else size
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, 0, 0]),
                            pos, np.eye(3).ravel(), np.array([1, 0, 0, 1], dtype=np.float32))
        scn.ngeom += 1


def run_viewer():
    """Robot turns slowly in the middle of the room; the dots are the processed scan in the world."""
    import mujoco.viewer
    robot = SimRobot(seed=2)
    robot.reset(0.0, 0.0, 0.0)
    spec, P_W = robot.lidar, None
    last = None
    with mujoco.viewer.launch_passive(robot.m, robot.d) as viewer:
        while viewer.is_running():
            t0 = time.time()
            robot.send_cmd(0.0, 0.4)
            robot.step(0.02)
            t_scan, rho = robot.read_scan()
            if t_scan != last:                                     # a new scan arrived (5 Hz)
                ok = lidar.valid_beams(rho, spec.range_min, spec.range_max)
                P_W = lidar.to_world(lidar.scan_to_points(rho, robot.scan_angles, ok), true_pose(robot))
                last = t_scan
            draw_hits(viewer.user_scn, P_W, cam=viewer.cam)        # every frame, so zooming keeps dot size
            viewer.sync()
            time.sleep(max(0.0, 0.02 - (time.time() - t0)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--viewer", action="store_true", help="open the MuJoCo viewer after the checks")
    sys.exit(main(ap.parse_args().viewer))