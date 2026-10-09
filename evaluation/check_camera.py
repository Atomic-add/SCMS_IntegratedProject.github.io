"""Check the simulated camera, its calibration and the ArUco range-bearing measurement.

    python experiments/calibrate_camera.py      # once: writes config/camera_sim.json
    python evaluation/check_camera.py           # checks + results/camera_check.png + results/aruco_noise.csv
    python evaluation/check_camera.py --viewer  # then drive at the dock, camera view in a second window

Perception uses the calibrated K from config/camera_sim.json, as the EKF will. Only this script
compares it with the true K of the MJCF camera.

1. Extrinsic: the camera in the model matches T_BC in src/perception/aruco.py.
2. Calibration: calibrated K vs the true K, and no invented lens distortion (W3 Part C: a small
   reprojection error is necessary but not sufficient).
3. Image noise: rendered frames minus noise-free frames match CameraSpec.pixel_noise.
4. Projection: detected corners vs the true marker corners projected with the true K (noise off),
   at three poses, one with the marker near the image edge.
5. Accuracy: range, bearing and marker yaw vs ground truth at five poses.
6. Noise vs distance: 20 frames per pose from 0.12 m to 3.4 m. Detection limits and the W3 law
   sigma_Z ~ Z^2 sigma_px / (f s). Writes results/aruco_noise.csv, the data for the EKF's Rn.
7. Planar-pose ambiguity (W3 Part E): both IPPE solutions agree on range and bearing, and the yaw
   is only wrong when it is flagged.
8. Frame chain: the marker in the world with the full T_BC, and the cost of a wrong extrinsic.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root, so "src" imports work

import cv2                                                      # noqa: E402
import matplotlib                                               # noqa: E402
import mujoco                                                   # noqa: E402
import numpy as np                                              # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402
from spatialmath import SE3                                     # noqa: E402

from evaluation.ground_truth import (true_camera_pose, true_intrinsics, true_marker_pose,  # noqa: E402
                                     true_pose_se3)
from src.perception import aruco, camera                        # noqa: E402
from src.robot.sim_robot import PROJECT_ROOT, CameraSpec, SimRobot  # noqa: E402

CALIB = PROJECT_ROOT / "config" / "camera_sim.json"
# Camera-to-marker distances [m] (horizontal) for the sweep. Head-on runs back to 3.4 m along the dock axis
# (3.55 m puts the robot against the west wall).
# The oblique line is 30 deg off the marker normal on the north side; beyond 1.4 m the robot would
# sit on obs_cyl, and on the south side obs_box2 blocks the view.
HEAD_ON = [0.12, 0.15, 0.18, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8,
           3.0, 3.2, 3.4]
OBLIQUE = [0.3, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
OBLIQUE_DEG = -30.0
N_FRAMES = 20
results = []


def check(name, ok, detail):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def wrap(a):
    """Angle wrapped to [-pi, pi) (W7 wrap)."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def place(robot, d, ang_deg=0.0, turn_deg=0.0):
    """Reset the robot so its camera is d metres (horizontally) from the marker centre, on a line
    ang_deg off the marker normal (anticlockwise seen from above), looking at the marker.
    turn_deg then turns the robot about the camera, which moves the marker towards the image edge.
    Raises if anything blocks the line of sight, so a changed room layout cannot pass silently.
    """
    T_WM = true_marker_pose(robot)
    n = T_WM.R[:2, 2] / np.linalg.norm(T_WM.R[:2, 2])           # marker normal in the floor plane
    a = np.deg2rad(ang_deg)
    u = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ n   # direction marker -> camera
    cam = T_WM.t[:2] + d * u
    yaw = np.arctan2(-u[1], -u[0]) + np.deg2rad(turn_deg)
    base = cam - aruco.T_BC.t[0] * np.array([np.cos(yaw), np.sin(yaw)])
    robot.reset(base[0], base[1], yaw)
    p_C = true_camera_pose(robot).t
    gid = np.zeros(1, np.int32)
    mujoco.mj_ray(robot.m, robot.d, p_C, T_WM.t - p_C, None, 1, robot.m.body("base").id, gid)
    if gid[0] < 0 or robot.m.geom(gid[0]).name != "aruco_marker":
        raise RuntimeError(f"line of sight to the marker blocked at d={d}, ang={ang_deg}, turn={turn_deg}")


def truth(robot):
    """The true measurement: z = [range, bearing] and marker yaw in {B}, T_CM, depth Z, T_WB, p_WM."""
    T_WB, T_WC, T_WM = true_pose_se3(robot), true_camera_pose(robot), true_marker_pose(robot)
    T_BM, T_CM = T_WB.inv() * T_WM, T_WC.inv() * T_WM
    p, n = T_BM.t, T_BM.R[:, 2]
    return dict(z=np.array([np.hypot(p[0], p[1]), np.arctan2(p[1], p[0])]), yaw=float(np.arctan2(n[1], n[0])),
                T_CM=T_CM, Z=float(T_CM.t[2]), T_WB=T_WB, p_WM=T_WM.t.copy())


def true_corners(tr, K_true):
    """Where the marker corners really are in the image: the true T_CM through the true K (4 x 2)."""
    uv = K_true @ (tr["T_CM"] * aruco.marker_corners().T.astype(float))
    return (uv[:2] / uv[2]).T


def rb(T_BM):
    """Range and bearing of a marker pose in {B}, in the floor plane (as in aruco.measure)."""
    p = T_BM.t
    return np.array([np.hypot(p[0], p[1]), np.arctan2(p[1], p[0])])


def rot_deg(R):
    """Rotation angle of R [deg]."""
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


def main(open_viewer):
    if not CALIB.exists():
        print(f"{CALIB.relative_to(PROJECT_ROOT)} not found: run  python experiments/calibrate_camera.py  first")
        return 1
    K, dist, size = camera.load_intrinsics(CALIB)
    rms_cal = json.loads(CALIB.read_text())["rms_px"]
    f_cal = (K[0, 0] + K[1, 1]) / 2

    # ------------------------------------------------------------ 1. extrinsic in the model vs T_BC
    exact = SimRobot(camera=CameraSpec(pixel_noise=0.0))
    place(exact, 0.8, OBLIQUE_DEG)
    T_BC_model = true_pose_se3(exact).inv() * true_camera_pose(exact)
    dt_mm = np.linalg.norm(T_BC_model.t - aruco.T_BC.t) * 1000
    dR = rot_deg(T_BC_model.R.T @ aruco.T_BC.R)
    check("extrinsic T_BC matches the model", dt_mm < 0.5 and dR < 0.1,
          f"model camera in base: ({T_BC_model.t[0]:+.4f}, {T_BC_model.t[1]:+.4f}, {T_BC_model.t[2]:+.4f}) m "
          f"| T_BC: ({aruco.T_BC.t[0]:+.4f}, {aruco.T_BC.t[1]:+.4f}, {aruco.T_BC.t[2]:+.4f}) m "
          f"(diff {dt_mm:.2f} mm, {dR:.3f} deg)")

    # ------------------------------------------------------------ 2. calibration vs the true K
    K_true = true_intrinsics(exact)
    df = max(abs(K[0, 0] / K_true[0, 0] - 1), abs(K[1, 1] / K_true[1, 1] - 1))
    dc = max(abs(K[0, 2] - K_true[0, 2]), abs(K[1, 2] - K_true[1, 2]))
    u, v = np.meshgrid(np.arange(0, size[0], 8.0), np.arange(0, size[1], 8.0))
    pix = np.stack([u.ravel(), v.ravel()], axis=1).reshape(-1, 1, 2)
    shift = np.linalg.norm(cv2.undistortPoints(pix, K, dist, P=K) - pix, axis=2).max()
    check("calibrated K vs true K", df < 0.005 and dc < 2.0 and rms_cal < 0.3 and shift < 1.0 and size == (exact.camera.width, exact.camera.height),
          f"f {K[0, 0]:.2f}/{K[1, 1]:.2f} px (true {K_true[0, 0]:.2f}, {100 * df:.3f} %), pp ({K[0, 2]:.2f}, {K[1, 2]:.2f}) "
          f"(true ({K_true[0, 2]:.1f}, {K_true[1, 2]:.1f})), reprojection rms {rms_cal:.3f} px, "
          f"fitted distortion moves pixels by at most {shift:.2f} px (true lens: none)")

    # ------------------------------------------------------------ 3. image noise
    noisy = SimRobot(seed=1)
    place(noisy, 0.8, OBLIQUE_DEG)                                # same state as `exact` (deterministic physics)
    assert np.allclose(noisy.d.qpos, exact.d.qpos)
    _, clean = exact.read_image()
    _, img = noisy.read_image()
    diff = img.astype(float) - clean.astype(float)
    mask = (clean > 10) & (clean < 245)                           # away from clipping at 0 and 255
    sd_expected = np.sqrt(noisy.camera.pixel_noise ** 2 + 1 / 12)    # + rounding to integer grey levels
    check("image noise matches CameraSpec.pixel_noise", abs(diff[mask].mean()) < 0.05 and abs(diff[mask].std() / sd_expected - 1) < 0.05,
          f"noisy - clean: mean {diff[mask].mean():+.3f}, std {diff[mask].std():.3f} grey levels over {mask.sum()} values "
          f"(spec {noisy.camera.pixel_noise:.1f}, with rounding {sd_expected:.3f})")

    # ------------------------------------------------------------ 4. projection, noise off
    errs = []
    for d, ang, turn in [(0.8, 0.0, 0.0), (0.6, OBLIQUE_DEG, 0.0), (0.8, 0.0, 20.0)]:
        place(exact, d, ang, turn)
        _, rgb = exact.read_image()
        c = aruco.detect(rgb)
        errs.append(np.linalg.norm(c - true_corners(truth(exact), K_true), axis=1) if c is not None else np.full(4, np.inf))
    errs = np.concatenate(errs)
    check("detected corners vs true projection (3 poses, noise off)", errs.mean() < 0.25 and errs.max() < 0.5,
          f"mean {errs.mean():.3f} px, max {errs.max():.3f} px (true K and true pose, corner order TL TR BR BL)")

    # ------------------------------------------------------------ 5. accuracy at five poses
    poses5 = [(0.4, 0.0, 0.0), (0.8, 0.0, 0.0), (1.5, 0.0, 0.0), (0.8, OBLIQUE_DEG, 0.0), (0.8, 0.0, 20.0)]
    rows5, frames5, ok5 = [], [], True
    for d, ang, turn in poses5:
        place(noisy, d, ang, turn)
        _, rgb = noisy.read_image()
        m, tr = aruco.measure(rgb, K, dist), truth(noisy)
        if m is None:
            ok5 = False
            rows5.append(f"      d {d:.1f} m, ang {ang:+.0f}, turn {turn:+.0f}: not detected")
            continue
        er, eb = m["z"][0] - tr["z"][0], wrap(m["z"][1] - tr["z"][1])
        ey = wrap(m["yaw"] - tr["yaw"])
        ok5 &= abs(er) < 0.01 * tr["z"][0] + 0.005 and abs(np.degrees(eb)) < 0.5 and (m["ambiguous"] or abs(np.degrees(ey)) < 2.0)
        frames5.append((m, tr))
        rows5.append(f"      d {d:.1f} m, ang {ang:+.0f}, turn {turn:+.0f}: range {m['z'][0]:.3f} m (true {tr['z'][0]:.3f}, "
                     f"{1000 * er:+.1f} mm), bearing {np.degrees(m['z'][1]):+.2f} deg ({np.degrees(eb):+.2f}), "
                     f"yaw err {np.degrees(ey):+.2f} deg{' (flagged ambiguous)' if m['ambiguous'] else ''}")
    check("range, bearing and yaw vs ground truth (5 poses, noise on)", ok5,
          "|range err| < 1 % + 5 mm, |bearing err| < 0.5 deg, |yaw err| < 2 deg when not flagged")
    print("\n".join(rows5))

    # ------------------------------------------------------------ 6. noise vs distance
    sweep = SimRobot(seed=2)
    rows, recs = [], []
    for ang, dists in [(0.0, HEAD_ON), (OBLIQUE_DEG, OBLIQUE)]:
        for d in dists:
            place(sweep, d, ang)
            fr = []
            for _ in range(N_FRAMES):
                _, rgb = sweep.next_image()
                m, tr = aruco.measure(rgb, K, dist), truth(sweep)
                if m is not None:
                    fr.append((m, tr, true_corners(tr, K_true)))
                    z_alt = rb(aruco.T_BC * m["T_CM_alt"]) if m["T_CM_alt"] is not None else None
                    recs.append(dict(d=d, ang=ang, Z=tr["Z"], amb=m["ambiguous"], ey=wrap(m["yaw"] - tr["yaw"]),
                                     dz_alt=None if z_alt is None else [z_alt[0] - m["z"][0], wrap(z_alt[1] - m["z"][1])]))
            rows.append(summarise(d, ang, sweep, fr, f_cal))
    out_csv = PROJECT_ROOT / "results" / "aruco_noise.csv"
    out_csv.parent.mkdir(exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    head = [r for r in rows if r["angle_deg"] == 0.0]
    good = [r for r in head if r["detection_rate"] >= 0.95]
    d_lo, d_hi = min(r["d_cam_m"] for r in good), max(r["d_cam_m"] for r in good)
    span_ok = all(r["detection_rate"] >= 0.95 for r in head if d_lo <= r["d_cam_m"] <= d_hi)
    ratio = np.array([r["range_sd_mm"] / r["range_sd_w3_mm"] for r in good])
    slope = np.polyfit(np.log([r["Z_m"] for r in good]), np.log([r["range_sd_mm"] for r in good]), 1)[0]
    sig = np.array([r["corner_sd_px"] for r in good])
    sig_px = float(np.median(sig))
    end = " (end of sweep)" if d_hi == max(HEAD_ON) else ""
    check("detection span, and range noise follows the W3 law pose by pose",
          d_lo <= 0.25 and d_hi >= 2.4 and span_ok and np.all((ratio > 1 / 3) & (ratio < 3)) and 0.5 < np.median(ratio) < 2,
          f"head-on, camera to marker: detected in >= 95 % of frames from {d_lo:.2f} to {d_hi:.2f} m{end}; "
          f"measured range sd / (Z^2 sigma_px / (f s)) = {ratio.min():.2f}-{ratio.max():.2f} (median {np.median(ratio):.2f}) "
          f"with each pose's own corner noise sigma_px ({sig.min():.3f}-{sig.max():.3f} px)")
    for r in head:
        if r["detection_rate"] < 0.95:
            print(f"      d {r['d_cam_m']:.2f} m: detected in {100 * r['detection_rate']:.0f} % of frames"
                  + (f", marker {r['size_px']:.0f} px wide" if r["detection_rate"] > 0 else ""))
    bias = {d: r["range_bias_mm"] for r in good for d in (0.5, 1.0, 2.0, 3.0) if abs(r["d_cam_m"] - d) < 1e-9}
    print(f"      overall the sd grows as Z^{slope:.2f}, steeper than Z^2 because sigma_px grows as the marker shrinks.\n"
          f"      The error is bias-dominated: range bias " + ", ".join(f"{v:+.1f} mm at {d:.0f} m" if d >= 1 else
                                                                      f"{v:+.1f} mm at {d} m" for d, v in bias.items())
          + f", |bias| > sd at {sum(abs(r['range_bias_mm']) > r['range_sd_mm'] for r in good)} of {len(good)} poses")

    # ------------------------------------------------------------ 7. planar-pose ambiguity
    flag = [r for r in recs if r["amb"]]
    alt = np.array([r["dz_alt"] for r in flag if r["dz_alt"] is not None]).reshape(-1, 2)
    ey_t = np.abs(np.degrees([r["ey"] for r in recs if not r["amb"]]))
    ey_f = np.abs(np.degrees([r["ey"] for r in flag]))
    worst_r, worst_b = (1000 * np.abs(alt[:, 0]).max(), np.degrees(np.abs(alt[:, 1]).max())) if len(alt) else (0.0, 0.0)
    check("ambiguity: range-bearing robust, yaw trusted only when unambiguous",
          worst_r < 10.0 and worst_b < 0.5 and np.percentile(ey_t, 95) < 2.0,
          f"in the {len(flag)} flagged frames the two IPPE solutions differ by at most {worst_r:.1f} mm in range and "
          f"{worst_b:.2f} deg in bearing; yaw error p95 {np.percentile(ey_t, 95):.2f} deg trusted ({len(ey_t)} frames) "
          f"vs {np.percentile(ey_f, 95) if len(ey_f) else float('nan'):.1f} deg flagged")

    # ------------------------------------------------------------ 8. frame chain and wrong extrinsics
    t0, R0 = aruco.T_BC.t, aruco.T_BC.R
    variants = {"full chain": aruco.T_BC,
                "translation forgotten": SE3.Rt(R0, [0.0, 0.0, 0.0]),
                "2 deg mounting yaw error": SE3(*t0) * SE3.Rz(np.deg2rad(2)) * SE3.Rt(R0, [0.0, 0.0, 0.0]),
                "MuJoCo camera axes": SE3.Rt(R0 @ np.diag([1.0, -1.0, -1.0]), t0)}
    rms = {}
    for k, T_BC in variants.items():
        e = [np.linalg.norm((tr["T_WB"] * T_BC * m["T_CM"]).t[:2] - tr["p_WM"][:2]) for m, tr in frames5]
        rms[k] = float(np.sqrt(np.mean(np.square(e))))
    check("marker in the world with the full frame chain", rms["full chain"] < 0.010,
          ", ".join(f"{k} {v * 1000:.0f} mm" for k, v in rms.items()) + " (rms horizontal distance to the true marker)")

    # ------------------------------------------------------------ figure
    out = PROJECT_ROOT / "results" / "camera_check.png"
    place(noisy, 0.8, OBLIQUE_DEG)
    _, rgb = noisy.read_image()
    make_figure(rgb, aruco.measure(rgb, K, dist), K, dist, rows, recs, sig_px, f_cal, (d_lo, d_hi), rms, out)
    for r in (exact, noisy, sweep):
        r.close()
    print(f"\nwrote {out.relative_to(PROJECT_ROOT)}")
    print(f"wrote {out_csv.relative_to(PROJECT_ROOT)}")
    print(f"{sum(results)}/{len(results)} checks passed")

    if open_viewer:
        run_viewer(K, dist)
    return 0 if all(results) else 1


def summarise(d, ang, robot, fr, f_px):
    """One row of results/aruco_noise.csv: detection and error statistics of the frames at one pose."""
    Z = truth(robot)["Z"]
    row = dict(d_cam_m=d, angle_deg=ang, Z_m=round(Z, 4), frames=N_FRAMES, detection_rate=len(fr) / N_FRAMES)
    keys = ["size_px", "corner_sd_px", "corner_bias_px", "range_bias_mm", "range_sd_mm", "range_sd_w3_mm",
            "bearing_bias_deg", "bearing_sd_deg", "yaw_trusted_frac", "yaw_err_trusted_p95_deg", "yaw_err_flagged_p95_deg"]
    if len(fr) < 2:
        return {**row, **{k: float("nan") for k in keys}}
    C = np.array([m["corners"] for m, _, _ in fr])                 # frames x 4 x 2
    Ct = np.array([c for _, _, c in fr])
    er = np.array([m["z"][0] - tr["z"][0] for m, tr, _ in fr])
    eb = np.degrees(wrap([m["z"][1] - tr["z"][1] for m, tr, _ in fr]))
    ey = np.abs(np.degrees(wrap([m["yaw"] - tr["yaw"] for m, tr, _ in fr])))
    amb = np.array([m["ambiguous"] for m, _, _ in fr])
    sd_px = float(np.sqrt(np.mean(np.var(C, axis=0, ddof=1))))   # corner scatter about each corner's mean
    vals = [np.mean([m["size_px"] for m, _, _ in fr]), sd_px, float(np.sqrt(np.mean(np.sum((C.mean(0) - Ct.mean(0)) ** 2, axis=1)))),
            1000 * er.mean(), 1000 * er.std(ddof=1), 1000 * float(aruco.sigma_range_model(Z, f_px, sigma_px=sd_px)),
            eb.mean(), eb.std(ddof=1), float(np.mean(~amb)),
            float(np.percentile(ey[~amb], 95)) if (~amb).any() else float("nan"),
            float(np.percentile(ey[amb], 95)) if amb.any() else float("nan")]
    return {**row, **{k: round(float(v), 5) for k, v in zip(keys, vals)}}


def make_figure(rgb, meas, K, dist, rows, recs, sig_px, f_px, span, rms, out):
    fig, ax = plt.subplots(2, 2, figsize=(14, 10.5))
    (a0, a1), (a2, a3) = ax

    a0.imshow(aruco.draw_overlay(rgb, meas, K, dist))
    a0.set_title("Camera frame, 0.8 m and 30 deg off the marker normal (axes: x red, y green)", loc="left", fontsize=11)
    a0.axis("off")

    head = [r for r in rows if r["angle_deg"] == 0.0 and r["detection_rate"] >= 0.95]
    obl = [r for r in rows if r["angle_deg"] != 0.0 and r["detection_rate"] >= 0.95]
    Zg = np.linspace(0.1, 3.2, 200)
    a1.plot([r["Z_m"] for r in head], [r["range_sd_mm"] for r in head], "o", color="#2a78d6", label="range sd, head-on")
    a1.plot([r["Z_m"] for r in obl], [r["range_sd_mm"] for r in obl], "s", color="#1baf7a", mfc="none", label="range sd, 30 deg")
    a1.plot([r["Z_m"] for r in head], np.abs([r["range_bias_mm"] for r in head]), "^", color="#eb6834", mfc="none",
            label="|range bias|, head-on")
    a1.plot(Zg, 1000 * aruco.sigma_range_model(Zg, f_px, sigma_px=sig_px), color="#333333", lw=1.3,
            label=f"W3: Z^2 sigma_px / (f s), sigma_px = {sig_px:.3f} px")
    x_max = max(HEAD_ON) * 1.08
    a1.axvspan(0.1, span[0], color="#eeeeee", zorder=0)
    a1.text(np.sqrt(0.1 * span[0]), 0.35, "marker not fully in view", transform=a1.get_xaxis_transform(), rotation=90,
            ha="center", va="center", fontsize=8.5, color="#777777")
    if span[1] < max(HEAD_ON):
        a1.axvspan(span[1], x_max, color="#eeeeee", zorder=0)
        a1.text(span[1] * 1.02, 0.03, "not detected", transform=a1.get_xaxis_transform(), fontsize=8.5, color="#777777")
    a1.set_xscale("log"); a1.set_yscale("log"); a1.set_xlim(0.1, x_max)
    a1.set_xticks([0.1, 0.2, 0.5, 1, 2, 3], ["0.1", "0.2", "0.5", "1", "2", "3"])
    a1.set_xlabel("camera to marker depth Z [m]"); a1.set_ylabel("range error [mm]")
    a1.set_title("Range error: bias dominates, sd follows the W3 law (20 frames each)", loc="left", fontsize=11)
    a1.legend(fontsize=8.5, frameon=False, loc="upper left")

    Zt = [r["Z"] for r in recs if not r["amb"]]
    Zf = [r["Z"] for r in recs if r["amb"]]
    a2.semilogy(Zt, np.maximum(np.abs(np.degrees([r["ey"] for r in recs if not r["amb"]])), 1e-3), ".", color="#2a78d6",
                ms=4, label="trusted")
    a2.semilogy(Zf, np.maximum(np.abs(np.degrees([r["ey"] for r in recs if r["amb"]])), 1e-3), "x", color="#eb6834",
                ms=4, label="flagged ambiguous (e1 > 0.1 e2)")
    a2.axhline(2.0, color="#333333", lw=1, ls="--", label="2 deg")
    a2.set_xlabel("camera to marker depth Z [m]"); a2.set_ylabel("|marker yaw error| [deg]")
    a2.set_title("Marker yaw: wrong only when IPPE is ambiguous (W3 Part E)", loc="left", fontsize=11)
    a2.legend(fontsize=9, frameon=False)

    names = list(rms)
    vals = [rms[k] * 1000 for k in names]
    bars = a3.bar(range(len(vals)), vals, color="#2a78d6", width=0.6)
    for b, v in zip(bars, vals):
        a3.text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:.0f} mm", ha="center", fontsize=9.5, color="#333333")
    a3.set_yscale("log"); a3.set_ylim(1, max(vals) * 3)
    a3.set_xticks(range(len(vals)), ["full chain", "translation\nforgotten", "2 deg yaw\nerror", "MuJoCo camera\naxes (no flip)"],
                  fontsize=9.5)
    a3.set_ylabel("rms distance to the true marker [mm]")
    a3.set_title("Cost of a wrong extrinsic T_BC", loc="left", fontsize=11)
    for a in (a1, a2, a3):
        a.grid(color="#eeeeee", lw=0.8, which="both")
        a.set_axisbelow(True)
        for sp in a.spines.values():
            sp.set_color("#cccccc")
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_viewer(K, dist):
    """The robot drives straight at the dock from 2.8 m and stops when the measured range drops below
    0.35 m (measurement only, no ground truth). A second window shows what the camera sees, with the
    detected marker and the measurement. Close the MuJoCo viewer to finish."""
    import mujoco.viewer
    robot = SimRobot(seed=3)
    place(robot, 2.8)                                              # 2.8 m from the marker, facing it
    robot.read_image()                                             # create the offscreen renderer first
    last, stop = None, False
    win = "camera rgb with the ArUco measurement"
    with mujoco.viewer.launch_passive(robot.m, robot.d) as viewer:
        with viewer.lock():                                        # look over the robot's path to the dock
            viewer.cam.lookat[:] = [0.3, 0.0, 0.0]
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 4.0, 35.0, -45.0
        while viewer.is_running():
            t0 = time.time()
            t_img, rgb = robot.read_image()
            if t_img != last:                                      # a new frame arrived (15 Hz)
                meas = aruco.measure(rgb, K, dist)
                stop |= meas is not None and meas["z"][0] < 0.35
                cv2.imshow(win, cv2.cvtColor(aruco.draw_overlay(rgb, meas, K, dist), cv2.COLOR_RGB2BGR))
                last = t_img
            cv2.waitKey(1)
            robot.send_cmd(0.0 if stop else 0.1, 0.0)
            robot.step(0.02)
            viewer.sync()
            time.sleep(max(0.0, 0.02 - (time.time() - t0)))
    cv2.destroyAllWindows()
    robot.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--viewer", action="store_true", help="open the MuJoCo viewer after the checks")
    sys.exit(main(ap.parse_args().viewer))