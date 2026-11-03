"""ArUco range on the real camera against tape-measured distances: bias and noise for the EKF's Rn.

    python experiments/aruco_real_noise.py           # reads results/hardware/aruco_cam<d>m/, marker ID 0
    python experiments/aruco_real_noise.py --write   # also stores the range correction in config/camera_real.json

Each folder holds frames saved by capture_frames.py with the robot looking straight at the marker,
<d> = tape distance from the camera lens to the marker face [m]. The camera is 0.076 m ahead of the base
origin (aruco.T_BC), so the true forward distance base -> marker is d + 0.076. For every frame,
aruco.measure (with config/camera_real.json, its K and its T_BC) gives the marker centre in {B}; per distance this prints
the forward-distance bias x_meas - x_true, the range and bearing spread, and the marker centre height.
The frames at one distance show a static scene, so the sd is pixel noise only and is a lower bound;
the bias across distances is what calibration (K, distortion, printed size) leaves behind.
A repeat set is a folder with a suffix (aruco_cam0.5m_v2): it is listed too, and the latest set at
each distance replaces the earlier ones in the fit, the csv and the plot.
The bias is fitted on the raw PnP range (measure without a correction) as bias_m + bias_per_m * x_true,
and the range sd for the EKF as a + b r^2: a = the rms residual of that fit (tape placement and what the
straight line cannot follow), b = the W3 corner-noise model at 0.3 px (aruco.sigma_range_model; the
static frames are well below it). --write stores both as "range_correction" in config/camera_real.json;
aruco.load_range_correction reads it there, so aruco.measure hands corrected ranges to the EKF and docking,
and models.load_estimation_config(camera=...) swaps in this range sd and drops the sim's range_bias_r2.
Run it after camera_pitch.py --write (the fit uses T_BC). The sim file has no correction and is unchanged.
Compares with the sim model in config/estimation.json and writes results/aruco_noise_real.csv / .png.

Two checks from the 41014 Week 3 wrap-up, which do not need the tape to be right:
  - apparent size: the marker spans n ~ f_px s / Z pixels, so n * d must be the same at every distance
    (n from the undistorted corners). A set whose n * d is off by more than SIZE_TOL from the median had
    its tape distance read wrong: it is listed, and left out of the fit that --write stores. n * d / f_px
    is the printed side s the camera sees: if it is not MARKER_SIDE, PnP scales every range by
    MARKER_SIDE / s, and measuring the printed marker is the physical fix.
  - noise: sigma_Z ~ Z^2 sigma_px / (f_px s) with sigma_px the corner spread over the static frames
    (aruco.sigma_range_model), against the measured range sd.
Adapted from 41014 Week 3 Part E (marker pose, PnP) and the Week 3 wrap-up (n ~ f s / Z, sigma_Z law).
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root, so "src" imports work

import cv2                                                      # noqa: E402
import matplotlib                                               # noqa: E402
import numpy as np                                              # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402

from src.perception import aruco, camera                        # noqa: E402
from src.robot.sim_robot import PROJECT_ROOT                    # noqa: E402

CAMERA_FILE = PROJECT_ROOT / "config" / "camera_real.json"
CAM_AHEAD = aruco.T_BC.t[0]      # 0.076 m, camera ahead of the base origin
SIZE_TOL = 0.01                  # relative spread of n * d allowed before a tape reading is called wrong


def undistorted_size(corners, K, dist):
    """Mean edge length [px] of the marker after removing the lens distortion (n in n ~ f s / Z)."""
    u = cv2.undistortPoints(corners.reshape(-1, 1, 2).astype(np.float32), K, dist, P=K).reshape(4, 2)
    return float(np.mean(np.linalg.norm(u - np.roll(u, 1, axis=0), axis=1)))


def measure_folder(folder, K, dist, T_BC, range_correction=None):
    """One row per frame where marker 0 is found: base-frame position, z = [range, bearing], corners."""
    rows = []
    for p in sorted(folder.glob("*.png")):
        rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        m = aruco.measure(rgb, K, dist, T_BC=T_BC, marker_id=0, range_correction=range_correction)
        if m is not None:
            rows.append(dict(x=m["T_BM"].t[0], z=m["T_BM"].t[2], r=m["z"][0], b=np.rad2deg(m["z"][1]),
                             size_px=m["size_px"], n_px=undistorted_size(m["corners"], K, dist),
                             corners=m["corners"], ambiguous=m["ambiguous"]))
    return rows


def corner_sigma(rows):
    """Pixel noise sigma_px: sd of each corner coordinate over the static frames, pooled (W3 wrap-up)."""
    if len(rows) < 2:
        return np.nan
    c = np.array([q["corners"] for q in rows])                   # frames x 4 x 2
    return float(np.sqrt(np.mean(c.var(axis=0, ddof=1))))


def main():
    ap = argparse.ArgumentParser(description="ArUco range bias and noise of the real camera against tape distances.")
    ap.add_argument("--write", action="store_true",
                    help="store the bias fit and range sd as range_correction in config/camera_real.json")
    args = ap.parse_args()

    K, dist, size = camera.load_intrinsics(CAMERA_FILE)
    T_BC = aruco.load_T_BC(CAMERA_FILE)                         # with the fitted tilt, if any
    est = json.loads((PROJECT_ROOT / "config" / "estimation.json").read_text())
    table, folders = [], {}
    for folder in PROJECT_ROOT.glob("results/hardware/aruco_cam*"):
        match = re.fullmatch(r"aruco_cam([\d.]+)m(?:_(\w+))?", folder.name)
        if match is None:
            continue
        d, tag = float(match.group(1)), match.group(2) or ""
        n = len(list(folder.glob("*.png")))
        rows = measure_folder(folder, K, dist, T_BC)            # raw PnP range: no correction
        if not rows:
            table.append(dict(d_cam=d, set=tag, frames=n, detected=0))
            continue
        x, r, b = (np.array([q[k] for q in rows]) for k in ("x", "r", "b"))
        x_true = d + CAM_AHEAD
        n_px, sig_px = np.mean([q["n_px"] for q in rows]), corner_sigma(rows)
        folders[(d, tag)] = folder
        table.append(dict(d_cam=d, set=tag, x_true=x_true, frames=n, detected=len(rows),
                          x_bias_mm=1000 * (x.mean() - x_true), x_bias_pct=100 * (x.mean() - x_true) / x_true,
                          range_sd_mm=1000 * r.std(ddof=1) if len(r) > 1 else np.nan,
                          bearing_deg=b.mean(), bearing_sd_deg=b.std(ddof=1) if len(b) > 1 else np.nan,
                          height_m=np.mean([q["z"] for q in rows]), size_px=np.mean([q["size_px"] for q in rows]),
                          n_px=n_px, side_seen_mm=1000 * n_px * d / K[0, 0], sigma_px=sig_px,
                          w3_sd_mm=1000 * aruco.sigma_range_model(d, K[0, 0], sigma_px=sig_px),
                          sim_bias_mm=1000 * est["range_bias_r2"] * x_true ** 2,
                          sim_sd_mm=1000 * (est["range_sd"][0] + est["range_sd"][1] * x_true ** 2)))
    table.sort(key=lambda t: (t["d_cam"], t["set"]))

    print(f"{'d_cam':>6} {'set':>4} {'x_true':>7} {'det':>5} {'bias mm':>8} {'bias %':>7} {'sd mm':>6} "
          f"{'bear deg':>8} {'h m':>6} {'px':>5} | sim bias, sd mm")
    for t in table:
        if not t["detected"]:
            print(f"{t['d_cam']:6.2f} {t['set']:>4}   marker 0 not found in {t['frames']} frames")
            continue
        print(f"{t['d_cam']:6.2f} {t['set']:>4} {t['x_true']:7.3f} {t['detected']:2d}/{t['frames']:<2d} {t['x_bias_mm']:8.1f} "
              f"{t['x_bias_pct']:7.2f} {t['range_sd_mm']:6.2f} {t['bearing_deg']:8.2f} {t['height_m']:6.3f} "
              f"{t['size_px']:5.0f} | {t['sim_bias_mm']:5.1f}, {t['sim_sd_mm']:4.1f}")

    latest = {t["d_cam"]: t for t in table if t["detected"]}     # sorted by set, so the last set wins
    good = [latest[d] for d in sorted(latest)]
    xt = np.array([t["x_true"] for t in good])
    bias = np.array([t["x_bias_mm"] for t in good]) / 1000

    # W3 apparent size: n * d is the same at every distance unless the tape reading is wrong
    fs = np.array([t["n_px"] * t["d_cam"] for t in good])
    ok = np.abs(fs / np.median(fs) - 1) <= SIZE_TOL
    for t, o in zip(good, ok):
        t["size_consistent"] = bool(o)
    print(f"\napparent size n * d [px m] (W3: n ~ f s / Z, so constant): "
          + ", ".join(f"{t['d_cam']:g}{t['set'] and '_' + t['set']} {t['n_px'] * t['d_cam']:.1f}" for t in good))
    for t in (t for t, o in zip(good, ok) if not o):
        print(f"  {t['d_cam']:g} m {t['set']}: {100 * (t['n_px'] * t['d_cam'] / np.median(fs) - 1):+.1f} % off the "
              f"median, i.e. the camera was about {1000 * t['d_cam'] * (np.median(fs) / (t['n_px'] * t['d_cam']) - 1):+.0f} mm "
              "from the tape reading: left out of the fit")
    side_seen = np.median(fs[ok]) / K[0, 0]
    print(f"printed side the camera sees: {1000 * side_seen:.1f} mm (code: MARKER_SIDE = {1000 * aruco.MARKER_SIDE:.1f} mm, "
          f"ratio {aruco.MARKER_SIDE / side_seen:.4f}). Measure the printed marker with a ruler: if it is this size, "
          "setting the side is the physical fix for the scale part of the bias")

    c_all = np.polyfit(xt, bias, 1)                              # every set, for comparison only
    c_lin, c0 = np.polyfit(xt[ok], bias[ok], 1)                  # the fit that is used: consistent sets
    resid = bias[ok] - (c0 + c_lin * xt[ok])
    sd_a = float(np.sqrt(np.sum(resid ** 2) / (ok.sum() - 2)))            # rms residual, 2 parameters fitted
    sd_b = float(aruco.sigma_range_model(1.0, (K[0, 0] + K[1, 1]) / 2))   # W3 model at 0.3 px, per m^2
    b_static = max(t["range_sd_mm"] / 1000 / t["x_true"] ** 2 for t in good)
    print(f"bias fit, all {len(xt)} sets: {1000 * c_all[1]:.1f} mm + {100 * c_all[0]:.2f} % of range (not used)")
    print(f"bias fit, {ok.sum()} size-consistent sets: {1000 * c0:.1f} mm + {100 * c_lin:.2f} % of range   "
          f"(residual rms {1000 * sd_a:.1f} mm)   (a constant = tape or lens offset; a % = focal length or printed "
          f"marker size)")
    print("noise: range sd measured vs W3 sigma_Z = Z^2 sigma_px / (f s): "
          + ", ".join(f"{t['d_cam']:g} m {t['range_sd_mm']:.2f}/{t['w3_sd_mm']:.2f} mm (sigma_px {t['sigma_px']:.3f})"
                      for t in good))
    print(f"range sd for the EKF: {1000 * sd_a:.1f} mm + {1000 * sd_b:.1f} mm * r^2   "
          f"(static frames need only {1000 * b_static:.1f} mm * r^2; sim model "
          f"{1000 * est['range_sd'][0]:.0f} mm + {1000 * est['range_sd'][1]:.0f} mm * r^2)")

    corr = dict(bias_m=round(float(c0), 4), bias_per_m=round(float(c_lin), 4), range_sd=[round(sd_a, 4), round(sd_b, 4)])
    print("after the correction (aruco.measure with range_correction), range - x_true:")
    for t in good:
        r = np.array([q["r"] for q in measure_folder(folders[(t["d_cam"], t["set"])], K, dist, T_BC, corr)])
        t["corr_bias_mm"] = 1000 * (r.mean() - t["x_true"])
        t["real_sd_mm"] = 1000 * (sd_a + sd_b * t["x_true"] ** 2)
        print(f"  x_true {t['x_true']:.3f} m: {t['corr_bias_mm']:+5.1f} mm   (model sd {t['real_sd_mm']:4.1f} mm)")

    out = PROJECT_ROOT / "results" / "aruco_noise_real.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(good[0].keys()))
        w.writeheader()
        w.writerows(good)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4))
    a0.plot(xt, 1000 * bias, "o-", label="real (tape)")
    a0.plot(xt, [t["sim_bias_mm"] for t in good], "s--", label="sim model (range_bias_r2)")
    a0.plot(xt, 1000 * np.polyval(c_all, xt), ":", color="0.6", label="fit through all sets (not used)")
    a0.plot(xt, 1000 * (c0 + c_lin * xt), "-", color="0.3", lw=1, label="fit through size-consistent sets")
    a0.plot(xt[~ok], 1000 * bias[~ok], "x", color="r", ms=10, mew=2, label="tape inconsistent with n * d")
    a0.plot(xt, [t["corr_bias_mm"] for t in good], "^-", label="real after range_correction")
    a0.set(xlabel="true base-to-marker distance [m]", ylabel="bias [mm]", title="ArUco range bias")
    a1.plot(xt, [t["range_sd_mm"] for t in good], "o-", label="real, static frames (lower bound)")
    a1.plot(xt, [t["sim_sd_mm"] for t in good], "s--", label="EKF range_sd model (sim)")
    a1.plot(xt, [t["real_sd_mm"] for t in good], "^-", label="range_sd for the real camera")
    a1.plot(xt, [t["w3_sd_mm"] for t in good], "v:", label="W3 law, measured corner sigma_px")
    a1.set_yscale("log")
    a1.set(xlabel="true base-to-marker distance [m]", ylabel="sd [mm]", title="ArUco range spread")
    for a in (a0, a1):
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=120)
    print(f"wrote {out.relative_to(PROJECT_ROOT)} and {out.with_suffix('.png').relative_to(PROJECT_ROOT)}")

    if args.write:
        dropped = ", ".join(f"{t['d_cam']:g} m {t['set']}".strip() for t, o in zip(good, ok) if not o) or "none"
        data = json.loads(CAMERA_FILE.read_text())
        data["range_correction"] = {
            **corr,
            "_doc": f"ArUco range bias of this camera, base origin to marker centre: r_meas = r_true + bias_m + "
                    f"bias_per_m * r_true; aruco.measure returns r_true. range_sd = [a, b]: EKF range sd a + b r^2 "
                    f"[m] for the corrected range (a = fit residual rms, b = W3 model at 0.3 px); it replaces "
                    f"range_sd, and range_bias_r2 becomes 0, in config/estimation.json. Fitted by "
                    f"experiments/aruco_real_noise.py from {ok.sum()} tape distances {xt[ok].min():.2f}-"
                    f"{xt[ok].max():.2f} m with this file's K and T_BC, residual rms {1000 * sd_a:.1f} mm; "
                    f"left out (tape inconsistent with the W3 apparent size n * d): "
                    f"{dropped}. "
                    f"The camera sees a {1000 * side_seen:.1f} mm marker: if the print measures that, set the side "
                    f"instead"}
        CAMERA_FILE.write_text(json.dumps(data, indent=2))
        print(f"wrote range_correction into {CAMERA_FILE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
