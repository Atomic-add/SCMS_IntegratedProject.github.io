"""ArUco range on the real camera against tape-measured distances: bias and noise for the EKF's Rn.

    python experiments/aruco_real_noise.py      # reads results/hardware/aruco_cam<d>m/, marker ID 0

Each folder holds frames saved by capture_frames.py with the robot looking straight at the marker,
<d> = tape distance from the camera lens to the marker face [m]. The camera is 0.076 m ahead of the base
origin (aruco.T_BC), so the true forward distance base -> marker is d + 0.076. For every frame,
aruco.measure (with config/camera_real.json, its K and its T_BC) gives the marker centre in {B}; per distance this prints
the forward-distance bias x_meas - x_true, the range and bearing spread, and the marker centre height.
The frames at one distance show a static scene, so the sd is pixel noise only and is a lower bound;
the bias across distances is what calibration (K, distortion, printed size) leaves behind.
A repeat set is a folder with a suffix (aruco_cam0.5m_v2): it is listed too, and the latest set at
each distance replaces the earlier ones in the fit, the csv and the plot.
Compares with the sim model in config/estimation.json and writes results/aruco_noise_real.csv / .png.
"""
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

CAM_AHEAD = aruco.T_BC.t[0]      # 0.076 m, camera ahead of the base origin


def measure_folder(folder, K, dist, T_BC):
    rows = []
    for p in sorted(folder.glob("*.png")):
        rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        m = aruco.measure(rgb, K, dist, T_BC=T_BC, marker_id=0)
        if m is not None:
            rows.append(dict(x=m["T_BM"].t[0], z=m["T_BM"].t[2], r=m["z"][0], b=np.rad2deg(m["z"][1]),
                             size_px=m["size_px"], ambiguous=m["ambiguous"]))
    return rows


def main():
    K, dist, size = camera.load_intrinsics(PROJECT_ROOT / "config" / "camera_real.json")
    T_BC = aruco.load_T_BC(PROJECT_ROOT / "config" / "camera_real.json")   # with the fitted tilt, if any
    est = json.loads((PROJECT_ROOT / "config" / "estimation.json").read_text())
    table = []
    for folder in PROJECT_ROOT.glob("results/hardware/aruco_cam*"):
        match = re.fullmatch(r"aruco_cam([\d.]+)m(?:_(\w+))?", folder.name)
        if match is None:
            continue
        d, tag = float(match.group(1)), match.group(2) or ""
        n = len(list(folder.glob("*.png")))
        rows = measure_folder(folder, K, dist, T_BC)
        if not rows:
            table.append(dict(d_cam=d, set=tag, frames=n, detected=0))
            continue
        x, r, b = (np.array([q[k] for q in rows]) for k in ("x", "r", "b"))
        x_true = d + CAM_AHEAD
        table.append(dict(d_cam=d, set=tag, x_true=x_true, frames=n, detected=len(rows),
                          x_bias_mm=1000 * (x.mean() - x_true), x_bias_pct=100 * (x.mean() - x_true) / x_true,
                          range_sd_mm=1000 * r.std(ddof=1) if len(r) > 1 else np.nan,
                          bearing_deg=b.mean(), bearing_sd_deg=b.std(ddof=1) if len(b) > 1 else np.nan,
                          height_m=np.mean([q["z"] for q in rows]), size_px=np.mean([q["size_px"] for q in rows]),
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
    c_lin, c0 = np.polyfit(xt, bias, 1)
    print(f"\nbias fit: {1000 * c0:.1f} mm + {100 * c_lin:.2f} % of range"
          f"   (a constant = tape or lens offset; a % = focal length or printed marker size)")

    out = PROJECT_ROOT / "results" / "aruco_noise_real.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(good[0].keys()))
        w.writeheader()
        w.writerows(good)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4))
    a0.plot(xt, 1000 * bias, "o-", label="real (tape)")
    a0.plot(xt, [t["sim_bias_mm"] for t in good], "s--", label="sim model (range_bias_r2)")
    a0.plot(xt, 1000 * (c0 + c_lin * xt), ":", color="0.5", label="linear fit")
    a0.set(xlabel="true base-to-marker distance [m]", ylabel="bias [mm]", title="ArUco range bias")
    a1.plot(xt, [t["range_sd_mm"] for t in good], "o-", label="real, static frames (lower bound)")
    a1.plot(xt, [t["sim_sd_mm"] for t in good], "s--", label="EKF range_sd model")
    a1.set(xlabel="true base-to-marker distance [m]", ylabel="sd [mm]", title="ArUco range spread")
    for a in (a0, a1):
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=120)
    print(f"wrote {out.relative_to(PROJECT_ROOT)} and {out.with_suffix('.png').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
