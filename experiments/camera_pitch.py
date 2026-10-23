"""How far the real camera is tilted down, from ArUco frames of one marker at several distances.

    python experiments/camera_pitch.py                                   # every results/hardware/aruco_cam*m* folder
    python experiments/camera_pitch.py --glob "pitch_cam*" --marker-height 0.10 --write

The robot looks straight at marker 0 from several distances; the marker centre stays at one height.
For each frame aruco.marker_pose gives the marker centre p_C in the camera, turned into "level camera"
axes (f forward, u up) with the URDF rotation. If the camera is really tilted down by theta, a point at a
fixed height difference dz from the camera reads
    u = tan(theta) * f + dz / cos(theta),
so a straight-line fit of u against f gives theta from its slope, whatever the marker height (and a
range scale error, e.g. the ~2 % ArUco bias, scales f and u together and leaves the slope alone).
With --marker-height [m] (the centre's height above the floor) the intercept also gives the camera
height. --write stores {"xyz", "pitch_deg"} as "T_BC" in config/camera_real.json; aruco.load_T_BC reads
it there, so the real robot's ArUco measurements and IBVS use it (the sim file has none and keeps the
URDF pose).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root, so "src" imports work

import cv2                                                      # noqa: E402
import numpy as np                                              # noqa: E402

from src.perception import aruco, camera                        # noqa: E402
from src.robot.sim_robot import PROJECT_ROOT                    # noqa: E402

CAMERA_FILE = PROJECT_ROOT / "config" / "camera_real.json"
R_LEVEL = aruco.camera_pose().R          # camera axes -> level base axes (x forward, y left, z up)


def frames(pattern, K, dist):
    """(folder, f, u) for every frame where marker 0 is found."""
    rows = []
    for folder in sorted(PROJECT_ROOT.glob(f"results/hardware/{pattern}")):
        for p in sorted(folder.glob("*.png")):
            corners = aruco.detect(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB), 0)
            if corners is None:
                continue
            T_CM = aruco.marker_pose(corners, K, dist)[0][0]
            f, _, u = R_LEVEL @ T_CM.t
            rows.append((folder.name, f, u))
    return rows


def main():
    ap = argparse.ArgumentParser(description="Fit the real camera's downward tilt from ArUco frames.")
    ap.add_argument("--glob", default="aruco_cam*m*", help="folders under results/hardware (default aruco_cam*m*)")
    ap.add_argument("--marker-height", type=float, default=None,
                    help="height of the marker centre above the floor the robot stands on [m]; gives the camera height")
    ap.add_argument("--write", action="store_true", help="store the result as T_BC in config/camera_real.json")
    args = ap.parse_args()

    K, dist, _ = camera.load_intrinsics(CAMERA_FILE)
    rows = frames(args.glob, K, dist)
    if len({r[0] for r in rows}) < 2:
        sys.exit(f"need frames at 2 or more distances, found {len(rows)} frames in {len({r[0] for r in rows})} folders")
    f = np.array([r[1] for r in rows])
    u = np.array([r[2] for r in rows])
    b, a = np.polyfit(f, u, 1)
    theta = np.arctan(b)
    resid = u - (a + b * f)
    print(f"{len(rows)} frames in {len({r[0] for r in rows})} folders, forward distance {f.min():.2f}-{f.max():.2f} m")
    for name in sorted({r[0] for r in rows}, key=lambda n: np.mean([r[1] for r in rows if r[0] == n])):
        sel = np.array([r[0] == name for r in rows])
        print(f"  {name:22s} f = {f[sel].mean():.3f} m   u = {1000 * u[sel].mean():6.1f} mm   "
              f"fit residual {1000 * resid[sel].mean():+5.1f} mm")
    print(f"\ncamera tilted DOWN by {np.rad2deg(theta):.2f} deg  (fit residual rms {1000 * resid.std():.1f} mm)")
    dz = a * np.cos(theta)
    print(f"marker centre is {1000 * dz:+.0f} mm relative to the camera height")

    z_cam = aruco.T_BC.t[2]
    if args.marker_height is not None:
        z_cam = args.marker_height - dz
        print(f"camera height {1000 * z_cam:.0f} mm above the floor (URDF {1000 * aruco.T_BC.t[2]:.0f} mm)")

    if args.write:
        data = json.loads(CAMERA_FILE.read_text())
        data["T_BC"] = {"xyz": [float(aruco.T_BC.t[0]), 0.0, round(float(z_cam), 4)],
                        "pitch_deg": round(float(np.rad2deg(theta)), 2),
                        "_doc": f"camera pose in base_footprint: URDF position"
                                f"{'' if args.marker_height is None else ' with the fitted height'}, tilted down "
                                f"by pitch_deg; fitted by experiments/camera_pitch.py from {len(rows)} frames "
                                f"({args.glob}), residual rms {1000 * resid.std():.1f} mm"}
        CAMERA_FILE.write_text(json.dumps(data, indent=2))
        print(f"wrote T_BC into {CAMERA_FILE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
