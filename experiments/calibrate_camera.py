"""Calibrate the camera from checkerboard images (W3 Parts B and C) and save the intrinsics.

    python experiments/calibrate_camera.py                      # sim: board waved in front of the robot
    python experiments/calibrate_camera.py --images DIR --out config/camera_real.json   # real Pi camera
    python experiments/calibrate_camera.py --camera rgb_lab --pitch-deg 6.3 --out config/camera_sim_lab.json
                                                                # sim camera with the lab robot's geometry

Sim mode places the 9 x 6, 25 mm board (models/robotis_tb3/scene_calib.xml) at random poses in front
of the robot, keeps the frames where the board is found, and calibrates from those frames only. It
never reads the camera's true intrinsics; evaluation/check_camera.py compares the result with them.
Writes config/camera_sim.json and results/camera_calibration.png.

--camera picks another MJCF camera (rgb_lab: main.py --sim-camera lab). --pitch-deg is that camera's
downward tilt as built into the MJCF: the board is held along it, and it is stored as "T_BC" (URDF
position, tilted) for aruco.load_T_BC, as experiments/camera_pitch.py does for the real camera.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root, so "src" imports work

import cv2                                                      # noqa: E402
import matplotlib                                               # noqa: E402
import numpy as np                                              # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402
from spatialmath import SO3, UnitQuaternion                     # noqa: E402

from src.perception import aruco, camera                        # noqa: E402
from src.robot.sim_robot import PROJECT_ROOT, CameraSpec, SimRobot  # noqa: E402

CALIB_SCENE = PROJECT_ROOT / "models" / "robotis_tb3" / "scene_calib.xml"


def sim_views(n_good=20, max_tries=200, seed=0, cam="rgb", pitch_deg=0.0):
    """Greyscale frames of the board at random poses 0.35-0.55 m in front of the robot camera.

    cam: the MJCF camera; pitch_deg: its downward tilt, so the board is held along the line of sight.
    A tilted camera sees the floor across the bottom of the image at these distances, so a board held
    low is cut by the floor and rejected and the bottom rows get no corners (rgb_lab: principal point
    8 px off, rms 0.4 px). For a tilted camera the floor is therefore hidden from the renderer (geom
    group 3); it still holds the robot up.
    """
    rng = np.random.default_rng(seed)
    R0 = SO3(np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float))   # board +z towards the robot (= xyaxes in MJCF)
    frames, gray = [], []
    with SimRobot(scene=CALIB_SCENE, seed=seed, camera=CameraSpec(name=cam)) as robot:
        if pitch_deg:
            robot.m.geom_group[robot.m.geom("floor").id] = 3
        robot.reset(0.0, 0.0, 0.0)
        bid = robot.m.body("calib_board").mocapid[0]
        for _ in range(max_tries):
            Z = rng.uniform(0.35, 0.55)
            # roughly in front of the camera (0.076 m ahead of the base, 0.103 m up), with tilts like a hand-held board
            robot.d.mocap_pos[bid] = [0.076 + Z, rng.uniform(-0.25, 0.25) * Z,
                                      0.103 + (rng.uniform(-0.15, 0.15) - np.tan(np.deg2rad(pitch_deg))) * Z]
            R = R0 * SO3.Rx(rng.uniform(-0.6, 0.6)) * SO3.Ry(rng.uniform(-0.6, 0.6)) * SO3.Rz(rng.uniform(-0.35, 0.35))
            robot.d.mocap_quat[bid] = UnitQuaternion(R).vec
            _, rgb = robot.next_image()                         # a fresh frame of the new board pose
            g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            if camera.measured_corners(g) is not None:
                frames.append(rgb)
                gray.append(g)
                if len(gray) == n_good:
                    break
    return gray, frames


def folder_views(folder):
    """Greyscale images from a folder of real camera frames (png or jpg)."""
    paths = sorted([p for ext in ("*.png", "*.jpg", "*.jpeg") for p in Path(folder).glob(ext)])
    imgs = [cv2.imread(str(p)) for p in paths]
    return [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in imgs], [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in imgs]


def plot(calib, frames, out):
    """One calibration frame with its corners, and the per-image reprojection error (W3 Part C)."""
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw=dict(width_ratios=[1.1, 1]))
    rgb = frames[calib["used"][0]]
    c = camera.measured_corners(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    a0.imshow(rgb)
    a0.plot(c[:, 0], c[:, 1], "+", color="#eb6834", ms=8, mew=1.5)
    a0.set_title(f"Board found in {len(calib['used'])} of {calib['n_images']} frames (frame shown: corners +)",
                 loc="left", fontsize=10.5)
    a0.axis("off")
    e = calib["per_image_rms"]
    a1.bar(range(len(e)), e, color="#2a78d6", width=0.7)
    a1.axhline(calib["rms"], color="#333333", lw=1.2, ls="--", label=f"overall rms {calib['rms']:.3f} px")
    a1.set_xlabel("calibration image"); a1.set_ylabel("reprojection error rms [px]")
    a1.set_title("Reprojection error per image (W3 Part C)", loc="left", fontsize=10.5)
    a1.legend(frameon=False)
    a1.grid(color="#eeeeee", lw=0.8); a1.set_axisbelow(True)
    for sp in a1.spines.values():
        sp.set_color("#cccccc")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", help="folder of real checkerboard images; default: capture in the sim")
    ap.add_argument("--out", help="output JSON (default config/camera_sim.json, or camera_real.json with --images)")
    ap.add_argument("--camera", default="rgb", help="sim: the MJCF camera to calibrate (default rgb)")
    ap.add_argument("--pitch-deg", type=float, default=None,
                    help="sim: the camera's downward tilt in the MJCF [deg]; stored as T_BC in the output")
    args = ap.parse_args()

    if args.images:
        gray, frames = folder_views(args.images)
        source, default_out = f"real images from {args.images}", "camera_real.json"
    else:
        gray, frames = sim_views(cam=args.camera, pitch_deg=args.pitch_deg or 0.0)
        cam = "" if args.camera == "rgb" else f"camera {args.camera}, "
        source, default_out = f"sim: scene_calib.xml, {cam}20 board poses, seed 0", "camera_sim.json"
    calib = camera.calibrate(gray)

    K, (w, h) = calib["K"], calib["image_size"]
    fov_h, fov_v = camera.fov_deg(K, (w, h))
    print(f"board found in {len(calib['used'])} of {calib['n_images']} images")
    print(f"f/rho_u = {K[0, 0]:.2f} px, f/rho_v = {K[1, 1]:.2f} px")
    print(f"principal point = ({K[0, 2]:.2f}, {K[1, 2]:.2f}) px, image centre = ({(w - 1) / 2:.1f}, {(h - 1) / 2:.1f}) px")
    print(f"field of view = {fov_h:.2f} x {fov_v:.2f} deg")
    print(f"distortion (k1, k2, p1, p2, k3) = {np.round(calib['dist'], 4)}")
    print(f"reprojection error: overall rms {calib['rms']:.3f} px, worst image {max(calib['per_image_rms']):.3f} px")

    out = Path(args.out) if args.out else PROJECT_ROOT / "config" / default_out
    camera.save_intrinsics(out, calib, source)
    if args.pitch_deg is not None and not args.images:
        data = json.loads(out.read_text())
        data["T_BC"] = {"xyz": aruco.T_BC.t.tolist(), "pitch_deg": args.pitch_deg,
                        "_doc": f"camera pose in base_footprint: URDF position, tilted down by pitch_deg as built "
                                f"into the MJCF camera {args.camera} (calibrate_camera.py --pitch-deg)"}
        out.write_text(json.dumps(data, indent=2))
    name = "_real" if args.images else ("" if args.camera == "rgb" else f"_{args.camera}")
    fig_out = PROJECT_ROOT / "results" / f"camera_calibration{name}.png"
    fig_out.parent.mkdir(exist_ok=True)
    plot(calib, frames, fig_out)
    print(f"\nwrote {out.relative_to(PROJECT_ROOT) if out.is_relative_to(PROJECT_ROOT) else out}")
    print(f"wrote {fig_out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()