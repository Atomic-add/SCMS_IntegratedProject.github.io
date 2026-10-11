"""Look at the L-corridor with the robot at the start: LiDAR hits in the viewer, camera view in a window.

    python experiments/view_corridor.py                    # no obstacle
    python experiments/view_corridor.py --obstacle centre  # none / left / right / centre
    python experiments/view_corridor.py --no-camera        # MuJoCo viewer only

Drive the robot with the wheel sliders under Control in the viewer's right panel. The red dots are the
processed LiDAR scan placed with the wheel-odometry pose (src/estimation/odometry.py), not with the
simulator's true pose, so they slide off the walls as odometry drifts. The camera window marks every
ArUco marker it detects. The whole mission (EKF, LQR, avoidance, docking) is `python main.py`.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root, so "src" imports work

import cv2                                                      # noqa: E402
import mujoco                                                   # noqa: E402
import mujoco.viewer                                            # noqa: E402
import numpy as np                                              # noqa: E402

from src.estimation.odometry import Odometry                    # noqa: E402
from src.perception import aruco, lidar                         # noqa: E402
from src.robot.corridor import load_config, make_robot          # noqa: E402

DICT = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(aruco.ARUCO_DICT), cv2.aruco.DetectorParameters())


def draw_points(scn, P_W, cam, z=0.132, px=0.003):
    """Red dot at every LiDAR point in the world (display only, no physics).

    The radius grows with the distance to the viewer camera (px metres per metre), so every dot has
    the same size on screen, like RViz's "Points" style.
    """
    az, el = np.deg2rad(cam.azimuth), np.deg2rad(cam.elevation)
    fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    cam_pos = np.asarray(cam.lookat) - cam.distance * fwd
    scn.ngeom = 0
    for p in P_W.T:
        if scn.ngeom >= scn.maxgeom:
            break
        pos = np.array([p[0], p[1], z])
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([px * np.linalg.norm(pos - cam_pos), 0, 0]), pos, np.eye(3).ravel(),
                            np.array([1, 0, 0, 1], dtype=np.float32))
        scn.ngeom += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obstacle", default="none", choices=[k for k in load_config()["obstacles"] if not k.startswith("_")])
    ap.add_argument("--no-camera", action="store_true", help="do not open the camera window")
    args = ap.parse_args()

    robot = make_robot(args.obstacle, seed=0)
    start = load_config()["start"]                              # the same prior the estimator starts from
    odo = Odometry(start)
    odo.step(robot.read_encoders())                             # first sample = encoder baseline
    spec, last_scan, last_img, P_W, t_prev = robot.lidar, None, None, None, robot.t
    if not args.no_camera:
        robot.read_image()                                      # create the offscreen renderer first
    with mujoco.viewer.launch_passive(robot.m, robot.d) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = [2.3, 2.1, 0.0]
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 7.5, 90.0, -65.0
        while viewer.is_running():
            t0 = time.time()
            if robot.t < t_prev:                                # viewer Reset: MuJoCo put the robot at the
                robot.reset(*start)                             # XML origin and the clock at 0; go back to the
                odo = Odometry(start)                           # start pose, restart odometry and the sensor
                odo.step(robot.read_encoders())                 # clocks
                last_scan = last_img = None
            robot.step(0.02)
            t_prev = robot.t
            pose = odo.step(robot.read_encoders())              # wheel odometry, never the true pose
            t_scan, rho = robot.read_scan()
            if t_scan != last_scan:
                ok = lidar.valid_beams(rho, spec.range_min, spec.range_max)
                P_W = lidar.to_world(lidar.scan_to_points(rho, robot.scan_angles, ok), pose)
                last_scan = t_scan
            draw_points(viewer.user_scn, P_W, viewer.cam)
            if not args.no_camera:
                t_img, rgb = robot.read_image()
                if t_img != last_img:
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    corners, ids, _ = DICT.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
                    if ids is not None:
                        cv2.aruco.drawDetectedMarkers(bgr, corners, ids)
                    cv2.putText(bgr, f"markers seen: {[] if ids is None else sorted(ids.ravel().tolist())}",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.imshow("robot camera", bgr)
                    last_img = t_img
                cv2.waitKey(1)
            viewer.sync()
            time.sleep(max(0.0, 0.02 - (time.time() - t0)))
    cv2.destroyAllWindows()
    robot.close()


if __name__ == "__main__":
    main()