"""Look at the L-corridor with the robot at the start, LiDAR hits drawn and the camera view in a window.

    python experiments/view_corridor.py                    # no obstacle
    python experiments/view_corridor.py --obstacle centre  # none / left / right / centre
    python experiments/view_corridor.py --no-camera        # MuJoCo viewer only

The robot stands still: drive it with the wheel sliders under Control in the viewer's right panel.
The red dots are the processed LiDAR scan; the camera window marks every ArUco marker it detects.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root, so "src" imports work

import cv2                                                      # noqa: E402
import mujoco.viewer                                            # noqa: E402

from evaluation.check_lidar import draw_hits                    # noqa: E402
from evaluation.ground_truth import true_pose                   # noqa: E402  (display only)
from src.perception import aruco, lidar                         # noqa: E402
from src.robot.corridor import load_config, make_robot          # noqa: E402

DICT = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(aruco.ARUCO_DICT), cv2.aruco.DetectorParameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obstacle", default="none", choices=[k for k in load_config()["obstacles"] if not k.startswith("_")])
    ap.add_argument("--no-camera", action="store_true", help="do not open the camera window")
    args = ap.parse_args()

    robot = make_robot(args.obstacle, seed=0)
    spec, last_scan, last_img, P_W = robot.lidar, None, None, None
    if not args.no_camera:
        robot.read_image()                                      # create the offscreen renderer first
    with mujoco.viewer.launch_passive(robot.m, robot.d) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = [2.3, 2.1, 0.0]
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 7.5, 90.0, -65.0
        while viewer.is_running():
            t0 = time.time()
            robot.step(0.02)
            t_scan, rho = robot.read_scan()
            if t_scan != last_scan:
                ok = lidar.valid_beams(rho, spec.range_min, spec.range_max)
                P_W = lidar.to_world(lidar.scan_to_points(rho, robot.scan_angles, ok), true_pose(robot))
                last_scan = t_scan
            draw_hits(viewer.user_scn, P_W, cam=viewer.cam)
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