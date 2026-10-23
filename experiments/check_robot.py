"""First check on the real TurtleBot: is every topic there, and does config/robot_real.json match it?

    python experiments/check_robot.py 192.168.1.213             # listen only, the robot does not move
    python experiments/check_robot.py 192.168.1.213 --drive     # also 0.05 m/s forward for 2 s (10 cm)
    python experiments/check_robot.py --drive --distance 0.5 --seconds 1   # 50 cm, to compare with a tape

Listening, it prints the wheel angles twice a second: push the robot forward by hand and both
should grow (otherwise flip wheel_sign in the config). Then it reports what the placeholders in
src/robot/sim_robot.py and config/ need from the lab robot: topic rates, joint names, the raw /scan
layout and how its missing returns are coded, the forward coverage the mission waits on (stand the
robot on the track's start pose to check it), the image size, and the ArUco IDs in view.
--drive checks /cmd_vel and the wheel signs together: the encoders should measure about 10 cm.
Everything it sees is saved under results/hardware/<time>/ (image.png, scan.npz, report.json).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root, so "src" imports work

import cv2                                                      # noqa: E402
import numpy as np                                              # noqa: E402

from src.control.lidar_obs import make_obs                      # noqa: E402
from src.control.models import load_control_config              # noqa: E402
from src.perception import aruco, lidar                         # noqa: E402
from src.robot.real_robot import RealRobot                      # noqa: E402
from src.robot.sim_robot import PROJECT_ROOT, WHEEL_RADIUS, WHEEL_SEPARATION   # noqa: E402


def listen(robot, seconds):
    """Print the wheel angles while the user pushes the robot; return the topic rates [Hz]."""
    print(f"\nlistening {seconds:.0f} s: push the robot FORWARD by hand, both wheel angles should grow")
    c0, t0 = robot.message_counts(), robot.t
    t_print = 0.0
    while robot.t - t0 < seconds:
        if robot.t >= t_print:
            _, phi_l, phi_r = robot.read_encoders()
            print(f"  t = {robot.t - t0:4.1f} s   phi_L = {phi_l:8.3f} rad   phi_R = {phi_r:8.3f} rad")
            t_print = robot.t + 0.5
        time.sleep(0.02)
    c1, dt = robot.message_counts(), robot.t - t0
    return {k: (c1[k] - c0.get(k, 0)) / dt for k in c1}


def scan_report(robot):
    """The raw /scan layout and how many beams are 0 / inf / NaN / outside the range limits."""
    _, scan = robot.latest("scan")
    r = scan.ranges
    with np.errstate(invalid="ignore"):
        rep = dict(n_beams=int(r.size), angle_min_deg=float(np.rad2deg(scan.angle_min)),
                   angle_max_deg=float(np.rad2deg(scan.angle_max)),
                   angle_increment_deg=float(np.rad2deg(scan.angle_increment)),
                   range_min=scan.range_min, range_max=scan.range_max, scan_time=scan.scan_time,
                   zero=int(np.sum(r == 0)), inf=int(np.sum(np.isinf(r))), nan=int(np.sum(np.isnan(r))),
                   out_of_range=int(np.sum(np.isfinite(r) & (r != 0) & ((r < scan.range_min) | (r > scan.range_max)))))
    t_scan, rho = robot.read_scan()
    ok = lidar.valid_beams(rho, robot.lidar.range_min, robot.lidar.range_max)
    rep["valid_after_resampling"] = int(ok.sum())
    rep["range_at_deg"] = {d: (float(rho[d]) if ok[d] else None) for d in (0, 90, 180, 270)}
    # What the mission sees: the obs the navigation waits on (coverage of the +-90 deg forward half)
    obs = make_obs(t_scan, rho, robot.lidar.range_min, robot.lidar.range_max)
    rep["forward_coverage"] = round(float(obs["coverage"]), 3)
    rep["min_coverage"] = load_control_config()["lidar_obs"]["min_coverage"]
    rep["scan_valid_for_driving"] = bool(obs["valid"])
    rep["free_ahead_m"] = round(float(obs["free_ahead"]), 3)
    return rep, scan


def markers_in_view(rgb):
    """IDs of the DICT_4X4_50 markers in the frame (no intrinsics needed)."""
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(aruco.ARUCO_DICT))
    _, ids, _ = det.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    return [] if ids is None else sorted(int(i) for i in ids.ravel())


def drive_test(robot, v=0.05, seconds=2.0):
    """Drive straight at v for the given time; return the distance and turn the encoders measured."""
    print(f"\n--drive: {v} m/s forward for {seconds} s in 3 s, keep the area in front clear")
    time.sleep(3.0)
    _, l0, r0 = robot.read_encoders()
    t_end = robot.t + seconds
    while robot.t < t_end:
        robot.send_cmd(v, 0.0)
        robot.step(0.05)
    robot.send_cmd(0.0, 0.0)
    time.sleep(0.5)                                             # let it stop and the last joint state arrive
    _, l1, r1 = robot.read_encoders()
    dl, dr = (l1 - l0) * WHEEL_RADIUS, (r1 - r0) * WHEEL_RADIUS
    return dict(commanded_m=v * seconds, left_m=dl, right_m=dr, distance_m=(dl + dr) / 2,
                turn_deg=float(np.rad2deg((dr - dl) / WHEEL_SEPARATION)))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Check the real TurtleBot against config/robot_real.json.")
    ap.add_argument("host", nargs="?", help="robot IP, 192.168.1.2XX (default: host in the config)")
    ap.add_argument("--seconds", type=float, default=10.0, help="how long to listen (default 10 s)")
    ap.add_argument("--drive", action="store_true", help="also drive 0.05 m/s forward (default 10 cm)")
    ap.add_argument("--distance", type=float, default=0.10, help="how far --drive goes [m] (default 0.10, max 1.0)")
    args = ap.parse_args(argv)

    out = PROJECT_ROOT / "results" / "hardware" / time.strftime("%Y%m%d_%H%M%S")
    with RealRobot(args.host, strict=False) as robot:      # report a second robot instead of refusing
        print(f"connected to {robot.host}; every sensor topic has a first message")
        names = list(robot.latest("joint_states")[1]["name"])
        print(f"/joint_states joints: {names}  (config wheel_joints {robot.wheel_joints})")

        rates = listen(robot, args.seconds)
        scan_rep, scan = scan_report(robot)
        t_img, rgb = robot.read_image()
        report = dict(host=robot.host, joints=names, rates_hz=rates, scan=scan_rep,
                      image_size=[robot.camera.width, robot.camera.height], markers_in_view=markers_in_view(rgb),
                      joint_jumps=robot.joint_jumps())
        two_robots = robot.joint_jumps() > 0
        if args.drive and two_robots:
            print("\n--drive skipped: the wheel angles jump (a second robot?), /cmd_vel could move it too")
        elif args.drive:
            report["drive"] = drive_test(robot, seconds=min(abs(args.distance), 1.0) / 0.05)

        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "image.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        np.savez(out / "scan.npz", angles=scan.angles, ranges=scan.ranges, intensities=scan.intensities)
        (out / "report.json").write_text(json.dumps(report, indent=2))

    print("\nrates [Hz]: " + ", ".join(f"{k} {v:.1f}" for k, v in rates.items()) +
          "   (sim: joint_states every step, scan 5, image 15)")
    if two_robots:
        print(f"\n!!! /joint_states jumped {report['joint_jumps']} times between messages: a SECOND ROBOT is "
              "publishing on the same topics\n!!! (same ROS_DOMAIN_ID on the lab wifi), or the robot restarted. "
              "Rates near twice the usual (joint_states 20, scan 5, image 30 Hz) point to a second robot.\n"
              "!!! main.py --real refuses to drive like this. Ask the tutor for a separate ROS_DOMAIN_ID per robot.\n")
    print("scan:", json.dumps(scan_rep))
    print(f"forward LiDAR coverage {100 * scan_rep['forward_coverage']:.0f} % (mission needs "
          f"{100 * scan_rep['min_coverage']:.0f} %): "
          + ("OK, the robot may drive here" if scan_rep["scan_valid_for_driving"] else
             "TOO LOW, the mission would wait here: panels farther than 3.5 m ahead give no return"))
    print(f"image {robot.camera.width}x{robot.camera.height} (sim 640x480), markers in view: {report['markers_in_view']}")
    if "drive" in report:
        d = report["drive"]
        print(f"drive: commanded {d['commanded_m']:.3f} m, encoders {d['left_m']:.3f} / {d['right_m']:.3f} m "
              f"(L / R), turn {d['turn_deg']:.1f} deg")
        if d["left_m"] < 0 or d["right_m"] < 0:
            print("  a wheel ran backwards in the encoders: flip its wheel_sign in config/robot_real.json "
                  "(both negative: the robot may have driven backwards, check /cmd_vel)")
    print(f"saved {out.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
