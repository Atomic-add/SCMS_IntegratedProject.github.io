"""Track A demo entry point: the whole mission in the MuJoCo simulation.

    python main.py                       # centre box, EKF, MuJoCo viewer + camera window, real time
    python main.py --case none           # obstacle case: none | left | centre | right
    python main.py --odometry            # steer on wheel odometry only (the baseline, no EKF)
    python main.py --seed 3              # another sensor-noise seed
    python main.py --no-camera           # MuJoCo viewer only
    python main.py --headless            # no windows (Linux without a display: MUJOCO_GL=egl is set)
    python main.py --randomise --seed 7  # a random scenario: route, start, boxes, noise, wheel error
    python main.py --dock pbvs           # dock on the LQR all the way (default: ibvs for the last part)

The robot starts from the prior in config/corridor.json, follows the waypoints with the LQR, goes
around the box with the LiDAR and docks at the ArUco station (marker ID 0). With --randomise every seed
gives a different scenario (src/robot/scenario.py): the robot is TOLD a new start spot (its prior) and a
new route; it is NOT told where it was really placed, where the boxes are, how noisy its sensors are or
how wrong its wheel radii are. The terminal prints both halves. The estimator is the EKF in
src/estimation (wheel odometry + ArUco range-bearing) and the controller acts on its estimate xh only.
Nothing here reads simulator ground truth; evaluation/ is the only place that does.

Exit code: 0 when the robot docked, 1 otherwise, so a script or CI can check the run.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))                      # project root, so "src" imports work from any folder

CASES = ("none", "left", "centre", "right")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Track A mission: waypoints, LiDAR avoidance, ArUco docking.")
    ap.add_argument("--case", default="centre", choices=CASES, help="obstacle case (default: centre)")
    ap.add_argument("--odometry", action="store_true", help="steer on wheel odometry only, without the EKF")
    ap.add_argument("--seed", type=int, default=0, help="sensor-noise seed, and the scenario with --randomise (default: 0)")
    ap.add_argument("--randomise", action="store_true",
                    help="random route, start placement, boxes, sensor noise and wheel error for this seed (ignores --case)")
    ap.add_argument("--dock", choices=("ibvs", "pbvs"), default=None,
                    help="docking: LQR then image-based visual servoing (ibvs), or the LQR all the way (pbvs); "
                         "default from config/control.json")
    ap.add_argument("--headless", action="store_true", help="no viewer or camera window")
    ap.add_argument("--no-camera", action="store_true", help="MuJoCo viewer only, no camera window")
    return ap.parse_args(argv)


def camera_window(K, dist, marker_ids, dock_id=0):
    """on_step hook for run_mission: show the robot camera with the markers it sees.

    It only calls robot.read_image(), the same sensor read the robot's own code uses. The docking
    marker is drawn with its axes and camera-frame position, the EKF landmarks with a yellow outline
    (aruco.draw_markers).
    """
    import cv2
    from src.perception import aruco

    last = {"t": None}

    def on_step(robot, row):
        t_frame, rgb = robot.read_image()           # latest camera frame (15 Hz), no new render if not due
        if t_frame == last["t"]:
            return
        last["t"] = t_frame
        img = aruco.draw_markers(rgb, K, dist, marker_ids, dock_id=dock_id)
        cv2.imshow("robot camera", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    return on_step


def main(argv=None):
    args = parse_args(argv)
    if args.headless and sys.platform.startswith("linux"):
        os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen rendering without a display; set before mujoco loads

    from src.control.behaviour import run_mission
    from src.control.models import load_control_config
    from src.estimation.ekf import EKF
    from src.perception import camera
    from src.robot.corridor import load_task

    task = load_task()                              # start prior, waypoints, marker map, dock goal
    case, setup, label = args.case, None, args.case
    if args.randomise:
        from src.robot.scenario import apply_world, describe, random_scenario
        sc = random_scenario(args.seed)
        print(describe(sc))
        task, case, label = sc["task"], "none", f"random seed {args.seed}"   # boxes come from the scenario
        setup = lambda robot: apply_world(robot, sc["world"])              # noqa: E731  (sim side only)
    estimator = None if args.odometry else EKF(task["start"], marker_map=task["markers"])
    cfg = load_control_config()
    if args.dock:
        cfg["docking"]["servo"] = args.dock
    on_step = None
    if not args.headless and not args.no_camera:
        K, dist, _ = camera.load_intrinsics(ROOT / "config" / "camera_sim.json")
        on_step = camera_window(K, dist, sorted(task["markers"]), task["dock"]["marker_id"])

    print(f"{label}, steering on {'odometry' if args.odometry else 'the EKF'}, docking {cfg['docking']['servo']}, "
          f"seed {args.seed}{', headless' if args.headless else ''}")
    info, t_end = run_mission(case, seed=args.seed, verbose=True, estimator=estimator, view=not args.headless,
                              on_step=on_step, setup=setup, task=task, label=label, cfg=cfg)
    if on_step is not None:
        import cv2
        cv2.destroyAllWindows()

    reason = info["stop_reason"] or "stopped before the end (window closed or time limit)"
    print(f"result: {reason} after {t_end:.1f} s of sim time, dock_complete = {info['dock_complete']}")
    return 0 if info["dock_complete"] else 1


if __name__ == "__main__":
    sys.exit(main())