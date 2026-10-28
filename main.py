"""Track A demo entry point: the whole mission in the MuJoCo simulation, or on the real TurtleBot.

    python main.py                       # the lab track as built in the lab, centre box, lab camera, viewer
    python main.py --case none           # obstacle case: none | left | centre | right
    python main.py --odometry            # steer on wheel odometry only (the baseline, no EKF)
    python main.py --seed 3              # another sensor-noise seed
    python main.py --no-camera           # MuJoCo viewer only
    python main.py --headless            # no windows (Linux without a display: MUJOCO_GL=egl is set)
    python main.py --dock pbvs           # dock on the LQR all the way (default: ibvs for the last part)
    python main.py --task config/stage1_dock.json --case none    # one stage of the lab track in the sim
    python main.py --randomise --seed 7  # a random lab scenario: route, start, box, noise, wheel error
    python main.py --real --task config/stage1_dock.json         # real robot, host from config/robot_real.json
    python main.py --real 192.168.1.213 --task config/arena_lab.json   # another robot, the whole lab track
    python main.py --headless --log results/run.npz              # also save the run log (always on with --real)

Without --task the sim runs config/arena_lab.json, the lab floor as the robot will meet it: the 0.5 m
tiles, the panels, the 20 cm box and the three markers, the LDS-01 range, and the lab robot's camera.
The robot starts from the task file's prior, follows the waypoints with the LQR, goes around the box with
the LiDAR and docks at the ArUco station (marker ID 0). There is one simulated world, the lab track, and
one simulated camera, the lab robot's (config/camera_sim.json). --randomise draws lab scenarios
(src/robot/scenario.py): the robot is TOLD a new start spot (its prior) and a new route; it is NOT told
where it was really placed, where the box is, how noisy its sensors are or how wrong its wheel radii
are. The terminal prints both halves. The estimator is the EKF in
src/estimation (wheel odometry + ArUco range-bearing) and the controller acts on its estimate xh only.
Nothing here reads simulator ground truth; evaluation/ is the only place that does.

--real swaps only the robot: the same run_mission, estimator and controllers drive the TurtleBot over
rosbridge. It needs config/camera_real.json (experiments/calibrate_camera.py --images ...) and a
task file measured in the lab arena (--task is required: config/arena_lab.json or a stage that
extends it). --case has no effect there, the box is wherever it was put. Run
experiments/check_robot.py first. Every real run is logged (--log, default results/hardware/run_<time>.npz),
also when it stops early, so a failure can be diagnosed afterwards.

The simulated camera is the lab robot's camera as measured on 23 Oct (49.7 x 38.2 deg, 6.3 deg down,
30 Hz), so a sim run sees what the real camera sees.

Exit code: 0 when the robot docked, 1 otherwise, so a script or CI can check the run.

Adapted from 41014 Week 8 C (run_landmark_lqg: LQR on the range-bearing EKF, closed loop).
Additions: one entry point for the sim and the real robot (--real), the camera window, the run log,
the start-pose warning and the printed docking checks.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LAB_TASK = ROOT / "config" / "arena_lab.json"           # the default: the lab floor as built in the lab
sys.path.insert(0, str(ROOT))                      # project root, so "src" imports work from any folder

CASES = ("none", "left", "centre", "right")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Track A mission: waypoints, LiDAR avoidance, ArUco docking.")
    ap.add_argument("--case", default="centre", choices=CASES, help="obstacle case (default: centre)")
    ap.add_argument("--odometry", action="store_true", help="steer on wheel odometry only, without the EKF")
    ap.add_argument("--seed", type=int, default=0, help="sensor-noise seed, and the scenario with --randomise (default: 0)")
    ap.add_argument("--randomise", action="store_true",
                    help="random lab route, start placement, box, sensor noise and wheel error for this seed "
                         "(ignores --case; runs config/arena_lab.json)")
    ap.add_argument("--dock", choices=("ibvs", "pbvs"), default=None,
                    help="docking: LQR then image-based visual servoing (ibvs), or the LQR all the way (pbvs); "
                         "default from config/control.json")
    ap.add_argument("--headless", action="store_true", help="no viewer or camera window")
    ap.add_argument("--no-camera", action="store_true", help="MuJoCo viewer only, no camera window")
    ap.add_argument("--real", nargs="?", const="", metavar="HOST",
                    help="drive the real TurtleBot over rosbridge (HOST default: config/robot_real.json)")
    ap.add_argument("--task", type=Path, default=None,
                    help="start prior, waypoints, markers and dock (default config/arena_lab.json, the lab floor; "
                         "or a stage, e.g. config/stage1_dock.json; required with --real)")
    ap.add_argument("--log", type=Path, default=None,
                    help="save the run (t, xh, P, v, w, state, EKF NIS) as .npz; default with --real: "
                         "results/hardware/run_<time>.npz")
    return ap.parse_args(argv)


def run_logger(estimator):
    """on_step hook that keeps every step's row, and save(path, **extra) that writes them as .npz.

    Only what the robot itself knows is logged (the estimate, its P, the commands, the state and the
    EKF's NIS per marker), so the same log comes from the sim and from the real robot. On the real
    robot the truth is measured by hand at the end (tape from the dock panel).
    """
    import numpy as np

    rows = []

    def on_step(robot, row):
        rows.append(row)

    def save(path, **extra):
        path.parent.mkdir(parents=True, exist_ok=True)
        nis = np.array(getattr(estimator, "log", []), dtype=float).reshape(-1, 4)   # (t, id, NIS, used)
        np.savez(path, t=np.array([r["t"] for r in rows]), xh=np.array([r["xh"] for r in rows]).reshape(-1, 3),
                 P=np.array([r["P"] for r in rows]).reshape(-1, 3, 3), v=np.array([r["v"] for r in rows]),
                 w=np.array([r["w"] for r in rows]), state=np.array([r["state"] for r in rows]),
                 nis=nis, **extra)
        print(f"log: {len(rows)} steps, {len(nis)} marker updates -> {path}")

    return on_step, save


def camera_window(K, dist, marker_ids, dock_id=0, T_BC=None, range_correction=None):
    """on_step hook for run_mission: show the robot camera with the markers it sees.

    It only calls robot.read_image(), the same sensor read the robot's own code uses. The docking
    marker is drawn with its axes and camera-frame position, the EKF landmarks with a yellow outline
    (aruco.draw_markers).
    """
    import cv2
    from src.perception import aruco

    last = {"t": None}

    def on_step(robot, row):
        t_frame, rgb = robot.read_image()           # latest camera frame (30 Hz), no new render if not due
        if t_frame == last["t"]:
            return
        last["t"] = t_frame
        img = aruco.draw_markers(rgb, K, dist, marker_ids, dock_id=dock_id, T_BC=T_BC if T_BC is not None else aruco.T_BC,
                                 range_correction=range_correction)
        cv2.imshow("robot camera", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    return on_step


def start_check(estimator, n=15):
    """on_step hook: say it loudly when the first n marker updates are all rejected by the EKF's NIS gate. That
    means the start pose in the task file disagrees with what the camera sees (28 Oct: the robot stood 0.25 m
    from where the task file put it, and the whole run went on odometry)."""
    said = []

    def check(robot, row):
        import numpy as np
        log = getattr(estimator, "log", [])
        if not said and len(log) >= n:
            said.append(True)
            first = np.array(log[:n], dtype=float)
            if first[:, 3].sum() == 0:
                print(f"WARNING: the EKF rejected the first {n} marker updates (median NIS {np.median(first[:, 2]):.0f}): "
                      f"the robot is not where the task file's start says. Stop (Ctrl+C), measure its distance to "
                      f"the marker and put it on the start pose.")
    return check


def main(argv=None):
    args = parse_args(argv)
    real = args.real is not None
    if real and args.randomise:
        sys.exit("--randomise changes the simulated world; it has no meaning with --real")
    if real and args.task is None:
        sys.exit("--real needs --task (e.g. config/stage1_dock.json): the task file is the measured lab floor "
                 "the robot will follow, so it is never assumed")
    if args.randomise and args.task is not None and Path(args.task).resolve() != LAB_TASK:
        sys.exit("--randomise draws scenarios of the whole lab track (src/robot/scenario.py): leave out --task")
    if args.headless and sys.platform.startswith("linux"):
        os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen rendering without a display; set before mujoco loads

    from src.robot.sim_robot import SIM_INTRINSICS
    task_path = Path(args.task) if args.task else LAB_TASK
    intrinsics = ROOT / "config" / "camera_real.json" if real else SIM_INTRINSICS
    if not intrinsics.exists():
        how = "--images DIR --out config/camera_real.json" if real else "--pitch-deg 6.3"
        sys.exit(f"{intrinsics.relative_to(ROOT)} is missing: calibrate the robot camera first with\n"
                 f"  python experiments/calibrate_camera.py {how}")

    from src.control.behaviour import run_mission
    from src.control.models import load_control_config
    from src.estimation.ekf import EKF
    from src.estimation.models import load_estimation_config
    from src.perception import camera
    from src.robot.task import load_task

    task = load_task(task_path)                     # start prior, waypoints, marker map, dock goal
    case, setup, label = args.case, None, args.case
    if args.randomise:
        from src.robot.scenario import apply_world, describe, random_scenario
        sc = random_scenario(args.seed)
        print(describe(sc))
        task, case, label = sc["task"], "none", f"random seed {args.seed}"   # the box comes from the scenario
        setup = lambda robot: apply_world(robot, sc["world"])              # noqa: E731  (sim side only)
    if real:
        from src.robot.real_robot import RealRobot
        robot = RealRobot(args.real or None)
        label = f"real robot {robot.host}"
    else:
        from src.robot.task import make_robot
        robot = make_robot(case, seed=args.seed, path=task_path)   # the task file's scene, box, LiDAR
    # the camera file decides the range model: the real one corrects its own bias (no sim range_bias_r2 on top)
    estimator = None if args.odometry else EKF(task["start"], marker_map=task["markers"],
                                               params=load_estimation_config(camera=intrinsics))
    from src.robot.task import apply_control_overrides
    if real and "control" in robot.cfg:
        # The real arena's own tuning on top of the task file's (a narrower lane than the design). Merged into the
        # task itself, because run_mission applies the task's control block again.
        task["control"] = apply_control_overrides(json.loads(json.dumps(task.get("control", {}))), {"control": robot.cfg["control"]})
        shown = {k: v for k, v in robot.cfg["control"].items() if not k.startswith("_")}
        print(f"note: real-robot control overrides {json.dumps(shown)} (config/robot_real.json)")
    cfg = apply_control_overrides(load_control_config(), task)   # a task file may tune the controller
    if args.dock:
        cfg["docking"]["servo"] = args.dock
    cfg["docking"]["camera_intrinsics"] = str(intrinsics.relative_to(ROOT))   # IBVS: the same K and T_BC as ArUco
    if real and "min_coverage" in robot.cfg:
        # The lab room is open: whole sectors have nothing within the LDS-01's 3.5 m and read 0, so the robot
        # needs its own coverage threshold (config/robot_real.json); the sim keeps control.json's.
        cfg["lidar_obs"]["min_coverage"] = robot.cfg["min_coverage"]
        print(f"note: LiDAR min_coverage {robot.cfg['min_coverage']} on the real robot (config/robot_real.json)")
    if real:
        # RealRobot scales any command above its caps down (v and w together, so the path keeps its shape).
        # Say so: the gains were designed at v_nom, and a sim twin for comparison must use the same speeds.
        fast = {k: s for k, s in (("v_nom", cfg["v_nom"]), ("avoidance.v_avoid", cfg["avoidance"]["v_avoid"]),
                                  ("docking.v_dock", cfg["docking"]["v_dock"])) if s > robot.v_max}
        if fast:
            print(f"note: {', '.join(f'{k} = {s}' for k, s in fast.items())} m/s exceed the robot cap "
                  f"v_max = {robot.v_max} m/s (config/robot_real.json): those commands are scaled down")
    cameras = []
    if not args.headless and not args.no_camera:
        from src.perception import aruco
        K, dist, _ = camera.load_intrinsics(intrinsics)
        cameras.append(camera_window(K, dist, sorted(task["markers"]), task["dock"]["marker_id"],
                                     T_BC=aruco.load_T_BC(intrinsics),
                                     range_correction=aruco.load_range_correction(intrinsics)))
    log_path = args.log or (ROOT / "results" / "hardware" / time.strftime("run_%Y%m%d_%H%M%S.npz") if real else None)
    log_step, save_log = run_logger(estimator)
    hooks = cameras + ([log_step] if log_path else []) + ([start_check(estimator)] if estimator is not None else [])

    def on_step(robot, row):
        for hook in hooks:
            hook(robot, row)

    print(f"{task_path.name}: {label}, steering on {'odometry' if args.odometry else 'the EKF'}, docking {cfg['docking']['servo']}, "
          f"seed {args.seed}{', headless' if args.headless else ''}")
    info, t_end, failure = None, 0.0, None
    try:
        info, t_end = run_mission(case, seed=args.seed, verbose=True, estimator=estimator,
                                  view=not args.headless and not real, on_step=on_step if hooks else None,
                                  setup=setup, task=task, label=label, cfg=cfg, robot=robot, intrinsics=intrinsics)
    except (ConnectionError, TimeoutError, RuntimeError) as e:
        if not real:
            raise
        failure = f"{type(e).__name__}: {e}"               # the robot was already stopped by RealRobot.close()
    finally:
        if cameras:
            import cv2
            cv2.destroyAllWindows()
        if log_path:
            save_log(Path(log_path), task=str(task_path.resolve().relative_to(ROOT)),
                     camera=str(intrinsics.relative_to(ROOT)),
                     result=str(failure or (info["stop_reason"] if info else "interrupted")),
                     overruns=getattr(robot, "overruns", -1))   # real robot: control steps that started late

    if failure:
        print(f"result: stopped, {failure}")
        return 1
    dock = info.get("dock") if info else None
    if dock and dock["checks"]:                                  # the last docking checks, so a failed dock says why
        print(f"docking checks (phase {dock['phase']}):")
        for k, ok in dock["checks"].items():
            print(f"  {'ok  ' if ok else 'FAIL'} {k:9s} {dock['values'].get(k, '')}")
    reason = info["stop_reason"] or "stopped before the end (window closed or time limit)"
    print(f"result: {reason} after {t_end:.1f} s{'' if real else ' of sim time'}, "
          f"dock_complete = {info['dock_complete']}")
    return 0 if info["dock_complete"] else 1


if __name__ == "__main__":
    sys.exit(main())