"""The navigation "boss": every control step it decides WHICH controller drives, and returns (v, w, info).

It does not steer by itself. The states and who is in charge:

    TRACK  waypoints.py follows start -> W1 -> W2 -> W3 -> W4 (LQR + turn on the spot at corners)
    AVOID  avoidance.py shifts the path around the box and rejoins (waypoints paused meanwhile)
    DOCK   docking.py drives from W4 onto the charger (marker ID 0) and confirms the dock; its last part
           is image-based visual servoing (shown as IBVS) when config docking.servo = "ibvs"
    STOP   v = w = 0 for good; stop_reason says why ("docked", "no_passage", "marker_lost", ...)

Transitions:
    TRACK -> AVOID   the LiDAR sees the box in the robot's lane before the next waypoint
    AVOID -> TRACK   avoidance finished rejoining (waypoints resume where they were)
    TRACK -> DOCK    the pre-dock point W4 is reached
    DOCK  -> STOP    dock_complete ("docked") or docking failed (its stop_reason)
    AVOID -> STOP    no side of the box has room for the robot ("no_passage")

Safety rules that apply everywhere:
    - While driving on the path (TRACK / AVOID), a LiDAR scan that is too old or not valid means
      "wait" (v = w = 0): missing returns do not prove the way is clear.
    - In DOCK the box-ahead trigger is off (the charger is SUPPOSED to be in front); docking stops on
      the camera range instead, and holds when the marker is lost.
    - The final command is always clipped to the Waffle Pi limits (0.26 m/s, 1.82 rad/s).

Interface (team plan):  v, w, info = navigation.step(xh, P, obs, marker_frame, dt, motion_feedback)
    xh, P            EKF estimate and covariance (odometry + a fixed P until the EKF is ready)
    obs              LiDAR obs dict (lidar_obs.py now, Perception's version later)
    marker_frame     {"t": t_frame, "markers": {id: {"z": [range, bearing], ...}}} or None
    dt               control period [s]
    motion_feedback  {"v", "w", "t"}: speeds from the encoders and the current time [s]
Ground truth is never used here.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # project root, so "src" imports work even
                                                                # when this file is run directly (VS Code Run)

from src.control.avoidance import DONE as AVOID_DONE, STOP as AVOID_STOP, AvoidanceController  # noqa: E402
from src.control.docking import COMPLETE, FAILED, DockingController  # noqa: E402
from src.control.models import TS, load_control_config  # noqa: E402
from src.control.waypoints import WaypointTracker  # noqa: E402
from src.robot.sim_robot import MAX_V, MAX_W  # noqa: E402

TRACK, AVOID, DOCK, STOP = "TRACK", "AVOID", "DOCK", "STOP"


class NavigationController:
    """TRACK / AVOID / DOCK / STOP on top of waypoints.py, avoidance.py and docking.py."""

    def __init__(self, task, cfg=None):
        cfg = cfg or load_control_config()
        self.tracker = WaypointTracker.from_task(task, cfg=cfg)
        self.avoid = AvoidanceController(cfg=cfg)
        self.dock = DockingController.from_task(task, cfg=cfg)
        self.max_scan_age = cfg["avoidance"]["max_scan_age"]
        self.state, self.stop_reason = TRACK, None

    def _scan_ok(self, obs, t):
        """A LiDAR obs we may move on: present, valid coverage, and recent."""
        return obs is not None and obs["valid"] and t - obs["t"] <= self.max_scan_age

    def _stop(self, reason):
        self.state, self.stop_reason = STOP, reason

    def step(self, xh, P, obs, marker_frame, dt, motion_feedback):
        """One control step: final (v, w) and info."""
        t = motion_feedback["t"]
        v = w = 0.0

        # ---- TRACK: follow the waypoints, hand over to AVOID or DOCK when needed
        if self.state == TRACK:
            if self.tracker.pre_dock_reached:
                self.dock.start()
                self.state = DOCK
            elif not self._scan_ok(obs, t):
                pass                                            # no trustworthy scan: wait (v = w = 0)
            else:
                seg = self.tracker.segment
                remaining = seg.length - seg.errors(xh)[2]
                if self.tracker.mode == "TRACK" and self.avoid.blocked(obs, remaining):
                    self.tracker.pause()                        # keep the waypoint progress
                    if self.avoid.start(obs, xh, seg):
                        self.state = AVOID
                    else:
                        self._stop(self.avoid.stop_reason)      # no side has room
                else:
                    v, w = self.tracker.step(xh)

        # ---- AVOID: go around the box, then give control back to the waypoints
        if self.state == AVOID:
            if not self._scan_ok(obs, t):
                v = w = 0.0                                     # wait for a good scan
            else:
                v, w = self.avoid.step(obs, xh, t - obs["t"])
                if self.avoid.phase == AVOID_DONE:
                    self.tracker.resume(xh)                     # turns first if facing away
                    self.state = TRACK
                    v, w = self.tracker.step(xh)
                elif self.avoid.phase == AVOID_STOP:
                    self._stop(self.avoid.stop_reason)

        # ---- DOCK: drive onto the charger; finished or failed means STOP
        if self.state == DOCK:
            v, w = self.dock.step(xh, P, marker_frame, motion_feedback, t)
            if self.dock.phase == COMPLETE:
                self._stop("docked")
            elif self.dock.phase == FAILED:
                self._stop(self.dock.stop_reason)

        if self.state == STOP:
            v = w = 0.0

        # ---- final safety: never exceed the Waffle Pi limits, whatever a controller asked for
        v = float(np.clip(v, -MAX_V, MAX_V))
        w = float(np.clip(w, -MAX_W, MAX_W))
        return v, w, self.info()

    def info(self):
        """The team's info fields, for main.py's log and run_demo.py."""
        state = self.state
        if state == AVOID:
            state = self.avoid.phase                                # AVOID or REJOIN
        elif state == DOCK and self.dock.phase == "IBVS":
            state = "IBVS"                                          # docking's visual-servoing phase
        return dict(state=state,
                    waypoint_index=self.tracker.i,
                    pre_dock_reached=self.tracker.pre_dock_reached,
                    dock_complete=self.dock.dock_complete,
                    stop_reason=self.stop_reason)


# ---------------------------------------------------------------- the whole mission in the sim
class DeadReckoning:
    """STAND-IN estimator with the EKF's interface (team plan): xh, P = est.step(encoders, marker_frame).

    It only integrates the wheel encoders and returns a fixed P. To drive on the EKF instead, pass
    run_mission(..., estimator=EKF(x0, P0, marker_map, params)): nothing else changes.
    """

    def __init__(self, x0):
        self.xh = np.asarray(x0, float).copy()
        self.P = np.diag([0.02**2, 0.02**2, np.deg2rad(2.0)**2])
        self.enc_prev = None

    def step(self, encoders, marker_frame=None):
        from src.control.waypoints import _dead_reckoning
        _, phi_l, phi_r = encoders
        if self.enc_prev is not None:                         # the first sample only sets the baseline
            self.xh = _dead_reckoning(self.xh, (phi_l, phi_r), self.enc_prev)
        self.enc_prev = (phi_l, phi_r)
        return self.xh.copy(), self.P


def _draw_overlay(viewer, obs, xh):
    """Display only: LiDAR returns as small red dots in the viewer, placed with the ESTIMATED pose xh
    (so you also see how good the estimate is: if xh drifts, the dots slide off the walls)."""
    import mujoco
    c, s = np.cos(xh[2]), np.sin(xh[2])
    P_W = np.array([[c, -s], [s, c]]) @ obs["points_B"] + np.asarray(xh[:2]).reshape(2, 1)
    scn = viewer.user_scn
    scn.ngeom = 0
    for x, y in P_W.T[: scn.maxgeom]:
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.015, 0, 0]),
                            np.array([x, y, 0.132]), np.eye(3).ravel(), np.array([1, 0, 0, 1], np.float32))
        scn.ngeom += 1


def _panel(label, info, v, w, P, est):
    """Top-left text: state, commands and the EKF's own uncertainty (sd from P, last NIS)."""
    sd = np.sqrt(np.maximum(np.diag(np.asarray(P, float)), 0.0))
    log = getattr(est, "log", None)                            # the EKF logs (t, id, NIS, used) per marker
    nis = f"{log[-1][2]:.2f} (ID {log[-1][1]}{'' if log[-1][3] else ', gated'})" if log else "-"
    left = "run\nstate\nwaypoint\nv, w\nsd x, y\nsd theta\nlast NIS"
    right = (f"{label}\n{info['state']}\n{info['waypoint_index']}\n{v:.2f} m/s, {w:+.2f} rad/s\n"
             f"{100 * sd[0]:.1f}, {100 * sd[1]:.1f} cm\n{np.degrees(sd[2]):.2f} deg\n{nis}")
    return left, right


def run_mission(case="centre", seed=0, t_max=180.0, verbose=True, estimator=None, view=False, on_step=None,
                setup=None, task=None, label=None, cfg=None, robot=None, intrinsics=None):
    """Start -> (avoid) -> W4 -> dock, in the corridor with the given obstacle case.

    Everything the robot uses comes from its sensors. marker_frame, obs and motion_feedback built here
    are STAND-INS for the perception glue; the estimator defaults to DeadReckoning (odometry only).
    Pass estimator=<the team's EKF> to drive on x_hat from the EKF. Returns (final info, time taken).
    view=True opens the MuJoCo viewer and runs in real time: red dots = LiDAR, top-left = state.
    on_step(robot, row) is called after every step with this step's log row (team log contract); it is
    how evaluation/run_demo.py records the truth next to the estimate without src/ ever reading it.
    setup(robot) is called once before the loop: evaluation/ uses it for repeated trials (a random
    true start pose, encoder scale errors). The controller still starts from the config prior.
    task: what the robot is told (start prior, waypoints, marker map, dock); default load_task(). A
    randomised scenario (src/robot/scenario.py) passes its own task and a setup that changes the world.
    label: the run's name on the viewer panel (default: the case).
    cfg: controller config (default config/control.json), e.g. with docking servo "ibvs" or "pbvs".
    robot: the robot to drive, default the corridor SimRobot for the case; main.py --real passes a
    RealRobot (src/robot/real_robot.py), which the with block connects and stops. view needs a SimRobot.
    intrinsics: camera calibration for the ArUco measurements (default config/camera_sim.json); its
    optional "T_BC" (the real camera's tilt, experiments/camera_pitch.py) replaces the URDF camera pose,
    and its optional "range_correction" (the real camera's range bias, experiments/aruco_real_noise.py)
    is removed from every ArUco range, so the EKF and docking both get corrected ranges. When given, the
    IBVS docking uses the same file (cfg docking.camera_intrinsics is overridden).
    """
    import contextlib
    import copy
    import time
    from src.control.lidar_obs import make_obs
    from src.perception import aruco, camera
    from src.robot.corridor import load_task, make_robot
    from src.robot.sim_robot import PROJECT_ROOT, WHEEL_RADIUS, WHEEL_SEPARATION

    task = task if task is not None else load_task()
    label = label or case
    if intrinsics is not None:                                # one camera file for ArUco AND the IBVS docking
        cfg = copy.deepcopy(cfg or load_control_config())
        cfg["docking"]["camera_intrinsics"] = str(Path(intrinsics).resolve())
    nav = NavigationController(task, cfg=cfg)
    intrinsics = intrinsics or PROJECT_ROOT / "config" / "camera_sim.json"
    K_cam, dist, _ = camera.load_intrinsics(intrinsics)
    T_BC = aruco.load_T_BC(intrinsics)                        # real camera: with its tilt
    range_corr = aruco.load_range_correction(intrinsics)      # real camera: its range bias; sim: None
    if range_corr is not None and getattr(estimator, "p", {}).get("range_bias_r2", 0.0):
        raise ValueError("the camera file corrects its own range bias, but the EKF would subtract the sim's "
                         "range_bias_r2 too: build it with params=load_estimation_config(camera=<that file>)")
    est = estimator or DeadReckoning(task["start"])           # start prior from the config, not the truth
    marker_ids = [0] if estimator is None else sorted(task["markers"])   # the EKF wants every known marker
    with (robot if robot is not None else make_robot(case, seed=seed)) as robot, contextlib.ExitStack() as stack:
        if setup is not None:
            setup(robot)                                      # sim-side changes for a trial (evaluation/)
        viewer = None
        if view:
            import mujoco
            import mujoco.viewer
            robot.read_image()                                # create the camera renderer before the viewer
            viewer = stack.enter_context(mujoco.viewer.launch_passive(robot.m, robot.d))
            with viewer.lock():                               # top view over the corridor
                viewer.cam.lookat[:] = [2.3, 2.1, 0.0]
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 7.5, 90.0, -65.0
        t_enc, *enc_prev = robot.read_encoders()
        frame, t_last_frame, state = None, None, None
        while nav.state != STOP and robot.t < t_max:
            # sensors -> stand-in perception
            obs = make_obs(*robot.read_scan(), robot.lidar.range_min, robot.lidar.range_max)
            if nav.state == DOCK or estimator is not None:   # odometry alone needs the camera only to dock
                t_frame, rgb = robot.read_image()
                if t_frame != t_last_frame:                   # run ArUco once per NEW frame
                    markers = {}
                    for mid in marker_ids:
                        m = aruco.measure(rgb, K_cam, dist, T_BC=T_BC, marker_id=mid, range_correction=range_corr)
                        if m is not None:                     # Rn is the EKF's own (R_marker, estimation.json)
                            markers[mid] = {"z": m["z"], "yaw": m["yaw"], "ambiguous": m["ambiguous"],
                                            "corners": m["corners"], "T_CM": m["T_CM"]}   # IBVS needs these
                    frame = {"t": t_frame, "markers": markers}
                    t_last_frame = t_frame
            encoders = robot.read_encoders()
            t_now, *enc = encoders
            dl, dr = (np.array(enc) - np.array(enc_prev)) * WHEEL_RADIUS
            feedback = {"v": (dl + dr) / 2 / max(t_now - t_enc, 1e-9),
                        "w": (dr - dl) / WHEEL_SEPARATION / max(t_now - t_enc, 1e-9), "t": t_now}
            xh, P = est.step(encoders, frame)                 # odometry now, the EKF later
            t_enc, enc_prev = t_now, enc

            v, w, info = nav.step(xh, P, obs, frame, TS, feedback)
            if verbose and info["state"] != state:
                state = info["state"]
                print(f"t = {robot.t:5.1f} s  {state:7s} waypoint {info['waypoint_index']}  "
                      f"xh = ({xh[0]:.2f}, {xh[1]:.2f}, {np.rad2deg(xh[2]):4.0f} deg)")
            robot.send_cmd(v, w)
            if on_step is not None:
                on_step(robot, dict(t=robot.t, xh=xh.copy(), P=np.array(P), v=v, w=w, state=info["state"],
                                    waypoint_index=info["waypoint_index"]))
            t_wall = time.time()
            robot.step(TS)
            if viewer is not None:
                if not viewer.is_running():
                    break                                     # window closed: stop the run
                _draw_overlay(viewer, obs, xh)
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                  *_panel(label, info, v, w, P, est)))
                viewer.sync()
                time.sleep(max(0.0, TS - (time.time() - t_wall)))   # real time
        if viewer is not None and viewer.is_running():
            viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                              "run\nresult\n", f"{label}\n{nav.stop_reason}\nclose the window to exit"))
            while viewer.is_running():                        # keep the final pose on screen
                viewer.sync()
                time.sleep(0.05)
        return nav.info(), robot.t


if __name__ == "__main__":
    # Run from the project root:  python -m src.control.behaviour [none|left|centre|right|all] [--view]
    #   --view  watch it in the MuJoCo viewer in real time (one window per case; close it for the next)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    arg = args[0] if args else "centre"
    for case in (["none", "left", "centre", "right"] if arg == "all" else [arg]):
        print(f"\n=== obstacle: {case}")
        info, t_end = run_mission(case, view="--view" in sys.argv)
        print(f"result: {info['stop_reason']} after {t_end:.1f} s, dock_complete = {info['dock_complete']}")