"""Docking: from the pre-dock point W4 onto the charger marked with ArUco ID 0, then stop and confirm.

Kept deliberately small. Three ideas:

1. STEER along the approach line through goal_pose (corridor.json) with the same LQR as the rest of the
   path, redesigned for the slower docking speed. The robot POSITION comes from the camera and the
   HEADING from x_hat (_dock_pose): odometry can drift a few cm, the ID 0 range is good to < 1 mm.
2. STOP on what the camera SEES: when the ID 0 range reaches stop_range_base = 0.356 m (0.28 m from the
   camera, which still sees ID 0 reliably; checked in the sim from 0.16 to 1.36 m, error < 1 mm).
3. CONFIRM (team plan "Docking completion"): dock_complete when, for dock_hold_sec without a break,
   - range and bearing to ID 0 are within tolerance          (what the camera sees)
   - lateral and heading error to goal_pose are within tolerance   (lateral from _dock_pose, heading
     from x_hat; bearing ~ 0 alone cannot tell "sideways but pointing at the marker" from docked)
   - the position and heading sd from P are small           (only trust a confident estimate)
   - the encoders say the robot has really stopped
   Marker lost -> stop and wait (never drive in blind); still lost after marker_lost_timeout -> FAILED.

Phases:  APPROACH -> SETTLE -> COMPLETE,  APPROACH <-> WAIT (marker lost),  WAIT / SETTLE -> FAILED

Inputs (team interface): x_hat from the EKF (odometry until it is ready); marker_frame from perception,
{"t": t_frame, "markers": {id: {"z": [range, bearing], ...}}}; motion_feedback {v, w, t} = speeds
measured by the encoders. Nothing here reads the simulator's ground truth.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # project root, so "src" imports work even
                                                                # when this file is run directly (VS Code Run)

from src.control.lqr import PathLQR, design_lqr  # noqa: E402
from src.control.models import TS, load_control_config  # noqa: E402
from src.control.waypoints import Segment  # noqa: E402

APPROACH, WAIT, SETTLE, COMPLETE, FAILED = "APPROACH", "WAIT", "SETTLE", "COMPLETE", "FAILED"


class DockingController:
    """Call start() when waypoints.py reports pre_dock_reached, then step() every Ts."""

    def __init__(self, dock, marker_xy, cfg=None):
        cfg = cfg or load_control_config()
        self.p, self.Ts = cfg["docking"], cfg["Ts"]
        self.a_max = cfg["waypoints"]["a_max"]               # same acceleration limit as normal driving
        self.marker_id = dock["marker_id"]
        self.stop_range = dock["stop_range_base"]
        self.marker_xy = np.asarray(marker_xy, float)[:2]   # known map position of ID 0 (corridor.json)
        gx, gy, gth = dock["goal_pose"]
        # Approach line: from 1 m behind the goal up to the goal, along the goal heading.
        self.line = Segment(np.array([gx, gy]) - np.array([np.cos(gth), np.sin(gth)]), np.array([gx, gy]))
        # The cruise gain was designed at 0.20 m/s; docking is slower, so K is redesigned at v_dock on the
        # same model (the first step of gain scheduling, see lqr.py).
        self.lqr = PathLQR(K=design_lqr(v=self.p["v_dock"], cfg=cfg)[0], integral=False, cfg=cfg)
        self.phase, self.stop_reason = None, None

    @classmethod
    def from_task(cls, task, **kw):
        """Build from load_task(): marker id, its map position, stop range and goal pose."""
        return cls(task["dock"], task["markers"][task["dock"]["marker_id"]], **kw)

    def start(self):
        """Hand-over from waypoint tracking (pre_dock_reached)."""
        self.phase, self.stop_reason = APPROACH, None
        self.v_cmd = 0.0
        self.marker, self.t_marker = None, -np.inf          # last ID 0 measurement and its frame time
        self.t_lost = self.t_settle = self.t_ok = None
        self.lqr.reset()

    @property
    def dock_complete(self):
        return self.phase == COMPLETE

    def _read_marker(self, marker_frame):
        """Keep ID 0 from the frame if the frame is NEW (each image is used once) and the value is valid."""
        if marker_frame is not None and marker_frame["t"] > self.t_marker:
            m = marker_frame["markers"].get(self.marker_id)
            if m is not None and np.all(np.isfinite(m["z"])):
                self.marker, self.t_marker = m, marker_frame["t"]

    def _dock_pose(self, x_hat):
        """Robot POSITION from the camera, HEADING from x_hat.

        With the marker's map position m and the measurement (range r, bearing b):
            position = m - r [cos(theta + b), sin(theta + b)],   theta = x_hat heading
        A 1 deg heading error moves this by only r * 1 deg ~ 6 mm at the stop range.
        (Once the EKF fuses ID 0, x_hat and this agree anyway.)
        """
        th = x_hat[2]
        r, b = self.marker["z"]
        return np.array([self.marker_xy[0] - r * np.cos(th + b), self.marker_xy[1] - r * np.sin(th + b), th])

    def step(self, x_hat, P, marker_frame, motion_feedback, t):
        """(v, omega) for this step. t = current time [s] (same clock as the camera timestamps).
        P = the estimate's covariance, used in the completion check."""
        if self.phase not in (APPROACH, WAIT, SETTLE):
            return 0.0, 0.0
        self._read_marker(marker_frame)
        fresh = t - self.t_marker <= self.p["marker_max_age"]

        # 1. Marker lost: stop and wait for it (never drive in blind); give up after marker_lost_timeout.
        if not fresh:
            self.t_lost = t if self.t_lost is None else self.t_lost
            if t - self.t_lost > self.p["marker_lost_timeout"]:
                self.phase, self.stop_reason = FAILED, "marker_lost"
            elif self.phase == APPROACH:
                self.phase = WAIT
            self.v_cmd = 0.0
            return 0.0, 0.0
        self.t_lost = None
        rng, brg = self.marker["z"]

        # 2. APPROACH: steer along the line, slow down as the SEEN range closes in, stop when it is reached.
        if self.phase in (APPROACH, WAIT):
            self.phase = APPROACH
            to_go = rng - self.stop_range                       # metres left, from the camera
            if to_go > self.p["stop_early"]:
                v = float(np.clip(self.p["v_dock"] * to_go / self.p["slow_down_dist"],
                                  self.p["v_min_dock"], self.p["v_dock"]))
                self.v_cmd = min(v, self.v_cmd + self.a_max * self.Ts)   # gentle start (acceleration limit)
                e_y, e_theta, _ = self.line.errors(self._dock_pose(x_hat))
                return self.v_cmd, self.lqr.step(e_y, e_theta, 0.0, self.v_cmd)
            self.phase, self.t_settle = SETTLE, t               # range reached: stop at once
            self.v_cmd = 0.0

        # 3. SETTLE: zero command; complete when every check holds for dock_hold_sec without a break.
        e_y, e_theta, _ = self.line.errors(np.array([*self._dock_pose(x_hat)[:2], x_hat[2]]))
        P = np.asarray(P, float)
        self.checks = dict(
            range=abs(rng - self.stop_range) <= self.p["dock_range_tol"],
            bearing=abs(brg) <= np.deg2rad(self.p["dock_bearing_tol_deg"]),
            lateral=abs(e_y) <= self.p["dock_lateral_tol"],
            heading=abs(e_theta) <= np.deg2rad(self.p["dock_heading_tol_deg"]),   # e_theta is wrapped
            confident=(np.sqrt(P[0, 0] + P[1, 1]) <= self.p["dock_pos_sigma_max"]
                       and np.sqrt(P[2, 2]) <= np.deg2rad(self.p["dock_heading_sigma_max_deg"])),
            stopped=(abs(motion_feedback["v"]) <= self.p["dock_linear_speed_tol"]
                     and abs(motion_feedback["w"]) <= self.p["dock_angular_speed_tol"]),
        )
        if all(self.checks.values()):
            self.t_ok = t if self.t_ok is None else self.t_ok
            if t - self.t_ok >= self.p["dock_hold_sec"]:      # held long enough: not one lucky frame
                self.phase = COMPLETE
        else:
            self.t_ok = None
        if self.phase == SETTLE and t - self.t_settle > self.p["settle_timeout"]:
            failing = [k for k, ok in self.checks.items() if not ok]
            self.phase, self.stop_reason = FAILED, "not_docked: " + ", ".join(failing)
        return 0.0, 0.0

    def info(self):
        """For behaviour.py / logging."""
        return dict(phase=self.phase, dock_complete=self.dock_complete, stop_reason=self.stop_reason)


# ---------------------------------------------------------------- try it on the robot
if __name__ == "__main__":
    # Run from the project root:  python -m src.control.docking      (or the VS Code Run button)
    # Starts at W4 (the pre-dock point, the prior from the config) and docks on dead reckoning plus the
    # camera. The marker_frame, P and motion_feedback built here are STAND-INS for the perception glue
    # and the EKF, which other teammates own; main.py will pass the real ones.
    from src.control.waypoints import _dead_reckoning
    from src.perception import aruco, camera
    from src.robot.corridor import load_task, make_robot
    from src.robot.sim_robot import PROJECT_ROOT, WHEEL_RADIUS, WHEEL_SEPARATION

    task = load_task()
    K_cam, dist, _ = camera.load_intrinsics(PROJECT_ROOT / "config" / "camera_sim.json")
    dock = DockingController.from_task(task)
    w4 = task["waypoints"][-1]
    with make_robot("none", seed=0) as robot:
        robot.reset(w4[0], w4[1], np.pi / 2)            # sim setup: put the robot at the pre-dock point
        pose = np.array([w4[0], w4[1], np.pi / 2])      # odometry starts from the same prior
        P = np.diag([0.02**2, 0.02**2, np.deg2rad(2.0)**2])   # stand-in covariance until the EKF exists
        t_enc, *enc_prev = robot.read_encoders()
        dock.start()
        phase = None
        while dock.phase not in (COMPLETE, FAILED) and robot.t < 60:
            # stand-in perception: ID 0 range-bearing from the newest camera frame
            t_frame, rgb = robot.read_image()
            m = aruco.measure(rgb, K_cam, dist, marker_id=dock.marker_id)
            frame = {"t": t_frame, "markers": {} if m is None else {dock.marker_id: {
                "z": m["z"], "R": np.diag([0.005**2, np.deg2rad(1.0)**2]), "yaw": m["yaw"], "ambiguous": m["ambiguous"]}}}
            # stand-in motion feedback: wheel speeds from the encoder increments
            t_now, *enc = robot.read_encoders()
            dl, dr = (np.array(enc) - np.array(enc_prev)) * WHEEL_RADIUS
            dt = max(t_now - t_enc, 1e-9)
            feedback = {"v": (dl + dr) / 2 / dt, "w": (dr - dl) / WHEEL_SEPARATION / dt, "t": t_now}
            pose = _dead_reckoning(pose, enc, enc_prev)
            t_enc, enc_prev = t_now, enc

            v, w = dock.step(pose, P, frame, feedback, robot.t)
            if dock.phase != phase:
                phase = dock.phase
                rng = f"{m['z'][0]:.3f} m" if m is not None else "not seen"
                print(f"t = {robot.t:5.2f} s  {phase:8s} marker range {rng}")
            robot.send_cmd(v, w)
            robot.step(TS)
        print(f"\ndock_complete = {dock.dock_complete}, stop_reason = {dock.stop_reason}")
        print("checks:", {k: bool(ok) for k, ok in getattr(dock, "checks", {}).items()})
        e_y, e_theta, _ = dock.line.errors(pose)
        print(f"odometry vs goal: lateral {e_y * 100:+.1f} cm, heading {np.rad2deg(e_theta):+.1f} deg; "
              f"last marker range {dock.marker['z'][0]:.3f} m (stop range {dock.stop_range} m)")
