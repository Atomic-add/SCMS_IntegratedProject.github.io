"""Docking: from the pre-dock point (the last waypoint) onto the charger marked with ArUco ID 0, then stop and confirm.

Kept deliberately small. Four ideas:

1. STEER along the approach line through goal_pose (the task file) with the same LQR as the rest of the
   path, redesigned for the slower docking speed. The robot POSITION comes from the camera and the
   HEADING from x_hat (_dock_pose): odometry can drift a few cm, the ID 0 range is good to < 1 mm.
   This brings the robot onto the marker's normal, which servoing alone cannot do (Week 9 D3/D4).
2. SERVO the last part on the image (servo = "ibvs", Week 9): once the ID 0 range is below
   ibvs_switch_range, (v, omega) = lam * J_mr^+ (p* - p) on the four marker corners, where p* is where the
   corners appear when docked and J_mr is the image Jacobian reduced to what the base can do
   (src/control/visual_servo.py). Saturated at v_dock and ibvs_w_max (Week 9 B3). It ends when the
   feature error is below ibvs_e_tol_px, or the commands have died out (the residual the base cannot
   remove), or the stop range is reached. servo = "pbvs" keeps idea 1 all the way to the stop.
3. STOP on what the camera SEES: never closer than stop_range_base = 0.356 m (0.28 m from the camera,
   which still sees ID 0 reliably; checked in the sim from 0.16 to 1.36 m, error < 1 mm).
4. CONFIRM (team plan "Docking completion"): dock_complete when, for dock_hold_sec without a break,
   - range and bearing to ID 0 are within tolerance          (what the camera sees)
   - lateral and heading error to goal_pose are within tolerance   (lateral from _dock_pose, heading
     from x_hat; bearing ~ 0 alone cannot tell "sideways but pointing at the marker" from docked)
   - the position and heading sd from P are small           (only trust a confident estimate)
   - the encoders say the robot has really stopped
   Marker lost -> stop and wait (never drive in blind); still lost after marker_lost_timeout -> FAILED.

Phases:  APPROACH -> (IBVS ->) SETTLE -> COMPLETE,  APPROACH / IBVS <-> WAIT (marker lost),
         WAIT / SETTLE -> FAILED

Inputs (team interface): x_hat from the EKF (odometry until it is ready); marker_frame from perception,
{"t": t_frame, "markers": {id: {"z": [range, bearing], "corners": 4 x 2 px, "T_CM": SE3, ...}}};
motion_feedback {v, w, t} = speeds measured by the encoders. IBVS needs "corners" and "T_CM" (the PnP
pose, for the corner depths); a frame without them keeps the LQR approach. Nothing here reads the
simulator's ground truth.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # project root, so "src" imports work even
                                                                # when this file is run directly (VS Code Run)

from src.control import visual_servo as vs  # noqa: E402
from src.control.lqr import PathLQR, design_lqr  # noqa: E402
from src.control.models import TS, load_control_config  # noqa: E402
from src.control.waypoints import Segment  # noqa: E402
from src.robot.sim_robot import PROJECT_ROOT  # noqa: E402

APPROACH, IBVS, WAIT, SETTLE, COMPLETE, FAILED = "APPROACH", "IBVS", "WAIT", "SETTLE", "COMPLETE", "FAILED"


class DockingController:
    """Call start() when waypoints.py reports pre_dock_reached, then step() every Ts."""

    def __init__(self, dock, marker, cfg=None, K=None, dist=None, T_BC=None):
        """K, dist, T_BC: the camera model for IBVS. Left out, they come from the calibration file
        cfg["docking"]["camera_intrinsics"], with its range_correction (run_mission points that key at the
        file its ArUco uses), so K, the camera tilt and the range bias can never come from two cameras."""
        cfg = cfg or load_control_config()
        self.p, self.Ts = cfg["docking"], cfg["Ts"]
        self.a_max = cfg["waypoints"]["a_max"]               # same acceleration limit as normal driving
        self.marker_id = dock["marker_id"]
        self.stop_range = dock["stop_range_base"]
        marker = np.asarray(marker, float)
        self.marker_xy = marker[:2]                          # known map position of ID 0 (the task file)
        gx, gy, gth = dock["goal_pose"]
        # Approach line: from 1 m behind the goal up to the goal, along the goal heading.
        self.line = Segment(np.array([gx, gy]) - np.array([np.cos(gth), np.sin(gth)]), np.array([gx, gy]))
        # The cruise gain was designed at 0.20 m/s; docking is slower, so K is redesigned at v_dock on the
        # same model (the first step of gain scheduling, see lqr.py).
        self.lqr = PathLQR(K=design_lqr(v=self.p["v_dock"], cfg=cfg)[0], integral=False, cfg=cfg)
        # IBVS (Week 9): calibrated intrinsics, p* from the docked geometry, (v, omega) -> camera twist.
        self.servo = self.p.get("servo", "pbvs")
        if self.servo == "ibvs":
            cam_file = PROJECT_ROOT / self.p["camera_intrinsics"]
            if K is None:
                from src.perception import camera
                K, dist, _ = camera.load_intrinsics(cam_file)
            if T_BC is None:
                T_BC = vs.aruco.load_T_BC(cam_file)          # the file's tilt (real camera), else the URDF pose
            p_star_range = self.stop_range
            corr = vs.aruco.load_range_correction(cam_file)
            if corr is not None:       # real camera: the ranges here are corrected, but the corners are raw, so
                # p* is drawn at the range PnP reads at the true stop range (else IBVS aims short of the stop)
                p_star_range = corr["bias_m"] + (1 + corr["bias_per_m"]) * self.stop_range
            self.K, self.dist = np.asarray(K, float), np.asarray(dist, float)
            marker_height = marker[2] if marker.size > 2 else 0.12
            self.p_star = vs.desired_features(self.K, p_star_range, marker_height, T_BC=T_BC)
            self.G = vs.robot_to_camera(T_BC)
        self.phase, self.stop_reason = None, None
        self.ibvs_log = []                                   # (t, corners 2x4, e_rms px, v, w) per IBVS step

    @classmethod
    def from_task(cls, task, **kw):
        """Build from load_task(): marker id, its map position and height, stop range and goal pose."""
        return cls(task["dock"], task["markers"][task["dock"]["marker_id"]], **kw)

    def start(self):
        """Hand-over from waypoint tracking (pre_dock_reached)."""
        self.phase, self.stop_reason = APPROACH, None
        self.resume = APPROACH                              # phase to go back to after a WAIT
        self.v_cmd = 0.0
        self.marker, self.t_marker = None, -np.inf          # last ID 0 measurement and its frame time
        self.t_lost = self.t_settle = self.t_ok = self.t_still = None
        self.ibvs_log = []
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

    def _ibvs_ready(self, rng):
        """Switch to IBVS: servo = ibvs, close enough, and the frame carries corners and the PnP pose."""
        return (self.servo == "ibvs" and rng <= self.p["ibvs_switch_range"]
                and "corners" in self.marker and "T_CM" in self.marker)

    def _ibvs_step(self, rng, t):
        """One IBVS command (Week 9 B1b / D4 with our camera), or None when IBVS has finished."""
        if "corners" not in self.marker or "T_CM" not in self.marker:
            self.phase = APPROACH                              # no corners in this frame: back to the LQR
            return None
        uv = vs.undistort(self.marker["corners"], self.K, self.dist)
        Z = vs.corner_depths(self.marker["T_CM"])              # depth from this frame's PnP (Week 9 B4)
        v, w, e, _ = vs.ibvs_command(uv, Z, self.p_star, self.K, self.G, self.p["ibvs_lambda"])
        e_rms = float(np.sqrt(np.mean(e ** 2)))
        # Saturate like Week 9 B3 (vmax), scaling v and omega together so the motion keeps its direction.
        s = max(1.0, abs(v) / self.p["v_dock"], abs(w) / self.p["ibvs_w_max"])
        v, w = v / s, w / s
        self.ibvs_log.append((t, uv, e_rms, v, w))
        still = abs(v) < self.p["ibvs_v_still"] and abs(w) < self.p["ibvs_w_still"]
        self.t_still = (self.t_still if self.t_still is not None else t) if still else None
        if (e_rms <= self.p["ibvs_e_tol_px"] or rng - self.stop_range <= self.p["stop_early"]
                or (self.t_still is not None and t - self.t_still >= self.p["ibvs_still_sec"])):
            return None                                        # converged, at the stop range, or stalled
        if v <= 0.0:
            # IBVS would back up (or hold): the camera says the robot is at the stop or a hair past it. A real
            # base cannot creep back by millimetres (28 Oct: -0.8 cm/s for 30 s at 2 cm from the stop), so end
            # IBVS here and let the SETTLE checks decide.
            return None
        self.v_cmd = min(v, self.v_cmd + self.a_max * self.Ts)  # gentle start (acceleration limit)
        if 0.0 < self.v_cmd < self.p["v_min_dock"]:
            # The real motors do not turn below ~1 cm/s: near the stop IBVS asked for 0.6 cm/s and the robot sat
            # 4 cm short for 30 s (28 Oct). Keep the approach's floor until the stop range ends IBVS.
            self.v_cmd = self.p["v_min_dock"]
        return self.v_cmd, w

    def step(self, x_hat, P, marker_frame, motion_feedback, t):
        """(v, omega) for this step. t = current time [s] (same clock as the camera timestamps).
        P = the estimate's covariance, used in the completion check."""
        if self.phase not in (APPROACH, IBVS, WAIT, SETTLE):
            return 0.0, 0.0
        self._read_marker(marker_frame)
        fresh = t - self.t_marker <= self.p["marker_max_age"]

        # 1. Marker lost: stop and wait for it (never drive in blind); give up after marker_lost_timeout.
        if not fresh:
            self.t_lost = t if self.t_lost is None else self.t_lost
            if t - self.t_lost > self.p["marker_lost_timeout"]:
                self.phase, self.stop_reason = FAILED, "marker_lost"
            elif self.phase in (APPROACH, IBVS):
                self.resume, self.phase = self.phase, WAIT
            self.v_cmd = 0.0
            return 0.0, 0.0
        self.t_lost = None
        rng, brg = self.marker["z"]
        if self.phase == WAIT:
            self.phase = self.resume

        # 2. APPROACH: steer along the line, slow down as the SEEN range closes in. Hand over to IBVS
        #    once close enough (servo = ibvs), otherwise stop when the stop range is reached.
        if self.phase == APPROACH and self._ibvs_ready(rng):
            self.phase = IBVS
        if self.phase == IBVS:
            cmd = self._ibvs_step(rng, t)
            if cmd is not None:
                return cmd
            if self.phase == IBVS:                              # finished: stop at once and confirm
                self.phase, self.t_settle, self.v_cmd = SETTLE, t, 0.0
        if self.phase == APPROACH:
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
    # Starts at the pre-dock point (the last waypoint, the prior from the config) and docks on dead reckoning plus the
    # camera. The marker_frame, P and motion_feedback built here are STAND-INS for the perception glue
    # and the EKF, which other teammates own; main.py will pass the real ones.
    from src.control.waypoints import _dead_reckoning
    from src.perception import aruco, camera
    from src.robot.task import load_task, make_robot
    from src.robot.sim_robot import SIM_INTRINSICS, WHEEL_RADIUS, WHEEL_SEPARATION

    task = load_task()
    K_cam, dist, _ = camera.load_intrinsics(SIM_INTRINSICS)
    dock = DockingController.from_task(task)
    w4, th0 = task["waypoints"][-1], task["dock"]["goal_pose"][2]
    with make_robot("none", seed=0) as robot:
        robot.reset(w4[0], w4[1], th0)                  # sim setup: put the robot at the pre-dock point
        pose = np.array([w4[0], w4[1], th0])            # odometry starts from the same prior
        P = np.diag([0.02**2, 0.02**2, np.deg2rad(2.0)**2])   # stand-in covariance until the EKF exists
        t_enc, *enc_prev = robot.read_encoders()
        dock.start()
        phase = None
        while dock.phase not in (COMPLETE, FAILED) and robot.t < 60:
            # stand-in perception: ID 0 range-bearing, corners and PnP pose from the newest camera frame
            t_frame, rgb = robot.read_image()
            m = aruco.measure(rgb, K_cam, dist, marker_id=dock.marker_id)
            frame = {"t": t_frame, "markers": {} if m is None else {dock.marker_id: {
                "z": m["z"], "corners": m["corners"], "T_CM": m["T_CM"], "yaw": m["yaw"], "ambiguous": m["ambiguous"]}}}
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
        if dock.ibvs_log:
            print(f"IBVS: {len(dock.ibvs_log)} steps, feature error {dock.ibvs_log[0][2]:.1f} -> {dock.ibvs_log[-1][2]:.2f} px rms")
        e_y, e_theta, _ = dock.line.errors(pose)
        print(f"odometry vs goal: lateral {e_y * 100:+.1f} cm, heading {np.rad2deg(e_theta):+.1f} deg; "
              f"last marker range {dock.marker['z'][0]:.3f} m (stop range {dock.stop_range} m)")
