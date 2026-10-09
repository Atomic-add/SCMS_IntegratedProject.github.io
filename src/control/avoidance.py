"""Obstacle avoidance with the LiDAR: shift the path sideways around the box, then rejoin.

The box is NOT in the waypoints or the map; the robot only knows it from the LiDAR obs (team plan
"LiDAR output specification", built for now by lidar_obs.py). The idea keeps the controller we already
have: instead of a new steering law, the reference line is shifted sideways by an offset d, and the
same LQR tracks the shifted line:

    e_y_shifted = e_y - d          (d > 0 = pass on the LEFT of the path direction)

Phases:
  (TRACK)  behaviour.py asks blocked(obs, ...) every step; when the box is in the swept corridor ahead
           and before the next waypoint, it pauses waypoints.py and calls start().
  AVOID    choose the side (wider of left_clear / right_clear first, the other one if the first has no
           room), check the shifted lane is free for the robot's footprint, then ramp d from 0 to the
           pass offset over ramp_len metres of travel, and hold it while passing.
  REJOIN   once the LiDAR shows the original lane free beside and behind the robot, ramp d back to 0
           over ramp_len (smooth, no jerk), then DONE: behaviour.py resumes waypoints.py.
  STOP     no side has room, or the scan is invalid / too old (missing returns do not prove clearance),
           or something is closer than stop_dist ahead: v = 0. stop_reason says why.

Points are compared in the PATH frame: along-track a and cross-track l to the active segment, computed
with the ESTIMATED pose x_hat (never the truth). The footprint is the robot half width plus a margin.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # project root, so "src" imports work even
                                                                # when this file is run directly (VS Code Run)

from src.control.lqr import PathLQR  # noqa: E402
from src.control.models import TS, load_control_config  # noqa: E402
from src.perception.lidar import ROBOT_FRONT, ROBOT_HALF_WIDTH  # noqa: E402

AVOID, REJOIN, DONE, STOP = "AVOID", "REJOIN", "DONE", "STOP"
ROBOT_REAR = 0.20     # m, base origin to the back of the Waffle Pi (about 0.28 m long, origin near the axle)


def points_in_path_frame(P_B, x_hat, segment):
    """LiDAR points from {B} into the path frame of a segment: rows (a = along-track, l = cross-track).

    1. {B} -> world with the estimated pose:  p_W = R(theta) p_B + [x, y]
    2. world -> path frame: a = t . (p_W - seg.a),  l = t x (p_W - seg.a)   (l > 0 = left of the path)
    """
    c, s = np.cos(x_hat[2]), np.sin(x_hat[2])
    p_W = np.array([[c, -s], [s, c]]) @ P_B + np.asarray(x_hat[:2], float).reshape(2, 1)
    r = p_W - segment.a.reshape(2, 1)
    t = segment.t
    return np.vstack([t[0] * r[0] + t[1] * r[1], t[0] * r[1] - t[1] * r[0]])


class AvoidanceController:
    """AVOID / REJOIN state machine on top of the LQR, for the active segment of waypoints.py."""

    def __init__(self, cfg=None, controller=None):
        cfg = cfg or load_control_config()
        self.p, self.Ts, self.v_nom = cfg["avoidance"], cfg["Ts"], cfg["v_nom"]
        self.rate = cfg["waypoints"]                           # same acceleration limits everywhere
        self.lqr = controller or PathLQR(cfg=cfg)
        self.half = ROBOT_HALF_WIDTH + self.p["side_margin"]   # half width of the robot's swept lane
        self.phase, self.stop_reason = None, None
        self.v_cmd = self.w_cmd = 0.0

    # ------------------------------------------------------------ detection (called in TRACK)
    def blocked(self, obs, remaining):
        """True when the box is in the swept corridor ahead, closer than trigger_dist, and before the
        segment's end waypoint (so the wall at the end of a leg is not treated as an obstacle)."""
        fa = obs["free_ahead"]
        return obs["valid"] and fa < self.p["trigger_dist"] and fa + ROBOT_FRONT < remaining - self.p["end_ignore"]

    # ------------------------------------------------------------ helpers
    def _lane_free(self, Q, d, a_from, a_to):
        """True if no LiDAR point lies in the robot's lane at offset d between along-track a_from..a_to."""
        sel = (Q[0] >= a_from) & (Q[0] <= a_to) & (np.abs(Q[1] - d) <= self.half)
        return not np.any(sel)

    def _rate_limit(self, v, w):
        dv, dw = self.rate["a_max"] * self.Ts, self.rate["alpha_max"] * self.Ts
        self.v_cmd += float(np.clip(v - self.v_cmd, -dv, dv))
        self.w_cmd += float(np.clip(w - self.w_cmd, -dw, dw))
        return self.v_cmd, self.w_cmd

    def _stop(self, reason):
        self.phase, self.stop_reason = STOP, reason
        self.v_cmd = self.w_cmd = 0.0
        return 0.0, 0.0

    # ------------------------------------------------------------ start: choose the pass side
    def start(self, obs, x_hat, segment):
        """Choose the pass side and offset. Returns False (and STOP) when neither side has room."""
        self.segment = segment
        self.lqr.reset()
        self.stop_reason = None
        Q = points_in_path_frame(obs["points_B"], x_hat, segment)
        _, _, a_robot = segment.errors(x_hat)

        # The obstacle = the points inside our current lane, ahead of the robot.
        ahead = (Q[0] > a_robot) & (Q[0] < a_robot + self.p["look_ahead"])
        box = ahead & (np.abs(Q[1]) <= self.half)
        if not np.any(box):
            self.phase = DONE                                   # nothing in the lane any more
            return True
        l_min, l_max = Q[1, box].min(), Q[1, box].max()        # its sideways extent, seen from the front
        self.a_box = Q[0, box].min()                            # where its front face is along the path

        # Offsets that put the robot's lane just beside the box, plus clearance.
        d_left = l_max + self.half + self.p["clearance"]
        d_right = l_min - self.half - self.p["clearance"]
        # Prefer the side the LiDAR says is wider (nan = no return = unknown, treated as not wider).
        lc, rc = obs["left_clear"], obs["right_clear"]
        order = [d_left, d_right] if np.nan_to_num(lc, nan=0.0) >= np.nan_to_num(rc, nan=0.0) else [d_right, d_left]

        for d in order:
            # The shifted lane must be free from here to past the box, for the robot's full footprint,
            # and stay inside what the LiDAR actually saw (a side with no returns proves nothing).
            if abs(d) <= self.p["max_offset"] and self._lane_free(Q, d, a_robot, self.a_box + self.p["box_depth"] + ROBOT_REAR):
                self.d_pass, self.d = d, 0.0
                self.a_start = a_robot
                self.phase = AVOID
                return True
        self._stop("no_passage")
        return False

    # ------------------------------------------------------------ one control step
    def step(self, obs, x_hat, scan_age):
        """(v, omega) while avoiding. obs = the latest LiDAR obs, scan_age = now - obs['t'] [s]."""
        if self.phase in (None, DONE, STOP):
            return self._stop(self.stop_reason or "idle") if self.phase == STOP else (0.0, 0.0)

        # Safety first: never move on an old or poorly covered scan, or with something right in front.
        if scan_age > self.p["max_scan_age"] or not obs["valid"]:
            self.v_cmd = self.w_cmd = 0.0
            return 0.0, 0.0                                      # wait for a good scan (not a STOP yet)
        if obs["free_ahead"] < self.p["stop_dist"] and self.d == self.d_pass:
            return self._stop("too_close")

        e_y, e_theta, a = self.segment.errors(x_hat)
        Q = points_in_path_frame(obs["points_B"], x_hat, self.segment)
        ramp = self.p["ramp_len"]

        if self.phase == AVOID:
            # Ramp d from 0 to d_pass over ramp_len of travel: the shifted reference is a gentle slant.
            frac = np.clip((a - self.a_start) / ramp, 0.0, 1.0)
            self.d = frac * self.d_pass
            slope = self.d_pass / ramp if frac < 1.0 else 0.0
            # Passed? The robot's REAR must be past the box's BACK face (front face a_box + box_depth;
            # the LiDAR cannot see the back, so box_depth is assumed), and the original lane must be free
            # from behind the rear to look_ahead. (Comparing with the front face only let the robot swing
            # back too early and clip the box's back corner in 1 of 20 random trials.)
            if frac >= 1.0 and a - ROBOT_REAR > self.a_box + self.p["box_depth"] and self._lane_free(Q, 0.0, a - ROBOT_REAR - self.p["box_depth"], a + self.p["rejoin_ahead"]):
                self.phase, self.a_rejoin = REJOIN, a
        if self.phase == REJOIN:
            frac = np.clip((a - self.a_rejoin) / ramp, 0.0, 1.0)
            self.d = (1.0 - frac) * self.d_pass
            slope = -self.d_pass / ramp if frac < 1.0 else 0.0
            if frac >= 1.0:
                self.phase = DONE
                self.d = 0.0
                return self.v_cmd, self.w_cmd                    # behaviour.py resumes waypoints now

        # Track the shifted line with the same LQR: the reference heading follows the slant of d.
        e_y_s = e_y - self.d
        e_theta_s = e_theta - np.arctan(slope)
        v = self.p["v_avoid"]
        w = self.lqr.step(e_y_s, e_theta_s, 0.0, v)
        return self._rate_limit(v, w)

    def info(self):
        return dict(phase=self.phase, offset=getattr(self, "d", 0.0), d_pass=getattr(self, "d_pass", None),
                    stop_reason=self.stop_reason)


# ---------------------------------------------------------------- try it on the robot
if __name__ == "__main__":
    # Run from the project root:  python -m src.control.avoidance [none|left|centre|right]
    # A minimal TRACK <-> AVOID loop on dead reckoning (behaviour.py will do this properly).
    from src.control.lidar_obs import make_obs
    from src.control.waypoints import WaypointTracker, _dead_reckoning
    from src.robot.corridor import load_task, make_robot

    case = sys.argv[1] if len(sys.argv) > 1 else "centre"
    task = load_task()
    tracker, avoid = WaypointTracker.from_task(task), AvoidanceController()
    with make_robot(case, seed=0) as robot:
        pose = np.array(task["start"], float)
        enc_prev = robot.read_encoders()[1:]
        state = "TRACK"
        while not tracker.pre_dock_reached and robot.t < 150 and avoid.phase != STOP:
            obs = make_obs(*robot.read_scan(), robot.lidar.range_min, robot.lidar.range_max)
            if state == "TRACK":
                remaining = tracker.segment.length - tracker.segment.errors(pose)[2]
                if tracker.mode == "TRACK" and avoid.blocked(obs, remaining):
                    tracker.pause()
                    avoid.start(obs, pose, tracker.segment)
                    state = "AVOID"
                    print(f"t = {robot.t:5.1f} s  AVOID  box ahead {obs['free_ahead']:.2f} m, "
                          f"left_clear {obs['left_clear']:.2f}, right_clear {obs['right_clear']:.2f} -> "
                          f"{'STOP ' + str(avoid.stop_reason) if avoid.phase == STOP else f'offset {avoid.d_pass:+.2f} m'}")
            if state == "AVOID":
                v, w = avoid.step(obs, pose, robot.t - obs["t"])
                if avoid.phase == DONE:
                    tracker.resume(pose)
                    state = "TRACK"
                    print(f"t = {robot.t:5.1f} s  REJOINED, back to waypoints")
            if state == "TRACK":
                v, w = tracker.step(pose)
            robot.send_cmd(v, w)
            robot.step(TS)
            enc = robot.read_encoders()[1:]
            pose = _dead_reckoning(pose, enc, enc_prev)
            enc_prev = enc
        print(f"\ncase {case}: pre-dock reached {tracker.pre_dock_reached} at t = {robot.t:.1f} s"
              f"{', stopped: ' + avoid.stop_reason if avoid.phase == STOP else ''}")
