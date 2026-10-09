"""Waypoint following: which segment the robot is on, its errors to it, and the corner turns.

lqr.py answers "how hard do I steer?"; this file answers "steer relative to WHAT?".

The path is the polyline start -> W1 -> W2 -> W3 -> W4 from config/corridor.json (load_task()).
Every control step (every Ts) the tracker:

  1. takes the ESTIMATED pose x_hat = [x, y, theta] (odometry now, the EKF later; never the truth),
  2. computes the errors to the active segment, e_y, e_theta and kappa (= 0, segments are straight),
  3. checks whether the segment's end waypoint has been reached, and moves on if so,
  4. returns the command (v, omega) for SimRobot.send_cmd().

Two driving modes (a small state machine):

  TRACK  drive forward at cruise speed and steer with the LQR (omega = v kappa - K e). When the next
         segment needs a sharp turn, slow down before the waypoint so the robot stops ON it.
  TURN   at a sharp corner: v = 0 and rotate on the spot until the robot faces the next segment,
         then back to TRACK. Turning in place is used because the LQR loses its grip on e_y as v -> 0
         (models.py: the controllability rank drops to 1 at v = 0), so it cannot steer a slow corner.
  DONE   the last waypoint (W4, the pre-dock point) is reached: stop and set pre_dock_reached.

pause() / resume() let behaviour.py hand control to avoidance or docking without the tracker
skipping waypoints meanwhile.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # project root, so "src" imports work even
                                                                # when this file is run directly (VS Code Run)

from src.control.lqr import PathLQR  # noqa: E402
from src.control.models import TS, load_control_config  # noqa: E402

TRACK, TURN, DONE = "TRACK", "TURN", "DONE"


def wrap(a):
    """Wrap an angle to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------- geometry of one segment
@dataclass
class Segment:
    """Straight segment from point a to point b (2D, world frame)."""
    a: np.ndarray
    b: np.ndarray

    def __post_init__(self):
        self.a, self.b = np.asarray(self.a, float), np.asarray(self.b, float)
        d = self.b - self.a
        self.length = float(np.hypot(*d))
        self.t = d / self.length                     # unit tangent: the path direction
        self.heading = float(np.arctan2(d[1], d[0]))  # theta_path
        self.kappa = 0.0                             # straight line: zero curvature

    def errors(self, pose):
        """(e_y, e_theta, s) of a pose [x, y, theta] to this segment.

        e_y      signed distance from the line, positive when the robot is LEFT of the path direction
                 (2D cross product of the tangent with the vector from a to the robot);
        e_theta  heading error theta - theta_path, wrapped (Assessment 3 conventions);
        s        progress along the segment from a [m] (dot product with the tangent);
                 s > length means the robot has passed the end waypoint.
        """
        r = np.asarray(pose[:2], float) - self.a
        e_y = self.t[0] * r[1] - self.t[1] * r[0]
        s = float(self.t @ r)
        return float(e_y), float(wrap(pose[2] - self.heading)), s


# ---------------------------------------------------------------- the tracker
class WaypointTracker:
    """TRACK / TURN state machine over the waypoint list, steering with PathLQR."""

    def __init__(self, waypoints, start_xy, cfg=None, controller=None):
        cfg = cfg or load_control_config()
        self.p = cfg["waypoints"]
        self.Ts, self.v_nom = cfg["Ts"], cfg["v_nom"]
        pts = [np.asarray(start_xy, float)[:2]] + [np.asarray(w, float) for w in waypoints]
        self.segments = [Segment(a, b) for a, b in zip(pts[:-1], pts[1:])]
        self.lqr = controller or PathLQR(cfg=cfg)
        self.reset()

    @classmethod
    def from_task(cls, task, **kw):
        """Build from load_task() (src/robot/corridor.py): start prior and waypoints from the config."""
        return cls(task["waypoints"], task["start"][:2], **kw)

    def reset(self):
        """Start again at the first segment."""
        self.i = 0                       # index of the active segment
        self.mode = TRACK
        self.paused = False
        self.v_cmd = self.w_cmd = 0.0    # last command sent, for the rate limits
        self.last = {}                   # errors of the last step, for logging and plots
        self.lqr.reset()

    @property
    def segment(self):
        return self.segments[self.i]

    @property
    def pre_dock_reached(self):
        """True once the last waypoint (the pre-dock point) is reached: docking can take over."""
        return self.mode == DONE

    def pause(self):
        """Another behaviour (avoidance, docking) is driving: keep the segment, stop updating."""
        self.paused = True

    def resume(self, pose):
        """Take back control. If the robot now faces well away from the segment (e.g. after an
        avoidance manoeuvre), turn on the spot first, otherwise carry on tracking."""
        self.paused = False
        self.lqr.reset()
        if self.mode != DONE:
            _, e_theta, _ = self.segment.errors(pose)
            self.mode = TURN if abs(e_theta) > np.deg2rad(self.p["turn_start_deg"]) else TRACK

    def _corner_angle(self):
        """Heading change from the active segment to the next one [rad] (0 after the last one)."""
        if self.i + 1 >= len(self.segments):
            return 0.0
        return float(wrap(self.segments[self.i + 1].heading - self.segment.heading))

    def _needs_turn(self):
        """Is the next corner sharp enough to stop and turn on the spot?"""
        return abs(self._corner_angle()) > np.deg2rad(self.p["turn_start_deg"])

    def _rate_limit(self, v, w):
        """Limit how fast v and omega may change per step (acceleration limits, Week 6 saturation).

        Wheels cannot jump in speed; a smooth command also keeps the camera image sharp and the
        odometry honest (less wheel slip).
        """
        dv, dw = self.p["a_max"] * self.Ts, self.p["alpha_max"] * self.Ts
        self.v_cmd += float(np.clip(v - self.v_cmd, -dv, dv))
        self.w_cmd += float(np.clip(w - self.w_cmd, -dw, dw))
        return self.v_cmd, self.w_cmd

    def step(self, pose):
        """One control step: (v, omega) to send, from the ESTIMATED pose [x, y, theta]."""
        if self.paused or self.mode == DONE:
            return self._rate_limit(0.0, 0.0)

        seg = self.segment
        e_y, e_theta, s = seg.errors(pose)
        remaining = seg.length - s                       # distance left to the end waypoint
        self.last = dict(seg=self.i, mode=self.mode, e_y=e_y, e_theta=e_theta, s=s)

        if self.mode == TURN:
            # Rotate on the spot towards this segment's heading: proportional on e_theta, capped.
            if abs(e_theta) < np.deg2rad(self.p["turn_done_deg"]):
                self.mode = TRACK                        # facing the segment: drive on
                self.lqr.reset()
            else:
                w = float(np.clip(-self.p["k_turn"] * e_theta, -self.p["w_turn_max"], self.p["w_turn_max"]))
                return self._rate_limit(0.0, w)

        # ---- TRACK
        # 1. Waypoint reached? "Close enough along the path" also covers having passed it, so a
        #    robot that is a bit off to the side still moves on instead of circling back.
        if remaining < self.p["arrive_tol"]:
            if self.i + 1 >= len(self.segments):
                self.mode = DONE                         # last waypoint = pre-dock point
                return self._rate_limit(0.0, 0.0)
            sharp = self._needs_turn()
            self.i += 1
            self.lqr.reset()
            if sharp:
                self.mode = TURN
                return self._rate_limit(0.0, 0.0)
            return self.step(pose)                       # gentle corner: just track the next segment

        # 2. Speed: cruise, but slow down before a corner where we will stop and turn, or before the
        #    end, so the robot stops on the waypoint instead of overshooting it.
        v = self.v_nom
        if self._needs_turn() or self.i + 1 == len(self.segments):
            v = float(np.clip(self.v_nom * remaining / self.p["slow_down_dist"], self.p["v_min"], self.v_nom))

        # 3. Steering from the LQR, with the v that is actually being sent.
        w = self.lqr.step(e_y, e_theta, seg.kappa, v)
        return self._rate_limit(v, w)


# ---------------------------------------------------------------- try it on the robot
def _dead_reckoning(pose, enc, enc_prev):
    """STAND-IN for the Estimation teammate's odometry, only for this demo.

    Differential-drive dead reckoning (W2 Part A): wheel arc lengths -> distance and heading change.
    Replace with their odometry / EKF x_hat in main.py.
    """
    from src.robot.sim_robot import WHEEL_RADIUS, WHEEL_SEPARATION
    dl, dr = (np.asarray(enc) - np.asarray(enc_prev)) * WHEEL_RADIUS
    ds, dth = (dl + dr) / 2, (dr - dl) / WHEEL_SEPARATION
    x, y, th = pose
    return np.array([x + ds * np.cos(th + dth / 2), y + ds * np.sin(th + dth / 2), wrap(th + dth)])


if __name__ == "__main__":
    # Run from the project root:  python -m src.control.waypoints      (or the VS Code Run button)
    # Drives start -> W1 -> W2 -> W3 -> W4 in the corridor (no obstacle) on dead reckoning only.
    from src.robot.corridor import load_task, make_robot

    task = load_task()
    tracker = WaypointTracker.from_task(task)
    with make_robot("none", seed=0) as robot:
        pose = np.array(task["start"], float)            # the prior from the config, not the truth
        enc_prev = robot.read_encoders()[1:]
        mode, seg = None, None
        while not tracker.pre_dock_reached and robot.t < 120:
            v, w = tracker.step(pose)
            if (tracker.mode, tracker.i) != (mode, seg):  # print each mode / segment change
                mode, seg = tracker.mode, tracker.i
                print(f"t = {robot.t:5.1f} s  {mode:5s} segment {seg} -> W{seg + 1}   "
                      f"odometry pose ({pose[0]:.2f}, {pose[1]:.2f}, {np.rad2deg(pose[2]):4.0f} deg)")
            robot.send_cmd(v, w)
            robot.step(TS)
            enc = robot.read_encoders()[1:]
            pose = _dead_reckoning(pose, enc, enc_prev)
            enc_prev = enc
        print(f"\npre-dock reached: {tracker.pre_dock_reached} at t = {robot.t:.1f} s, "
              f"odometry says ({pose[0]:.3f}, {pose[1]:.3f}), W4 is {tuple(task['waypoints'][-1])}")
