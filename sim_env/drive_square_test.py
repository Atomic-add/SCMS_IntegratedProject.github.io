"""Drive the simulated Waffle Pi the same way as the real robot, around a 0.5 m square.

Run from the project root:
    python sim/drive_square.py              # with the MuJoCo viewer
    python sim/drive_square.py --no-viewer  # numbers only

What makes this "like the real robot":
  * commands are (v, w) like /cmd_vel, sent at 20 Hz, never wheel speeds directly
  * Waffle Pi limits: 0.26 m/s, 1.82 rad/s, plus an acceleration limit (placeholder, see below)
  * the robot knows where it is ONLY from wheel-encoder odometry (4096 ticks/rev),
    like the real /joint_states; ground truth is used only to grade the odometry at the end
"""
import argparse
import time
from pathlib import Path

import mujoco
import numpy as np

SCENE = Path(__file__).resolve().parents[1] / "models" / "robotis_tb3" / "scene_turtlebot3_waffle_pi.xml"

# Waffle Pi (turtlebot3_waffle_pi.urdf, ROBOTIS e-Manual)
WHEEL_R, WHEEL_SEP = 0.033, 0.288
MAX_V, MAX_W = 0.26, 1.82
TICKS_PER_REV = 4096                  # XM430-W210 encoder
# PLACEHOLDERS: replace with values fitted from a step test on the real robot
MAX_ACC, MAX_ALPHA = 0.5, 2.0         # m/s^2, rad/s^2
CMD_RATE = 20.0                       # Hz, how often a new /cmd_vel is applied


class SimWafflePi:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(str(SCENE))
        self.d = mujoco.MjData(self.m)
        self.qadr = [self.m.jnt_qposadr[self.m.joint(n).id] for n in ("wheel_left", "wheel_right")]
        self.v = self.w = 0.0                     # what the "motors" are currently doing (after ramp)
        mujoco.mj_step(self.m, self.d, 100)       # settle on the floor

    def send_cmd(self, v_cmd, w_cmd, dt):
        """Like /cmd_vel: saturate, ramp with the acceleration limit, convert to wheel speeds."""
        v_cmd = np.clip(v_cmd, -MAX_V, MAX_V)
        w_cmd = np.clip(w_cmd, -MAX_W, MAX_W)
        self.v += np.clip(v_cmd - self.v, -MAX_ACC * dt, MAX_ACC * dt)
        self.w += np.clip(w_cmd - self.w, -MAX_ALPHA * dt, MAX_ALPHA * dt)
        self.d.ctrl[:] = [(self.v - self.w * WHEEL_SEP / 2) / WHEEL_R,
                          (self.v + self.w * WHEEL_SEP / 2) / WHEEL_R]

    def step(self, dt):
        mujoco.mj_step(self.m, self.d, int(round(dt / self.m.opt.timestep)))

    def read_encoders(self):
        tick = 2 * np.pi / TICKS_PER_REV
        return np.round(self.d.qpos[self.qadr] / tick) * tick     # left, right (rad)

    def ground_truth(self):
        """EVALUATION ONLY - never use this to decide what the robot does."""
        q = self.d.qpos[3:7]
        yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        return np.array([self.d.qpos[0], self.d.qpos[1], yaw])


class EncoderOdometry:
    """Dead reckoning from wheel encoders - this is the EKF prediction step later."""
    def __init__(self, enc):
        self.prev = enc.copy()
        self.pose = np.zeros(3)

    def update(self, enc):
        dl, dr = (enc - self.prev) * WHEEL_R
        self.prev = enc.copy()
        ds, dth = (dl + dr) / 2, (dr - dl) / WHEEL_SEP
        x, y, th = self.pose
        self.pose = np.array([x + ds * np.cos(th + dth / 2), y + ds * np.sin(th + dth / 2), th + dth])
        return self.pose


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def main(viewer_on):
    robot = SimWafflePi()
    odom = EncoderOdometry(robot.read_encoders())
    gt0 = robot.ground_truth()
    dt = 1.0 / CMD_RATE
    plan = [("forward", 0.5), ("turn", np.pi / 2)] * 4    # 0.5 m square, counter-clockwise
    seg, seg_start = 0, odom.pose.copy()
    heading_ref = 0.0                                     # absolute heading target, so turn errors don't add up

    viewer = None
    if viewer_on:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(robot.m, robot.d)

    t = 0.0
    while seg < len(plan) and t < 60:
        pose = odom.update(robot.read_encoders())
        kind, target = plan[seg]
        # Brake early: the robot cannot stop instantly (acceleration limit), so the speed is
        # capped at sqrt(2 * a * remaining) - the speed from which it can still stop in time.
        if kind == "forward":
            rem = target - np.hypot(*(pose[:2] - seg_start[:2]))
            v, w = min(0.15, np.sqrt(2 * MAX_ACC * max(rem, 0))), 0.0
            done = rem < 0.002
        else:
            rem = wrap(heading_ref + target - pose[2])
            v, w = 0.0, min(0.6, np.sqrt(2 * MAX_ALPHA * max(rem, 0)))
            done = rem < 0.003
            if done:
                heading_ref += target
        if done:
            seg, seg_start = seg + 1, pose.copy()
            v = w = 0.0
        robot.send_cmd(v, w, dt)
        robot.step(dt)
        t += dt
        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            time.sleep(dt)

    for _ in range(int(2 * CMD_RATE)):                      # stop and let it settle
        robot.send_cmd(0, 0, dt); robot.step(dt); odom.update(robot.read_encoders())

    # ---- evaluation (ground truth allowed here only)
    gt = robot.ground_truth() - gt0
    c, s = np.cos(-gt0[2]), np.sin(-gt0[2])
    gt_local = np.array([c * gt[0] - s * gt[1], s * gt[0] + c * gt[1], wrap(gt[2])])
    est = odom.pose
    print(f"finished {seg}/{len(plan)} segments in {t:.1f} s")
    print(f"odometry says : x {est[0]:+.3f} m  y {est[1]:+.3f} m  yaw {np.degrees(wrap(est[2])):+.1f} deg")
    print(f"ground truth  : x {gt_local[0]:+.3f} m  y {gt_local[1]:+.3f} m  yaw {np.degrees(gt_local[2]):+.1f} deg")
    print(f"odometry error: {np.hypot(*(est[:2] - gt_local[:2])) * 1000:.1f} mm, "
          f"{np.degrees(abs(wrap(est[2] - gt_local[2]))):.2f} deg")
    print(f"square closure (truth, should be ~0): {np.hypot(*gt_local[:2]) * 1000:.1f} mm")
    if viewer is not None:
        print("close the viewer window to exit")
        while viewer.is_running():
            viewer.sync(); time.sleep(0.05)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-viewer", action="store_true")
    main(not ap.parse_args().no_viewer)