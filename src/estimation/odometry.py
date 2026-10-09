"""Wheel odometry: encoder angles -> distance and heading change -> new pose (midpoint integration).

Adapted from sim_env/drive_square_test.py (EncoderOdometry) and the Week 7 unicycle model f_uni.
This is the EKF's PREDICTION of the pose (src/estimation/ekf.py calls it); on its own, without any
marker updates, it is plain dead reckoning, the baseline the EKF is compared against.

With wheel radius r, wheel separation b and wheel angle increments dphi_L, dphi_R [rad]:

    d_L = r dphi_L,   d_R = r dphi_R                  distance each wheel rolled [m]
    ds  = (d_L + d_R) / 2                             distance the base origin moved [m]
    dth = (d_R - d_L) / b                             heading change [rad] (left turn > 0)

Midpoint integration (move along the AVERAGE heading of the step, theta + dth/2):

    x'  = x + ds cos(theta + dth/2)
    y'  = y + ds sin(theta + dth/2)
    th' = wrap(theta + dth)

Using the wheel increments (what the robot really did) instead of the command (v, omega) means the
prediction includes the motor lag and the speed limits for free.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # project root, so "src" imports work even
                                                                # when this file is run directly (VS Code Run)

from src.robot.sim_robot import WHEEL_RADIUS, WHEEL_SEPARATION  # noqa: E402


def wrap(a):
    """Angle to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def increments(dphi_l, dphi_r):
    """Wheel angle increments [rad] -> (d_L, d_R, ds, dth) in metres and radians."""
    d_l, d_r = WHEEL_RADIUS * dphi_l, WHEEL_RADIUS * dphi_r
    return d_l, d_r, (d_l + d_r) / 2, (d_r - d_l) / WHEEL_SEPARATION


def move(pose, ds, dth):
    """The motion model f: the pose after moving ds along the midpoint heading and turning dth."""
    x, y, th = pose
    m = th + dth / 2                                   # midpoint heading of this step
    return np.array([x + ds * np.cos(m), y + ds * np.sin(m), wrap(th + dth)])


class Odometry:
    """Dead reckoning on its own: step(encoders) -> pose. The first sample only sets the baseline."""

    def __init__(self, x0):
        self.pose = np.array(x0, float)
        self.prev = None                               # last (t, phi_L, phi_R)

    def step(self, encoders):
        t, phi_l, phi_r = encoders
        if self.prev is not None and t > self.prev[0]:  # a NEW encoder sample: move once
            _, _, ds, dth = increments(phi_l - self.prev[1], phi_r - self.prev[2])
            self.pose = move(self.pose, ds, dth)
        if self.prev is None or t > self.prev[0]:
            self.prev = (t, phi_l, phi_r)
        return self.pose.copy()


if __name__ == "__main__":
    # Run from the project root:  python -m src.estimation.odometry
    # Both wheels turn by the same angle for 1 m, then opposite ways for a 90 deg turn on the spot.
    odo = Odometry([0.0, 0.0, 0.0])
    odo.step((0.0, 0.0, 0.0))
    a = 1.0 / WHEEL_RADIUS                             # wheel angle for 1 m
    print("after 1 m straight:", odo.step((1.0, a, a)).round(4))
    b = (np.pi / 2) * WHEEL_SEPARATION / 2 / WHEEL_RADIUS   # wheel angle for a 90 deg spin
    print("after +90 deg spin:", odo.step((2.0, a - b, a + b)).round(4))
