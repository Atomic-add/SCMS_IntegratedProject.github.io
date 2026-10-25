"""EKF models: motion Jacobians, process noise Qd from wheel noise, and the marker range-bearing model.

Adapted from Week 7 C.4 (F_uni, h_rb, H_rb) and Assessment 3 Part B (Qd reasoning). The motion model
itself, f = odometry.move(), is in odometry.py; this file holds what the EKF needs AROUND it.

State x = [x, y, theta] in the world frame {W} (corridor.json), theta wrapped to (-pi, pi].

Motion Jacobian F = df/dx, with m = theta + dth/2 the midpoint heading:

    F = [[1, 0, -ds sin m],
         [0, 1,  ds cos m],
         [0, 0,  1       ]]

Process noise Qd from the WHEELS (team plan): each wheel's rolled distance has an error that grows
with how far it rolled, sd = k |d| (slip, radius error). Pushed through G = df/d(d_L, d_R):

    G  = [[cos m / 2, cos m / 2],
          [sin m / 2, sin m / 2],
          [-1 / b,    1 / b    ]]          (the small ds sin m / 2b terms are dropped)
    Qd = G diag((k d_L)^2, (k d_R)^2) G^T + q_floor I

Qd scales with the step by itself: a longer dt means more wheel travel and more noise; standing
still adds only q_floor (so P can still grow a little and the filter never locks up).

Measurement: range r and bearing b of a marker at the known map position (mx, my), from the base
origin (as aruco.measure() returns it):

    h(x) = [ sqrt(q),  wrap(atan2(dy, dx) - theta) ],   dx = mx - x, dy = my - y, q = dx^2 + dy^2
    H    = [[-dx/sqrt(q), -dy/sqrt(q),  0],
            [ dy/q,       -dx/q,       -1]]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # project root, so "src" imports work even
                                                                # when this file is run directly (VS Code Run)

from src.estimation.odometry import increments, move, wrap  # noqa: E402
from src.robot.sim_robot import PROJECT_ROOT, WHEEL_SEPARATION  # noqa: E402

ESTIMATION_CONFIG = PROJECT_ROOT / "config" / "estimation.json"


def load_estimation_config(path=ESTIMATION_CONFIG, camera=None):
    """The EKF tuning as a dict (config/estimation.json).

    camera: the intrinsics file the ArUco ranges come from. When it has its own "range_correction" (the
    real camera, experiments/aruco_real_noise.py), aruco.measure has already removed that bias, so
    range_bias_r2 (the sim camera's bias) becomes 0, not subtracted twice, and range_sd is the camera's.
    When it has an "ekf_noise" block (range_bias_r2, range_sd, bearing_sd_deg measured for that camera by
    evaluation/consistency_sweep.py --write-noise), those values replace the config's. A file with
    neither (config/camera_sim.json) leaves the config unchanged.
    """
    cfg = json.loads(Path(path).read_text())
    cam = {} if camera is None else json.loads(Path(camera).read_text())
    corr = cam.get("range_correction")
    if corr is not None:
        cfg["range_bias_r2"], cfg["range_sd"] = 0.0, list(corr["range_sd"])
    for k, v in cam.get("ekf_noise", {}).items():
        if not k.startswith("_"):
            cfg[k] = v
    return cfg


# ---------------------------------------------------------------- motion
def F_jac(pose, ds, dth):
    """Motion Jacobian df/dx at the current estimate."""
    m = pose[2] + dth / 2
    return np.array([[1.0, 0.0, -ds * np.sin(m)],
                     [0.0, 1.0, ds * np.cos(m)],
                     [0.0, 0.0, 1.0]])


def Qd_wheels(pose, d_l, d_r, dth, k_wheel, q_floor=1e-8):
    """Process noise of one step from wheel distance noise sd = k_wheel |d| (see the module doc)."""
    m = pose[2] + dth / 2
    G = np.array([[np.cos(m) / 2, np.cos(m) / 2],
                  [np.sin(m) / 2, np.sin(m) / 2],
                  [-1 / WHEEL_SEPARATION, 1 / WHEEL_SEPARATION]])
    return G @ np.diag([(k_wheel * d_l) ** 2, (k_wheel * d_r) ** 2]) @ G.T + q_floor * np.eye(3)


# ---------------------------------------------------------------- measurement (Week 7, unchanged)
def h_rb(pose, mk):
    """Expected range and bearing from pose to the marker at map position mk = (mx, my)."""
    dx, dy = mk[0] - pose[0], mk[1] - pose[1]
    return np.array([np.hypot(dx, dy), wrap(np.arctan2(dy, dx) - pose[2])])


def H_rb(pose, mk):
    """Measurement Jacobian dh/dx at the current estimate."""
    dx, dy = mk[0] - pose[0], mk[1] - pose[1]
    q = dx ** 2 + dy ** 2
    r = np.sqrt(q)
    return np.array([[-dx / r, -dy / r, 0.0],
                     [dy / q, -dx / q, -1.0]])


def R_marker(r, cfg):
    """Measurement noise for a marker seen at range r: range sd a + b r^2, bearing sd fixed."""
    a, b = cfg["range_sd"]
    return np.diag([(a + b * r ** 2) ** 2, np.deg2rad(cfg["bearing_sd_deg"]) ** 2])


def numerical_jacobian(fun, x, eps=1e-6):
    """Central differences, for checking F_jac and H_rb in estimation_test.py."""
    x = np.asarray(x, float)
    cols = []
    for i in range(len(x)):
        d = np.zeros_like(x)
        d[i] = eps
        diff = fun(x + d) - fun(x - d)
        if len(diff) == 3:
            diff[2] = wrap(diff[2])
        else:
            diff[1] = wrap(diff[1])
        cols.append(diff / (2 * eps))
    return np.array(cols).T


if __name__ == "__main__":
    # Run from the project root:  python -m src.estimation.models
    # Prints Qd for 5 cm straight and the Jacobians, checked against finite differences.
    cfg = load_estimation_config()
    pose = np.array([1.0, 0.5, 0.3])
    d_l, d_r, ds, dth = increments(0.05 / 0.033, 0.052 / 0.033)
    print("Qd for ~5 cm of travel:\n", Qd_wheels(pose, d_l, d_r, dth, cfg["k_wheel"]).round(8))
    F_num = numerical_jacobian(lambda p: move(p, ds, dth), pose)
    print("F matches finite differences:", np.allclose(F_jac(pose, ds, dth), F_num, atol=1e-6))
    mk = np.array([3.6, 4.599])
    print("H matches finite differences:", np.allclose(H_rb(pose, mk), numerical_jacobian(lambda p: h_rb(p, mk), pose), atol=1e-6))
