"""Path-following error model of the Waffle Pi: linearise, discretise (ZOH), check controllability.

Adapted from 41014 Assessment 3 Part A.1 (error_model on the unicycle path-following errors) and
Week 5 (Jacobians at an equilibrium, zero-order-hold discretisation, controllability rank).
The LQR / LQI gains are designed on these matrices in src/control/lqr.py.

Why an ERROR model and not a pose model
---------------------------------------
Following a path is regulation of the error to the path, not of the pose. For the segment the
robot is on, take the closest point on it and define (Assessment 3 conventions):

    e_y      cross-track error [m], positive when the robot is LEFT of the path direction
    e_theta  heading error theta - theta_path [rad], wrapped to (-pi, pi]
    kappa    path curvature [1/m]: 0 on a straight segment, 1/R on an arc of radius R

With forward speed v held constant and steering by the yaw rate omega, the error dynamics are

    de_y/dt     = v sin(e_theta)
    de_theta/dt = omega - v kappa cos(e_theta) / (1 - kappa e_y)

On the path (e = 0) the robot must turn at exactly omega = v kappa to stay on it, so the command
is split into a feedforward part known from the path and a feedback part the LQR designs:

    omega = v kappa + d_omega,      d_omega = -K e

Linearising at e = 0, d_omega = 0 gives the 2-state linear model used here:

    d/dt [e_y    ]   [ 0          v ] [e_y    ]   [0]
         [e_theta] = [ -v kappa^2 0 ] [e_theta] + [1] d_omega

The -v kappa^2 term comes from differentiating 1 / (1 - kappa e_y) with respect to e_y. It is
zero on a straight segment (kappa = 0), where the model is a double integrator.

Frames and inputs: the state is computed from the ESTIMATED pose (EKF x_hat, or odometry until the
EKF is ready), never from the simulator's true pose. Only evaluation/ may read the truth.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import cont2discrete

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # project root, so "src" imports work even
                                                                # when this file is run directly (VS Code Run)

from src.robot.sim_robot import MAX_V, MAX_W, PROJECT_ROOT  # noqa: E402

# ---------------------------------------------------------------- design point
# Tuning lives in config/control.json so every value is in one place (and in one table for the report).
CONTROL_CONFIG = PROJECT_ROOT / "config" / "control.json"


def load_control_config(path=CONTROL_CONFIG):
    """The controller tuning as a dict (config/control.json)."""
    return json.loads(path.read_text())


_cfg = load_control_config()
TS = _cfg["Ts"]        # s, controller sample time (20 Hz, the /cmd_vel rate used in sim_env/drive_square_test.py)
V_NOM = _cfg["v_nom"]  # m/s, cruise speed; below the Waffle Pi limit MAX_V = 0.26 m/s so the wheels
                       # keep some headroom for the steering correction
OMEGA_MAX = MAX_W      # rad/s, steering limit the controller must clip to (Waffle Pi: 1.82 rad/s)


def continuous_error_model(v, kappa):
    """Continuous Jacobians (A, B) of the error dynamics at e = 0, d_omega = 0.

    State e = [e_y, e_theta], input d_omega = omega - v*kappa (Assessment 3 Task A.1).
    """
    A = np.array([[0.0,           v],      # de_y/dt     = v * e_theta            (sin e ~ e)
                  [-v * kappa**2, 0.0]])   # de_theta/dt = -v kappa^2 e_y + d_omega (from 1/(1 - kappa e_y))
    B = np.array([[0.0],
                  [1.0]])                  # d_omega acts directly on the heading error only
    return A, B


def error_model(v, kappa, Ts=TS):
    """Discrete error model (Ad, Bd) with a zero-order hold at sample time Ts.

    ZOH is exact for a command that is held constant over each sample, which is what send_cmd()
    does between control updates. Forward Euler (Ad = I + A Ts, Bd = B Ts) would give Bd[0] = 0,
    i.e. it would say steering does not move e_y within one sample; ZOH gives Bd[0] = v Ts^2 / 2.
    """
    A, B = continuous_error_model(v, kappa)
    C, D = np.eye(2), np.zeros((2, 1))     # C and D are not used, cont2discrete just needs them
    Ad, Bd, _, _, _ = cont2discrete((A, B, C, D), Ts, method="zoh")
    return Ad, Bd


def integral_augmented(Ad, Bd, Ts=TS):
    """Add an integrator on the cross-track error, z[k+1] = z[k] + Ts * e_y[k] (Assessment 3 Task A.3).

    The augmented state is [e_y, e_theta, z]. It removes a steady offset caused by a constant
    disturbance (wheel mismatch, slope, motor lag). Only use it if the plain LQR shows an offset.
    """
    Ad_i = np.block([[Ad,                       np.zeros((2, 1))],
                     [np.array([[Ts, 0.0]]),    np.ones((1, 1))]])
    Bd_i = np.vstack([Bd, np.zeros((1, 1))])
    return Ad_i, Bd_i


def ctrb(Ad, Bd):
    """Controllability matrix [B, AB, A^2 B, ...] (what control.ctrb returns, Week 5)."""
    n = Ad.shape[0]
    return np.hstack([np.linalg.matrix_power(Ad, k) @ Bd for k in range(n)])


def controllability_rank(Ad, Bd):
    """Rank of the controllability matrix; full rank (= number of states) means controllable."""
    return int(np.linalg.matrix_rank(ctrb(Ad, Bd)))


if __name__ == "__main__":
    # Run from the project root:  python -m src.control.models
    np.set_printoptions(precision=5, suppress=True)

    print(f"Design point: Ts = {TS} s, v = {V_NOM} m/s, omega limit = {OMEGA_MAX} rad/s (MAX_V = {MAX_V})\n")

    # 1. The model on a straight segment and on an arc (e.g. a smoothed corner of radius 0.5 m)
    for name, kappa in [("straight (kappa = 0)", 0.0), ("arc R = 0.5 m (kappa = 2)", 2.0)]:
        Ad, Bd = error_model(V_NOM, kappa)
        print(f"{name}:\nAd =\n{Ad}\nBd =\n{Bd}")
        print(f"|eig(Ad)| = {np.abs(np.linalg.eigvals(Ad))}   (on the unit circle: open loop is not stable)")
        print(f"controllability rank = {controllability_rank(Ad, Bd)} of 2\n")

    # 2. Low-speed check: the controller steers e_y only THROUGH the heading (de_y/dt = v e_theta),
    #    so the authority over e_y shrinks with v and vanishes at v = 0. This matters for the slow
    #    docking approach and for turning on the spot at corners.
    print("Low-speed controllability on a straight (smallest singular value of the ctrb matrix):")
    for v in [V_NOM, 0.10, 0.05, 0.02, 0.0]:
        Ad, Bd = error_model(v, 0.0)
        sv_min = np.linalg.svd(ctrb(Ad, Bd), compute_uv=False)[-1]
        print(f"  v = {v:4.2f} m/s: rank {controllability_rank(Ad, Bd)}, smallest singular value {sv_min:.2e}")

    # 3. The augmented model for integral action (used only if a steady offset shows up)
    Ad_i, Bd_i = integral_augmented(*error_model(V_NOM, 0.0))
    print(f"\nWith the e_y integrator: controllability rank = {controllability_rank(Ad_i, Bd_i)} of 3")
