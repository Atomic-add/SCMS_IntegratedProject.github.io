"""Discrete LQR (and optional LQI) steering for path following, designed with Bryson's rule.

Adapted from 41014 Assessment 3 Part A.2 / A.3 (LQRController, LQIController) and Week 6
(Bryson's rule, discrete LQR, integral action, actuator saturation with anti-windup).

The plant is the discrete error model in src/control/models.py, state e = [e_y, e_theta].
The steering command is feedforward plus state feedback:

    omega = v * kappa  -  K e          (LQR)
    omega = v * kappa  -  K [e; z]     (LQI, z = running sum of Ts * e_y)

clipped to the Waffle Pi limit. The forward speed v is NOT chosen here: waypoints.py / behaviour.py
decide v (cruise, slow down, stop) and this file only steers.

Inputs come from the ESTIMATED pose (odometry, later the EKF x_hat). Nothing here reads ground truth.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_are

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # project root, so "src" imports work even
                                                                # when this file is run directly (VS Code Run)

from src.control.models import (OMEGA_MAX, TS, V_NOM, error_model, integral_augmented,  # noqa: E402
                                load_control_config)


# ---------------------------------------------------------------- design tools
def bryson(*x_max):
    """Bryson's rule weight matrix (Week 6): diag(1 / x_max_i^2).

    Each x_max is 'the largest value of this quantity we are willing to accept'. Dividing by it
    makes every term of the cost dimensionless, so a 0.10 m position error and a 10 deg heading
    error cost the same when they are each at their limit.
    """
    return np.diag([1.0 / m**2 for m in x_max])


def dlqr(Ad, Bd, Q, R):
    """Discrete LQR gain K for u[k] = -K x[k] (what control.dlqr returns, Week 6).

    1. Solve the discrete algebraic Riccati equation for P.
    2. K = (R + B'PB)^-1 B'PA.
    Returns (K, P, closed-loop eigenvalues of Ad - Bd K).
    """
    P = solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    return K, P, np.linalg.eigvals(Ad - Bd @ K)


def design_weights(cfg=None):
    """Q (2x2) and R (1x1) for the error model, from the limits in config/control.json.

    Why these limits (see the _doc in control.json too):
      e_y_max     0.05 m   the corridor this was tuned on was 1.0 m wide (the lab band is 1.5 m) and the robot
                           is 0.31 m wide, so it had about 0.35 m each side. 0.10 m was tried first: it settled a 0.20 m offset only
                           after ~2.3 m of travel, longer than the 1.6 m between W1 and W2. 0.05 m halves
                           that (~1.2 m); 0.03 m is faster still but asks for > 35 deg of heading error.
      e_theta_max 10 deg   a noticeable misalignment; bigger and the camera starts to lose the markers.
      d_omega_max 0.5 rad/s the feedback correction we are willing to spend, well below the 1.82 rad/s
                           limit so normal errors never saturate the wheels.
    """
    c = (cfg or load_control_config())["lqr"]
    Q = bryson(c["e_y_max"], np.deg2rad(c["e_theta_max_deg"]))
    R = bryson(c["d_omega_max"])
    return Q, R


def design_lqr(v=V_NOM, kappa=0.0, Ts=TS, cfg=None):
    """LQR gain K (1x2) on the error model at speed v and curvature kappa.

    The baseline design is at the cruise speed on a straight segment (kappa = 0), and the SAME gain
    is then used everywhere (Assessment 3 approach). Redesigning K for the current v and kappa is
    gain scheduling: the HD extension, compared against this fixed gain.
    """
    Ad, Bd = error_model(v, kappa, Ts)
    Q, R = design_weights(cfg)
    K, _, eig = dlqr(Ad, Bd, Q, R)
    return K, eig


def design_lqi(v=V_NOM, Ts=TS, cfg=None):
    """LQI gain K_i (1x3) on the error model augmented with z[k+1] = z[k] + Ts * e_y[k]."""
    cfg = cfg or load_control_config()
    Ad_i, Bd_i = integral_augmented(*error_model(v, 0.0, Ts), Ts)
    Q, R = design_weights(cfg)
    Q_i = np.block([[Q,                np.zeros((2, 1))],
                    [np.zeros((1, 2)), bryson(cfg["lqi"]["z_max"])]])
    K_i, _, eig = dlqr(Ad_i, Bd_i, Q_i, R)
    return K_i, eig


# ---------------------------------------------------------------- the controller
class PathLQR:
    """Steering controller: omega = v*kappa - K e (LQR), or with an e_y integrator (LQI).

    Call reset() when a new segment or a new run starts, then step() once every Ts seconds.
    """

    def __init__(self, K=None, integral=None, Ts=TS, omega_max=OMEGA_MAX, cfg=None):
        cfg = cfg or load_control_config()
        self.integral = cfg["lqi"]["enabled"] if integral is None else integral
        if K is None:
            K = design_lqi(Ts=Ts, cfg=cfg)[0] if self.integral else design_lqr(Ts=Ts, cfg=cfg)[0]
        self.K = np.asarray(K, dtype=float).reshape(1, -1)
        self.Ts, self.omega_max = Ts, omega_max
        self.z = 0.0                       # integrator state (LQI only)

    def reset(self):
        """Zero the integrator (a fresh start must not carry an old offset)."""
        self.z = 0.0

    def step(self, e_y, e_theta, kappa, v):
        """Yaw-rate command [rad/s] for the current errors to the path.

        e_y, e_theta: errors of the ESTIMATED pose to the current segment (waypoints.py computes them,
                      e_theta already wrapped to (-pi, pi]).
        kappa:        curvature of that segment (0 for a straight line).
        v:            the forward speed that will be sent with this omega.
        """
        e = [e_y, e_theta, self.z] if self.integral else [e_y, e_theta]
        # 1. Feedforward v*kappa keeps the robot on a curved path with zero error,
        #    feedback -K e corrects the error (sign: omega = v*kappa MINUS K e).
        omega_raw = v * kappa - (self.K @ np.asarray(e)).item()
        # 2. Clip to what the wheels can do (Waffle Pi 1.82 rad/s).
        omega = float(np.clip(omega_raw, -self.omega_max, self.omega_max))
        # 3. Anti-windup (Week 6): only integrate while the command is NOT clipped, otherwise z grows
        #    during saturation (e.g. starting far off the path) and causes a big overshoot afterwards.
        #    Same increment Ts * e_y as in the augmented model, so the code matches the design.
        if self.integral and omega == omega_raw:
            self.z += self.Ts * e_y
        return omega


# ---------------------------------------------------------------- design report
def _linear_response(K, v, kappa=0.0, e0=(0.20, 0.0), t_end=30.0, Ts=TS):
    """Closed-loop LINEAR error model from e0: returns time, e history, d_omega history.

    A quick check of the tuning before driving the robot: how fast an error decays and how much
    steering that costs.
    """
    Ad, Bd = error_model(v, kappa, Ts)
    e, E, U = np.array(e0, float), [], []
    for _ in range(int(t_end / Ts)):
        u = -(K @ e).item()
        E.append(e.copy()); U.append(u)
        e = Ad @ e + Bd[:, 0] * u
    return Ts * np.arange(len(E)), np.array(E), np.array(U)


if __name__ == "__main__":
    # Run from the project root:  python -m src.control.lqr
    np.set_printoptions(precision=4, suppress=True)
    Q, R = design_weights()
    print(f"Bryson weights: Q = diag({Q[0, 0]:.1f}, {Q[1, 1]:.1f}), R = {R[0, 0]:.2f}")

    # 1. The baseline gain, designed at cruise speed on a straight segment
    K, eig = design_lqr()
    print(f"\nK (v = {V_NOM} m/s, straight) = {K.ravel()}   [rad/s per m, rad/s per rad]")
    print(f"|closed-loop eig|, straight      = {np.abs(eig)}   (all < 1: stable)")

    # 2. The same fixed gain on an arc and at lower speeds (what it will meet on the track)
    for v, kappa, what in [(V_NOM, 2.0, "arc R = 0.5 m"), (0.10, 0.0, "v = 0.10 m/s"), (0.05, 0.0, "v = 0.05 m/s")]:
        Ad, Bd = error_model(v, kappa)
        print(f"|closed-loop eig|, {what:13s} = {np.abs(np.linalg.eigvals(Ad - Bd @ K))}")

    # 3. Time response from a 0.20 m sideways offset: settling time and steering cost
    for v in [V_NOM, 0.10]:
        t, E, U = _linear_response(K, v)
        outside = np.nonzero(np.abs(E[:, 0]) > 0.02 * 0.20)[0]          # 2% settling band
        t_settle = t[min(outside[-1] + 1, len(t) - 1)] if len(outside) else 0.0
        print(f"\nfrom e_y = 0.20 m at v = {v:.2f} m/s: 2% settling {t_settle:.1f} s "
              f"(~{v * t_settle:.2f} m of travel), peak |d_omega| {np.abs(U).max():.2f} rad/s, "
              f"peak |e_theta| {np.rad2deg(np.abs(E[:, 1]).max()):.0f} deg, "
              f"overshoot {max(0.0, -E[:, 0].min()) * 100:.1f} cm")

    # 4. LQI, only used if a steady offset shows up in the simulator
    K_i, eig_i = design_lqi()
    print(f"\nLQI K_i = {K_i.ravel()}, |closed-loop eig| = {np.abs(eig_i)}")
