"""Observability of the EKF's pose from the camera's markers: rank, Gramian and P growth without markers.

Adapted from 41014 Week 5 C4 (ct.obsv and its rank: which sensors make the state observable), Week 7
A.2 (unobservable directions from the SVD of O) and D.5 (range-only localisation), and Week 8 A.7 -
A.8 (singular values / Gramian: HOW observable; pbh_detectable).
Additions: the time-varying observability matrix of the EKF's own linearisation (F_jac, H_rb in
models.py) along a driven path, range-only rows, the Rn-weighted Gramian, and the predict-only P growth
when every marker is lost.

The EKF state is x = [x, y, theta] in {W} (m, rad). One step of the EKF's linearisation:

    x[k+1] ~ F_k x[k]            F_k = F_jac(xh_k, ds_k, dth_k)          (motion, models.py)
    z_k    ~ H_k x[k]            H_k = [H_rb(xh_k, m_i) for markers m_i in view]   (2 rows per marker)

Time-invariant case (robot still, F = I), Week 5 C4:   O = [H; H F; H F^2] = ct.obsv(F, H)
Time-varying case (robot moving), over steps 0 .. N-1:

    O_N = [ H_0 ; H_1 F_0 ; H_2 F_1 F_0 ; ... ; H_{N-1} F_{N-2} ... F_0 ]

rank(O_N) = 3 means the whole pose at step 0 can be reconstructed from those N frames of measurements.
Week 8 A.7 grades how well: the singular values of O_N (smallest = weakest direction). With the
measurement noise included,

    W_N = sum_k Phi_k^T H_k^T Rn_k^-1 H_k Phi_k,   Phi_k = F_{k-1} ... F_0      (information, 1/m^2 and 1/rad^2)

and 1 / sqrt(lambda_min(W_N)) is the best sd any estimator can reach in the weakest direction from
those frames alone (no prior, no process noise).

One marker, range and bearing, robot still: H_rb has 2 independent rows on 3 states, so rank 2: the
robot can sit anywhere on the circle of radius r around the marker, each spot with the heading that
gives the same bearing. Driving does NOT add the third constraint: odometry only measures motion in
the robot's own frame, so turning the whole run rigidly about the marker (position and heading by the
same angle) changes no range and no bearing. The unobservable direction of O_N is that rotation,
[-(y - my), x - mx, 1] (normalised). A second marker, in the same frame or a later one, removes it.

Marker lost: no H rows, so only predict runs, P <- F P F^T + Qd, and P grows with the distance driven
(Qd_wheels). The modes of F are all at z = 1 (integrators): with no measurement they are not detectable
(Week 8 A.8), so P grows without bound.

Pure math on the EKF's own models: no simulator, nothing here reads ground truth.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # project root, so "src" imports work even
                                                                # when this file is run directly (VS Code Run)

from src.estimation.models import F_jac, H_rb, Qd_wheels, R_marker, h_rb  # noqa: E402
from src.estimation.odometry import move  # noqa: E402
from src.robot.sim_robot import WHEEL_SEPARATION  # noqa: E402


def obsv(F, H):
    """Observability matrix [H; H F; H F^2] of a time-invariant pair (what control.obsv returns, Week 5 C4)."""
    n = F.shape[0]
    return np.vstack([H @ np.linalg.matrix_power(F, k) for k in range(n)])


def pbh_detectable(F, H, tol=1e-8):
    """Discrete PBH test (Week 8 A.8): True iff rank [F - lambda I; H] = n for every |lambda| >= 1."""
    n = F.shape[0]
    for lam in np.linalg.eigvals(F):
        if abs(lam) >= 1.0 - tol:
            if H.size == 0 or np.linalg.matrix_rank(np.vstack([F - lam * np.eye(n), H])) < n:
                return False, lam
    return True, None


def H_range(pose, mk):
    """Range-only row dh/dx (Week 7 D.5, the Assessment 3 filter): the first row of H_rb."""
    return H_rb(pose, mk)[:1]


def H_stack(pose, markers, bearing=True):
    """H rows for every marker map position in markers at this pose (0 rows when none is in view)."""
    rows = [H_rb(pose, mk) if bearing else H_range(pose, mk) for mk in markers]
    return np.vstack(rows) if rows else np.zeros((0, 3))


def R_stack(pose, markers, cfg, bearing=True):
    """Block-diagonal Rn for the same rows as H_stack (range sd a + b r^2, bearing sd from cfg)."""
    blocks = []
    for mk in markers:
        Rm = R_marker(h_rb(pose, mk)[0], cfg)
        blocks.append(Rm if bearing else Rm[:1, :1])
    n = sum(b.shape[0] for b in blocks)
    R = np.zeros((n, n))
    i = 0
    for b in blocks:
        R[i:i + b.shape[0], i:i + b.shape[0]] = b
        i += b.shape[0]
    return R


def steps_along(pose0, v, w, Ts, N):
    """N poses of the EKF's motion model from pose0 at constant (v [m/s], w [rad/s]), and each step's (ds, dth)."""
    poses, inc = [np.asarray(pose0, float)], []
    for _ in range(N - 1):
        ds, dth = v * Ts, w * Ts
        inc.append((ds, dth))
        poses.append(move(poses[-1], ds, dth))
    return np.array(poses), inc


def obsv_tv(poses, inc, markers_per_step, bearing=True, cfg=None):
    """Time-varying observability matrix O_N and (if cfg is given) the Rn-weighted Gramian W_N.

    poses: N x 3 linearisation poses; inc: N-1 (ds, dth) between them; markers_per_step: N lists of map
    positions (x, y) [m] in view at each pose. Returns (O, W) with W None when cfg is None."""
    Phi = np.eye(3)
    rows, W = [], np.zeros((3, 3))
    for k, pose in enumerate(poses):
        if k > 0:
            ds, dth = inc[k - 1]
            Phi = F_jac(poses[k - 1], ds, dth) @ Phi
        H = H_stack(pose, markers_per_step[k], bearing)
        if H.size:
            rows.append(H @ Phi)
            if cfg is not None:
                W += Phi.T @ H.T @ np.linalg.solve(R_stack(pose, markers_per_step[k], cfg, bearing), H) @ Phi
    O = np.vstack(rows) if rows else np.zeros((0, 3))
    return O, (W if cfg is not None else None)


def rank_and_sv(O, tol=1e-9):
    """Rank and smallest singular value of O relative to the largest (0 rows -> rank 0, sv 0)."""
    if O.size == 0:
        return 0, 0.0
    s = np.linalg.svd(O, compute_uv=False)
    rank = int(np.sum(s > tol * max(s[0], 1.0)))
    return rank, float(s[-1]) if len(s) == 3 else 0.0


def weakest_sd(W):
    """Best reachable sd in the weakest direction, 1 / sqrt(lambda_min(W)) [m or rad]; inf if W is singular."""
    lam = np.linalg.eigvalsh(W)
    return float(1.0 / np.sqrt(lam[0])) if lam[0] > 1e-9 * max(lam[-1], 1e-300) else float("inf")


def predict_only(P0, pose0, v, w, Ts, T, k_wheel, q_floor=1e-8):
    """Marker lost: P after the EKF's predict alone (P <- F P F^T + Qd) at constant (v, w) for T seconds.

    Returns (t [s], P history N x 3 x 3). Wheel distances per step: d_L, d_R = ds -/+ dth b / 2."""
    pose, P = np.asarray(pose0, float), np.asarray(P0, float)
    t, Ps = [0.0], [P.copy()]
    for k in range(int(round(T / Ts))):
        ds, dth = v * Ts, w * Ts
        d_l, d_r = ds - dth * WHEEL_SEPARATION / 2, ds + dth * WHEEL_SEPARATION / 2
        F = F_jac(pose, ds, dth)
        P = F @ P @ F.T + Qd_wheels(pose, d_l, d_r, dth, k_wheel, q_floor)
        pose = move(pose, ds, dth)
        t.append((k + 1) * Ts)
        Ps.append(P.copy())
    return np.array(t), np.array(Ps)


def converged_P(P0, pose, markers, cfg, Ts, T=5.0):
    """P after T seconds standing at pose with the given markers in view every step (predict + update).

    The covariance update does not depend on the measured values, only on H and Rn, so this is the
    P the EKF would report there. Used as the starting P for the marker-lost case."""
    P = np.asarray(P0, float)
    H = H_stack(pose, markers)
    R = R_stack(pose, markers, cfg)
    for _ in range(int(round(T / Ts))):
        P = P + Qd_wheels(pose, 0.0, 0.0, 0.0, cfg["k_wheel"], cfg["q_floor"])
        if H.size:
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            P = (np.eye(3) - K @ H) @ P
            P = (P + P.T) / 2
    return P


if __name__ == "__main__":
    # Run from the project root:  python -m src.estimation.observability
    np.set_printoptions(precision=4, suppress=True)
    mk = np.array([0.0, 0.75])
    pose = np.array([1.5, 1.25, np.pi])
    H = H_rb(pose, mk)
    print("one marker, robot still: rank O =", np.linalg.matrix_rank(obsv(np.eye(3), H)), "of 3")
    poses, inc = steps_along(pose, 0.2, 0.0, 0.05, 20)
    O, _ = obsv_tv(poses, inc, [[mk]] * 20)
    print("one marker, 1 s at 0.2 m/s: rank O =", rank_and_sv(O)[0], "of 3 (rotation about the marker)")
    O, _ = obsv_tv(poses, inc, [[mk]] * 10 + [[np.array([0.0, 1.75])]] * 10)
    print("marker 2 for 0.5 s, then marker 1: rank O =", rank_and_sv(O)[0], "of 3")
