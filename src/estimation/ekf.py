"""Range-bearing EKF: wheel odometry predicts the pose, ArUco markers with known positions correct it.

Adapted from Week 7 C.3 / C.4 (ekf_predict, ekf_update, h_rb, H_rb) and Assessment 3 Task B.3 (one
measurement at a time, wrapped residuals, symmetric P, NIS for every update).

    PREDICT  once per NEW encoder sample (the first sample only sets the baseline):
             x <- f(x, wheel increments)            odometry.move()
             P <- F P F^T + Qd                      models.F_jac(), models.Qd_wheels()
    UPDATE   once per NEW camera frame, one marker at a time, known IDs only:
             nu = [r - r_hat, wrap(b - b_hat)]      residual, bearing wrapped
             S  = H P H^T + R                       innovation covariance
             NIS = nu^T S^-1 nu  > nis_gate  ->  skip this marker (outlier, e.g. a bad PnP solution)
             K  = P H^T S^-1,  x <- x + K nu,  P <- (I - K H) P,  P <- (P + P^T) / 2

Team interface (team plan "Runtime interface"):
    ekf = EKF(x0, P0, marker_map, params)
    xh, P = ekf.step(encoders, marker_frame=None)   encoders = (t, phi_L, phi_R) [s, rad]
    ekf.last_update_ids                              marker IDs fused in this step

marker_frame = {"t": t_frame, "markers": {id: {"z": [range, bearing], ...}}} from perception. An empty
markers dict is fine. A frame is fused once (repeated frames are ignored), and a frame older than
max_frame_age compared with the encoder time is NOT fused, rather than fusing it at the wrong time.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # project root, so "src" imports work even
                                                                # when this file is run directly (VS Code Run)

from src.estimation.models import F_jac, H_rb, Qd_wheels, R_marker, h_rb, load_estimation_config  # noqa: E402
from src.estimation.odometry import increments, move, wrap  # noqa: E402


class EKF:
    def __init__(self, x0, P0=None, marker_map=None, params=None):
        self.p = params or load_estimation_config()
        if P0 is None:                                  # start uncertainty from the config
            sx, sy, sth = self.p["P0_sd"]
            P0 = np.diag([sx ** 2, sy ** 2, np.deg2rad(sth) ** 2])
        self.xh, self.P = np.array(x0, float), np.array(P0, float)
        # known map: id -> (x, y); keys starting with "_" are comments in the task file
        self.map = {int(i): np.array(m[:2], float) for i, m in (marker_map or {}).items() if not str(i).startswith("_")}
        self.enc_prev = None                            # last (t, phi_L, phi_R)
        self.t_frame = -np.inf                          # time of the last camera frame fused
        self.last_update_ids = []
        self.log = []                                   # (t, id, NIS, used) for every marker seen

    # ------------------------------------------------------------ predict
    def predict(self, dphi_l, dphi_r):
        """Move the estimate with the wheel increments and grow P by the wheel noise."""
        d_l, d_r, ds, dth = increments(dphi_l, dphi_r)
        F = F_jac(self.xh, ds, dth)                     # Jacobians at the estimate BEFORE moving
        Qd = Qd_wheels(self.xh, d_l, d_r, dth, self.p["k_wheel"], self.p["q_floor"])
        self.xh = move(self.xh, ds, dth)
        self.P = F @ self.P @ F.T + Qd

    # ------------------------------------------------------------ update
    def update(self, marker_id, z, t=None):
        """Fuse one marker's [range, bearing]. Returns its NIS; skipped (not fused) above the gate."""
        mk = self.map[marker_id]
        # ArUco range reads long by about c * r^2 (bias in results/aruco_noise.csv): remove it first
        z = np.array([z[0] - self.p.get("range_bias_r2", 0.0) * z[0] ** 2, z[1]])
        H = H_rb(self.xh, mk)                           # recomputed at the CURRENT estimate
        S = H @ self.P @ H.T + R_marker(z[0], self.p)
        z_hat = h_rb(self.xh, mk)
        nu = np.array([z[0] - z_hat[0], wrap(z[1] - z_hat[1])])
        nis = float(nu @ np.linalg.solve(S, nu))
        used = nis <= self.p["nis_gate"]
        self.log.append((t, marker_id, nis, used))
        if used:
            K = self.P @ H.T @ np.linalg.inv(S)
            self.xh = self.xh + K @ nu
            self.xh[2] = wrap(self.xh[2])
            self.P = (np.eye(3) - K @ H) @ self.P
            self.P = (self.P + self.P.T) / 2            # keep P symmetric against round-off
            self.last_update_ids.append(marker_id)
        return nis

    # ------------------------------------------------------------ one control-loop step
    def step(self, encoders, marker_frame=None):
        t, phi_l, phi_r = encoders
        self.last_update_ids = []
        if self.enc_prev is None:                       # first sample: baseline only
            self.enc_prev = (t, phi_l, phi_r)
        elif t > self.enc_prev[0]:                      # new encoder sample: predict once
            self.predict(phi_l - self.enc_prev[1], phi_r - self.enc_prev[2])
            self.enc_prev = (t, phi_l, phi_r)
        if (marker_frame is not None and marker_frame["t"] > self.t_frame
                and t - marker_frame["t"] <= self.p["max_frame_age"]):
            self.t_frame = marker_frame["t"]            # new, recent frame: each marker once
            for mid, m in marker_frame["markers"].items():
                z = np.asarray(m["z"], float)
                if int(mid) in self.map and z.shape == (2,) and np.all(np.isfinite(z)):
                    self.update(int(mid), z, marker_frame["t"])
        return self.xh.copy(), self.P.copy()


# The whole-mission check against simulator truth lives in evaluation/check_ekf.py, because only
# evaluation/ may read ground truth.
