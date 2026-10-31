"""Quick pass/fail checks of the LQR at the three control weights of evaluation/sweep_lqr.py (no simulator).

Run from the project root:   python -m src.control.lqr_weights_test

For R x10, R (config) and R / 10 (Q fixed, R = 1 / d_omega_max^2 scaled): the repo's scipy dlqr gives the
same K as control.dlqr (the tutorials' toolbox), the discrete Riccati residual is ~0, the closed loop is
stable at v_nom and at 0.05 m/s, and K behaves as LQR theory says (smaller R -> larger gains).
The exit code is the number of failures (check_rules reads it).

Adapted from 41014 Week 6 A.4 - A.5 (ct.dlqr, Riccati residual, "only the ratio of Q and R matters").
Additions: the check on the repo's own error model and Bryson weights at the three sweep weights.
Nothing here reads ground truth.
"""
import copy
import sys
from pathlib import Path

import control as ct
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # project root, so "src" imports work

from src.control.lqr import design_weights, dlqr  # noqa: E402
from src.control.models import V_NOM, error_model, load_control_config  # noqa: E402

FAILS = []
SCALES = [("R x10", 10.0), ("R (config)", 1.0), ("R / 10", 0.1)]   # same as evaluation/sweep_lqr.py


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"   ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


Ad, Bd = error_model(V_NOM, 0.0)
Ad_slow, Bd_slow = error_model(0.05, 0.0)
gains = []
for label, scale in SCALES:
    cfg = copy.deepcopy(load_control_config())
    cfg["lqr"]["d_omega_max"] /= np.sqrt(scale)
    Q, R = design_weights(cfg)
    K, P, eig = dlqr(Ad, Bd, Q, R)
    K_ct, P_ct, _ = ct.dlqr(Ad, Bd, Q, R)
    err = np.abs(K - K_ct).max()
    check(f"{label}: K matches control.dlqr", err < 1e-8, f"K = {K.ravel().round(4)}, max diff {err:.1e}")
    res = Ad.T @ P @ Ad - P - Ad.T @ P @ Bd @ np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad) + Q
    check(f"{label}: Riccati residual ~0", np.abs(res).max() < 1e-8 * np.abs(P).max(), f"{np.abs(res).max():.1e}")
    check(f"{label}: |eig(Ad - Bd K)| < 1 at v_nom", bool(np.all(np.abs(eig) < 1)), f"{np.abs(eig).round(4)}")
    eig_slow = np.abs(np.linalg.eigvals(Ad_slow - Bd_slow @ K))
    check(f"{label}: |eig| < 1 at 0.05 m/s", bool(np.all(eig_slow < 1)), f"{eig_slow.round(4)}")
    gains.append(K.ravel())

gains = np.array(gains)
check("smaller R gives larger gains on both e_y and e_theta",
      bool(np.all(np.diff(gains[:, 0]) > 0) and np.all(np.diff(gains[:, 1]) > 0)), f"K_y {gains[:, 0].round(3)}")

print(f"\n{len(FAILS)} failed" + (": " + ", ".join(FAILS) if FAILS else " - all checks passed"))
sys.exit(len(FAILS))                                             # check_rules reads the exit code
