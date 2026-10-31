"""Quick pass/fail checks of the observability claims (no simulator). Run: python -m src.estimation.observability_test

Each check prints PASS or FAIL; the exit code is the number of failures (check_rules reads it).
The ranks are computed on the EKF's own F_jac and H_rb at a pose on the lab lane, with the lab markers
1 at (0, 1.75) and 2 at (0, 0.75) (config/arena_lab.json).

Adapted from 41014 Week 5 C4 (ct.obsv rank), Week 7 C.4 (num_jac against the analytic Jacobians) and
Week 8 A.8 (pbh_detectable).
Additions: the time-varying rank along a driven path, range-only rows, the rotation-about-the-marker
null direction, predict-only P growth.
"""
import sys
from pathlib import Path

import control as ct
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # project root, so "src" imports work

from src.estimation.models import F_jac, H_rb, h_rb, load_estimation_config, numerical_jacobian  # noqa: E402
from src.estimation.observability import (H_stack, obsv, obsv_tv, pbh_detectable, predict_only,  # noqa: E402
                                          rank_and_sv, steps_along)
from src.estimation.odometry import move  # noqa: E402

FAILS = []
M1, M2 = np.array([0.0, 1.75]), np.array([0.0, 0.75])
POSE = np.array([1.5, 1.25, np.pi])          # on the lab lane, facing the left end panel (-x)
TS, V = 0.05, 0.2                            # s, m/s: control period and cruise speed (config/control.json)
N = 20                                       # steps = 1 s of frames


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"   ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def null_dir(O):
    """Unit vector of the weakest direction of O (last right singular vector)."""
    return np.linalg.svd(O)[2][-1]


def rotation_about(pose, mk):
    """Direction in which turning the pose about marker mk changes nothing: [-(y - my), x - mx, 1]."""
    d = np.array([-(pose[1] - mk[1]), pose[0] - mk[0], 1.0])
    return d / np.linalg.norm(d)


cfg = load_estimation_config()

# 1. The EKF's own Jacobians against central differences (Week 7 C.4)
ds, dth = V * TS, 0.3 * TS
err_F = np.abs(F_jac(POSE, ds, dth) - numerical_jacobian(lambda p: move(p, ds, dth), POSE)).max()
check("F_jac matches the numerical Jacobian", err_F < 1e-6, f"max error {err_F:.1e}")
err_H = np.abs(H_rb(POSE, M2) - numerical_jacobian(lambda p: h_rb(p, M2), POSE)).max()
check("H_rb matches the numerical Jacobian", err_H < 1e-6, f"max error {err_H:.1e}")

# 2. obsv() is ct.obsv (Week 5 C4)
H = H_rb(POSE, M2)
check("obsv(F, H) equals control.obsv", np.allclose(obsv(np.eye(3), H), ct.obsv(np.eye(3), H)))

# 3. (a) no marker in view: nothing observable, and not detectable (integrators at z = 1)
check("(a) no marker: rank 0", rank_and_sv(H_stack(POSE, []))[0] == 0)
check("(a) no marker: not detectable (PBH, lambda = 1)", not pbh_detectable(np.eye(3), np.zeros((0, 3)))[0])

# 4. (b) one marker, range + bearing, robot still: 2 constraints on 3 DOF
O_b = obsv(np.eye(3), H)
check("(b) one marker, still: rank 2 of 3", np.linalg.matrix_rank(O_b) == 2)
check("(b) the hidden direction is the rotation about the marker",
      abs(null_dir(O_b) @ rotation_about(POSE, M2)) > 1 - 1e-9)

# 5. (c) one marker, moving: still rank 2, the same rotation stays hidden (straight and on an arc)
for w, what in [(0.0, "straight"), (0.5, "arc w = 0.5 rad/s")]:
    poses, inc = steps_along(POSE, V, w, TS, N)
    O_c, _ = obsv_tv(poses, inc, [[M2]] * N)
    r, _ = rank_and_sv(O_c)
    check(f"(c) one marker, 1 s {what}: rank 2 of 3", r == 2, f"rank {r}")
    check(f"(c) {what}: hidden direction = rotation about the marker",
          abs(null_dir(O_c) @ rotation_about(POSE, M2)) > 1 - 1e-6)

# 6. (d) range only (Assessment 3 filter): rank 1 still, rank 2 moving
check("(d) range only, still: rank 1", np.linalg.matrix_rank(H_stack(POSE, [M2], bearing=False)) == 1)
poses, inc = steps_along(POSE, V, 0.0, TS, N)
O_d, _ = obsv_tv(poses, inc, [[M2]] * N, bearing=False)
check("(d) range only, moving 1 s: rank 2", rank_and_sv(O_d)[0] == 2, f"rank {rank_and_sv(O_d)[0]}")

# 7. (e) two markers in view: rank 3 at one instant, detectable
H_e = H_stack(POSE, [M1, M2])
check("(e) markers 1 and 2: rank 3 at one instant", np.linalg.matrix_rank(H_e) == 3)
check("(e) markers 1 and 2: detectable", pbh_detectable(np.eye(3), H_e)[0])

# 8. one marker after the other also gives rank 3 (the switch breaks the rotation)
O_s, _ = obsv_tv(poses, inc, [[M2]] * (N // 2) + [[M1]] * (N - N // 2))
check("marker 2 then marker 1 (one at a time): rank 3", rank_and_sv(O_s)[0] == 3)

# 9. Marker lost: predict only, P grows every step
sx, sy, sth = cfg["P0_sd"]
t, Ps = predict_only(np.diag([sx**2, sy**2, np.deg2rad(sth)**2]), POSE, V, 0.0, TS, 10.0, cfg["k_wheel"], cfg["q_floor"])
tr = np.trace(Ps, axis1=1, axis2=2)
check("marker lost: trace(P) grows at every predict", bool(np.all(np.diff(tr) > 0)),
      f"{tr[0]:.2e} -> {tr[-1]:.2e} in {t[-1]:.0f} s")

print(f"\n{len(FAILS)} failed" + (": " + ", ".join(FAILS) if FAILS else " - all checks passed"))
sys.exit(len(FAILS))                                             # check_rules reads the exit code
