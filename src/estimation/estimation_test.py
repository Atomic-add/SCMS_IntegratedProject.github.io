"""Quick pass/fail checks for src/estimation (no simulator, runs in a second).

Run from the project root:   python -m src.estimation.estimation_test      (or the VS Code Run button)

Each check prints PASS or FAIL with what it tested; the exit code is the number of failures (0 = all
passed). The checks are the team plan's list: baseline, straight and turning motion, angle wrapping,
Jacobians, missing / unknown / repeated / old markers, the correction itself, the gate, and P's health.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # project root, so "src" imports work even
                                                                # when this file is run directly (VS Code Run)
from src.estimation.ekf import EKF  # noqa: E402
from src.estimation.models import F_jac, H_rb, h_rb, increments, move, numerical_jacobian, wrap  # noqa: E402
from src.robot.sim_robot import WHEEL_RADIUS, WHEEL_SEPARATION  # noqa: E402

MAP = {"0": [3.6, 4.599, 0.12, -np.pi / 2], "1": [4.199, 0.0, 0.12, np.pi], "2": [2.95, 4.699, 0.12, -np.pi / 2],
       "_doc": "comment"}
FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"   ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def new_ekf(x0=(3.6, 3.9, np.pi / 2)):
    ekf = EKF(x0, marker_map=MAP)
    ekf.step((0.0, 0.0, 0.0))                       # baseline
    return ekf


def frame(t, markers):
    return {"t": t, "markers": {i: {"z": np.array(z, float)} for i, z in markers.items()}}


A = 1.0 / WHEEL_RADIUS                               # wheel angle for 1 m
SPIN = (np.pi / 2) * WHEEL_SEPARATION / 2 / WHEEL_RADIUS   # wheel angle for a 90 deg spin

# ---------------------------------------------------------------- odometry
ekf = EKF([1.0, 2.0, 0.3], marker_map=MAP)
xh, _ = ekf.step((0.0, 5.0, 7.0))
check("baseline: the first encoder sample does not move the estimate", np.allclose(xh, [1.0, 2.0, 0.3]))
ekf = new_ekf((0.0, 0.0, 0.0))
xh, _ = ekf.step((1.0, A, A))
check("straight: equal wheel turns move 1 m along x, heading unchanged", np.allclose(xh, [1.0, 0.0, 0.0], atol=1e-9))
xh, _ = ekf.step((2.0, A - SPIN, A + SPIN))
check("turn: opposite wheel turns spin +90 deg on the spot", np.allclose(xh, [1.0, 0.0, np.pi / 2], atol=1e-9))
xh2, _ = ekf.step((2.0, A + 5, A + 5))
check("repeated encoder timestamp: no second prediction", np.allclose(xh2, xh))

# ---------------------------------------------------------------- angles and Jacobians
check("wrap: 179 deg - (-179 deg) is -2 deg, not 358 deg",
      np.isclose(wrap(np.deg2rad(179) - np.deg2rad(-179)), np.deg2rad(-2)))
z_hat = h_rb([0.0, 0.0, np.deg2rad(179)], [-1.0, -0.01])
check("bearing residual is wrapped near +-180 deg", abs(z_hat[1]) < np.deg2rad(3), f"bearing {np.rad2deg(z_hat[1]):.2f} deg")
pose, (_, _, ds, dth) = np.array([1.0, 0.5, 0.3]), increments(1.4, 1.7)
check("F matches finite differences", np.allclose(F_jac(pose, ds, dth), numerical_jacobian(lambda p: move(p, ds, dth), pose), atol=1e-6))
mk = np.array([3.6, 4.599])
check("H matches finite differences", np.allclose(H_rb(pose, mk), numerical_jacobian(lambda p: h_rb(p, mk), pose), atol=1e-6))

# ---------------------------------------------------------------- missing, unknown, repeated and old markers
ekf = new_ekf()
x0 = ekf.xh.copy()
for name, f in [("empty markers dict", frame(0.0, {})), ("unknown ID 7", frame(0.0, {7: [0.7, 0.0]})),
                ("nan measurement", frame(0.0, {0: [np.nan, 0.0]}))]:
    ekf.t_frame = -np.inf
    xh, _ = ekf.step((0.0, 0.0, 0.0), f)
    check(f"missing data: {name} leaves the estimate unchanged", np.allclose(xh, x0) and ekf.last_update_ids == [])

z_true = h_rb([3.6, 3.9, np.pi / 2], MAP["0"])
ekf = new_ekf((3.55, 3.85, np.pi / 2 + 0.03))       # 7 cm and 2 deg off
f = frame(0.05, {0: z_true})
ekf.step((0.05, 0.0, 0.0), f)
first = ekf.last_update_ids
ekf.step((0.10, 0.0, 0.0), f)
check("repeated frame: fused once only", first == [0] and ekf.last_update_ids == [])
ekf = new_ekf((3.55, 3.85, np.pi / 2))
ekf.step((1.5, 0.0, 0.0), frame(0.5, {0: z_true}))
check("old frame (1 s behind the encoders): not fused", ekf.last_update_ids == [])

# ---------------------------------------------------------------- the correction and the gate
# The robot stands in the dock area facing the charger, where the camera sees markers 0 and 2 (both
# inside the +-31 deg horizontal field of view). The fake measurements read long by the same r^2 bias
# the EKF subtracts (config range_bias_r2), as real ArUco ranges do. The start is 11 cm and 3 deg off,
# about 2 sigma of P0: a placement error the prior admits. (The old version of this test started 5 sigma
# outside a 2 cm / 2 deg P0, fed unbiased ranges, and used marker 1, which is BEHIND the robot here.)
TRUE = np.array([3.6, 3.3, np.pi / 2])
bias = EKF(TRUE, marker_map=MAP).p["range_bias_r2"]


def seen(mid):
    """[range, bearing] of a marker from TRUE, with the ArUco range bias added."""
    z = h_rb(TRUE, MAP[mid])
    return z + [bias * z[0] ** 2, 0.0]


fov_ok = all(abs(np.degrees(h_rb(TRUE, MAP[m])[1])) < 31.0 for m in ("0", "2"))
ekf = new_ekf((3.50, 3.25, np.pi / 2 - 0.05))       # 11 cm and 3 deg off
for k in range(1, 30):
    ekf.step((0.05 * k, 0.0, 0.0), frame(0.05 * k, {0: seen("0"), 2: seen("2")}))
err = np.hypot(*(ekf.xh[:2] - TRUE[:2]))
err_th = np.degrees(abs(wrap(ekf.xh[2] - TRUE[2])))
fused = sum(used for *_, used in ekf.log)
check("correction: 11 cm / 3 deg off, markers 0 and 2 in view -> below 1 cm and 0.5 deg",
      fov_ok and err < 0.01 and err_th < 0.5, f"{100 * err:.2f} cm, {err_th:.2f} deg, {fused}/{len(ekf.log)} fused")
ekf = new_ekf()
ekf.step((0.05, 0.0, 0.0), frame(0.05, {0: z_true + [1.0, 0.0]}))
check("gate: a 1 m range outlier is rejected", ekf.last_update_ids == [] and ekf.log[-1][2] > ekf.p["nis_gate"],
      f"NIS {ekf.log[-1][2]:.0f}")

# ---------------------------------------------------------------- P stays healthy
rng = np.random.default_rng(0)
ekf = new_ekf((2.0, 2.0, 0.0))
phi = np.zeros(2)
for k in range(1, 1001):
    phi += rng.uniform(-0.5, 1.5, 2)
    pose_guess = ekf.xh
    f = frame(0.05 * k, {0: h_rb(pose_guess, MAP["0"]) + rng.normal(0, [0.01, 0.01])}) if k % 3 == 0 else None
    xh, P = ekf.step((0.05 * k, *phi), f)
eig = np.linalg.eigvalsh(P)
check("1000 random steps: x_hat finite, P symmetric, eigenvalues >= 0",
      np.all(np.isfinite(xh)) and np.allclose(P, P.T) and eig.min() >= -1e-12, f"min eig {eig.min():.1e}")

# ---------------------------------------------------------------- the camera file picks the marker noise
from src.estimation.models import load_estimation_config  # noqa: E402
from src.robot.sim_robot import PROJECT_ROOT  # noqa: E402
base = load_estimation_config()
sim = load_estimation_config(camera=PROJECT_ROOT / "config" / "camera_sim.json")
lab = load_estimation_config(camera=PROJECT_ROOT / "config" / "camera_sim_lab.json")
real = load_estimation_config(camera=PROJECT_ROOT / "config" / "camera_real.json")
check("config: the URDF sim camera keeps estimation.json", sim == base)
noise = {k: v for k, v in __import__("json").loads((PROJECT_ROOT / "config" / "camera_sim_lab.json").read_text())
         .get("ekf_noise", {}).items() if not k.startswith("_")}
check("config: the lab sim camera's ekf_noise replaces range bias, range sd and bearing sd",
      bool(noise) and all(lab[k] == v for k, v in noise.items()) and lab["k_wheel"] == base["k_wheel"], str(noise))
check("config: the real camera's range_correction drops the sim bias (never subtracted twice)",
      real["range_bias_r2"] == 0.0)

print(f"\n{len(FAILS)} failed" + (": " + ", ".join(FAILS) if FAILS else " - all checks passed"))
sys.exit(len(FAILS))
