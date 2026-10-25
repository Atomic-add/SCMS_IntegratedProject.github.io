"""Quick pass/fail checks for src/control (no simulator, runs in a few seconds).

Run from the project root:   python -m src.control.control_test      (or the VS Code Run button)

Each check prints PASS or FAIL with what it tested; the exit code is the number of failures (0 = all
passed), so it can also be used before every merge. Inputs are made up on purpose (synthetic poses,
LiDAR obs and marker frames), so each rule is tested on its own, without the physics.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # project root, so "src" imports work even
                                                                # when this file is run directly (VS Code Run)

from src.control import avoidance, docking  # noqa: E402
from src.control.behaviour import AVOID, DOCK, STOP, TRACK, NavigationController  # noqa: E402
from src.control.lqr import PathLQR, design_lqr  # noqa: E402
from src.control.models import controllability_rank, error_model, load_control_config  # noqa: E402
from src.control.waypoints import TURN, Segment, WaypointTracker  # noqa: E402
from src.robot.corridor import load_task  # noqa: E402
from src.robot.sim_robot import MAX_V, MAX_W  # noqa: E402

TASK, CFG = load_task(), load_control_config()
FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"   ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def clear_obs(t=0.0):
    """A valid scan with nothing in the way (walls far to the sides)."""
    pts = np.array([[0.5, 1.0, 1.5, 0.5, 1.0, 1.5], [0.6, 0.6, 0.6, -0.6, -0.6, -0.6]])
    return dict(t=t, valid=True, coverage=1.0, points_B=pts, nearest=(0.78, 0.88), free_ahead=np.inf,
                left_clear=0.7, right_clear=0.7)


def box_obs(t=0.0, walls=0.75):
    """A valid scan with a 0.3 m wide box 0.5 m ahead, centred slightly right; walls at +-walls."""
    box = np.array([[0.6] * 7, np.linspace(-0.2, 0.1, 7)])
    wall_x = np.linspace(0.0, 2.0, 21)
    pts = np.hstack([box, np.vstack([wall_x, np.full(21, walls)]), np.vstack([wall_x, np.full(21, -walls)])])
    return dict(t=t, valid=True, coverage=1.0, points_B=pts, nearest=(0.6, 0.0), free_ahead=0.6 - 0.086,
                left_clear=walls, right_clear=walls)


def marker_frame(t, rng, brg=0.0):
    return {"t": t, "markers": {0: {"z": np.array([rng, brg]), "R": np.eye(2) * 1e-4, "yaw": np.pi, "ambiguous": False}}}


# ---------------------------------------------------------------- models and LQR
Ad, Bd = error_model(CFG["v_nom"], 0.0)
check("model: controllable at cruise speed", controllability_rank(Ad, Bd) == 2)
check("model: NOT controllable at v = 0 (why corners turn on the spot)", controllability_rank(*error_model(0.0, 0.0)) == 1)
K, eig = design_lqr()
check("LQR: closed loop stable at cruise speed", np.all(np.abs(eig) < 1), f"|eig| = {np.abs(eig).round(3)}")
check("LQR: same gain still stable at 0.05 m/s", np.all(np.abs(np.linalg.eigvals(error_model(0.05, 0)[0] - error_model(0.05, 0)[1] @ K)) < 1))

# ---------------------------------------------------------------- the tutorials' toolbox gives the same answers
# models.py and lqr.py use scipy; Week 5 and Week 6 use SymPy and python-control. Same model, same gain.
try:
    import control as ct
    import sympy as sp
except ImportError as e:
    check(f"toolbox: {e.name} is installed (python -m pip install -r requirements.txt)", False)
else:
    from src.control.lqr import design_weights
    from src.control.models import continuous_error_model
    e_y, e_th, d_om, v_s, k_s = sp.symbols("e_y e_theta d_omega v kappa")
    f = sp.Matrix([v_s * sp.sin(e_th),                                         # full nonlinear error dynamics,
                   v_s * k_s + d_om - v_s * k_s * sp.cos(e_th) / (1 - k_s * e_y)])   # omega = v kappa + d_omega
    at = {e_y: 0, e_th: 0, d_om: 0, v_s: 0.2, k_s: 0.5}
    A_sym = np.array(f.jacobian([e_y, e_th]).subs(at), dtype=float)
    B_sym = np.array(f.jacobian([d_om]).subs(at), dtype=float)
    A_num, B_num = continuous_error_model(0.2, 0.5)
    check("toolbox: SymPy Jacobians of the nonlinear error model = continuous_error_model (W5)",
          np.allclose(A_sym, A_num) and np.allclose(B_sym, B_num), f"A = {A_sym.round(3).tolist()}")
    sysd = ct.c2d(ct.ss(*continuous_error_model(CFG["v_nom"], 0.0), np.eye(2), np.zeros((2, 1))), CFG["Ts"], method="zoh")
    check("toolbox: ct.c2d ZOH = models.error_model (W5)", np.allclose(sysd.A, Ad) and np.allclose(sysd.B, Bd))
    check("toolbox: ct.ctrb rank 2 at cruise speed, 1 at v = 0 (W5)",
          np.linalg.matrix_rank(ct.ctrb(Ad, Bd)) == 2 and np.linalg.matrix_rank(ct.ctrb(*error_model(0.0, 0.0))) == 1)
    K_ct, S_ct, _ = ct.dlqr(Ad, Bd, *design_weights(CFG))
    Q_w, R_w = design_weights(CFG)
    res = Ad.T @ S_ct @ Ad - S_ct - Ad.T @ S_ct @ Bd @ np.linalg.solve(R_w + Bd.T @ S_ct @ Bd, Bd.T @ S_ct @ Ad) + Q_w
    check("toolbox: ct.dlqr gain = lqr.design_lqr, Riccati residual ~0 (W6)",
          np.allclose(K_ct, K) and np.abs(res).max() < 1e-8, f"K = {np.ravel(K_ct).round(3)}, residual {np.abs(res).max():.1e}")
seg = Segment([0, 0], [2, 0])
e_y, e_th, _ = seg.errors([1.0, 0.1, 0.0])
check("errors: robot LEFT of the path gives e_y > 0", e_y > 0, f"e_y = {e_y:.2f}")
check("LQR: robot left of the path steers RIGHT (omega < 0)", PathLQR(cfg=CFG).step(e_y, e_th, 0.0, 0.2) < 0)
check("LQR: huge error is clipped to 1.82 rad/s", abs(PathLQR(cfg=CFG).step(5.0, 0.0, 0.0, 0.2)) <= MAX_W + 1e-9)

# ---------------------------------------------------------------- command bounds
nav = NavigationController(TASK, cfg=CFG)
rng = np.random.default_rng(0)
worst_v = worst_w = 0.0
for k in range(300):
    xh = np.array([rng.uniform(0, 4), rng.uniform(-0.4, 4), rng.uniform(-np.pi, np.pi)])
    v, w, _ = nav.step(xh, np.eye(3) * 1e-4, clear_obs(t=0.05 * k), None, 0.05, {"v": 0, "w": 0, "t": 0.05 * k})
    worst_v, worst_w = max(worst_v, abs(v)), max(worst_w, abs(w))
check("bounds: |v| <= 0.26 m/s and |w| <= 1.82 rad/s for random poses", worst_v <= MAX_V and worst_w <= MAX_W,
      f"max |v| {worst_v:.3f}, max |w| {worst_w:.3f}")

# ---------------------------------------------------------------- sensor loss
nav = NavigationController(TASK, cfg=CFG)
start = np.array(TASK["start"], float)
fb = {"v": 0, "w": 0, "t": 1.0}
v, w, info = nav.step(start, np.eye(3) * 1e-4, dict(clear_obs(t=1.0), valid=False), None, 0.05, fb)
check("sensor loss: invalid LiDAR scan -> robot waits (v = w = 0)", (v, w) == (0.0, 0.0) and info["state"] == TRACK)
v, w, info = nav.step(start, np.eye(3) * 1e-4, clear_obs(t=0.0), None, 0.05, {"v": 0, "w": 0, "t": 5.0})
check("sensor loss: 5 s old LiDAR scan -> robot waits", (v, w) == (0.0, 0.0))
v, w, info = nav.step(start, np.eye(3) * 1e-4, None, None, 0.05, fb)
check("sensor loss: no LiDAR scan at all -> robot waits", (v, w) == (0.0, 0.0))

# ---------------------------------------------------------------- transitions
nav = NavigationController(TASK, cfg=CFG)
for k in range(10):                                         # build up some speed on the first segment
    nav.step(start, np.eye(3) * 1e-4, clear_obs(t=0.05 * k), None, 0.05, {"v": 0, "w": 0, "t": 0.05 * k})
check("transition: clear path stays in TRACK", nav.state == TRACK)
_, _, info = nav.step(start, np.eye(3) * 1e-4, box_obs(t=0.5), None, 0.05, {"v": 0, "w": 0, "t": 0.5})
check("transition: box in the lane -> AVOID", nav.state == AVOID, f"state {info['state']}")
check("pause: waypoints paused while avoiding, segment kept", nav.tracker.paused and nav.tracker.i == 0)

nav = NavigationController(TASK, cfg=CFG)
nav.step(start, np.eye(3) * 1e-4, box_obs(t=0.0, walls=0.3), None, 0.05, {"v": 0, "w": 0, "t": 0.0})
check("transition: box and no room on either side -> STOP no_passage",
      nav.state == STOP and nav.stop_reason == "no_passage", f"{nav.state}, {nav.stop_reason}")

nav = NavigationController(TASK, cfg=CFG)
w4 = TASK["waypoints"][-1]
nav.tracker.i = len(nav.tracker.segments) - 1               # on the last segment, standing on W4
nav.step(np.array([w4[0], w4[1], np.pi / 2]), np.eye(3) * 1e-4, clear_obs(), None, 0.05, {"v": 0, "w": 0, "t": 0.0})
nav.step(np.array([w4[0], w4[1], np.pi / 2]), np.eye(3) * 1e-4, clear_obs(), None, 0.05, {"v": 0, "w": 0, "t": 0.05})
check("transition: W4 reached -> pre_dock_reached and DOCK", nav.tracker.pre_dock_reached and nav.state == DOCK)

# ---------------------------------------------------------------- pause / resume
tr = WaypointTracker.from_task(TASK, cfg=CFG)
for _ in range(20):
    tr.step(start)
tr.pause()
for _ in range(20):
    v, w = tr.step(start)
check("pause: command ramps down to zero", (v, w) == (0.0, 0.0))
tr.resume(np.array([start[0], start[1], np.pi / 2]))        # facing 90 deg away from the segment
check("resume: facing away from the path -> TURN on the spot first", tr.mode == TURN and tr.i == 0)

# ---------------------------------------------------------------- docking
P_ok = np.diag([0.01**2, 0.01**2, np.deg2rad(1.0)**2])
goal = np.array(TASK["dock"]["goal_pose"])


def run_dock(ranges, feedback_v=0.0, lose_marker_after=None, steps=200, P=P_ok, x_off=0.0):
    """Feed DockingController a scripted marker range (one camera frame per step).
    x_off: robot that far sideways of the approach line, but turned to point straight at the marker."""
    d = docking.DockingController.from_task(TASK, cfg=CFG)
    d.start()
    m = TASK["markers"][0]
    for k in range(steps):
        t = 0.05 * k
        r = ranges[min(k, len(ranges) - 1)]
        frame = None if (lose_marker_after is not None and k >= lose_marker_after) else marker_frame(t, r)
        # place x_hat consistently with the marker range (bearing 0 = pointing at the marker)
        x, y = goal[0] + x_off, m[1] - np.sqrt(r**2 - x_off**2)
        xh = np.array([x, y, np.arctan2(m[1] - y, m[0] - x)])
        d.step(xh, P, frame, {"v": feedback_v, "w": 0.0, "t": t}, t)
        if d.phase in (docking.COMPLETE, docking.FAILED):
            break
    return d


d = run_dock(list(np.linspace(0.70, 0.356, 60)))
check("docking: range reaches 0.356 m, robot still -> dock_complete", d.dock_complete, f"phase {d.phase}")
d = run_dock(list(np.linspace(0.70, 0.356, 60)), feedback_v=0.05)
check("docking: encoders say it is still moving -> NOT complete", not d.dock_complete and "stopped" in (d.stop_reason or ""),
      f"{d.phase}, {d.stop_reason}")
d = run_dock(list(np.linspace(0.70, 0.356, 60)), x_off=0.05)
check("docking: 5 cm sideways but pointing at the marker -> NOT complete (bearing alone would pass)",
      not d.dock_complete and "lateral" in (d.stop_reason or ""), f"{d.phase}, {d.stop_reason}")
d = run_dock(list(np.linspace(0.70, 0.356, 60)), P=np.diag([0.05**2, 0.05**2, np.deg2rad(1.0)**2]))
check("docking: estimate not confident (5 cm sd) -> NOT complete", not d.dock_complete and "confident" in (d.stop_reason or ""),
      f"{d.phase}, {d.stop_reason}")
d = docking.DockingController.from_task(TASK, cfg=CFG)
d.start()
xh_stop = np.array([goal[0], TASK["markers"][0][1] - 0.356, np.pi / 2])
for k in range(4):                                          # range reached, all checks true for 0.15 s only
    d.step(xh_stop, P_ok, marker_frame(0.05 * k, 0.356), {"v": 0, "w": 0, "t": 0.05 * k}, 0.05 * k)
check("docking: checks true for only 0.15 s -> not complete yet (dock_hold_sec)", d.phase == docking.SETTLE, d.phase)
d = run_dock(list(np.linspace(0.70, 0.356, 60)), lose_marker_after=20)
check("docking: marker lost for good -> FAILED marker_lost (never drives in blind)",
      d.phase == docking.FAILED and d.stop_reason == "marker_lost", f"{d.phase}, {d.stop_reason}")
d = docking.DockingController.from_task(TASK, cfg=CFG)
d.start()
d.step(np.array([goal[0], 3.9, np.pi / 2]), P_ok, marker_frame(0.0, 0.7), {"v": 0, "w": 0, "t": 0.0}, 0.0)
v, w = d.step(np.array([goal[0], 3.9, np.pi / 2]), P_ok, None, {"v": 0, "w": 0, "t": 0.5}, 0.5)
check("docking: marker briefly missing -> no forward motion (WAIT)", v == 0.0 and d.phase == docking.WAIT)

# ---------------------------------------------------------------- docking: IBVS hand-over (Week 9)
from spatialmath import SE3  # noqa: E402
from src.control import visual_servo as vs  # noqa: E402
CFG_IBVS = {**CFG, "docking": {**CFG["docking"], "servo": "ibvs"}}


K_IBVS = docking.DockingController.from_task(TASK, cfg=CFG_IBVS).K


def ibvs_frame(t, rng, shift_x=0.0):
    """A frame with corners and PnP pose: the marker square on at base range rng, slid shift_x along x_C."""
    T_CM = SE3(shift_x, 0, 0) * vs.desired_pose(rng, TASK["markers"][0][2])
    p = (vs.aruco.T_BC * T_CM).t                                # marker centre in the base frame
    z = np.array([np.hypot(p[0], p[1]), np.arctan2(p[1], p[0])])
    corners = vs.project(K_IBVS, T_CM).T                        # 4 x 2, as aruco.detect returns them
    return {"t": t, "markers": {0: {"z": z, "corners": corners, "T_CM": T_CM}}}


xh_line = np.array([goal[0], TASK["markers"][0][1] - 0.50, np.pi / 2])
d = docking.DockingController.from_task(TASK, cfg=CFG_IBVS)
d.start()
v, w = d.step(xh_line, P_ok, ibvs_frame(0.0, 0.50), {"v": 0, "w": 0, "t": 0.0}, 0.0)
v, w = d.step(xh_line, P_ok, ibvs_frame(0.1, 0.50), {"v": 0, "w": 0, "t": 0.1}, 0.1)
check("docking: below ibvs_switch_range with corners -> IBVS, drives forward, does not turn",
      d.phase == docking.IBVS and v > 0 and abs(w) < 1e-3, f"{d.phase}, v {v:.3f}, w {w:+.4f}")
d = docking.DockingController.from_task(TASK, cfg=CFG_IBVS)
d.start()
d.step(xh_line, P_ok, ibvs_frame(0.0, 0.50, shift_x=-0.03), {"v": 0, "w": 0, "t": 0.0}, 0.0)
v, w = d.step(xh_line, P_ok, ibvs_frame(0.1, 0.50, shift_x=-0.03), {"v": 0, "w": 0, "t": 0.1}, 0.1)
check("docking: IBVS turns LEFT (omega > 0) towards a marker left of the image centre", w > 0, f"w {w:+.3f}")
d = docking.DockingController.from_task(TASK, cfg=CFG_IBVS)
d.start()
v, w = d.step(xh_line, P_ok, marker_frame(0.0, 0.50), {"v": 0, "w": 0, "t": 0.0}, 0.0)
check("docking: frame without corners -> stays on the LQR approach (no IBVS)", d.phase == docking.APPROACH, d.phase)

# ---------------------------------------------------------------- avoidance on its own
a = avoidance.AvoidanceController(cfg=CFG)
ok = a.start(box_obs(), np.array([0.0, 0.0, 0.0]), Segment([0, 0], [3, 0]))
check("avoidance: picks a side with room and an offset that clears the box",
      ok and a.phase == avoidance.AVOID and abs(a.d_pass) >= 0.2 + 0.15, f"offset {getattr(a, 'd_pass', None)}")

print(f"\n{len(FAILS)} failed" + (": " + ", ".join(FAILS) if FAILS else " - all checks passed"))
sys.exit(len(FAILS))
