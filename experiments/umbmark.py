"""UMBmark on the real robot (Week 2 B.3-B.4): drive a square on the encoders, tape the end error, fit the wheels.

    python experiments/umbmark.py --cw  --runs 5            # 5 squares turning right (clockwise)
    python experiments/umbmark.py --ccw --runs 5            # 5 squares turning left (counter-clockwise)
    python experiments/umbmark.py --straight                # 2 m straight: the overall wheel scale
    python experiments/umbmark.py --analyse results/hardware/umbmark_*.json   # fit everything recorded
    python experiments/umbmark.py --sim --cw --runs 5 --sim-left 0.997 --sim-right 1.003   # try it in the sim

The robot drives the square on its own odometry only (WHEEL_RADIUS, WHEEL_SEPARATION): straight legs hold
the odometry heading, turns stop when the odometry says 90 deg. A perfect robot would come back to its
start; a real one does not, and the end error tells which wheel parameter is wrong:
  - left and right wheels of different size (E_d): every leg curves the same way, so the robot ends off
    to the same side in both directions;
  - wrong track width (E_b): every turn is too short or too long, so the error flips side between
    clockwise and counter-clockwise.
A closed square cannot see the overall wheel scale (a robot with both wheels 1 % too big drives a 1 %
bigger square and still closes it). --straight measures that: 2 m by the odometry, the true distance by tape.

Measuring (no need to put the robot exactly on the tape): tape an L on the floor, line A across behind
the robot, line B along its left side. Before AND after each run, measure with a tape [cm]:
    a = rear-left corner of the chassis to line A    (+ in front of the line)
    b = rear-left corner to line B                   (+ to the right of the line, i.e. inside the L)
    c = rear-right corner to line A                  (+ in front of the line)
The script turns the start and end readings into the pose change of the base origin (the wheel axle
centre). The rear corners are (REAR_X, +-CORNER_Y) in the base frame (Waffle Pi chassis in the URDF);
--rear-x / --corner-y take your own measurement of the robot.

Fit (--analyse, also printed after every recording): the end errors of all runs are matched by an
odometry model with true wheel distances (1 - d) and (1 + d) times the encoder ones and a true track
width (1 + e) times WHEEL_SEPARATION, by least squares (scipy). The model replays each run's recorded
encoder increments, so it drives exactly what the robot did (its turns stop within 0.2 deg, not on 90). The random part left over gives k_wheel:
the wheel distance noise sd = k_wheel |distance per step| that src/estimation/models.Qd_wheels assumes,
with steps of TS at this speed (Monte Carlo of the same squares). Writes results/hardware/umbmark_<time>.json
per session; config files are NOT changed, the printout says what to change.

Adapted from 41014 Week 2 B.3-B.4 (square, straight, UMBmark both ways) and Week 7 D.4 (Qd from data).
Additions: tape readings from two chassis corners, the fit that replays the recorded encoder increments,
k_wheel from the residual, the sim test with known wheel errors.
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root, so "src" imports work

import numpy as np                                              # noqa: E402
from scipy.optimize import least_squares                       # noqa: E402

from src.control.models import TS                              # noqa: E402
from src.estimation.odometry import wrap                       # noqa: E402
from src.robot.sim_robot import PROJECT_ROOT, WHEEL_RADIUS, WHEEL_SEPARATION   # noqa: E402

OUT = PROJECT_ROOT / "results" / "hardware"
REAR_X = -0.197          # m, rear face of the chassis in the base frame (URDF chassis box: -0.064 - 0.133)
CORNER_Y = 0.133         # m, half width of the chassis at its rear corners (the tyres stick out to 0.153)
V, W_TURN = 0.08, 0.4    # m/s, rad/s: slow enough that the wheels do not slip; below the robot_real.json caps
ACCEL = 0.2              # m/s^2 and rad/s^2 ramp, so the start and stop of each leg do not slip either
K_HEADING = 2.0          # 1/s, heading hold on straight legs (on the odometry heading)
TURN_TOL = np.deg2rad(0.2)
STRAIGHT_M = 2.0


# ---------------------------------------------------------------- driving on the encoders
class Odo:
    """Dead reckoning with the model's wheel radius and separation (sep_scale: sim test of a wrong track)."""

    def __init__(self, robot, sep_scale=1.0):
        self.robot, self.sep = robot, WHEEL_SEPARATION * sep_scale
        _, *self.prev = robot.read_encoders()
        self.pose = np.zeros(3)
        self.steps = []                                    # every (d_L, d_R) [m]: the fit replays them

    def update(self):
        _, *phi = self.robot.read_encoders()
        dl, dr = (np.array(phi) - np.array(self.prev)) * WHEEL_RADIUS
        self.prev = phi
        if dl or dr:
            self.steps.append((float(dl), float(dr)))
        ds, dth = (dl + dr) / 2, (dr - dl) / self.sep
        m = self.pose[2] + dth / 2
        self.pose = np.array([self.pose[0] + ds * np.cos(m), self.pose[1] + ds * np.sin(m), self.pose[2] + dth])
        return self.pose


def drive_straight(robot, odo, length):
    """Drive `length` metres by the odometry, holding the odometry heading, with speed ramps."""
    p0, th0, v = odo.update().copy(), odo.pose[2], 0.0
    while True:
        p = odo.update()
        left = length - np.hypot(*(p[:2] - p0[:2]))
        if left <= 0.0005:
            break
        v = min(V, v + ACCEL * TS, np.sqrt(2 * ACCEL * left) + 0.005)     # ramp up, then down to the end
        robot.send_cmd(v, K_HEADING * (th0 - p[2]))
        robot.step(TS)
    settle(robot, odo)


def turn(robot, odo, angle):
    """Turn on the spot by `angle` [rad] by the odometry, with ramps; stops within TURN_TOL."""
    target, w = odo.update()[2] + angle, 0.0
    while True:
        err = target - odo.update()[2]
        if abs(err) < TURN_TOL:
            break
        w_ramp = min(W_TURN, abs(w) + ACCEL * TS, np.sqrt(2 * ACCEL * abs(err)) + 0.02)
        w = np.sign(err) * w_ramp
        robot.send_cmd(0.0, w)
        robot.step(TS)
    settle(robot, odo)


def settle(robot, odo, seconds=1.0):
    """Stop and wait, still integrating the encoders (the robot rolls a little after the stop)."""
    t_end = robot.t + seconds
    while robot.t < t_end:
        robot.send_cmd(0.0, 0.0)
        robot.step(TS)
        odo.update()


def drive_square(robot, side, direction, sep_scale=1.0):
    """Four legs and four 90 deg turns, +1 = counter-clockwise (left turns), -1 = clockwise."""
    odo = Odo(robot, sep_scale)
    for _ in range(4):
        drive_straight(robot, odo, side)
        turn(robot, odo, direction * np.pi / 2)
    return odo


# ---------------------------------------------------------------- tape readings -> base pose
def base_pose(a, b, c, rear_x=REAR_X, corner_y=CORNER_Y):
    """Base origin pose (x, y, theta) in the tape frame (x along line B, forward; y along line A, left; origin
    at the L corner) from a, b, c in metres. The rear-left corner sits at (a, -b), the rear-right at x = c."""
    th = np.arcsin(np.clip((c - a) / (2 * corner_y), -1, 1))         # rear edge turned by th: right corner moves forward
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    xy = np.array([a, -b]) - R @ np.array([rear_x, corner_y])
    return np.array([xy[0], xy[1], th])


def pose_change(start, end):
    """End pose of the base in the START base frame: the UMBmark end error (0 for a perfect robot)."""
    d = end[:2] - start[:2]
    c, s = np.cos(start[2]), np.sin(start[2])
    return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1], wrap(end[2] - start[2])])


# ---------------------------------------------------------------- model fit
def model_square(steps, d, e, k=0.0, rng=None):
    """End pose of a recorded run if the true wheel distances were (1 - d) and (1 + d) times the encoder ones
    and the true track (1 + e) WHEEL_SEPARATION. steps: the run's encoder increments (d_L, d_R) [m], so the
    model drives exactly what the robot's odometry controller did (its early stops and heading corrections
    included). k > 0 adds wheel noise sd = k |distance per step| (Qd_wheels; the steps are TS long)."""
    st = np.asarray(steps, float)
    dl, dr = (1 - d) * st[:, 0], (1 + d) * st[:, 1]
    if k:
        dl, dr = dl + rng.normal(0, k * np.abs(dl)), dr + rng.normal(0, k * np.abs(dr))
    ds, dth = (dl + dr) / 2, (dr - dl) / (WHEEL_SEPARATION * (1 + e))
    th = np.concatenate([[0.0], np.cumsum(dth)])
    m = th[:-1] + dth / 2
    return np.array([np.sum(ds * np.cos(m)), np.sum(ds * np.sin(m)), wrap(th[-1])])


def fit(runs):
    """Least squares for d (left/right size) and e (track width) on the end errors; k_wheel from the rest."""
    squares = [r for r in runs if r["kind"] == "square"]
    if len(squares) < 2 or len({r["direction"] for r in squares}) < 2:
        return None
    meas = np.array([r["error"] for r in squares])
    dirs = [r["direction"] for r in squares]
    steps = [r["steps"] for r in squares]
    w = np.array([1.0, 1.0, 0.3])                                    # m, m, rad (0.3 m per rad ~ the square's size)

    def residual(p):
        return np.concatenate([(model_square(st, *p) - m) / w for st, m in zip(steps, meas)])

    sol = least_squares(residual, [0.0, 0.0], x_scale=[1e-3, 1e-2])
    d, e = sol.x
    resid = np.array([m - model_square(st, d, e) for st, m in zip(steps, meas)])
    rms_xy = float(np.sqrt(np.mean(np.sum(resid[:, :2] ** 2, axis=1))))
    rng = np.random.default_rng(0)                                   # k_wheel: scatter of noisy model squares
    k0 = 0.01
    sim = np.array([model_square(st, d, e, k0, rng) - model_square(st, d, e) for st in steps for _ in range(40)])
    rms0 = float(np.sqrt(np.mean(np.sum(sim[:, :2] ** 2, axis=1))))
    n_par = 2
    dof = max(len(squares) - n_par / 3, 1)                           # 3 numbers per run, 2 parameters fitted
    k_wheel = k0 * rms_xy / rms0 * np.sqrt(len(squares) / dof)
    return dict(d=float(d), e=float(e), left_scale=float(1 - d), right_scale=float(1 + d),
                track_m=float(WHEEL_SEPARATION * (1 + e)), residual_rms_cm=100 * rms_xy, k_wheel=float(k_wheel),
                n_cw=dirs.count(-1), n_ccw=dirs.count(1))


def report(runs):
    """Print the end errors per direction, the fit and what to change in the config."""
    for name, di in (("clockwise", -1), ("counter-clockwise", 1)):
        e = np.array([r["error"] for r in runs if r["kind"] == "square" and r["direction"] == di])
        if len(e):
            print(f"{name:18s} n {len(e)}: end x {100 * e[:, 0].mean():+.1f} +- {100 * e[:, 0].std():.1f} cm, "
                  f"y {100 * e[:, 1].mean():+.1f} +- {100 * e[:, 1].std():.1f} cm, "
                  f"heading {np.degrees(e[:, 2]).mean():+.1f} +- {np.degrees(e[:, 2]).std():.1f} deg")
    straight = [r for r in runs if r["kind"] == "straight"]
    scale = float(np.mean([r["true_m"] / r["odometry_m"] for r in straight])) if straight else None
    if scale is not None:
        print(f"straight, n {len(straight)}: true / odometry distance = {scale:.4f} "
              f"(the encoders {'over' if scale < 1 else 'under'}-count by {100 * abs(1 - scale):.1f} %)")
    f = fit(runs)
    if f is None:
        print("fit: needs squares in BOTH directions (--cw and --ccw)")
        return None
    s = scale or 1.0
    print(f"fit: right / left wheel size {f['right_scale'] / f['left_scale']:.4f}, true track width "
          f"{1000 * f['track_m'] * s:.1f} mm (model {1000 * WHEEL_SEPARATION:.0f}), residual {f['residual_rms_cm']:.1f} cm rms")
    print(f"     wheel radius left {1000 * WHEEL_RADIUS * f['left_scale'] * s:.2f} mm, right "
          f"{1000 * WHEEL_RADIUS * f['right_scale'] * s:.2f} mm (model {1000 * WHEEL_RADIUS:.1f}"
          f"{'' if scale else ', overall scale not measured: run --straight'})")
    print(f"     k_wheel (random part, for config/estimation.json) {f['k_wheel']:.4f} "
          f"(now {json.loads((PROJECT_ROOT / 'config' / 'estimation.json').read_text())['k_wheel']}); "
          f"from {f['n_cw'] + f['n_ccw']} squares: an upper bound, the tape readings' own error is in it too")
    return dict(fit=f, straight_scale=scale)


# ---------------------------------------------------------------- recording session
def ask_abc(prompt):
    """Read 'a b c' in cm from the terminal; returns metres."""
    while True:
        raw = input(f"  {prompt} a b c [cm]: ").replace(",", " ").split()
        try:
            a, b, c = (float(v) / 100 for v in raw)
            return a, b, c
        except ValueError:
            print("    three numbers please, e.g.  12.5 3.0 12.8")


def sim_robot(args):
    """SimRobot on the empty floor with encoder scale errors (the sim's stand-in for different wheels)."""
    from src.robot.sim_robot import SimRobot
    robot = SimRobot(scene=PROJECT_ROOT / "models" / "robotis_tb3" / "scene_turtlebot3_waffle_pi.xml", seed=args.seed)
    robot.reset(0.0, 0.0, 0.0)
    read = robot.read_encoders

    def read_scaled():                       # encoders that count 1/scale: the wheel really rolls scale x more
        t, l, r = read()
        return t, l / args.sim_left, r / args.sim_right
    robot.read_encoders = read_scaled
    return robot


def sim_tape(robot, rng):
    """The tape readings a, b, c of the simulated robot (truth plus 2 mm reading noise), tape at the origin."""
    from evaluation.ground_truth import true_pose
    x, y, th = true_pose(robot)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    cl = np.array([x, y]) + R @ [REAR_X, CORNER_Y]
    cr = np.array([x, y]) + R @ [REAR_X, -CORNER_Y]
    ox, oy = REAR_X - 0.02, CORNER_Y + 0.02                          # line A 2 cm behind, line B 2 cm left
    a, b, c = cl[0] - ox, oy - cl[1], cr[0] - ox
    return tuple(np.array([a, b, c]) + rng.normal(0, 0.002, 3))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--cw", action="store_true", help="squares turning right (clockwise)")
    g.add_argument("--ccw", action="store_true", help="squares turning left (counter-clockwise)")
    g.add_argument("--straight", action="store_true", help=f"drive {STRAIGHT_M} m straight: overall wheel scale")
    g.add_argument("--analyse", nargs="+", metavar="JSON", help="fit saved sessions, no driving")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--side", type=float, default=1.0, help="square side [m] (default 1.0)")
    ap.add_argument("--host", help="robot IP (default config/robot_real.json)")
    ap.add_argument("--rear-x", type=float, default=REAR_X, help=f"rear face of the chassis to the axle [m] (default {REAR_X})")
    ap.add_argument("--corner-y", type=float, default=CORNER_Y, help=f"half the distance between the rear corners [m] (default {CORNER_Y})")
    ap.add_argument("--sim", action="store_true", help="drive the MuJoCo robot instead (tape read from the sim)")
    ap.add_argument("--sim-left", type=float, default=1.0, help="sim: true / encoder distance, left wheel")
    ap.add_argument("--sim-right", type=float, default=1.0, help="sim: true / encoder distance, right wheel")
    ap.add_argument("--sim-track", type=float, default=1.0, help="sim: odometry track width / true (wrong WHEEL_SEPARATION)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    corner = dict(rear_x=args.rear_x, corner_y=args.corner_y)

    if args.analyse:
        runs = []
        for pattern in args.analyse:
            for f in sorted(glob.glob(pattern)):
                runs += json.loads(Path(f).read_text())["runs"]
        print(f"{len(runs)} runs from {len(args.analyse)} pattern(s)")
        report(runs)
        return 0
    if not (args.cw or args.ccw or args.straight):
        ap.error("pick --cw, --ccw, --straight or --analyse")

    direction = -1 if args.cw else 1
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / time.strftime(f"umbmark_%Y%m%d_%H%M%S{'_sim' if args.sim else ''}.json")
    session = dict(side_m=args.side, sim=args.sim, rear_x=args.rear_x, corner_y=args.corner_y, runs=[],
                   wheel_radius=WHEEL_RADIUS, wheel_separation=WHEEL_SEPARATION, v=V, w_turn=W_TURN)
    rng = np.random.default_rng(args.seed)
    if args.sim:
        robot = sim_robot(args)
    else:
        from src.robot.real_robot import RealRobot
        robot = RealRobot(args.host)
    what = (f"{STRAIGHT_M} m straight" if args.straight else
            f"{args.side} m square, {'clockwise (right turns)' if args.cw else 'counter-clockwise (left turns)'}")
    print(f"{what}, {args.runs} runs; results go to {out.relative_to(PROJECT_ROOT)} after every run")
    if not args.sim:
        print(f"keep about {args.side + 0.5:.1f} x {args.side + 0.5:.1f} m free "
              f"{'to the right' if args.cw else 'to the left'} of and in front of the robot")
    with robot:
        for k in range(1, args.runs + 1):
            print(f"\nrun {k}/{args.runs}")
            if args.straight:
                if not args.sim:
                    input("  put the robot on the start mark, then press Enter (it drives in 3 s)")
                    time.sleep(3.0)
                odo = Odo(robot)
                x0 = robot.d.qpos[0] if args.sim else None
                drive_straight(robot, odo, STRAIGHT_M)
                true_m = (robot.d.qpos[0] - x0) if args.sim else float(input("  true distance driven [cm]: ")) / 100
                session["runs"].append(dict(kind="straight", odometry_m=float(np.hypot(*odo.pose[:2])), true_m=float(true_m)))
            else:
                if args.sim:
                    start = base_pose(*sim_tape(robot, rng), **corner)
                else:
                    start = base_pose(*ask_abc("START (robot at rest, before it drives)"), **corner)
                    input("  press Enter to drive (it starts in 3 s)")
                    time.sleep(3.0)
                odo = drive_square(robot, args.side, direction, args.sim_track)
                odo_end = odo.pose
                end = base_pose(*(sim_tape(robot, rng) if args.sim else ask_abc("END")), **corner)
                err = pose_change(start, end)
                print(f"  end error: x {100 * err[0]:+.1f} cm, y {100 * err[1]:+.1f} cm, heading {np.degrees(err[2]):+.1f} deg "
                      f"(odometry thinks {100 * odo_end[0]:+.1f}, {100 * odo_end[1]:+.1f} cm)")
                session["runs"].append(dict(kind="square", direction=direction, start=start.tolist(), end=end.tolist(),
                                            error=err.tolist(), odometry_end=odo_end.tolist(),
                                            steps=np.round(odo.steps, 7).tolist()))
                if args.sim:
                    robot.reset(0.0, 0.0, 0.0)
            out.write_text(json.dumps(session, indent=1))
    print()
    report(session["runs"])
    print(f"saved {out.relative_to(PROJECT_ROOT)}; fit both directions together with\n"
          f"  python experiments/umbmark.py --analyse 'results/hardware/umbmark_*.json'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
