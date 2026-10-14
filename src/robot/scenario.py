"""Randomised scenarios: a different route, start and obstacle layout for every seed.

    python main.py --randomise --seed 7          # the demo
    python -m evaluation.random_trials --n 20    # repeated trials

Simulator side only, like corridor.py. Two kinds of randomness are kept apart on purpose:

KNOWN to the robot = the mission it is handed (a different start spot marked on the floor and a
different route from the planner):
    start spot      x0 = the EKF prior, stated before the run
    waypoints       W1, W2, W3 jittered inside the corridor; W4 (pre-dock) stays on the docking line
They go into `task`, the same dict load_task() returns, so the controller and the EKF need no change.

UNKNOWN to the robot = the world it meets. Only the simulator changes (apply_world):
    placement error true start = start spot + e,  e ~ N(0, P0) clipped at 2 sigma, so the prior is
                    honest by construction (P0 from config/estimation.json)
    obstacles       box A always, inside the lane of the obstacle section, one side >= 0.7 m wide;
                    with leg1_box=True also box B (probability 0.6) inside the lane of leg 1 (stress test)
    sensor noise    LiDAR range sd and dropout, camera pixel noise
    wheel scale     per-wheel encoder scale error = a wheel-radius calibration error the odometry
                    model does not know about

The robot's software never sees the unknown part: it gets `task` only. Obstacles are found with the
LiDAR, the placement error is corrected by the EKF, the wheel error is absorbed by Qd.
"""
import numpy as np

from src.robot.corridor import load_config, load_task, place_obstacles

# Ranges, all in the world frame of config/corridor.json [m, deg]. Why these numbers:
START_X, START_Y, START_YAW_DEG = (0.25, 0.45), (-0.15, 0.15), (-15.0, 15.0)   # start spot inside leg 1
W1_X, W1_Y = (2.5, 2.8), (-0.15, 0.15)       # leg 1 is 1.0 m wide: +-0.15 keeps 0.35 m to each wall
W2_X, W2_Y = (3.5, 3.7), (0.0, 0.2)          # the corner square is x 3.0-4.2, y -0.5-0.7
W3_X, W3_Y = (3.45, 3.75), (2.4, 2.55)       # obstacle section is x 2.85-4.35 (1.5 m wide)
BOX_A_ALONG = (1.7, 1.95)    # box A: y range. Nearer the corner (1.4-1.6) the avoidance starts while the robot
                             # is still in the corner square and its shifted lane hits the corner wall (seed 9)
BOX_A_ACROSS = 0.25          # box A: max |x - lane| [m]
SECTION_X = (2.85, 4.35)     # obstacle-section walls
WIDE_GAP = 0.70              # box A leaves one side at least this wide (the robot needs 0.41 m). Two similar
                             # ~0.6 m gaps made the avoidance pick the narrower side and stop too_close (seed 7)
BOX_B_X = (1.55, 1.85)       # box B: x range in leg 1
BOX_B_ACROSS = (0.22, 0.27)  # box B: |y| from the corridor axis. The far side keeps >= 0.57 m; 0.15-0.25 left
                             # only 0.52 m, too tight with the pose uncertainty at the start (no_passage, seed 2)
BOX_B_PROB = 0.6             # used only with leg1_box=True: a stress test, not the default. In the 1.0 m wide leg 1
                             # the avoidance failed in 4 of 6 early trials (no_passage / too_close right after the
                             # start, while the pose is still uncertain by P0); see evaluation/random_trials.py --leg1-box
LANE_HALF = 0.153 + 0.05 + 0.15              # robot half width + side margin + box half: box overlaps the lane
LIDAR_SIGMA_SCALE, LIDAR_DROPOUT = (1.0, 2.0), (0.01, 0.05)
PIXEL_NOISE = (2.0, 4.0)                     # grey levels
WHEEL_SCALE_SD = 0.005                       # 0.5 % per wheel, clipped at 2 sigma


def _lane_x(wa, wb, y):
    """x of the straight segment wa -> wb at height y (segments in the obstacle section run along +y)."""
    s = (y - wa[1]) / (wb[1] - wa[1])
    return wa[0] + s * (wb[0] - wa[0])


def _lane_y(wa, wb, x):
    """y of the straight segment wa -> wb at x (leg 1 runs along +x)."""
    s = (x - wa[0]) / (wb[0] - wa[0])
    return wa[1] + s * (wb[1] - wa[1])


def random_scenario(seed, p0_sd=None, leg1_box=False):
    """Scenario for this seed: dict(task=..., world=...). Same seed, same scenario, on any computer.

    leg1_box=True adds the second box in leg 1 with probability BOX_B_PROB (stress test).
    """
    rng = np.random.default_rng(10_000 + seed)                  # separate stream from the sensor noise
    u = lambda lo_hi: float(rng.uniform(*lo_hi))               # noqa: E731
    task = load_task()                                         # marker map and dock stay the same

    # ---- KNOWN: start spot (the prior) and route
    start = np.array([u(START_X), u(START_Y), np.deg2rad(u(START_YAW_DEG))])
    w1, w2, w3 = [u(W1_X), u(W1_Y)], [u(W2_X), u(W2_Y)], [u(W3_X), u(W3_Y)]
    w4 = list(task["waypoints"][-1])                           # pre-dock point: on the docking line
    task["start"], task["waypoints"] = start, np.array([w1, w2, w3, w4])

    # ---- UNKNOWN: where the robot really is, what is in the way, how noisy it is
    if p0_sd is None:
        from src.estimation.models import load_estimation_config   # P0 is stated in the EKF config
        p0_sd = load_estimation_config()["P0_sd"]
    sd = np.array([p0_sd[0], p0_sd[1], np.deg2rad(p0_sd[2])])
    placement = np.clip(rng.normal(0.0, sd), -2 * sd, 2 * sd)

    by = u(BOX_A_ALONG)
    lane = _lane_x(w2, w3, by)
    for _ in range(100):                                       # inside the lane, one side clearly passable
        bx = lane + float(rng.uniform(-BOX_A_ACROSS, BOX_A_ACROSS))
        if max(bx - 0.15 - SECTION_X[0], SECTION_X[1] - bx - 0.15) >= WIDE_GAP:
            break
    box_a = [bx, by]
    boxes = [box_a]
    if rng.random() < BOX_B_PROB and leg1_box:
        bx = u(BOX_B_X)
        lane = _lane_y(start, w1, bx)
        side = np.sign(lane) if abs(lane) > 0.02 else (1.0 if rng.random() < 0.5 else -1.0)
        b_y = side * u(BOX_B_ACROSS)                           # on the lane's side: it overlaps the lane by
                                                               # >= 8 cm, so the robot must avoid it
        boxes.append([bx, b_y])

    world = dict(
        true_start=(start + placement).tolist(),
        placement_error=placement.tolist(),
        obstacles=boxes,
        lidar_sigma_scale=u(LIDAR_SIGMA_SCALE),
        lidar_dropout=u(LIDAR_DROPOUT),
        pixel_noise=u(PIXEL_NOISE),
        wheel_scale=(1.0 + np.clip(rng.normal(0.0, WHEEL_SCALE_SD, 2), -2 * WHEEL_SCALE_SD,
                                   2 * WHEEL_SCALE_SD)).tolist(),
    )
    return dict(seed=seed, task=task, world=world)


def apply_world(robot, world):
    """Sim only: place the boxes, put the robot at its TRUE start, set the noise and the wheel error.

    Pass as run_mission(..., setup=lambda robot: apply_world(robot, sc["world"])).
    """
    place_obstacles(robot, world["obstacles"])
    robot.reset(*world["true_start"])
    robot.lidar.sigma *= world["lidar_sigma_scale"]
    robot.lidar.dropout = world["lidar_dropout"]
    robot.camera.pixel_noise = world["pixel_noise"]
    read, (sl, sr) = robot.read_encoders, world["wheel_scale"]

    def read_scaled():                                         # encoders that over/under-count a little
        t, phi_l, phi_r = read()
        return t, phi_l * sl, phi_r * sr
    robot.read_encoders = read_scaled


def describe(sc):
    """Human-readable lines for the terminal: what the robot is told, and what it is not."""
    t, w = sc["task"], sc["world"]
    s, e = t["start"], w["placement_error"]
    lines = [f"scenario seed {sc['seed']}",
             f"  told to the robot : start spot ({s[0]:.2f}, {s[1]:+.2f}, {np.degrees(s[2]):+.0f} deg), route "
             + " -> ".join(f"({x:.2f}, {y:.2f})" for x, y in t["waypoints"]),
             f"  hidden from it    : placed {100 * np.hypot(e[0], e[1]):.1f} cm / {np.degrees(e[2]):+.1f} deg off the spot, "
             f"{len(w['obstacles'])} box(es) at " + ", ".join(f"({x:.2f}, {y:.2f})" for x, y in w["obstacles"]),
             f"                      LiDAR sd x{w['lidar_sigma_scale']:.2f}, dropout {100 * w['lidar_dropout']:.1f} %, "
             f"pixel noise {w['pixel_noise']:.1f}, wheel scale L {100 * (w['wheel_scale'][0] - 1):+.2f} % "
             f"R {100 * (w['wheel_scale'][1] - 1):+.2f} %"]
    return "\n".join(lines)


if __name__ == "__main__":
    # Run from the project root:  python -m src.robot.scenario       (prints the first 5 scenarios)
    for k in range(1, 6):
        print(describe(random_scenario(k)), "\n")
    cfg = load_config()
    print("config start (non-random runs):", cfg["start"])
