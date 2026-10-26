"""Randomised scenarios on the lab track: a different route, start and box layout for every seed.

    python main.py --randomise --seed 7          # the demo
    python -m evaluation.random_trials --n 20    # repeated trials

Simulator side only, like task.py. Two kinds of randomness are kept apart on purpose:

KNOWN to the robot = the mission it is handed (a different start spot marked on the floor and a
different route from the planner):
    start spot      x0 = the EKF prior, stated before the run
    waypoints       W1 and the corner W2 jittered inside the 1.5 m band; W3 (pre-dock) stays on the docking line
They go into `task`, the same dict load_task() returns, so the controller and the EKF need no change.

UNKNOWN to the robot = the world it meets. Only the simulator changes (apply_world):
    placement error true start = start spot + e,  e ~ N(0, P0) clipped at 2 sigma, so the prior is
                    honest by construction (P0 from config/estimation.json)
    obstacle        the one printed box, inside the lane of the long leg, one side >= 0.7 m wide
    sensor noise    LiDAR range sd and dropout, camera pixel noise
    wheel scale     per-wheel encoder scale error = a wheel-radius calibration error the odometry
                    model does not know about

The robot's software never sees the unknown part: it gets `task` only. Obstacles are found with the
LiDAR, the placement error is corrected by the EKF, the wheel error is absorbed by Qd.
"""
import numpy as np

from src.estimation.odometry import wrap
from src.robot.task import load_config, load_task, place_obstacles

# Ranges, all in the world frame of config/arena_lab.json [m, deg]. The robot starts near the right end
# facing -x, drives the long leg (the band y 0.5-2.0 m) to the corner W2 and turns left into the pocket.
START_X, START_Y, START_YAW_DEG = (4.35, 4.65), (1.1, 1.4), (165.0, 195.0)   # start spot: the 0.5 m around
                             # the red dot; x <= 4.65 keeps the robot's back 0.2 m off the right panel (x 5.0)
W1_X, W1_Y = (3.8, 4.1), (1.1, 1.4)          # band 1.5 m wide: +-0.15 keeps 0.6 m to each panel
W2_X, W2_Y = (0.4, 0.6), (1.15, 1.35)        # the corner: >= 0.4 m from the left panel (x 0) for the turn
BOX_ALONG = (1.75, 3.0)      # the box: x range on the long leg. >= 0.8 m (trigger_dist) after W1 so the
                             # avoidance starts on a straight lane, and >= 1.15 m before the corner W2
BOX_ACROSS = 0.25            # the box: max |y - lane| [m]
BAND_Y = (0.5, 2.0)          # long-leg panels (the pocket below y 0.5 only starts at x < 2.0)
WIDE_GAP = 0.70              # the box leaves one side at least this wide (the robot needs 0.41 m), as in
                             # the old corridor: two similar ~0.55 m gaps can make the avoidance pick the tight side
BOX_HALF = 0.10              # the printed 0.20 m box
LANE_HALF = 0.153 + 0.05 + BOX_HALF          # robot half width + side margin + box half: box overlaps the lane
LIDAR_SIGMA_SCALE, LIDAR_DROPOUT = (1.0, 2.0), (0.01, 0.05)
PIXEL_NOISE = (2.0, 4.0)                     # grey levels
WHEEL_SCALE_SD = 0.005                       # 0.5 % per wheel, clipped at 2 sigma


def _lane_y(wa, wb, x):
    """y of the straight segment wa -> wb at x (the long leg runs along -x)."""
    s = (x - wa[0]) / (wb[0] - wa[0])
    return wa[1] + s * (wb[1] - wa[1])


def _box_in_lane(rng, x_range, w1, w2):
    """A box centre inside the lane W1 -> W2 at a random x, with one side at least WIDE_GAP wide."""
    bx = float(rng.uniform(*x_range))
    lane = _lane_y(w1, w2, bx)
    for _ in range(100):
        by = lane + float(rng.uniform(-BOX_ACROSS, BOX_ACROSS))
        if max(by - BOX_HALF - BAND_Y[0], BAND_Y[1] - by - BOX_HALF) >= WIDE_GAP:
            break
    return [bx, by]


def random_scenario(seed, p0_sd=None):
    """Scenario for this seed: dict(task=..., world=...). Same seed, same scenario, on any computer."""
    rng = np.random.default_rng(10_000 + seed)                  # separate stream from the sensor noise
    u = lambda lo_hi: float(rng.uniform(*lo_hi))               # noqa: E731
    task = load_task()                                         # marker map and dock stay the same

    # ---- KNOWN: start spot (the prior) and route
    start = np.array([u(START_X), u(START_Y), wrap(np.deg2rad(u(START_YAW_DEG)))])
    w1, w2 = [u(W1_X), u(W1_Y)], [u(W2_X), u(W2_Y)]
    w3 = list(task["waypoints"][-1])                           # pre-dock point: on the docking line
    task["start"], task["waypoints"] = start, np.array([w1, w2, w3])

    # ---- UNKNOWN: where the robot really is, what is in the way, how noisy it is
    if p0_sd is None:
        from src.estimation.models import load_estimation_config   # P0 is stated in the EKF config
        p0_sd = load_estimation_config()["P0_sd"]
    sd = np.array([p0_sd[0], p0_sd[1], np.deg2rad(p0_sd[2])])
    placement = np.clip(rng.normal(0.0, sd), -2 * sd, 2 * sd)

    boxes = [_box_in_lane(rng, BOX_ALONG, w1, w2)]          # the lab has one box
    rng.random()                                               # was the second-box draw (removed): kept so every
                                                               # seed draws the same noise as in results/random_trials*.json

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
    """Sim only: place the box, put the robot at its TRUE start, set the noise and the wheel error.

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
