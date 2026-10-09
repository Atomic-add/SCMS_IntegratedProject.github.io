"""The L-corridor task: load config/corridor.json and set up a trial in the simulator.

Sim side only (like SimRobot.reset): it moves the obstacle box, which the robot's software must not know
about. The marker map and the waypoints in the same file are what the estimator and controller may use;
they get them from load_task(), never the obstacle positions.
"""
import json

import mujoco
import numpy as np

from src.robot.sim_robot import PROJECT_ROOT, SimRobot

CONFIG = PROJECT_ROOT / "config" / "corridor.json"
PARKED = (3.6, 1.7, -1.0)          # obstacle hidden under the floor = config "none"


def load_config(path=CONFIG):
    """The whole corridor config as a dict."""
    return json.loads(path.read_text())


def load_task(path=CONFIG):
    """What the robot's software may know: start prior, waypoints, marker map, dock goal."""
    cfg = load_config(path)
    markers = {int(k): np.array(v) for k, v in cfg["markers"].items() if not k.startswith("_")}
    return dict(start=np.array(cfg["start"]), waypoints=np.array(cfg["waypoints"]), markers=markers,
                dock=cfg["dock"])


def set_obstacle(robot, name, path=CONFIG):
    """Sim only: put the obstacle box at one of the configs in corridor.json (none / left / right / centre)."""
    xy = load_config(path)["obstacles"][name]
    gid = robot.m.geom("obstacle").id
    half_h = robot.m.geom_size[gid][2]
    robot.m.geom_pos[gid] = PARKED if xy is None else (xy[0], xy[1], half_h)
    mujoco.mj_forward(robot.m, robot.d)


def make_robot(obstacle="none", seed=0, path=CONFIG, **kw):
    """SimRobot in the corridor, obstacle placed, robot at the start pose."""
    cfg = load_config(path)
    robot = SimRobot(scene=PROJECT_ROOT / cfg["scene"], seed=seed, **kw)
    set_obstacle(robot, obstacle, path)
    robot.reset(*cfg["start"])
    return robot