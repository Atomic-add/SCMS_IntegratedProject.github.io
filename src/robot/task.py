"""The lab task: load config/arena_lab.json (or a stage file that extends it) and set up a sim trial.

Sim side only (like SimRobot.reset): it moves the obstacle box, which the robot's software must not know
about. The marker map and the waypoints in the same file are what the estimator and controller may use;
they get them from load_task(), never the obstacle positions.
"""
import json
from pathlib import Path

import mujoco
import numpy as np

from src.robot.sim_robot import PROJECT_ROOT, LidarSpec, SimRobot

CONFIG = PROJECT_ROOT / "config" / "arena_lab.json"
PARKED = (2.75, 1.25, -1.0)        # obstacle hidden under the floor = config "none"


def load_config(path=CONFIG):
    """The whole task config as a dict.

    A file with "extends": "<task file>" takes every top-level key it does not set from that file, so a
    staged run (config/stage*.json) changes only its start and waypoints and keeps the arena's markers.
    """
    cfg = json.loads(Path(path).read_text())
    base = cfg.pop("extends", None)
    if base is not None:
        cfg = {**load_config(PROJECT_ROOT / base), **cfg}
    return cfg


def load_task(path=CONFIG):
    """What the robot's software may know: start prior, waypoints, marker map, dock goal."""
    cfg = load_config(path)
    markers = {int(k): np.array(v) for k, v in cfg["markers"].items() if not k.startswith("_")}
    return dict(start=np.array(cfg["start"]), waypoints=np.array(cfg["waypoints"]), markers=markers,
                dock=cfg["dock"], control=cfg.get("control", {}))


def apply_control_overrides(cfg, task):
    """Controller config with the task file's "control" block on top (e.g. a wider avoidance clearance
    for a smaller box). Nested dicts merge key by key; keys starting with "_" are comments."""
    def merge(base, extra):
        for k, v in extra.items():
            if k.startswith("_"):
                continue
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                merge(base[k], v)
            else:
                base[k] = v
    merge(cfg, task.get("control", {}))
    return cfg


OBSTACLE_GEOMS = ("obstacle",)               # the movable box in scene_lab.xml


def place_obstacles(robot, centres):
    """Sim only: put the movable boxes at the given (x, y) centres, in order; boxes not given are parked.

    Used by src/robot/scenario.py for randomised trials. The robot's software never sees these positions.
    """
    centres = list(centres)
    if len(centres) > len(OBSTACLE_GEOMS):
        raise ValueError(f"the scene has {len(OBSTACLE_GEOMS)} movable boxes, {len(centres)} requested")
    for k, name in enumerate(OBSTACLE_GEOMS):
        gid = robot.m.geom(name).id
        half_h = robot.m.geom_size[gid][2]
        xy = centres[k] if k < len(centres) else None
        robot.m.geom_pos[gid] = PARKED if xy is None else (xy[0], xy[1], half_h)
    mujoco.mj_forward(robot.m, robot.d)


def set_obstacle(robot, name, path=CONFIG):
    """Sim only: put the obstacle box at one of the task file's cases (none / left / centre / right)."""
    xy = load_config(path)["obstacles"][name]
    place_obstacles(robot, [] if xy is None else [xy])


def make_robot(obstacle="none", seed=0, path=CONFIG, **kw):
    """SimRobot on the task file's floor, obstacle placed, robot at the start pose, lab camera.

    The task file's "lidar" block (the lab's LDS-01 range in arena_lab.json) sets the sim LiDAR unless
    the caller passes its own lidar=.
    """
    cfg = load_config(path)
    if "lidar" in cfg and "lidar" not in kw:
        kw["lidar"] = LidarSpec(**{k: v for k, v in cfg["lidar"].items() if not k.startswith("_")})
    robot = SimRobot(scene=PROJECT_ROOT / cfg["scene"], seed=seed, **kw)
    set_obstacle(robot, obstacle, path)
    robot.reset(*cfg["start"])
    return robot