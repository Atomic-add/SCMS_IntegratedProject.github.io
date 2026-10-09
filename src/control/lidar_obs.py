"""STAND-IN for the Perception teammate's LiDAR output: builds the team plan's `obs` fields from one scan.

The team plan ("LiDAR output specification") gives Perception the job of turning /scan into:

    t, valid      scan time [s]; True only if the agreed beam-coverage check passed
    points_B      (2, M) valid return points in base_footprint {B} [m]
    nearest       (range, bearing) of the closest valid return; (inf, nan) and valid=False if none
    free_ahead    metres from the robot's front to a return in its forward swept corridor
                  (robot width + safety margin), inf if none is seen there
    left_clear    minimum valid range in the forward-left sector [15, 75] deg, nan if no return there
    right_clear   same in the forward-right sector [-75, -15] deg

Nobody owns that glue yet, so this file builds the same dict from the existing, checked helpers in
src/perception/lidar.py (nothing there is changed). When Perception delivers theirs, avoidance.py and
behaviour.py keep working unchanged: delete this file and pass their obs instead.

Settings (sector limits, coverage, margin) are in config/control.json "lidar_obs".
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # project root, so "src" imports work even
                                                                # when this file is run directly (VS Code Run)

from src.control.models import load_control_config  # noqa: E402
from src.perception import lidar  # noqa: E402


def make_obs(t_scan, rho, range_min, range_max, cfg=None):
    """The plan's obs dict from one scan (t_scan, rho) as SimRobot.read_scan() or /scan gives it."""
    c = (cfg or load_control_config())["lidar_obs"]
    alpha = lidar.beam_angles(len(rho))                               # beam i at i deg in the sensor frame
    ok = lidar.valid_beams(rho, range_min, range_max)                 # drop 0 / inf / nan / out of range
    P_B = np.asarray(lidar.to_base(lidar.scan_to_points(rho, alpha, ok)), float).reshape(2, -1)

    # Coverage check: "missing returns do not prove clearance". In the forward half (+-90 deg) at least
    # min_coverage of the beams must return something, otherwise the scan is not trusted for moving.
    fwd = np.cos(alpha) >= 0.0                                        # beams pointing within +-90 deg
    coverage = float(np.mean(ok[fwd])) if np.any(fwd) else 0.0

    nearest = lidar.nearest_obstacle(P_B)                            # (inf, nan) when there is no return
    bearing_B = np.degrees(np.arctan2(P_B[1], P_B[0]))
    range_B = np.hypot(P_B[0], P_B[1])

    def sector_min(lo, hi):
        """Minimum range of the returns whose bearing in {B} is within [lo, hi] deg; nan if none."""
        sel = (bearing_B >= lo) & (bearing_B <= hi)
        return float(np.min(range_B[sel])) if np.any(sel) else np.nan

    return dict(
        t=float(t_scan),
        valid=bool(coverage >= c["min_coverage"] and np.isfinite(nearest[0])),
        coverage=coverage,
        points_B=P_B,
        nearest=nearest,
        free_ahead=lidar.free_distance_ahead(P_B, margin=c["margin"]),
        left_clear=sector_min(*c["left_sector_deg"]),
        right_clear=sector_min(*c["right_sector_deg"]),
    )


if __name__ == "__main__":
    # Run from the project root:  python -m src.control.lidar_obs
    # Prints the obs fields at the start pose and just in front of the centre obstacle.
    from src.robot.corridor import make_robot

    with make_robot("centre", seed=0) as robot:
        for x, y, yaw, where in [(0.5, 0.0, 0.0, "start, facing +x"), (3.6, 0.9, np.pi / 2, "0.65 m before the centre box")]:
            robot.reset(x, y, yaw)
            t, rho = robot.read_scan()
            o = make_obs(t, rho, robot.lidar.range_min, robot.lidar.range_max)
            print(f"{where}: valid {o['valid']} (coverage {o['coverage']:.2f}), free_ahead {o['free_ahead']:.2f} m, "
                  f"left_clear {o['left_clear']:.2f} m, right_clear {o['right_clear']:.2f} m, "
                  f"nearest {o['nearest'][0]:.2f} m at {np.degrees(o['nearest'][1]):.0f} deg, {o['points_B'].shape[1]} points")
