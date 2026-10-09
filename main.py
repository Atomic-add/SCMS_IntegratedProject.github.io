"""Track A demo entry point.

    python main.py                       # L-corridor, no obstacle
    python main.py --obstacle centre     # none / left / right / centre
    python main.py --no-camera           # MuJoCo viewer only

Current stage: the environment and the sensors only. The robot stands still; drive it with the
wheel sliders (viewer, right panel > Control). The estimation -> control -> docking pipeline will be
wired in here (odometry + EKF, LQR waypoint tracking, LiDAR avoidance, ArUco docking).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))      # project root, so "src" imports work

from experiments.view_corridor import main as view_environment  # noqa: E402

if __name__ == "__main__":
    view_environment()