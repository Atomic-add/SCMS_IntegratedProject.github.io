"""Watch the whole mission live in the MuJoCo viewer (real time).

    python experiments/simulate.py                      # centre case, EKF steering
    python experiments/simulate.py none                 # another case: none | left | centre | right
    python experiments/simulate.py centre --odometry    # odometry only (the baseline)
    python experiments/simulate.py centre --speed 1     # real time (default 4x: about 15 s per case)

Or open this file in VS Code and press Run (centre, EKF, 4x). Red dots = LiDAR, top-left = state and
commands. Drag to rotate, scroll to zoom. Close the window to finish; the result prints here.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root, so "src" imports work

from src.control.behaviour import run_mission                   # noqa: E402
from src.estimation.ekf import EKF                              # noqa: E402
from src.robot.corridor import load_task                        # noqa: E402

if __name__ == "__main__":
    case = next((a for a in sys.argv[1:] if a in ("none", "left", "centre", "right")), "centre")
    task = load_task()
    estimator = None if "--odometry" in sys.argv else EKF(task["start"], marker_map=task["markers"])
    speed = float(sys.argv[sys.argv.index("--speed") + 1]) if "--speed" in sys.argv else 4.0
    # the viewer waits one control step (50 ms) per step to run in real time; wait 1/speed of it instead
    real_sleep = time.sleep
    time.sleep = lambda s: real_sleep(s / speed)
    print(f"case {case}, steering on {'odometry' if estimator is None else 'the EKF'}, {speed:g}x speed")
    info, t_end = run_mission(case, verbose=True, estimator=estimator, view=True)
    print(f"result: {info['stop_reason']} after {t_end:.1f} s, dock_complete = {info['dock_complete']}")
