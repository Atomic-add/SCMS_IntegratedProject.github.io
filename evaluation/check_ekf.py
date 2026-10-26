"""Whole-mission check of the EKF against simulator truth (evaluation only).

    python -m evaluation.check_ekf [none|left|centre|right]

Drives the whole mission with the controller steering on the EKF, then prints how far the estimate
was from the simulator's truth. Truth is read here, through evaluation/ground_truth.py, and nowhere
in src/estimation or src/control.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root, so "src" imports work even
                                                                # when this file is run directly (VS Code Run)

from evaluation.ground_truth import true_pose  # noqa: E402
from src.control.behaviour import run_mission  # noqa: E402
from src.estimation.ekf import EKF  # noqa: E402
from src.robot.task import load_task  # noqa: E402

if __name__ == "__main__":
    case = sys.argv[1] if len(sys.argv) > 1 else "centre"
    task = load_task()
    ekf = EKF(task["start"], marker_map=task["markers"])
    err = []

    def on_step(robot, row):
        e = np.asarray(true_pose(robot), float) - row["xh"]
        err.append(np.hypot(e[0], e[1]))

    info, t_end = run_mission(case, verbose=True, estimator=ekf, on_step=on_step)
    nis = np.array([n for _, _, n, _ in ekf.log])
    print(f"\nresult: {info['stop_reason']} after {t_end:.1f} s, dock_complete = {info['dock_complete']}")
    print(f"position error |x_hat - true|: max {100 * max(err):.1f} cm, final {100 * err[-1]:.1f} cm")
    print(f"marker updates: {len(nis)}, gated out {int(np.sum(nis > ekf.p['nis_gate']))}, mean NIS {nis.mean():.2f} "
          f"(about 2 if R is right)")
