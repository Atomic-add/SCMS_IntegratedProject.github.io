"""Project rule check: run before every merge.

    python -m evaluation.check_rules

1. Ground truth: no file in src/estimation, src/control, src/perception, src/robot/real_robot.py or
   main.py imports evaluation/ or names a ground-truth helper (true_pose, ground_truth, ...). Comments count too: commented-out truth code
   looks like truth code to a reader, so it has to go.
2. The unit checks pass: src/estimation/estimation_test.py, src/control/control_test.py, the
   visual-servoing checks in src/control/visual_servo.py (Week 9 Jacobian against finite differences)
   and the real-robot adapter checks in src/robot/real_robot_test.py (fake rosbridge link, no robot).

Exit code 0 = all pass. Pure Python, so it runs the same in PowerShell and in a Linux shell.

No tutorial counterpart (the team's rule check).
Additions: the ground-truth scan of the guarded files and the unit-check runner.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUARDED = [ROOT / "src" / "estimation", ROOT / "src" / "control", ROOT / "src" / "perception",
           ROOT / "src" / "robot" / "real_robot.py", ROOT / "main.py"]
# Words that only appear when something reaches for the simulator's truth.
FORBIDDEN = re.compile(r"\b(evaluation\.|from evaluation|import evaluation|ground_truth|true_pose|true_marker_pose|"
                       r"true_camera_pose|true_intrinsics|true_sensor_pose|\.qpos\b|\.xpos\b|\.xquat\b)")
# Prose that merely says where the truth lives is fine; an import or a call is not.
ALLOWED_PROSE = re.compile(r"(only (in|place|evaluation)|never|not .*truth|evaluation/ (is|may|uses|records)|"
                           r"lives in evaluation/)", re.IGNORECASE)


def scan():
    """(path, line number, text) of every suspicious line in the guarded files."""
    files = []
    for g in GUARDED:
        files += [g] if g.is_file() else sorted(g.rglob("*.py"))
    bad = []
    for f in files:
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if FORBIDDEN.search(line) and not ALLOWED_PROSE.search(line):
                bad.append((f.relative_to(ROOT), n, line.strip()))
    return bad


def run(module):
    """Run a unit-check module from the project root; True when it exits with 0 failures."""
    r = subprocess.run([sys.executable, "-m", module], cwd=ROOT, capture_output=True, text=True)
    last = (r.stdout.strip().splitlines() or ["(no output)"])[-1]
    return r.returncode == 0, last


if __name__ == "__main__":
    ok = True
    bad = scan()
    if bad:
        ok = False
        print("[FAIL] ground truth referenced outside evaluation/:")
        for f, n, line in bad:
            print(f"         {f}:{n}: {line}")
    else:
        print("[PASS] no ground truth in src/estimation, src/control, src/perception, real_robot.py or main.py")
    for module in ("src.estimation.estimation_test", "src.control.control_test", "src.control.visual_servo",
                   "src.robot.real_robot_test"):
        passed, last = run(module)
        ok &= passed
        print(f"[{'PASS' if passed else 'FAIL'}] {module}: {last}")
    print("all rules pass" if ok else "RULE CHECK FAILED")
    sys.exit(0 if ok else 1)
