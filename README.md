# TurtleBot3 Navigation and Visual Docking

**41014 Sensors and Control in Mechatronics Systems - Integrated Project, Track A, Team 12**

A TurtleBot3 Waffle Pi follows waypoints, avoids an obstacle using 2D LiDAR, and docks at an ArUco marker. The system combines wheel odometry, an ArUco range-and-bearing EKF, LQR path tracking and image-based visual servoing.

The same perception, estimation and control logic is used in MuJoCo and on the real TurtleBot3, with separate robot interfaces and configuration files. The controller uses estimated states and sensor measurements; simulator ground truth is used only for evaluation.

## Project page

See the [project page](index.md) for the model, calibration, estimator and controller design, results, demo video, team contributions and generative AI declaration.

## Installation

Use **Python 3.10**.

```bash
git clone https://github.com/Atomic-add/IntegratedProject_SensorsAndControlsForMechatronicSystems_SPR2026.git
cd IntegratedProject_SensorsAndControlsForMechatronicSystems_SPR2026
python -m venv .venv
```

Activate the environment:

**Linux:**
```bash
source .venv/bin/activate
```

**Windows PowerShell:**
```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

For real-robot operation:

```bash
python -m pip install -r requirements-real.txt
python -m pip install --no-deps "git+https://github.com/UTS-RI/turtlebot_rosbridge.git@8a6c481c506d1b817937d865af0d794ea0d45d51"
```

`--no-deps` keeps `opencv-contrib-python` (and `cv2.aruco`) from being replaced by `opencv-python`.

## Run the simulation

| Command | What it runs |
|---|---|
| `python main.py` | Default mission, box centred in the lane |
| `python main.py --randomise --seed 7` | Randomised start, route, obstacle and sensor conditions |
| `python main.py --odometry` | Odometry-only baseline, without EKF corrections |

## Run the real robot

Connect the laptop and TurtleBot to the same network, start rosbridge on the robot, and set the robot host in `config/robot_real.json`.

```bash
python main.py --real --task config/arena_lab.json
```

Each real run is logged to `results/hardware/run_<time>.npz`.

## Reproduce results and run checks

Run the 20 randomised trials and the simulation–hardware docking comparison:

```bash
python -m evaluation.random_trials --n 20
python -m evaluation.real_vs_sim
```

Run the project checks:

```bash
python -m evaluation.check_rules
```

See the Reproducibility section of the project page for the remaining evaluation commands.

## Repository structure

| Folder | Contents |
|---|---|
| `src/` | Robot interfaces, perception, estimation and control |
| `config/` | Calibration, task and tuning parameters |
| `models/` | MuJoCo scenes and robot assets |
| `evaluation/` | Validation scripts and performance evaluation |
| `experiments/` | Calibration and hardware experiment tools |
| `results/` | Saved metrics, figures and hardware logs |

## Team

- Addison Possamai - ID: 25840164
- Nguyen Kim Uyen Phan - ID: 25596519
- Nhat Thien Mai - ID: 25161391

