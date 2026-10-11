# 41014 Sensors and Control: Integrated Project, Track A

TurtleBot3 **Waffle Pi** in **MuJoCo**: follow waypoints along an L-shaped corridor, avoid an obstacle
with the 2D LiDAR, and dock at a charging station marked with an ArUco marker. The estimator fuses
wheel-encoder odometry with camera range-bearing measurements to known ArUco markers (EKF); a
state-feedback (LQR) controller tracks the waypoints on the EKF estimate.

**Status (11 Oct):** the whole mission runs in simulation: odometry + EKF with ArUco range-bearing,
LQR waypoint tracking, LiDAR avoidance and docking (`python main.py`). Sensor checks: `check_lidar`
6/6, `check_camera` 8/8; unit checks: `control_test` and `estimation_test`. Next: model and
observability analysis, repeated trials, hardware.

## 1. Set up (once per computer)

You need **Python 3.10** (64-bit) and **Git**. Each member creates their own virtual environment
`.venv_mujoco` inside the project folder; it is never committed.

### Linux (Ubuntu)

```bash
git clone https://github.com/Atomic-add/IntegratedProject_SensorsAndControlsForMechatronicSystems_SPR2026.git
cd IntegratedProject_SensorsAndControlsForMechatronicSystems_SPR2026
python3.10 -m venv .venv_mujoco            # Ubuntu: sudo apt install python3.10-venv if this fails
source .venv_mujoco/bin/activate
python -m pip install --upgrade pip
python -m pip uninstall -y opencv-python  # only matters if it was installed before ("Skipping" is fine)
python -m pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
git clone https://github.com/Atomic-add/IntegratedProject_SensorsAndControlsForMechatronicSystems_SPR2026.git
cd IntegratedProject_SensorsAndControlsForMechatronicSystems_SPR2026
py -3.10 -m venv .venv_mujoco
.\.venv_mujoco\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip uninstall -y opencv-python
python -m pip install -r requirements.txt
```

If PowerShell blocks `Activate.ps1`, skip activation and call the environment's Python directly,
e.g. `.\.venv_mujoco\Scripts\python.exe main.py`. More detail: `mujoco_installation_guide/`.

### Check the installation

```bash
python -c "import sys, mujoco, cv2; print(sys.version.split()[0], mujoco.__version__, cv2.__version__)"
```

Expected: `3.10.x 3.15.0 4.10.0`.

`requirements.txt` pins every package (cp310 wheels exist for Windows and Linux):
mujoco 3.15.0, numpy 2.2.6, scipy 1.15.3, matplotlib 3.10.9, spatialmath-python 1.1.18,
opencv-contrib-python 4.10.0.84 (the Week 3 pin). Never install `opencv-python` next to it: both
provide `cv2` and break each other. After a `git pull` that changes `requirements.txt`, run the
install line again.

**Every time** you open a new terminal: go to the project folder and activate the environment
(`source .venv_mujoco/bin/activate` or `.\.venv_mujoco\Scripts\Activate.ps1`).

## 2. Run the demo

The same commands work in a Linux terminal and in Windows PowerShell:

```bash
python main.py                      # centre box, EKF, MuJoCo viewer + camera window, real time
python main.py --case none          # obstacle case: none | left | centre | right
python main.py --odometry           # steer on wheel odometry only (the baseline, no EKF)
python main.py --seed 3             # another sensor-noise seed
python main.py --no-camera          # MuJoCo viewer only
python main.py --headless           # no windows, prints the result only
```

Linux without a display (SSH, CI): `MUJOCO_GL=egl python main.py --headless` (`--headless` sets
`MUJOCO_GL=egl` itself when it is not set).

What happens: the robot starts from the prior in `config/corridor.json`, follows start -> W1 -> W2 -> W3
-> W4 with the LQR, goes around the box with the LiDAR, then docks at the charging station (marker ID 0).
The terminal prints each state change (TRACK, AVOID, REJOIN, DOCK, STOP) and finally, for example:

```
result: docked after 57.1 s of sim time, dock_complete = True
```

The exit code is 0 when the robot docked and 1 otherwise. In the viewer the red dots are the LiDAR
returns placed with the EKF estimate (if the estimate drifts, the dots slide off the walls) and the
top-left panel shows the state and the commands. The **robot camera** window outlines every ArUco
marker it sees, with the range and bearing that perception hands to the EKF. Close the viewer to exit.
Viewer: left-drag rotates, right-drag pans, scroll zooms, Tab / Shift+Tab hide the panels.

Drive the robot by hand instead (wheel sliders under **Control** in the viewer's right panel; LiDAR
drawn with wheel odometry):

```bash
python experiments/view_corridor.py --obstacle centre    # none / left / right / centre
```

### The corridor (`models/robotis_tb3/scene_corridor.xml`, numbers in `config/corridor.json`)

| Part | Size | Notes |
|---|---|---|
| Leg 1 (+x) | 4.2 x 1.0 m | start (0.5, 0, 0 deg) -> W1 -> W2 |
| Corner | 1.2 x 1.2 m | 90 deg turn at W2 |
| Obstacle section (+y) | 1.5 m wide | one 0.3 x 0.3 x 0.25 m box: none / left / right / centre |
| Dock area | 2.0 x 2.0 m | W4 pre-dock, 1.9 m straight approach, stop at 0.356 m base-to-marker |
| Walls | 0.35 m high | LiDAR scan plane is 0.132 m above the floor |

Three fixed ArUco markers (DICT_4X4_50, 0.10 m) form the EKF's map: id 1 at the end of leg 1,
id 2 on the dock-area wall, id 0 on the charging station. All ranges are measured from the robot
base origin (`base_footprint`). Only the centre box blocks the robot's lane; the left and right boxes
leave it free, so those cases drive the same path as `none`.

## 3. Checks

Before every merge, from the project root:

```bash
python -m evaluation.check_rules             # ground-truth rule + both unit checks, exit code 0 = all pass
python -m src.estimation.estimation_test     # EKF and odometry unit checks
python -m src.control.control_test           # controller, avoidance and docking unit checks
```

After changing anything in `models/`, `src/robot/` or `src/perception/`:

```bash
python experiments/make_textures.py      # marker and checkerboard images (scenes need them)
python experiments/calibrate_camera.py   # camera K from checkerboard views -> config/camera_sim.json
python evaluation/check_lidar.py         # expect 6/6 checks passed -> results/lidar_check.png
python evaluation/check_camera.py        # expect 8/8 checks passed (1-2 min) -> results/camera_check.png
python evaluation/drive_square_test.py --no-viewer   # odometry around a 0.5 m square, graded on truth
```

Headless Linux (no display): prefix with `MUJOCO_GL=egl`, e.g. `MUJOCO_GL=egl python evaluation/check_camera.py`.

## 4. Results

```bash
python -m evaluation.run_demo                # everything: 4 cases on odometry and on the EKF, repeated trials, comparison
python -m evaluation.run_demo centre --ekf   # one case on the EKF -> results/demo_centre_ekf.png
```

Figures and metrics go to `results/` (`demo_*.png`, `demo_metrics*.json`, `demo_trials_*.json`,
`demo_compare.*`). An EKF run takes about 2 min on a machine without a GPU, because rendering the
camera dominates; odometry-only runs take about 20 s.

## 5. Repository layout

```
main.py                 the demo: python main.py (section 2)
config/                 corridor task, camera calibration, EKF and controller tuning
models/robotis_tb3/     Waffle Pi model (ROBOTIS, fixed), scenes, textures
src/robot/              SimRobot: read_encoders(), read_scan(), read_image(), send_cmd(v, w), step()
src/perception/         camera calibration (W3 B/C), ArUco range-bearing (W3 E), LiDAR (W2 C, W4 P5)
src/estimation/         wheel odometry, EKF with ArUco range-bearing (W7)
src/control/            error model + LQR (W5, W6), waypoints, LiDAR avoidance, docking, mission loop
experiments/            textures, calibration, viewers, trial scripts
evaluation/             checks, metrics and trials: the ONLY place that reads simulator ground truth
results/                generated figures and CSV
mujoco_installation_guide/   step-by-step setup for Windows
```

## 6. Rules for everyone

- **No ground truth outside `evaluation/`.** The estimator and controller start from the prior in
  `config/` and use sensor data only. `python -m evaluation.check_rules` fails if `src/estimation` or
  `src/control` imports `evaluation` or uses the simulator's true pose.
- Import the robot from one place: `from src.robot.sim_robot import SimRobot`.
- Fuse a camera or LiDAR measurement only when its timestamp is new (`read_image()` / `read_scan()`
  return `t`); otherwise the same frame is fused several times.
- One feature branch per task (`feature/ekf`, `feature/lqr`, ...), merged by pull request after a
  teammate's review. `main` must always run, and the checks in section 3 must pass before a merge.
- Paths with `pathlib`, no absolute paths; code, comments and commits in English.

## 7. Acknowledgements

- Robot model and meshes: [ROBOTIS robotis_mujoco_menagerie](https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie),
  Apache-2.0 (`models/robotis_tb3/LICENSE`); sensor poses from ROBOTIS `turtlebot3_waffle_pi.urdf`.
- Code adapted from the 41014 Week 2-8 tutorial solutions (cited in each module's docstring).
- Generative AI: Claude (Anthropic) was used to scaffold the simulation environment, the sensor
  models and their checks; see the project page's GenAI declaration.