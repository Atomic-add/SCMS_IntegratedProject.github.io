# 41014 Sensors and Control: Integrated Project, Track A

TurtleBot3 **Waffle Pi** in **MuJoCo** and on the lab robot: follow waypoints across the lab floor (or the
original L-shaped corridor), avoid an obstacle with the 2D LiDAR, and dock at a charging station marked
with an ArUco marker. The estimator fuses
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
python main.py                      # the lab track as built in the lab: centre box, EKF, lab camera, viewer
python main.py --case none          # obstacle case: none | left | centre | right
python main.py --task config/corridor.json   # the original L corridor (level URDF camera)
python main.py --odometry           # steer on wheel odometry only (the baseline, no EKF)
python main.py --seed 3             # another sensor-noise seed
python main.py --no-camera          # MuJoCo viewer only
python main.py --headless           # no windows, prints the result only
python main.py --randomise --seed 7 # random corridor scenario: route, start placement, boxes, noise, wheel error
python main.py --dock pbvs          # dock on the LQR all the way instead of finishing with IBVS
```

**Docking and visual servoing.** From W4 the LQR (EKF heading, camera position) puts the robot on the
marker's normal; below 0.55 m the last part is **image-based visual servoing** (Week 9,
`src/control/visual_servo.py`): (v, omega) = lambda J_mr^+ (p* - p) on the four marker corners, with the
image Jacobian reduced to what the base can do and corrected for the camera sitting 7.6 cm ahead of the
axle. In the camera window the docking marker (ID 0) is outlined in green with its axes and its
position in the camera frame; the EKF's landmark markers are outlined in yellow. `python -m evaluation.check_ibvs` compares IBVS and
PBVS docking on the same seeds -> `results/ibvs_docking.png`.

`--randomise` (`src/robot/scenario.py`): the robot is **told** a new start spot (its EKF prior) and a new
route; it is **not told** where it was really placed (drawn from N(0, P0)), where the one or two boxes
are (always inside its lane), how noisy the LiDAR and camera are, or how wrong each wheel radius is.
The terminal prints both halves. In the viewer the red dots are the LiDAR returns placed with the EKF
estimate, and the panel shows the state, the commands, the sd from P and the last NIS.

Linux without a display (SSH, CI): `MUJOCO_GL=egl python main.py --headless` (`--headless` sets
`MUJOCO_GL=egl` itself when it is not set).

What happens: without `--task` the sim is `config/arena_lab.json`, the lab floor as the robot will meet
it (0.5 m tiles, panels, the 20 cm box, three markers, the LDS-01 range and the lab robot's camera, whose
choice the task file's `sim_camera` makes). The robot starts from the task file's prior, follows the
waypoints with the LQR, goes around the box with the LiDAR, turns left at the far end and docks at the
charging station (marker ID 0). The terminal prints each state change (TRACK, AVOID, REJOIN, DOCK, STOP)
and finally, for example:

```
result: docked after 46.0 s of sim time, dock_complete = True
```

`--task config/corridor.json` runs the original corridor instead (57.1 s on seed 0), with the level URDF
camera its 0.12 m markers were designed for; `--sim-camera lab|urdf` overrides the task file's camera.

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

### The lab track and the real TurtleBot

The lab floor (10 x 4 tiles of 0.5 m) has its own task file, `config/arena_lab.json`, and scene,
`models/robotis_tb3/scene_lab.xml`; `config/stage1_dock.json` (dock only) and `config/stage2_corner.json`
(left turn + dock) reuse it through `"extends"`. The same file runs in the sim and on the robot:

```bash
python main.py                                                # the lab track in the sim, lab robot's camera (f 692 px, 6.3 deg down, 30 Hz)
python main.py --sim-camera urdf                              # ... with the level URDF camera instead
python -m evaluation.lab_camera_trials                        # 4 box cases x 10 seeds per sim camera -> results/lab_camera_trials.json
python experiments/plot_setup.py                              # layout sheet -> results/setup_arena_lab.png
```

`--sim-camera lab` uses the MJCF camera `rgb_lab` and its own calibration `config/camera_sim_lab.json`
(`python experiments/calibrate_camera.py --camera rgb_lab --pitch-deg 6.3 --out config/camera_sim_lab.json`).
It has no lens distortion and no range bias, and its pixel noise is still the 2.0 grey-level placeholder.

The real robot uses `src/robot/real_robot.py` (`RealRobot`, the same interface as `SimRobot`) over the
lab's rosbridge (UTS-RI/turtlebot_rosbridge). Install it once, after `requirements.txt`:

```bash
python -m pip install -r requirements-real.txt
python -m pip install --no-deps "git+https://github.com/UTS-RI/turtlebot_rosbridge.git@8a6c481c506d1b817937d865af0d794ea0d45d51"
```

`--no-deps` matters: the package asks for `opencv-python`, which would remove `cv2.aruco`. In the lab
(laptop on the `turtlebot-MMR` wifi, robot IP `192.168.1.2XX`, default in `config/robot_real.json`):

```bash
python experiments/check_robot.py 192.168.1.2XX --drive          # topics, joint names, scan layout, 10 cm drive
python experiments/capture_frames.py 192.168.1.2XX               # checkerboard frames -> results/hardware/calib
python experiments/calibrate_camera.py --images results/hardware/calib --out config/camera_real.json
python experiments/camera_pitch.py --write                       # camera tilt from ArUco frames -> camera_real.json
python experiments/aruco_real_noise.py --write                   # ArUco range bias + sd vs tape -> camera_real.json range_correction
python main.py --real --task config/stage1_dock.json             # then stage2_corner.json, then arena_lab.json
```

`--real` requires `--task`, caps the speed at `v_max` / `w_max` from `config/robot_real.json`, stops the
robot when `/joint_states` or `/scan` goes quiet, when the link drops, on Ctrl+C, or when a second robot
publishes on the same topics, and saves the run to `results/hardware/run_<time>.npz` (estimate, P,
commands, NIS). The robot's true end pose is measured by hand (tape from the dock panel).

## 3. Checks

Before every merge, from the project root:

```bash
python -m evaluation.check_rules             # ground-truth rule + unit checks + IBVS checks, exit code 0 = all pass
python -m src.control.visual_servo           # Week 9 image Jacobian vs finite differences, IBVS loop on geometry
python -m src.estimation.estimation_test     # EKF and odometry unit checks
python -m src.control.control_test           # controller, avoidance and docking unit checks
python -m src.robot.real_robot_test          # RealRobot on a fake rosbridge link (no robot needed)
python main.py --headless                    # the lab track must dock (exit code 0)
python main.py --headless --task config/corridor.json   # and so must the corridor
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
python -m evaluation.random_trials --n 20    # 20 random scenarios on the EKF -> results/random_trials.png/.json
python -m evaluation.random_trials --n 20 --odometry   # the same 20 on odometry only
python -m evaluation.sweep_r --n 20          # R x10, x1, /10: NEES, NIS, error -> results/sweep_r.png/.json
python -m evaluation.marker_layout           # where the lab landmarks go: markers in view + EKF replay -> results/marker_layout.*
python -m evaluation.consistency_sweep       # Qd x Rn grid on recorded lab runs: ANEES, ANIS, 2-sigma coverage -> results/consistency_*.*
python -m evaluation.lab_camera_trials       # lab track, both sim cameras, 4 cases x 10 seeds -> results/lab_camera_trials.json
```

Figures and metrics go to `results/` (`demo_*.png`, `demo_metrics*.json`, `demo_trials_*.json`,
`demo_compare.*`). An EKF run takes about 2 min on a machine without a GPU, because rendering the
camera dominates; odometry-only runs take about 20 s.

## 5. Repository layout

```
main.py                 the demo: python main.py (section 2)
config/                 corridor task, camera calibration, EKF and controller tuning
models/robotis_tb3/     Waffle Pi model (ROBOTIS, fixed), scenes, textures
src/robot/              SimRobot and RealRobot (rosbridge), one interface: read_encoders(), read_scan(),
                        read_image(), send_cmd(v, w), step(); task files and scenes (corridor.py)
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
- Code adapted from the 41014 Week 2-9 tutorial solutions (cited in each module's docstring).
- Real-robot link: [UTS-RI/turtlebot_rosbridge](https://github.com/UTS-RI/turtlebot_rosbridge) (`rosbridge_api`,
  MIT), used unmodified; `src/robot/real_robot.py` hooks its message callbacks to keep arrival times.
- Generative AI: Claude (Anthropic) was used to scaffold the simulation environment, the sensor
  models and their checks; see the project page's GenAI declaration.