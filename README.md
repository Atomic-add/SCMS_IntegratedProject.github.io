# 41014 Sensors and Control: Integrated Project, Track A

TurtleBot3 **Waffle Pi** in **MuJoCo**: follow waypoints along an L-shaped corridor, avoid an obstacle
with the 2D LiDAR, and dock at a charging station marked with an ArUco marker. The estimator fuses
wheel-encoder odometry with camera range-bearing measurements to known ArUco markers (EKF); a
state-feedback (LQR) controller tracks the waypoints.

**Status (9 Oct):** environment, sensors and perception are done and verified (`check_lidar` 6/6,
`check_camera` 8/8). Odometry + EKF, LQR, avoidance and docking are in progress.

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

## 2. See the environment

```bash
python main.py                      # L-corridor, no obstacle
python main.py --obstacle centre    # none / left / right / centre
```

Two windows open: the MuJoCo viewer (top view, red dots = LiDAR hits) and **robot camera**, which
outlines every ArUco marker the camera detects. The robot stands still for now: drive it with the
`wheel_left` / `wheel_right` sliders under **Control** in the viewer's right panel.
Viewer: left-drag rotates, right-drag pans, scroll zooms, Tab / Shift+Tab hide the panels.

Other views:

```bash
python -m mujoco.viewer --mjcf=models/robotis_tb3/scene_corridor.xml   # the scene file alone
python evaluation/check_lidar.py --viewer     # robot turning in the test room, LiDAR dots
python evaluation/check_camera.py --viewer    # robot driving at a dock, camera + ArUco window
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
base origin (`base_footprint`).

## 3. Sensor checks

Run these after changing anything in `models/`, `src/robot/` or `src/perception/`:

```bash
python experiments/make_textures.py      # marker and checkerboard images (scenes need them)
python experiments/calibrate_camera.py   # camera K from checkerboard views -> config/camera_sim.json
python evaluation/check_lidar.py         # expect 6/6 checks passed -> results/lidar_check.png
python evaluation/check_camera.py        # expect 8/8 checks passed (1-2 min) -> results/camera_check.png
```

Headless Linux (no display): prefix with `MUJOCO_GL=egl`, e.g. `MUJOCO_GL=egl python evaluation/check_camera.py`.

## 4. Repository layout

```
main.py                 demo entry point (currently: the environment viewer)
config/                 corridor task, camera calibration
models/robotis_tb3/     Waffle Pi model (ROBOTIS, fixed), scenes, textures
src/robot/              SimRobot: read_encoders(), read_scan(), read_image(), send_cmd(v, w), step()
src/perception/         camera calibration (W3 B/C), ArUco range-bearing (W3 E), LiDAR (W2 C, W4 P5)
src/estimation/         odometry, EKF                     (in progress)
src/control/            LQR, waypoints, avoidance, docking (in progress)
experiments/            textures, calibration, viewers, trial scripts
evaluation/             checks and metrics: the ONLY place that reads simulator ground truth
results/                generated figures and CSV
```

## 5. Rules for everyone

- **No ground truth in `src/`.** The estimator and controller start from the prior in `config/` and
  use sensor data only. Ground truth is read only in `evaluation/`.
- Import the robot from one place: `from src.robot.sim_robot import SimRobot`.
- Fuse a camera or LiDAR measurement only when its timestamp is new (`read_image()` / `read_scan()`
  return `t`); otherwise the same frame is fused several times.
- One feature branch per task (`feature/ekf`, `feature/lqr`, ...), merged by pull request after a
  teammate's review. `main` must always run.
- Paths with `pathlib`, no absolute paths; code, comments and commits in English.

## 6. Acknowledgements

- Robot model and meshes: [ROBOTIS robotis_mujoco_menagerie](https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie),
  Apache-2.0 (`models/robotis_tb3/LICENSE`); sensor poses from ROBOTIS `turtlebot3_waffle_pi.urdf`.
- Code adapted from the 41014 Week 2-8 tutorial solutions (cited in each module's docstring).
- Generative AI: Claude (Anthropic) was used to scaffold the simulation environment, the sensor
  models and their checks; see the project page's GenAI declaration.