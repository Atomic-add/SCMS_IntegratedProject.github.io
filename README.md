# 41014 Sensors and Control: Integrated Project, Track A

TurtleBot3 **Waffle Pi** in **MuJoCo** and on the lab robot: follow waypoints across the lab floor, avoid
an obstacle with the 2D LiDAR, and dock at a charging station marked with an ArUco marker. There is one
simulated world, the lab track as built in the lab, seen through one simulated camera, the lab robot's.
(The original L-shaped corridor and its level URDF camera live on the branch `feature/induction-ibvs`.) The estimator fuses
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
python main.py --odometry           # steer on wheel odometry only (the baseline, no EKF)
python main.py --seed 3             # another sensor-noise seed
python main.py --no-camera          # MuJoCo viewer only
python main.py --headless           # no windows, prints the result only
python main.py --randomise --seed 7 # random lab scenario: route, start placement, box, noise, wheel error
python main.py --dock pbvs          # dock on the LQR all the way instead of finishing with IBVS
```

**Docking and visual servoing.** From W3 (the pre-dock point) the LQR (EKF heading, camera position) puts the robot on the
marker's normal; below 0.55 m the last part is **image-based visual servoing** (Week 9,
`src/control/visual_servo.py`): (v, omega) = lambda J_mr^+ (p* - p) on the four marker corners, with the
image Jacobian reduced to what the base can do and corrected for the camera sitting 7.6 cm ahead of the
axle. In the camera window the docking marker (ID 0) is outlined in green with its axes and its
position in the camera frame; the EKF's landmark markers are outlined in yellow. `python -m evaluation.check_ibvs` compares IBVS and
PBVS docking on the same seeds -> `results/ibvs_docking.png`.

`--randomise` (`src/robot/scenario.py`): the robot is **told** a new start spot (its EKF prior) and a new
route (the start spot within 0.15 m and 15 deg of the red dot, W1 and the corner W2 within 0.15 m and
0.1 m); it is **not told** where it was really placed (drawn from N(0, P0)), where the box is (always
inside its lane on the long leg, x 1.75-3.0 m, with one side at least 0.7 m wide), how noisy the LiDAR and camera are, or how wrong each wheel radius is.
The terminal prints both halves. In the viewer the red dots are the LiDAR returns placed with the EKF
estimate, and the panel shows the state, the commands, the sd from P and the last NIS.

Linux without a display (SSH, CI): `MUJOCO_GL=egl python main.py --headless` (`--headless` sets
`MUJOCO_GL=egl` itself when it is not set).

What happens: without `--task` the sim is `config/arena_lab.json`, the lab floor as the robot will meet
it (0.5 m tiles, panels, the 20 cm box, three markers, the LDS-01 range and the lab robot's camera:
f 692 px, 49.7 x 38.2 deg, 6.3 deg down, 30 Hz, calibrated into `config/camera_sim.json`). The robot starts from the task file's prior, follows the
waypoints with the LQR, goes around the box with the LiDAR, turns left at the far end and docks at the
charging station (marker ID 0). The terminal prints each state change (TRACK, AVOID, REJOIN, DOCK, STOP)
and finally, for example:

```
result: docked after 46.0 s of sim time, dock_complete = True
```

The exit code is 0 when the robot docked and 1 otherwise. In the viewer the red dots are the LiDAR
returns placed with the EKF estimate (if the estimate drifts, the dots slide off the walls) and the
top-left panel shows the state and the commands. The **robot camera** window outlines every ArUco
marker it sees, with the range and bearing that perception hands to the EKF. Close the viewer to exit.
Viewer: left-drag rotates, right-drag pans, scroll zooms, Tab / Shift+Tab hide the panels.

### The lab track and the real TurtleBot

The lab floor (10 x 4 tiles of 0.5 m) has its own task file, `config/arena_lab.json`, and scene,
`models/robotis_tb3/scene_lab.xml`; `config/stage1_dock.json` (dock only) and `config/stage2_corner.json`
(left turn + dock) reuse it through `"extends"`. The same file runs in the sim and on the robot:

```bash
python main.py                                                # the lab track in the sim
python main.py --task config/stage1_dock.json --case none     # one stage in the sim
python experiments/plot_setup.py                              # layout sheet -> results/setup_arena_lab.png
```

The sim camera is the MJCF camera `rgb`, calibrated into `config/camera_sim.json`
(`python experiments/calibrate_camera.py`, which holds the board along the 6.3 deg tilt). It has no lens
distortion and no range bias, and its pixel noise is still the 2.0 grey-level placeholder. Its marker
noise for the EKF (`ekf_noise`) was fitted by `evaluation/consistency_sweep.py --write-noise`.

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

Wheel calibration (UMBmark, Week 2 B.3-B.4): `experiments/umbmark.py` drives 1 m squares on the encoders
only and asks for three tape readings (two rear chassis corners against an L of tape on the floor) before
and after each run; it fits the left/right wheel sizes and the true track width, `--straight` gives the
overall wheel scale, and the scatter left over gives `k_wheel`. Config files are not changed by it.

```bash
python experiments/umbmark.py --cw --runs 5        # right turns; then --ccw --runs 5, then --straight --runs 3
python experiments/umbmark.py --analyse 'results/hardware/umbmark_*.json'
```

Sim vs real (30 Oct): the docked pose of five real runs, taped at both wheels
(`results/hardware/final_pose_tape_30oct.csv`), against the same runs in the sim with the real control
overrides. Lateral error agrees within 0.6 cm and heading within 0.2 deg on average; the real robot stops
1.2 cm closer to the marker (its coast after the stop command, which `stop_early` compensates).

```bash
python -m evaluation.real_vs_sim      # -> results/real_vs_sim.png / .json
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
python -m src.estimation.observability_test # observability ranks per marker case, F_jac / H_rb vs numerical Jacobians
python -m src.control.lqr_weights_test      # LQR at R x10, x1, /10: K vs control.dlqr, Riccati residual, |eig| < 1
python main.py --headless                    # the lab track must dock (exit code 0)
python main.py --headless --randomise --seed 7   # and so must a random lab scenario
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
python -m evaluation.observability          # rank / Gramian per marker case, P growth with no marker, rank along the path -> results/observability.*
python -m evaluation.sweep_lqr --n 20       # LQR weight R x10, x1, /10 on the EKF: tracking error vs effort -> results/sweep_lqr.*
```

Figures and metrics go to `results/` (`demo_*.png`, `demo_metrics*.json`, `demo_trials_*.json`,
`demo_compare.*`). An EKF run takes about 2 min on a machine without a GPU, because rendering the
camera dominates; odometry-only runs take about 20 s. The trial scripts run `--workers 4` sims at a time
(`evaluation/__init__.py`): one per core renders 20 cameras at once and freezes a laptop.

**Laptops with an NVIDIA GPU (Linux):** with several sims rendering through EGL at once, the NVIDIA driver
(open kernel module 595.84) crashed in `nvidia_unlocked_ioctl` and the machine hung until it was powered off
(26 Oct, twice). Render on Mesa (the Intel iGPU) instead, about 2x slower per frame and stable:

```bash
export MUJOCO_GL=egl __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
python -m evaluation.sweep_r --n 20 --workers 1
```

Random lab scenarios (`random_trials`, seeds 1-20, re-run 31 Oct after the pass-side change):

| Run | Docked | Contacts | Notes |
|---|---|---|---|
| EKF | 20 / 20 | 0 | 42.0 +- 1.9 s, dock lateral 0.23 +- 0.44 cm, yaw -0.37 +- 0.76 deg, ANEES 3.8; seed 5 (box beside the lane) used to stop `too_close` on its narrow side; the pass side is now the one needing the smaller sideways offset (`avoidance.py`) |
| Odometry only | 9 / 20 | 2 | the hidden placement and wheel errors are never corrected (estimate off by up to 74 cm) |

Fixed track (`run_demo`, re-run 30 Oct; repeated trials = 10 seeds of a random true start and encoder scale error):

| Run | none | left | centre | right |
|---|---|---|---|---|
| Nominal, odometry | docked 37.4 s | docked 37.2 s | docked 43.1 s | docked 37.4 s |
| Nominal, EKF | docked 37.4 s, ANEES 2.79 | docked 37.2 s, ANEES 2.62 | docked 43.2 s, ANEES 3.66 | docked 37.4 s, ANEES 2.79 |
| 10 trials, odometry | 9 / 10 | - | 9 / 10 | - |
| 10 trials, EKF | 10 / 10, ANEES 2.78 | - | 10 / 10, ANEES 3.21 | - |

No contact in any run. Every EKF dock ends about 0.43 cm to the side and -0.7 deg off the goal, the same in
every trial (a systematic offset, inside the 3 cm / 5 deg tolerance). NIS stays low (ANIS 0.1) because the
ArUco errors of consecutive frames are correlated (`evaluation/consistency_sweep.py`); NEES against the
truth is inside its 95 % bound in 88-89 % of the steps (95-96 % before the 28 Oct changes).

`sweep_r` (EKF, centre, 20 seeds per row, all docked, 30 Oct): R x10 ANEES 1.47 (worst error 0.81 cm), R as
configured 3.24 (1.80 cm), R with the measured bearing sd 6.34 (3.21 cm), R / 10 9.42 (3.06 cm, 12.7 marker
updates gated per run). The configured R is the consistent one. R x10 has the smaller error only because the
sim odometry has the model's exact wheel radius; the real encoders over-count by about 1.3 %.
`check_ibvs` (10 seeds each, 30 Oct): PBVS and IBVS both dock 10 / 10, lateral +0.26 +- 0.45 cm; IBVS docks in 7.1 s, PBVS in 8.25 s.
`consistency_sweep` (40 recorded runs, 30 Oct): the configured noise gives ANEES 3.00, ANIS 0.12, 2-sigma coverage 0.979, rms 0.69 cm
(fitted noise: ANEES 2.95, coverage 1.000, rms 0.92 cm); the config was not rewritten.

All simulation numbers above were regenerated on 30 Oct on the code after the 28 Oct docking and avoidance changes (Linux, Mesa
software rendering, `--workers 4`).

`sweep_lqr` (31 Oct; EKF, case centre, seeds 1-20 with a random true start and encoder scale error; Q fixed, R = 1 / d_omega_max^2 scaled;
every LQR in the mission changes: tracking, avoidance and the docking approach):

| R | K [e_y, e_theta] | Docked | Cross-track RMS (max), true pose | RMS w | Jitter RMS (w[k]-w[k-1])/Ts | Linear peak d_omega from 20 cm |
|---|---|---|---|---|---|---|
| R x10 (d_omega_max 0.16 rad/s) | [3.05, 1.41] | 20 / 20 | 3.13 +- 0.11 cm (6.35) | 0.098 rad/s | 0.28 rad/s^2 | 0.61 rad/s |
| R (config, 0.5 rad/s) | [9.16, 3.25] | 20 / 20 | 1.73 +- 0.12 cm (5.03) | 0.077 rad/s | 0.38 rad/s^2 | 1.83 rad/s |
| R / 10 (1.58 rad/s) | [24.84, 7.78] | 20 / 20 | 1.42 +- 0.12 cm (4.87) | 0.070 rad/s | 0.57 rad/s^2 | 4.97 rad/s (above the 1.82 rad/s limit) |

Only LQR tracking steps count (state TRACK with v > 0). A 10x smaller R buys 0.3 cm of tracking error for 1.5x the
command jitter (the larger gain passes more of the EKF's estimate noise to the wheels); a 10x larger R doubles the
tracking error. No run saturated at 1.82 rad/s, because the tracking errors stay near 1-6 cm, far below the 20 cm of the
linear test. RMS w does not rise with the gain: the low gain leaves a larger standing error (no integral action), so it
keeps steering against it. The slow closed-loop pole (|z| 0.965) hardly moves with R, so the settling distance from
20 cm stays 1.15-1.21 m; R mostly moves the faster pole (|z| 0.96, 0.87, 0.64).

`observability` (31 Oct; EKF's own F_jac and H_rb at (1.5, 1.25) facing the left panel, 1 s = 20 frames):

| Markers in view | Rank of O, 1 frame | Rank, 1 s of driving | Best sd, weakest direction (1 s) |
|---|---|---|---|
| none | 0 | 0 | unobservable (P grows) |
| one, range + bearing (still, straight or arc) | 2 | 2 | unobservable: turning the pose about the marker |
| one, range only | 1 | 2 | unobservable |
| markers 1 + 2 together | 3 | 3 | 1.35 cm |
| marker 2, then marker 1 | 2 | 3 | 1.77 cm |

One marker gives 2 constraints on the 3-DOF pose, and driving does not add the third: odometry measures motion in the
robot's own frame, so the whole run turned about the marker gives the same ranges and bearings. A second marker, at the
same time or later, does. With no marker (predict only at 0.2 m/s, from the P the EKF reaches with markers 1 + 2) the
3-sigma position bound passes the 3 cm dock tolerance after 3.5 s (0.7 m) and the heading bound passes 5 deg after
4.9 s; the sd itself exceeds the 3 cm docking gate on P after 7.3 s. Along the recorded lab paths (25 Oct, marker_layout.py) the
pose is observable within the next 1 s in 34 % (none) and 33 % (centre) of the frames; no marker is in view in 29 % /
33 % of them; on the final approach only the dock marker is in view, so rank 2 there (docking steers on the camera's
own range and bearing to that marker).

### Placeholders (numbers not measured yet)

| Value | Where | Replaced by |
|---|---|---|
| `k_wheel` = 0.06 (wheel distance noise) | `config/estimation.json` | `python experiments/umbmark.py` on the lab robot |
| `bearing_sd_deg` = 0.1 for the real camera | `config/estimation.json` | moving real runs with a tape-measured truth |
| `pixel_noise` = 2.0 grey levels (sim camera) | `src/robot/sim_robot.py` CameraSpec | a static frame sequence from the real camera |

## 5. Repository layout

```
main.py                 the demo: python main.py (section 2)
config/                 lab task (and its stages), camera calibration, EKF and controller tuning
models/robotis_tb3/     Waffle Pi model (ROBOTIS, fixed), scenes, textures
src/robot/              SimRobot and RealRobot (rosbridge), one interface: read_encoders(), read_scan(),
                        read_image(), send_cmd(v, w), step(); task files and scenes (task.py)
src/perception/         camera calibration (W3 B/C), ArUco range-bearing (W3 E), LiDAR (W2 C, W4 P5)
src/estimation/         wheel odometry, EKF with ArUco range-bearing (W7)
src/control/            error model + LQR (W5, W6), waypoints, LiDAR avoidance, docking, mission loop
experiments/            textures, calibration, lab tools (robot check, frames, prints, plans), viewers
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