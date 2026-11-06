# Track A project page structure (final)

## GROUP NUMBER / TEAM MEMBERS
**TEAM 12**
- Addison Possamai (25840164)
- Nguyen Kim Uyen Phan (25596519)
- Nhat Thien Mai (25161391)

## Contents

- **[1. Overview](#1-overview)**
  - [1.1 Introduction](#11-introduction)
  - [1.2 Pipeline diagram, simulation and real-robot photos](#12-pipeline-diagram-simulation-and-real-robot-photos)
- **[2. Modelling (A)](#2-modelling-a-state-space-model-controllability-and-observability-analysis)**
  - [2.1 State-space model](#21-state-space-model)
  - [2.2 Controllability analysis](#22-controllability-analysis)
  - [2.3 Observability analysis](#23-observability-analysis)
  - [2.4 Model correctness](#24-model-correctness)
- **[3. Sensing (B)](#3-sensing-b-calibration-noise-handling-and-frame-management)**
  - [3.1 Calibration: camera, plus LiDAR validated against ray-cast truth](#31-calibration-camera-plus-lidar-validated-against-ray-cast-truth)
  - [3.2 Noise handling: measured noise characterisation](#32-noise-handling-measured-noise-characterisation)
  - [3.3 Noise handling: outlier rejection and marker-loss handling](#33-noise-handling-outlier-rejection-and-marker-loss-handling)
  - [3.4 Frame management: frame chain from sensor measurement to controller input](#34-frame-management-frame-chain-from-sensor-measurement-to-controller-input)
  - [3.5 Simulated vs real sensor noise, and why they differ](#35-simulated-vs-real-sensor-noise-and-why-they-differ)
- **[4. Estimation (C)](#4-estimation-c-ekf-design-tuning-and-reported-performance)**
  - [4.1 EKF design](#41-ekf-design)
  - [4.2 Tuning: Q, R and P0, and where each value comes from](#42-tuning-q-r-and-p0-and-where-each-value-comes-from)
  - [4.3 Reported performance: RMSE, NEES, NIS, ±3σ bounds](#43-reported-performance-rmse-nees-nis-3σ-bounds)
  - [4.4 Tuning trade-off: consistency vs responsiveness (R sweep)](#44-tuning-trade-off-consistency-vs-responsiveness-r-sweep)
  - [4.5 Comparison with a second estimator: EKF vs odometry only, same seeds](#45-comparison-with-a-second-estimator-ekf-vs-odometry-only-same-seeds)
- **[5. Control (D)](#5-control-d-controller-design-tuning-and-tracking-results)**
  - [5.1 Controller design: LQR, Turn, Avoid and IBVS](#51-controller-design-lqr-turn-avoid-and-ibvs)
  - [5.2 Tuning](#52-tuning)
  - [5.3 Tracking Results in Simulation](#53-tracking-results-in-simulation)
  - [5.4 Tuning trade-off: control effort vs tracking error](#54-tuning-trade-off-control-effort-vs-tracking-error)
- **[6. Integration and Robustness (E)](#6-integration-and-robustness-e-performance-under-disturbance-and-noise)**
  - [6.1 Performance under disturbance and noise](#61-performance-under-disturbance-and-noise)
  - [6.2 Failure analysis traced to model, sensing, estimator or controller](#62-failure-analysis-traced-to-model-sensing-estimator-or-controller)
- **[7. Real Robot](#7-real-robot)**
  - [7.1 Setup and differences from simulation](#71-setup-and-differences-from-simulation)
  - [7.2 Real-robot results with the same metrics](#72-real-robot-results-with-the-same-metrics)
  - [7.3 Quantitative simulation vs real comparison](#73-quantitative-simulation-vs-real-comparison)
- **[8. Limitations and Lessons](#8-limitations-and-lessons)**
  - [8.1 What the results do and do not support](#81-what-the-results-do-and-do-not-support)
- **[9. Reproducibility](#9-reproducibility)**
  - [9.1 Install and run from a fresh environment](#91-install-and-run-from-a-fresh-environment)
  - [9.2 Commands that regenerate each figure and table](#92-commands-that-regenerate-each-figure-and-table)
  - [9.3 Ground-truth rule: truth is read only in evaluation/](#93-ground-truth-rule-truth-is-read-only-in-evaluation)
- **[10. Team and Declarations](#10-team-and-declarations)**
  - [10.1 Group number and team members](#101-group-number-and-team-members)
  - [10.2 Individual contribution statement](#102-individual-contribution-statement)
  - [10.3 Generative AI declaration](#103-generative-ai-declaration)
  - [10.4 Acknowledgements](#104-acknowledgements)

## 1. Overview
### 1.1 Introduction


This is **Track A, a mobile robot**. This task has a TurtleBot navigating a room of obstacles to reach its ArUco-marked charging station. This task has been performed both in MuJoCo and in real life. Utilising: 
- 2D LiDAR
- Camera
- Wheel-encoder odometry

This task’s constraint is that the controller must use the estimated state, never the ground truth.

**Video link:**
[41014 Integrated Project - Team 12, Track A | TurtleBot3 Navigation and Visual Docking](https://youtu.be/6AC7xNN-bZU)

**GitHub repository (code and README):** https://github.com/Atomic-add/SCMS_IntegratedProject.github.io


### 1.2 Pipeline diagram, simulation and real-robot photos

**Pipeline diagram**

<img src="results/page/SCMS_PipelineDiagram.drawio.png" width="70%" alt="Sense, estimate, decide, act pipeline">

*Figure 1.2. Pipeline diagram*

**Photos**

<img src="results/page/Overview_Sim_Track.png" width="90%" alt="MuJoCo simulation track">

*Figure 1.3. MuJoCo sim track*

<img src="results/page/Overview_Real_Robot.png" width="50%" alt="Real TurtleBot in the lab arena">

*Figure 1.4. Real TurtleBot implementation*

## 2. Modelling (A): state-space model, controllability and observability analysis
### 2.1 State-space model
**Error dynamics:** Path following regulates the error to the path. With cross-track error e<sub>y</sub>, heading error e<sub>θ</sub> and curvature κ, the error dynamics are (A3 A.1)

$$\dot e_y = v\sin e_\theta, \qquad \dot e_\theta = \omega - \frac{v\kappa\cos e_\theta}{1-\kappa e_y}$$

**Linearisation:** Substituting ω = vκ + δω and writing the error dynamics above as ė = f(e, δω), A and B are the Jacobians of f at the operating point e = 0, δω = 0 (W5):

$$A = \left.\frac{\partial f}{\partial e}\right|_{e=0}, \qquad B = \left.\frac{\partial f}{\partial\,\delta\omega}\right|_{e=0}$$

```math
\dot e = \begin{bmatrix}0 & v\\ -v\kappa^2 & 0\end{bmatrix} e + \begin{bmatrix}0\\ 1\end{bmatrix}\delta\omega
```

On the lab's straight segments κ = 0, so A = [[0, v], [0, 0]] and B = [0, 1]: v sits inside A, and at v = 0 the robot cannot steer out a lateral error.

**Discretisation:** The controller runs every T<sub>s</sub> = 0.05 s and holds each command until the next, so the model is discretised with a zero-order hold:

$$A_d = e^{AT_s}, \qquad B_d = \int_0^{T_s} e^{A\tau}B\,d\tau$$

Because A² = 0 on a straight, the series e<sup>AT<sub>s</sub></sup> = I + AT<sub>s</sub> + (AT<sub>s</sub>)²/2 + … stops after two terms, so the result is exact:

```math
A_d = I + AT_s = \begin{bmatrix}1 & vT_s\\ 0 & 1\end{bmatrix}, \qquad e^{A\tau}B = \begin{bmatrix}v\tau\\ 1\end{bmatrix} \;\Rightarrow\; B_d = \begin{bmatrix}vT_s^2/2\\ T_s\end{bmatrix}
```

At v = 0.2 m/s:

```math
A_d = \begin{bmatrix}1 & 0.01\\ 0 & 1\end{bmatrix}, \qquad B_d = \begin{bmatrix}0.00025\\ 0.05\end{bmatrix}, \qquad e[k+1] = A_d\,e[k] + B_d\,\delta\omega[k]
```

Steering changes e<sub>θ</sub> directly (T<sub>s</sub> = 0.05 per unit δω), but moves e<sub>y</sub> only by turning the heading first, with coupling vT<sub>s</sub> = 0.01, so the authority over e<sub>y</sub> shrinks with speed and vanishes at v = 0.

**Assumptions:** Kinematics only (no slip, no motor lag), constant v, small errors (checked in 2.4), nominal track width 288 mm (UMBmark: 292.6 mm). The controller sees only the EKF estimate.

**Open-loop behaviour:** Without feedback control, the robot cannot correct its heading error, causing the cross-track error to increase as the robot travels. Therefore, feedback is needed to keep the robot on the planned path.

### 2.2 Controllability analysis 
With A = [[0, v], [0, 0]] and B = [0, 1] from 2.1 (on a straight), C = [B AB] = [[0, v], [1, 0]], so det C = −v.

| Speed | Rank of C | det C |
|---|---|---|
| 0.20 m/s (cruise) | 2 | −0.20 |
| 0.08 m/s (sim docking) | 2 | −0.08 |
| 0.05 m/s (real-robot docking) | 2 | −0.05 |
| 0 (turn on the spot) | **1** | 0 |

**Interpretation:** Controllable at every speed except zero: at v = 0, the only path from steering to e<sub>y</sub> (ė<sub>y</sub> = v e<sub>θ</sub>) is cut, so rank C drops to 1. This is the nonholonomic constraint: the robot cannot move sideways.

The rank test only says whether the system is controllable, not how easy it is to control. The singular values of C are 1 and v (calculated in the code). Therefore, slow driving is weakly controllable, not uncontrollable. So, the LQR is used only when the robot is driving on a straight. At the corners, the robot stops and turns on the spot with a P-controller, as e<sub>θ</sub> stays controllable at v = 0.

### 2.3 Observability analysis
**Measurement:** The EKF measures range and bearing to each marker centre, not the full ArUco pose, so each marker in view adds two rows to the measurement Jacobian H<sub>k</sub>. The pose has three states, (x, y, θ).

**Rank test:** A linear model with sensor matrix C (y = Cx) is observable when the observability matrix has full rank:

```math
\mathcal{O} = \begin{bmatrix}C\\ CA\\ \vdots\\ CA^{n-1}\end{bmatrix}, \qquad \mathrm{rank}(\mathcal{O}) = n
```

In the EKF, the measurement Jacobian H<sub>k</sub> and the motion Jacobian F<sub>k</sub> change every frame, so the same idea is applied to the linearised EKF, frame by frame. A small error δx<sub>0</sub> in the initial pose changes each frame's predicted measurement by:

$$\delta z_0 = H_0\,\delta x_0, \qquad \delta z_1 = H_1F_0\,\delta x_0, \qquad \delta z_2 = H_2F_1F_0\,\delta x_0, \ \dots$$

Stacking these gives

```math
\begin{bmatrix}\delta z_0\\ \delta z_1\\ \delta z_2\\ \vdots\end{bmatrix} = \mathcal{O}_N\,\delta x_0, \qquad \mathcal{O}_N = \begin{bmatrix}H_0\\ H_1F_0\\ H_2F_1F_0\\ \vdots\end{bmatrix}
```

If rank(O<sub>N</sub>) = 3, any small pose error δx<sub>0</sub> shows up in the measurements, so the pose is locally observable along that trajectory.

The rank says whether the pose is recoverable, and the null space shows what stays hidden. Test pose (1.5, 1.25) m on the lab lane, 1 s = 20 frames ([observability.py](evaluation/observability.py), results in [observability.json](results/observability.json)).

| Markers in view | Rank, 1 frame | Rank, 1 s driving | Hidden direction |
|---|---|---|---|
| none | 0 | 0 | everything (not detectable) |
| one | 2 | 2 | rotation about the marker |
| two | 3 | 3 | none |

**Marker lost:** With no marker in view, the EKF only predicts:

$$P_{k+1} = F_kP_kF_k^T + Q_d$$

From the converged P, the position standard deviation σ (worst horizontal direction) grows with every prediction step. If the robot kept driving at 0.2 m/s, the 3σ bound would pass the 3 cm dock tolerance after 3.5 s (0.7 m).

**Interpretation:** One marker gives two measurements, range and bearing, but the pose has three states, so one direction is always missing. That direction is a rotation of the robot about the marker: every pose on a circle around the marker, with the heading turned by the same angle, gives the same range and bearing. Odometry only measures motion in the robot's own frame, so the same run started anywhere on that circle gives the same measurements and the same wheel counts, and the rank stays 2.

A second marker at a different place breaks the circle and results in rank 3. This works whether both markers are seen at the same time or one after the other, with odometry linking the two views.

With no marker, the EKF can only predict from wheel odometry; no external measurement corrects the accumulated error, so it drifts instead of decaying and the pose is not even detectable. Driving blind would let the 3σ bound pass the 3 cm tolerance after about 3.5 s, so docking never drives without a marker. It stops at once, waits for the marker, and declares failure if none returns within 3 s ([docking.py](src/control/docking.py), `marker_lost_timeout` = 3 s in [control.json](config/control.json)).

### 2.4 Model correctness

#### 1. How does speed affect the model and controller?

The LQR gain (K) was designed at 0.20 m/s and tested without any modification at three different speeds, with 20 test runs for each speed ([model_limits.py](evaluation/model_limits.py)). On a straight path, the closed-loop poles satisfy:

$$\lambda^2 + k_\theta\lambda + k_y v = 0$$

Therefore, the closed-loop poles change as the robot's speed changes, even though the LQR gain (K) remains fixed.

| Speed | Slow pole \|z\| | Cross-track RMS | NEES ≤ 7.81 | Docked |
|---|---|---|---|---|
| 0.10 m/s | 0.985 | 1.76 cm | 91 % | 20/20 |
| 0.20 m/s (design) | 0.965 | 1.84 cm | 96 % | 20/20 |
| 0.26 m/s (top speed) | 0.947 | 1.92 cm | 98 % | 20/20 |

<p align="center">
  <img src="results/model_limits.png" width="90%" alt="Model limits">
  <br><em>Model limits: (a) closed-loop poles vs speed, (b) linear vs nonlinear response, (c) tracking and NEES vs speed.</em>
</p>

**Controller:** all poles stay inside the unit circle, indicating the closed-loop's stability across the tested speeds. Cross-track RMS increased only 9 %, and all 60 runs successfully docked. Therefore, the same LQR gain (K) remained effective without redesign.

#### 2. When does the linear model become inaccurate?

We started the linear and nonlinear models from the same initial cross-track error (v = 0.20 m/s, same K). The nonlinear model uses sin e<sub>θ</sub> and clips ω at 1.82 rad/s. Values are from [model_limits.json](results/model_limits.json), "validity".

| Initial cross-track error | Peak unsaturated command | Linear vs nonlinear gap |
|---|---|---|
| 5 cm | 0.46 rad/s | 0.13 % |
| 20 cm | 1.83 rad/s (applied limit: 1.82) | 1.2 % |
| 50 cm | 4.6 rad/s (applied limit: 1.82) | 12 % |

The linear and nonlinear models differed by only 0.13 % at an initial cross-track error of 5 cm. The gap increased at larger errors because the small-angle approximation sin e<sub>θ</sub> ≈ e<sub>θ</sub> became less accurate and the angular velocity command reached its saturation limit. The difference became significant at an initial cross-track error of 50 cm, reaching 12 %. However, in these 60 runs, the maximum cross-track error was only 5.3 cm, indicating that the linear model remained sufficiently accurate for the robot's normal operation. In addition, Figure (b) shows the comparison for initial cross-track errors of 10, 30, 50 and 80 cm.

#### 3. What the model leaves out

- **Per-wheel speed limits:** Wheel saturation can prevent the robot from achieving commanded motion.
- **Camera latency:** Delayed measurements can affect the pose estimation, potentially increasing the errors.
- **A small steady offset:** A steady cross-track offset of about +0.6 cm was observed at the design speed, but its cause has not been established.

An LQI could remove a steady-state error caused by a disturbance, but not a tracking error caused by a biased EKF. This is because the LQI integrates the estimated error rather than the ground-truth error.

## 3. Sensing (B): calibration, noise handling and frame management
The robot in this project uses three sensors: wheel encoders, a camera and a LiDAR. Both the [simulator](src/robot/sim_robot.py) and the [real robot](src/robot/real_robot.py) read sensor data through the same three functions, read_encoders(), read_image() and read_scan(); the real robot sends its data through rosbridge. As a result, both platforms process the data with the same perception and estimation logic.

- Wheel encoders (20 Hz): provide wheel rotation measurements for odometry, which the EKF uses to predict the robot's pose.

- Pi camera (640 x 480 pixels, 30 Hz): measures range and bearing to visible ArUco markers for EKF updates. Our track uses three markers from the DICT_4x4_50 dictionary, with a 100 mm side length.

- LDS-01 LiDAR (360 beams, 1° interval, 5 Hz): measures the free distance ahead, obstacle edges and side clearances for obstacle avoidance.
### 3.1 Calibration: camera, plus LiDAR validated against ray-cast truth 
**Camera intrinsics**: We calibrated both cameras using a 9 x 6 checkerboard with 25 mm squares ([calibrate_camera.py](experiments/calibrate_camera.py)).

**Simulated camera**: In simulation, the board was placed at 20 random poses in front of the camera.
<p align="center">
  <img src="results/camera_calibration.png" width="80%" alt="Simulated camera calibration">
  <br><em>Simulated camera: board found in 20 of 20 frames.</em>
</p>

**Real camera**: For the real camera, the board was held by hand at different positions and angles in front of it. 51 images were captured for calibration.
<p align="center">
  <img src="results/page/camera_calibration_real.png" width="80%" alt="Real camera calibration">
  <br><em>Real camera: board found in 40 of 51 captured images; those 40 were used for calibration.</em>
</p>

These images were then used to estimate the camera matrix K and the lens distortion. To assess the calibration quality, we used the reprojection error; the table below compares the results for both cameras.

| Parameter | Simulation | Real camera |
|---|---|---|
| Frames used | 20 / 20 | 40 / 51 |
| Reprojection RMS | 0.046 px | 0.41 px |
| Focal lengths f<sub>x</sub>, f<sub>y</sub> | 692.07, 692.01 px | 691.5, 692.3 px |
| Principal point (c<sub>x</sub>, c<sub>y</sub>) | (302.2, 245.2) px | (302.2, 245.2) px |

The estimated simulated camera was also compared with the true values of the MuJoCo camera: the estimated focal lengths matched the true value (692.00 px) within 0.010 %.

**Camera mounting pose**: Both the simulated and real cameras use the URDF mounting position: 0.076 m forward and 0.103 m above the robot's base origin. For the real camera, we also estimated the camera pitch ([camera_pitch.py](experiments/camera_pitch.py)): 50 ArUco images taken at different distances, with the marker at a fixed height, gave a downward tilt of 6.3° ([camera tilt image data](results/hardware)). The simulated camera was then set to the same 6.3° tilt to match the real one.

**Real-camera range correction**: To verify the camera's distance measurement, we compared the raw ArUco range estimates with tape-measured reference distances ([aruco_real_noise.py](experiments/aruco_real_noise.py)). The camera overestimated the range by about 0.6 mm + 3.2 % of the reference distance, so on the real robot [aruco.measure()](src/perception/aruco.py) corrects this offset and scale error before each range is passed to the EKF.

**Simulated LiDAR validation**: To check whether our LiDAR measurements, noise and dropout matched the configured model, we compared its ranges with reference distances calculated by a separate 2D ray caster using the same map geometry, at five poses on the robot's route ([check_sensors_lab.py](evaluation/check_sensors_lab.py), ray caster in [ground_truth.py](evaluation/ground_truth.py)).
With noise off, the scans matched the ray caster to 0.002 mm. With noise on, the range-noise standard deviation was 9.95 mm, close to the configured 10 mm, and the beam dropout rate was 1.01 %, compared with the configured 1 %.
<p align="center">
  <img src="results/page/lab_raycast.png" width="65%" alt="Simulated LiDAR vs ray caster">
</p>

### 3.2 Noise handling: measured noise characterisation 


**Planar-pose ambiguity**:
We use only the ArUco range and bearing for the EKF correction. For a flat marker viewed nearly straight on, IPPE can return two similar pose solutions with different marker orientations, so we do not use the marker yaw, and we evaluate the range and bearing below.

**ArUco measurement errors in simulation**:
We tested how the ArUco range and bearing errors change with distance, to check the EKF's range-bias correction and the measurement-noise settings in R<sub>n</sub>. At each robot pose we took 20 frames and compared them with the simulation ground truth.

The results show that the simulated camera tends to overestimate the marker distance, with mean range errors of about 65 mm at 3 m and 131 mm at 4.5 m (left). The bearing bias stayed below 0.02° at the tested poses (right).
Consecutive frames can share the same bias, so consistent readings are not always accurate. Therefore, the EKF first subtracts the known bias (4.5 mm·r²), and its range noise is set larger than the frame-to-frame spread (middle), so that it does not become overconfident from repeated readings.
<p align="center">
  <img src="results/page/lab_aruco.png" width="90%" alt="ArUco errors on the lab track">
</p>

**Real-camera range measurements**:
We tested the camera range readings at seven distances, keeping the robot and the marker stationary at each distance. Across repeated images, the range standard deviation was 0.05–3.4 mm. We then used tape-measured distances as reference values and fitted a linear correction to the raw ArUco ranges ([aruco_real_noise.py](experiments/aruco_real_noise.py)). This removes a fixed offset and adjusts the range scale. After applying it, the mean difference was within ±3 mm for the datasets used to fit the correction.
<p align="center">
  <img src="results/page/aruco_real_simple.png" width="60%" alt="Real camera ArUco range">
</p>

**Wheel-encoder evaluation**:
To evaluate the wheel odometry, we used [umbmark.py](experiments/umbmark.py). The robot drove a 0.5 m square three times clockwise and three times counter-clockwise, and after each run we measured the final position and heading errors.

The results give a right-to-left wheel size ratio of 1.006 and a wheel separation of 292.6 mm, compared with the nominal 288 mm. The estimated wheel distance noise was 5.2 %; to allow for manual measurement error, we kept k_wheel = 6 %. We kept the nominal wheel parameters and rely on the ArUco updates to correct the pose drift.
### 3.3 Noise handling: outlier rejection and marker-loss handling 

Before a measurement reaches the EKF, it must pass the checks below. The robot also stops when the dock marker is unavailable during docking ([docking.py](src/control/docking.py)), or when the LiDAR returns cover less than 80 % of the forward half in simulation, or 20 % on the real robot ([lidar_obs.py](src/control/lidar_obs.py)).

<p align="center">
  <img src="results/page/camera_measurement_flow.png" width="30%" alt="Camera Measurement">
</p>

(Frame age, bias, NIS gate from [ekf.py](src/estimation/ekf.py)), 
(Detection, IPPE from [aruco.py](src/perception/aruco.py)).

**Real-robot validation**: To test the noise handling on the real robot, we compared a docked run with a failed run (section 7.2). In the docked run (09:28), 96 % of 1,092 marker measurements passed the NIS check. In the failed run (09:41), all 30 detections of the dock marker exceeded the NIS threshold and were rejected, while measurements from markers 1 and 2 still passed. The robot then waited 3 seconds and stopped with marker_lost.

<p align="center">
  <img src="results/page/real_nis.png" width="100%" alt="NIS TEST">
</p>
 
**LiDAR data checks**: During tracking and obstacle avoidance, the robot stops if too few valid beams return in the forward half. The real robot's LiDAR coverage threshold was reduced from 80 % to 20 % because the lab's black foam panels return very few beams to the LDS-01. This did not stop the obstacle detection: the box still triggered avoidance in all nine filmed runs ([lidar_obs.py](src/control/lidar_obs.py)). The robot also stops if a scan is more than 0.5 seconds old.
### 3.4 Frame management: frame chain from sensor measurement to controller input 
<p align="center">
  <img src="results/page/frame_chain_3_4.png" width="80%" alt="Frame chain">
  <br><em>Blue: known or estimated transforms, Red: T<sub>C</sub><sup>M</sup>, measured every camera frame by ArUco + PnP.</em>
</p>

**Frame chain**: Camera and LiDAR sensors start their measurements in their own coordinate frames, {C} and {S}. We then use each sensor's mounting pose, T<sub>B</sub><sup>C</sup> or T<sub>B</sub><sup>S</sup>, to transform its data into the robot's base frame {B}, centred between the two drive wheels. Next, the EKF combines the marker range and bearing with the wheel encoders to estimate the robot pose T<sub>W</sub><sup>B</sup> in the map {W}, using the known marker poses T<sub>W</sub><sup>M</sup>. The path tracker then uses this estimate to compute the cross-track and heading errors, and the LiDAR points in {B} are used for obstacle avoidance.
### 3.5 Simulated vs real sensor noise, and why they differ 
**Sensor performance: simulation vs real robot**

| Check | Simulation | Real robot | Why they differ |
|---|---|---|---|
|Camera calibration (reprojection RMS)| 0.046 px | 0.41 px | The simulated camera is an ideal pinhole model, while real images include lens distortion, board imperfections and a hand-held board |
|ArUco range bias near 1 m| +1.8 mm | +33 mm raw, after correction +- 3 mm on fitted datasets|Residual calibration errors, tape measurement uncertainty may affect the result| 
|LiDAR forward coverage| 88–100 % | about 23 % | The real LiDAR receives few returns from the black foam panels |
|Wheel geometry| +- 0.1%|Right-to-left wheel ratio 1.006; wheel separation +1.6 % relative to the nominal value| Real tyres and assembly may introduce geometry differences |
## 4. Estimation (C): EKF design, tuning and reported performance
### 4.1 EKF design 

The estimation section is adapted from **Week 7 C.3 / C.4**, and **Assessment 3, Task B.3**, borrowing the following variables:
- ekf_predict
- ekf_update
- F_uni
- h_rb
- H_rb
- Qd reasoning

The design has 2 main functions: predict and update.

**Predict** is generated once per new encoder sample. Utilised to move the estimate with the wheel increments and grow P by the wheel noise. 

**Update** is generated once per new camera frame. This fuses one marker's range and bearing, and returns its NIS.

### 4.2 Tuning: Q, R and P0, and where each value comes from 

- **Q (process noise)**: This refers to uncertainty in the motion prediction at each step. This includes wheel slip and radius error, which grows with the distance travelled.
  
- **R (sensor noise)**: These are random, unwanted variations of the sensor's output. Affecting the accuracy and reliability of measurements

- **P0 (filter's starting uncertainty)**: A non-zero setup value for placement accuracy. Dictated by the trustworthiness of the robot's starting position. If it were 0, it would be overconfident and slow to correct any starting offsets. This can throw the robot off course if left unchecked.

The table below is an expansion upon these 3 variable groups. It illustrates what each variable belongs to and what it does.

|  Group   |    Variable     | Value             |                   What it is                     |
| :------- | :-------------- | :---------------- |  :---------------------------------------------- |
|   Q      | k_wheel         |   0.06            | Wheel error, which grows with distance travelled |
|          | q_floor         |   1e-8            | noise, which is added to Qd every step. Keeps P positive |                                        
|   R      | range_sd [a,b]  | [0.003, 0.004]    | range noise = a + b*r^2                          |
|          | bearing_sd_deg  | 0.1               | bearing noise                                    |
|          | range_bias_r2   | 0.0055            | Known camera range bias.                         |
|   P0     | P0_sd           | [0.05, 0.05, 5.0] | Starting uncertainty: x, y, heading              |    

All variables listed come from ([estimation.json](config/estimation.json))

### 4.3 Reported performance: RMSE, NEES, NIS, ±3σ bounds 

To analyse and report the performance, ([run_demo.py](evaluation/run_demo.py)) is utilised. Running a simulated test of the robot and reporting back both its odometry and EKF. This will be further compared within 4.5

**RMSE (Root Mean Square Error)**: Reports the average distance between predicted values of the model and actual values.

**NEES (Normalised Estimation Error Squared)**: Utilised to evaluate the consistency of the Kalman filter. Through comparing the estimation error to the expected covariance of the estimator.

**NIS (Normalised Innovation Squared)**: Checks whether the Kalman filter's noise assumption aligns with the actual measurements.

**Odometry** 

![Your Alt Text](./results/demo_centre.png)

|   Case   | Docked | Time [s] | Cross-track RMS [cm] | RMSE [cm]  | Dock lateral [cm]| Heading [deg] | Collisions |
| :--------| :------| :--------| :--------------------| :--------- |:-----------------| :-------------| :----------|
|   none   | Yes    | 37.4     | 1.37                 | 1.16       | +0.7             | -1.1          | None       |
|   left   | Yes    | 37.2     | 1.32                 | 1.03       | +0.6             | -1.1          | None       |
|   centre | Yes    | 43.1     | 1.62                 | 0.76       |+0.5              | -0.8          | None       |
|   right  | Yes    | 37.4     | 1.37                 | 1.16       |+0.7              | -1.1          | None       |


**EKF (nominal run, 4 cases)**

![Your Alt Text](./results/demo_centre_ekf.png)
 
|   Case   | Docked | Time [s] | Cross-track RMS [cm] | RMSE [cm] | Dock lateral [cm] | Heading [deg] | Collisions | ANEES (95 % band) | NEES inside | ANIS | NIS inside |
| :--------| :------| :--------| :--------------------| :-------- | :-----------------| :-------------| :----------| :-----------------| :-----------| :----| :----------|
|   none   | Yes    | 37.4     | 1.01                 | 0.43      | +0.4              | -0.7          | None       | 2.61 [2.83, 3.18] | 88 %        | 0.10 | 46 %       |
|   left   | Yes    | 37.2     | 1.06                 | 0.58      | +0.4              | -0.7          | None       | 2.66 [2.83, 3.18] | 88 %        | 0.09 | 44 %       |
|   centre | Yes    | 43.3     | 1.58                 | 0.83      | +0.4              | -0.7          | None       | 3.50 [2.84, 3.17] | 89 %        | 0.13 | 41 %       |
|   right  | Yes    | 37.4     | 1.01                 | 0.43      | +0.4              | -0.7          | None       | 2.61 [2.83, 3.18] | 88 %        | 0.10 | 46 %       |

To summarise. The robot performs relatively well. As displayed by this simple pass/fail check from ([estimation_test.py](src/estimation/estimation_test.py)). Showcasing that it is able to reach its charging station, and pass every test.

-----------------------------------------------------------------------------------

PASS  baseline: the first encoder sample does not move the estimate

PASS  straight: equal wheel turns move 1 m along x, heading unchanged

PASS  turn: opposite wheel turns spin +90 deg on the spot

PASS  repeated encoder timestamp: no second prediction

PASS  wrap: 179 deg - (-179 deg) is -2 deg, not 358 deg

PASS  bearing residual is wrapped near +-180 deg   (bearing 1.57 deg)

PASS  F matches finite differences

PASS  H matches finite differences

PASS  missing data: empty markers dict leaves the estimate unchanged

PASS  missing data: unknown ID 7 leaves the estimate unchanged

PASS  missing data: nan measurement leaves the estimate unchanged

PASS  repeated frame: fused once only

PASS  old frame (1 s behind the encoders): not fused

PASS  correction: 11 cm / 3 deg off, markers 0 and 2 in view -> below 1 cm and 0.5 deg   (0.63 cm, 0.23 deg, 58/58 fused)

PASS  gate: a 1 m range outlier is rejected   (NIS 311)

PASS  1000 random steps: x_hat finite, P symmetric, eigenvalues >= 0   (min eig 1.9e-04)

PASS  config: by default the sim camera's ekf_noise replaces range bias, range sd and bearing sd   ({'range_bias_r2': 0.0045, 'range_sd': [0.00911, 0.00563], 'bearing_sd_deg': 0.0571})

PASS  config: the real camera's range_correction drops the sim bias (never subtracted twice)

**0 failed - all checks passed**

------------------------------------------------------------------------------------

### 4.4 Tuning trade-off: consistency vs responsiveness (R sweep) 

This runs a test to compare Consistency to Responsiveness. For this test, **R (EKF's measurement noise)** is tested across its 4 conditions. **R x10, R (config), R (measured bearing sd)**, and **R / 10**. 

To elaborate:

**Consistency** refers to whether the error matches the set uncertainty. If the real errors are above it, that being the ANEES, it's overconfident. Then, overly cautious if it's below the ANEES. Further helping prevent collisions, at the cost of the robot's speed.

**Responsiveness** refers to how fast the filter chases the measurements. A small R results in the filter reacting to every sighting, including noise. This results in more jitter. Whereas a large R results in a slower filter. This makes it slower to correct drift, but it is an overall smoother experience.

![Your Alt text](./results/sweep_r.png)

These test results are from ([sweep_r](evaluation/sweep_r.py)).

|R                      |	Docked | ANEES      |	NIS	|Gated|	Worst error [cm] |
| :-------------------- | :----- | :--------- | :-- | :-- | :--------------- |
|R x10	                |  100 % |1.54 ± 0.06	|0.02 |0.0  |	0.82 ± 0.08      |
|R (config)	            | 100 %	 |3.08 ± 0.49 |0.12 |0.2  |	1.78 ± 0.65      |
|R, measured bearing sd |	100 %	 |4.76 ± 0.70 |0.21 |0.6  |	3.17 ± 0.83      |
|R / 10	                | 100 %	 |10.85 ± 1.99|1.25 |12.0 |	3.75 ± 0.78      |

The table below is an expansion upon the data above. Analysing their consistency and responsiveness. Then concluding how that would impact the robot's performance on the track.

| R setting              | Consistency       | Responsiveness   | Verdict                      |
| :--------------------- | :---------------  | :--------------- | :--------------------------- | 
|    R X10               | too cautious      | Unresponsive     | staggent                     |
| R (config)             | consistent        | responsive       | slow but steady movement     |
| R, measured bearing sd | too confident     | responsive       | fast but unreliable movement |
| R / 10                 | overly confident  | Overly responsive| uncontrollable movement      |

When taking into account track A. Where the robot's main objective is to reach the charging station without any collisions. No matter the time taken. R (config) would make the most logical sense. For, whilst slower. Being aligned with consistency means that it will measure less noise and move smoothly. At the cost of speed, which isn't a requirement for this track.

As a result, **consistency succeeds reliability** for this track.

### 4.5 Comparison with a second estimator: EKF vs odometry only, same seeds 

This compares the EKF to odometry through 20 random tests. Observing through simulation whether it could dock successfully.

To determine this, 2 tests were run:

Test 1 utilises ([random_trials.py](evaluation/random_trials.py)). Observing how the odometry compares to the EKF. When placed on randomised routes and starting positions. This has generated 2 graphs.

**EKF output**
![Your Alt text](./results/random_trials.png)

**Odometry output**
![Your Alt text](./results/random_trials_odometry.png)

Below is the data for both:

| Run | Docked | Contacts | Notes |
|---|---|---|---|
| EKF | 20 / 20 | 0 | 42.0 +- 1.9 s, dock lateral 0.23 +- 0.44 cm, yaw -0.37 +- 0.76 deg, ANEES 3.8; seed 5 (box beside the lane) used to stop `too_close` on its narrow side; the pass side is now the one needing the smaller sideways offset (`avoidance.py`) |
| Odometry only | 9 / 20 | 2 | the hidden placement and wheel errors are never corrected (estimate off by up to 74 cm) |

Test 2 follows ([run_demo](evaluation/run_demo.py)). The graph showcases how not only does the EKF have a higher docking rate, but also its errors are more consistently close to 0. In comparison to the odometry readings. With very inconsistent errors

![Your Alt Text](./results/demo_compare.png)

Below is the data behind the graph:
| Case   | Docked (odo / EKF) | Cross-track RMS [cm] (odo / EKF) | Dock lateral [cm] (odo / EKF) | Heading [deg] (odo / EKF) | Time [s] (odo / EKF) |
| :----- | :----------------- | :------------------------------- | :---------------------------- | :------------------------ | :------------------- |
| none   | Yes / Yes          | 1.37 / 1.01                      | +0.7 / +0.4                   | -1.1 / -0.7               | 37.4 / 37.4          |
| left   | Yes / Yes          | 1.32 / 1.06                      | +0.6 / +0.4                   | -1.1 / -0.7               | 37.2 / 37.2          |
| centre | Yes / Yes          | 1.62 / 1.58                      | +0.5 / +0.4                   | -0.8 / -0.7               | 43.1 / 43.3          |
| right  | Yes / Yes          | 1.37 / 1.01                      | +0.7 / +0.4                   | -1.1 / -0.7               | 37.4 / 37.4          |

From these 2 tests, it is made clear that the **EKF outperforms the odometry** as an estimator. Not only has a higher success rate of docking. But also produces fewer errors.



## 5. Control (D): controller design, tuning and tracking results
### 5.1 Controller design: LQR, Turn, Avoid and IBVS 
The robot has four modes (TRACK, AVOID, TURN and DOCK), handled by three controllers: TRACK and AVOID share the same LQR, TURN uses a P-controller, and DOCK uses IBVS. All three controllers use only EKF-estimated states or camera measurements (for IBVS), never the ground-truth pose.

| Mode | Controller | Command | Why this controller |
|---|---|---|---|
| Path following | Discrete LQR | ω = vκ − Kê | Balances tracking error against steering effort |
| Obstacle avoidance (LiDAR) | Same LQR, shifted line | ω = vκ − Kê, with e<sub>y</sub> → e<sub>y</sub> − d | No new controller: the reference is shifted by d and rejoined |
| Turning (corners > 30°) | P-controller | v = 0, ω = −k<sub>turn</sub>e<sub>θ</sub> | Allows the robot to rotate in place. At v = 0, the path-following model cannot control e<sub>y</sub> (rank C = 1, Section 2.2). |
| Docking (< 0.55 m from the marker) | IBVS | (v, ω) = λ J<sub>mr</sub><sup>+</sup>(p* − p) | Minimises the image-feature errors directly. |

**LQR (path following):** The LQR gain K is calculated from the discrete state-space model in Section 2.1 (ct.dlqr). It is combined with curvature feedforward and clipped to the motor limit:

$$\omega = v\kappa - K\hat e$$

At v = 0.20 m/s, the calculated gain is K = [9.16, 3.25], and the angular velocity is limited to |ω| ≤ 1.82 rad/s.

- **Stability check:** The closed-loop eigenvalue magnitudes are below 1 for both the straight path (0.965, 0.871) and a test arc of radius 0.5 m (0.960, 0.875). This confirms stability of the nominal linear model, while motor saturation, estimation errors and nonlinear dynamics are checked separately in simulation (Sections 2.4 and 5.3).
- **Approaching the dock:** After the pre-dock waypoint W3 (about 0.70 m from the marker), the LQR gain is recalculated at v<sub>dock</sub> = 0.08 m/s, which gives K = [9.25, 2.92]. It drives the robot until the marker is 0.55 m away, where IBVS takes over.
- **Why LQI is not used:** An LQI controller was also implemented in the code but disabled because the LQR already shows no large steady-state offset (mean cross-track error ≤ 0.83 cm, Section 5.4). Furthermore, LQI integrates the estimated cross-track error ê<sub>y</sub>, so it cannot reliably remove an offset caused by the estimation bias (Section 2.4).

**Obstacle avoidance:** When LiDAR detects an obstacle in the robot's path, the robot shifts its reference path sideways by an offset d to avoid it. The same LQR controller follows this shifted path using the modified cross-track error ê<sub>y</sub> − d. If neither side has enough clearance, or the LiDAR scan is invalid, the robot stops for safety (src/control/avoidance.py).

**Proportional turn (corners):** At a corner sharper than 30°, the robot stops and turns on the spot:

$$v = 0, \qquad \omega = -k_{turn}\, e_\theta$$

with k<sub>turn</sub> = 2.0 s⁻¹ and |ω| ≤ 1.0 rad/s. Control returns to the LQR once |e<sub>θ</sub>| < 3° (src/control/waypoints.py).

**IBVS (final docking):**

- **Switching:** The LQR first drives the robot onto the marker's normal, because IBVS alone may leave a unicycle facing the marker but offset sideways. IBVS takes over once the marker is less than 0.55 m from the robot and all four corners and a valid PnP pose are detected. However, the system does not verify the robot's alignment with the marker before switching, so perfect alignment is not guaranteed.
- **Control law:** IBVS uses the four marker corners to minimise image-feature errors:

  $$(v, \omega) = \lambda\, J_{mr}^{+}\,(p^* - p)$$

  with λ = 0.8 s⁻¹, |v| ≤ 0.08 m/s and |ω| ≤ 0.5 rad/s. Here p is the current positions of the four marker corners in the camera image, and p* is the desired positions when the robot is docked. J<sub>mr</sub> is the image Jacobian adapted to the robot's two control inputs (v, ω). The corner depths are estimated from the PnP pose in each frame.
- **Stopping:** IBVS ends at whichever comes first: the corner error falls below 1.5 px RMS, the camera-measured range reaches the stop range, the commands stay near zero for 1 s, or IBVS would command the robot backwards.

Docking is confirmed only after the SETTLE checks verify that the robot's docking position, heading, EKF uncertainty and wheel speed remain within tolerance for the required hold time. These checks use camera measurements and EKF estimates, never the true pose (src/control/visual_servo.py, src/control/docking.py).

**Interpretation:** LQR was chosen because it balances tracking errors and steering effort through the weighting matrices Q and R. Since our waypoint path consists mainly of straight segments (κ = 0), steering is primarily controlled by LQR feedback.

For obstacle avoidance, the same LQR follows a shifted reference path without requiring a new controller. The P-controller handles sharp turns in place, while IBVS handles the final docking by minimising image-feature errors. Together, the three controllers handle different operating conditions of the robot.

### 5.2 Tuning:
**LQR tuning: where Q and R come from.** We used Bryson's rule to select the weighting matrices Q and R. Each tracking error and control input is weighted by the inverse square of its maximum acceptable value:

$$Q = \mathrm{diag}\left(\frac{1}{e_{y,max}^2}, \frac{1}{e_{\theta,max}^2}\right) = \mathrm{diag}\left(\frac{1}{0.05^2}, \frac{1}{0.1745^2}\right) = \mathrm{diag}(400,\ 32.8)$$

$$R = \frac{1}{\delta\omega_{max}^2} = \frac{1}{0.5^2} = 4$$

with e<sub>θ,max</sub> = 10° = 0.1745 rad. Q penalises the cross-track and heading errors, while R penalises the steering effort. The selected limits are justified below.

| Limit | Value | Why |
|---|---|---|
| e<sub>y,max</sub> | 0.05 m | Reduces the distance needed to correct a 20 cm offset from ~2.3 m (e<sub>y,max</sub> = 0.10 m), longer than the 1.6 m between W1 and W2, to ~1.2 m (src/control/lqr.py, design_weights). |
| e<sub>θ,max</sub> | 10° | Limits heading errors to help keep the docking marker visible in the camera |
| δω<sub>max</sub> | 0.5 rad/s | Limits steering effort to avoid overly aggressive corrections, well below the 1.82 rad/s motor limit |

**LQR: what limits the gains.**

1. **Motor saturation:** For a 20 cm lateral offset with zero heading error on a straight path, the calculated unsaturated LQR command is 1.83 rad/s, slightly above the Waffle Pi's 1.82 rad/s maximum angular velocity. The steering command is therefore clipped (Section 2.4).
2. **Estimation noise:** LQR uses tracking errors estimated by the EKF. Higher gains make the steering command more sensitive to the estimation noise. In our tests, higher gains were also associated with a slightly lower NEES pass rate (Section 5.4).
3. **Forward speed:** LQR corrects cross-track error more slowly at lower speed. In Section 2.4, the cross-track error e<sub>y</sub> takes 3.2 s to decay at v = 0.10 m/s, compared with 0.92 s at 0.26 m/s. Therefore, sharp corners are handled by the P-controller, which rotates the robot in place at v = 0, rather than using the path-following LQR at low speeds.

**P-turn parameters:** The proportional gain is k<sub>turn</sub> = 2.0 s⁻¹. The angular velocity is limited to 1.0 rad/s and the angular acceleration is limited to 2 rad/s². These limits prevent the robot from turning aggressively and keep the angular velocity below the motor's limit of 1.82 rad/s.

**IBVS parameters:** The gain is λ = 0.8 to provide gradual correction of image-feature errors without reacting too aggressively to camera noise.

### 5.3 Tracking Results in Simulation 
This section evaluates how accurately the LQR makes the robot follow the path and how effectively IBVS performs docking. The controllers use only the estimated states and the camera measurements, while the ground-truth pose is used to evaluate their performance.

**Path tracking: how accurately does LQR follow the path?**

| Run | Box | Docked | Cross-track RMS | Dock lateral error | Source |
|---|---|---|---|---|---|
| None | none | Yes (37.4 s) | 1.01 cm | 0.43 cm | [demo_metrics_ekf.json](results/demo_metrics_ekf.json) |
| Left | left of lane | Yes (37.2 s) | 1.06 cm | 0.43 cm | [demo_metrics_ekf.json](results/demo_metrics_ekf.json) |
| Right | right of lane | Yes (37.4 s) | 1.01 cm | 0.43 cm | [demo_metrics_ekf.json](results/demo_metrics_ekf.json) |
| Centre | in the lane | Yes (43.1 s) | 1.7 cm (outside avoidance) | 0.45 cm | [a3style_centre.npz](results/a3style_centre.npz) |

All four cases were run on the final code with the EKF in the loop (seed 0).

<p align="center">
  <img src="results/ekf_summary_centre.png" width="90%" alt="Centre case summary">
  <br><em>Figure 5.3a: Centre case: robot trajectory and reference path (top left), true and estimated cross-track errors (top right), and EKF estimation results (bottom, discussed in Section 4.3).</em>
</p>

**Results:** In the none, left and right cases, the box did not completely block the lane, so the robot stayed on the path and the LQR maintained the cross-track RMS at 1.0–1.1 cm. In the centre case, the box blocked the lane, and the robot had to avoid it. The cross-track RMS was 1.7 cm (excluding the avoidance process). This slightly higher value likely came from rejoining the original track after avoiding the box (top right, Figure 5.3a, after about 22 s).

During obstacle avoidance, the robot intentionally moved about 40 cm away from the original path to pass the box, as shown by the shaded region (top right, Figure 5.3a). This deviation is intentional rather than a failure of LQR tracking, because the controller follows a laterally shifted reference path during obstacle avoidance and then drives the robot back to the original path.

The estimated cross-track error closely followed the true error (top right, Figure 5.3a), supporting the use of EKF estimates for LQR feedback in this run.

**Docking: does IBVS improve docking performance?**

| Controller | Docked | Lateral error | Yaw error | Docking time |
|---|---|---|---|---|
| PBVS | 10/10 | 0.26 ± 0.45 cm | −0.4 ± 0.7° | 8.25 s |
| IBVS | 10/10 | 0.26 ± 0.45 cm | −0.4 ± 0.7° | 7.10 s |

Source: [ibvs_docking.json](results/ibvs_docking.json).

<p align="center">
  <img src="results/ibvs_docking.png" width="80%" alt="IBVS docking">
  <br><em>Figure 5.3b: IBVS docking: (a) marker corner movement towards the desired image positions; (b) image-feature error during IBVS; (c) control commands during the LQR-to-IBVS transition; and (d) final docking errors of IBVS and PBVS.</em>
</p>

**Results:** The centre case successfully docked with a lateral error of 0.45 cm. To compare docking performance, IBVS and PBVS were tested using the same 10 random seeds with hidden placement errors.

Both controllers successfully docked in all 10 trials, with almost identical final lateral and heading errors. However, IBVS completed docking 1.15 s faster on average (7.10 s compared with PBVS's 8.25 s).

The image-feature errors decreased from approximately 93 to 57 px, but IBVS stopped when the robot reached a camera-based stopping distance, before the image-feature error reached the 1.5 px tolerance (Figure 5.3b). This suggests the final docking accuracy also depended on the earlier LQR alignment and the camera-based stopping distance.

Overall, IBVS reduced docking time in these 10 tests but did not show a measurable improvement in the final docking accuracy. Under the randomised 20 runs, the system also achieved 20/20 successful dockings (Section 6.1).

### 5.4 Tuning trade-off: control effort vs tracking error 
The LQR weight R was tested at three values with Q fixed: R = 40, R = 4 (chosen) and R = 0.4. Each value was tested on the centre case with 20 seeds, each seed adding a random initial position and encoder-scale error, with LQR using EKF-estimated states for feedback rather than ground truth. All 60 runs completed docking successfully. Cross-track errors were evaluated using the ground-truth pose, while control effort was evaluated using the steering commands. The tracking and steering metrics include only LQR path-tracking periods, while docking success and the NEES share are computed over the whole run (evaluation/sweep_lqr.py).

| R | Cross-track RMS | Steady offset | RMS feedback steering | NEES ≤ 7.81 | Command rate |
|---|---|---|---|---|---|
| 40 | 3.13 cm | +0.36 cm | 0.098 rad/s | 96 % | 0.28 rad/s² |
| 4 (chosen) | 1.73 cm | +0.57 cm | 0.077 rad/s | 96 % | 0.38 rad/s² |
| 0.4 | 1.42 cm | +0.83 cm | 0.070 rad/s | 93 % | 0.57 rad/s² |

Source: [sweep_lqr.json](results/sweep_lqr.json).

<p align="center">
  <img src="results/sweep_lqr.png" width="100%" alt="LQR weight sweep">
  <br><em>Figure 5.4: LQR weight sweep. R x10, R (config) and R / 10 correspond to R = 40, 4 and 0.4. (a) Linear-model response to a 20 cm offset at 0.2 m/s, with the resulting gains K. (b) Cross-track error against steering effort; error bars show the spread over seeds. (c) Docking success rate (green) and RMS rate of change of the angular velocity command (amber).</em>
</p>

**Interpretation:**

**1. How did R affect tracking?** Reducing R places less penalty on control effort, allowing LQR to correct cross-track errors more aggressively. The feedback gain increased (K = [3.05, 1.41], [9.16, 3.25] and [24.84, 7.78]) and errors are corrected faster (Figure 5.4a). The cross-track RMS fell from 3.13 cm at R = 40 to 1.73 cm at R = 4. However, reducing R by another factor of 10, from 4 to 0.4, decreased cross-track RMS error by only 0.31 cm.

**2. What did the higher gain cost?** At R = 0.4, the command rate increased from 0.38 to 0.57 rad/s², indicating less smooth steering (Figure 5.4c). The percentage of NEES samples below 7.81 also fell from 96 % to 93 %.

Meanwhile, RMS feedback steering decreased slightly from 0.077 to 0.070 rad/s. Although the steering commands had a lower RMS magnitude, they changed more rapidly. The higher gain therefore improved tracking accuracy at the cost of steering smoothness.

**3. Why R = 4?** All three R values docked in every run. However, R = 40 produced a much higher cross-track RMS error (3.13 cm) than R = 4 (1.73 cm). Reducing R from 4 to 0.4 improved the cross-track RMS error by only 0.31 cm, while increasing the command rate from 0.38 to 0.57 rad/s². R = 4 therefore provided a better balance between tracking accuracy and steering smoothness.

**4. What remains uncertain?** The steady-state offset increased from +0.36 to +0.83 cm as R decreased. Although reducing R improved the cross-track RMS error, it did not reduce the steady-state offset. The reason for this behaviour was not determined.

The weight sweep also kept Q fixed to isolate the effect of R, so the effect of changing Q was not evaluated.

## 6. Integration and Robustness (E): performance under disturbance and noise

### 6.1 Performance under disturbance and noise 

This section tests whether the complete system can still navigate and dock when the starting position, the obstacle location and the sensor measurements vary. Twenty randomised scenarios were tested with the EKF.

Each of the 20 randomised scenarios (seeds 1–20) varied two types of conditions: those known to the robot and those hidden from it ([random_trials.py](evaluation/random_trials.py), [scenario.py](src/robot/scenario.py)).

- **Known to the robot:** The initial position estimate provided to the EKF and the planned route (W1 and W2)
- **Hidden from the robot:**
  - Initial placement error: Random placement error (0.9-10.6 cm, up to 10°), sampled from N (0, P₀) and clipped at 2σ.
  - The box position: Randomly placed within the lane.
  - LiDAR noise: Noise scaled by 1–2, with 1–5% dropout.
  - Camera noise: Gaussian intensity noise added to every pixel (SD 2–4 grey levels)
  - Encoder-scale error: Up to ±1% per wheel.

The robot only uses sensor measurements and estimated states. Ground-truth pose is used only for performance evaluation.

A run is considered completed when the docking check passes. It is considered a docking success only if the docking check passes, the final ground-truth position and heading errors are within 3 cm and 5°, respectively, and no collision occurs with the walls or obstacle box.

| Condition | Trials | Completed | Docking success | Runs with contact | Notes | Source |
|---|---|---|---|---|---|---|
| Nominal, 4 box cases (none, left, centre, right) | 4 (1 per case) | 4/4 | 4/4 | 0/4 | 37.2–43.1 s<br>Docking lateral error: 0.43–0.45 cm | [demo_metrics_ekf.json](results/demo_metrics_ekf.json) |
| Disturbed, EKF | 20 | 20/20 | 20/20 | 0/20 | 42.0 ± 1.9 s<br>Docking lateral position error: 0.23 ± 0.44 cm<br>Docking heading error: −0.37 ± 0.76° | [random_trials.json](results/random_trials.json) |

These trials evaluated the robustness of the complete system under multiple random disturbances. However, they did not isolate the individual effect of each disturbance.

<p align="center">
  <img src="results/random_trials.png" width="80%" alt="20 random scenarios on the EKF">
  <br><em>Figure 6.1: True paths of the 20 randomised trials with the EKF.</em>
</p>

**Results:**

All four nominal runs (none, left, right and centre box cases) docked successfully without collision, confirming that the system worked under nominal conditions.

With EKF, all 20 random runs docked successfully without collision. The mean docking lateral error was 0.23 cm, with a standard deviation of 0.44 cm across the 20 trials. The maximum position estimation error during these trials was 10.9 cm (seed 14, with a 7.8 cm initial placement error). However, the error decreased to below 0.4 cm by the end of every run, as marker measurements helped correct the position estimates.

**Why it looks this way:** Marker measurements let the EKF correct the accumulated estimation errors, so the robot followed the planned route and docked reliably under multiple random disturbances. The comparison with odometry alone on the same 20 seeds is in Section 4.5.

### 6.2 Failure analysis traced to model, sensing, estimator or controller 

The observed failures and safety stops were investigated using the simulation ground truth and the real-robot sensor logs to identify their most likely causes. This section presents an analysis of failures and safety stops observed in the final tests, problems identified and fixed during development and remaining limitations.

#### A. Failures and safety stops observed in the final tests

| Observed issue          | Source                    | Evidence                                                                                        | Primary cause         |
|-------------------------|---------------------------|-------------------------------------------------------------------------------------------------|-----------------------|
| Odometry drift          | Simulation, odometry only | 11/20 failed; position estimation error: 20–74 cm (see Section 4.5)                                               | Estimator             |
| Floor vent cover        | Real robot                | Dock-marker detections rejected by the NIS gate (30 in run 09:41); `marker_lost` stop        | Environment           |
| Low LiDAR scan coverage | Real robot                | Forward scan coverage 0.34–0.55, below the required threshold of 0.8; robot waited at the start | Sensing               |

**1. Odometry drift (estimator):** Using the same 20 random seeds, all 20 runs with EKF completed the task successfully, compared with only 9 out of 20 odometry-only runs (Section 4.5). In the 11 failed odometry-only runs, the position estimation error ranged from 20 to 74 cm at the pre-dock point. These estimation errors led to three different stop reasons:

| Stop reason   | Runs (seeds)            | What happened                                                                                                                                                                                                                                              |
|---------------|-------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `marker_lost` | 6 (2, 5, 7, 13, 15, 19) | The docking marker was outside the camera view, so the robot waited and then gave up. Seeds 2 and 19 also collided with a corridor wall.                                                                                                                   |
| `not_docked`  | 4 (3, 4, 18, 20)        | The final docking check failed because the robot relied on the estimated pose. In seeds 4 and 18, the robot’s ground-truth pose was within 1.6 cm and 2.0° of the goal, but its position estimate was 24–31 cm away from the true position                 |
| `too_close`   | 1 (14)                  | The position estimation error reached 66cm on the long straight path. Because the robot used these incorrect estimated poses, it moved away from its planned path. Then, the LiDAR safety check detected an obstacle directly ahead and stopped the robot. |

The comparison between EKF and odometry-only runs showed that accumulated position estimation errors were the primary cause of the odometry-only failures. Depending on the run, these errors caused the robot to lose sight of the docking marker, fail the final docking check or trigger the LiDAR safety stop.

**2. Floor vent cover (environment):** In runs 09:40–09:43, the wheels were slipping but the encoders kept counting. The side markers were out of sight and the EKF only used wheel odometry, so the estimated pose drifted. The dock-marker measurements then disagreed with the prediction and were rejected by the NIS gate (all 30 detections in run 09:41) while Markers 1 and 2 were still accepted. The robot had no valid dock marker update so it safely stopped and reported `marker_lost`. The track was moved away from the vent cover and the next three runs docked with no change in code (see Section 7.2). The suggested solution is to check the floor along the track for floor obstacles prior to each run.

**3. Low LiDAR scan coverage (sensing):** During real-robot testing on 28 October, the forward LiDAR scan coverage was only 0.34-0.55, compared with 0.88-1.00 in simulation. This was likely because the LDS-01 received too few valid measurements from the black foam panels forming the corridor walls.

Because the scan coverage was below the required threshold of 0.8, the robot remained at its starting position. The threshold was then reduced to 0.2 for real-robot operation. In subsequent tests, the robot showed multiple successful runs with LiDAR being able to detect obstacles and trigger AVOID state.

#### B. Failures identified and fixed during development

These problems were identified and addressed during development, before the final trials reported in Section 6.1.

| Earlier problem                                                                  | Primary Cause                                                                                             | Fix                                                                                                                                         |
|----------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| EKF rejected marker measurements when the initial position was 11 cm off         | Estimator: initial covariance P0 too small (2 cm / 2° initial standard deviations), causing NIS rejection | Increased the EKF's initial position and heading uncertainty to 5 cm / 5° and generated random placement errors consistent with the new P0. |
| Robot collided with the box corner when returning to the path (3/10 tested runs) | Controller: avoidance clearance too small (0.05 m)                                                        | Increased avoidance clearance to 0.10 m. All 20 subsequent runs completed the task successfully.                                            |
| Robot selected the narrower side when avoiding a box (too_close, seed 5)         | Controller: Incorrect obstacle avoidance side selection                                                   | Changed the obstacle avoidance logic to select the side requiring less sideways movement ([avoidance.py](src/control/avoidance.py))         |

#### C. Remaining limitations

- **Model mismatch not confirmed:** No failures in the final trials were directly linked to differences between the mathematical model and the real robot's behaviour. Small docking errors remained in the nominal runs, but their causes were not determined. The limitations of the linear model are discussed in Section 2.4.
- **Lower LiDAR coverage threshold not re-tested:** The threshold was reduced from 0.8 to 0.2 to allow the robot to operate despite the low LiDAR scan coverage caused by the black foam panels. Previously, the robot required at least 80% valid scan coverage to move, but after the change, only 20% was required. Although this allowed the robot to operate, it also meant that the robot could continue moving with less valid LiDAR data. The system was not re-tested with a blocked or faulty LiDAR, so its safety under these conditions remains unverified.

## 7. Real Robot
### 7.1 Setup and differences from simulation 
We tested the system on a TurtleBot3 Waffle Pi running ROS 2, with an LDS-01 LiDAR and a Pi camera. Our laptop receives the sensor data and sends velocity commands over the lab wifi through rosbridge (UTS-RI rosbridge_api). The real robot interface ([real_robot.py](src/robot/real_robot.py)) provides the same sensor-reading and command functions as the simulation, so the EKF, the controllers and the mission logic run unchanged on both platforms. Wheel joint data from /joint_states is used to compute odometry, camera images are used to detect ArUco markers, and LiDAR data from /scan is used for obstacle avoidance.

The laptop controls the robot's linear and angular velocity through /cmd_vel.
The real track contained three 100 mm ArUco markers and a 20 cm box (obstacle).

The run command is `python main.py --real --task config/arena_lab.json`.

<p align="center">
  <img src="results/page/robot_start.jpg" width="30%" alt="Real robot at the start">
  <br><em>The real robot on its start mark, with markers 1 and 2 on the front wall and the obstacle (box) in the driving lane.</em>
</p>


| Setting | Simulation | Real robot | Why they differ |
|---|---|---|---|
|Speed limits| Linear: 0.26 m/s / Angular: 1.82 rad/s | Linear: 0.1 m/s / Angular: 1.0 rad/s| Safety caps for the first lab runs; slower driving also reduces the effect of communication delays |
|Minimum valid LiDAR forward coverage| 80% | 20% | The real LiDAR does not get returns from every surface; the return depends on factors such as material and distance. In our lab, the black foam panels produced very few returns |
|Docking| v: 0.08 m/s / lateral tolerance: 3 cm / heading tolerance: 5°|v: 0.05 m/s / lateral tolerance: 4 cm / heading tolerance: 8°, stop 1.5 cm early| The real robot moves more slowly and stops early to reduce overshoot at the docking position. Wider tolerances allow for the localisation and control errors seen in the hardware tests |
|Avoidance clearance| 0.1 m, side margin 0.05 m | 0.08 m, side margin 0.03 m| A smaller clearance and side margin let the planned avoidance path fit within the narrower real lane |
### 7.2 Real-robot results with the same metrics
**Real-robot trials**: We filmed 9 full-track runs on 30 October. In three runs, the box was centred in the lane; in the other six, it was shifted 15 cm left or right.

The real robot has no external ground-truth system, so the cross-track error in the table is the EKF estimate and was not independently verified. For five of the six successful docks, we measured the robot's final position and heading with a tape to compare them with the simulation (7.3).

| Run | Box position | Outcome | Run time | Estimated cross-track RMS during TRACK | Marker measurements accepted |
|---|---|---|---|---|---|
| 09:28 | Centre | Docked | 62.4 s | 1.9 cm | 96% |
| 09:32 | Centre | Docked | 62.2 s | 2.4 cm | 95% |
| 09:36 | Centre | Docked | 62.3 s | 2.1 cm | 98% |
| 09:40 | Right | marker_lost | 53.5 s | 1.7 cm | 93% |
| 09:41 | Right | marker_lost | 53.3 s | 1.2 cm | 93% |
| 09:43 | Left | marker_lost | 56.4 s | 2.6 cm | 83% |
| 10:48 | Right | Docked | 60.8 s | 1.4 cm | 96% |
| 10:54 | Right | Docked | 61.8 s | 1.4 cm | 96% |
| 10:59 | Left | Docked | 61.5 s | 1.7 cm | 99% |

The three unsuccessful runs (09:40–09:43) failed during the final approach, where the side markers were out of view and the EKF relied on wheel odometry alone. The robot caught on a raised floor vent cover while its wheel odometry kept counting, so the estimated pose became inaccurate, and the dock-marker measurements were missing or rejected by the NIS gate. The robot then stopped safely.

After the track was moved away from the vent cover, the next three runs docked without any code change. This shows that the NIS gate protects the estimate from bad measurements, but once the estimate itself is wrong and only one marker is in view, it cannot recover on its own.

### 7.3 Quantitative simulation vs real comparison 
The table below shows the difference between the simulation and the real robot in the docking stage.

| Metric (5 measured hardware docks and their simulation counterparts) | Simulation | Real robot |
|---|---|---|
| Final distance to dock marker — target: 35 cm | 36.55 ± 0.03 cm | 35.35 ± 0.25 cm |
| Lateral docking error | +0.46 ± 0.02 cm | +0.52 ± 0.35 cm |
| Heading docking error | −0.77 ± 0.03° | −0.65 ± 1.11° |
| Completion time | 43.8–44.9 s | 61.5–62.4 s |

Both systems used a 35 cm target distance and were set to stop 1.5 cm early, at about 36.5 cm from the marker. The simulation stopped at 36.55 cm, close to this value, while the real robot stopped at 35.35 cm. This is consistent with the real robot rolling about 1.2 cm after the stop command. The real robot took longer to finish because its interface limits the speed to 0.10 m/s, while the simulation drives at 0.2 m/s.


## 8. Limitations and Lessons
### 8.1 What the results do and do not support 

| results              | supports            | doesnt support                             |
| :------------------- | :------------------ | :----------------------------------------- |
| Nominal run, 4 cases | system completes mission in this setup               | repeatability |
| Cross-track RMS      | EKF is slightly better in 3/4 cases                  | gap in the centre case |
| Trials: EKF 100%     | EKF is more robust to start-pose, and encoder errors | Whenever it is always safe or not. With the trials only covering a limited range |
| ANEES                | ANEES is close to the target (3)                              | Consistency: 3/4 sit just below the ANEES band |
| NEES inside 88 - 89% | estimator is mostly honest                           | P is reliable at every step |
| ANIS about 0.1, NIS inside 41-46%       | estimator is cautious about the camera               | Whenever R is right or not |

These results have been gathered from ([run_demo](evaluation/run_demo.py)). All of these results are generated from the simulation from 10-20 seeds. 

## 9. Reproducibility
### 9.1 Install and run from a fresh environment 

You need **Python 3.10** (64-bit) and **Git**. Each member creates their own virtual environment
`.venv_mujoco` inside the project folder; it is never committed.

##### Linux (Ubuntu)

```bash
git clone https://github.com/Atomic-add/IntegratedProject_SensorsAndControlsForMechatronicSystems_SPR2026.git
cd IntegratedProject_SensorsAndControlsForMechatronicSystems_SPR2026
python3.10 -m venv .venv_mujoco            # Ubuntu: sudo apt install python3.10-venv if this fails
source .venv_mujoco/bin/activate
python -m pip install --upgrade pip
python -m pip uninstall -y opencv-python  # only matters if it was installed before ("Skipping" is fine)
python -m pip install -r requirements.txt
```

##### Windows (PowerShell)

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

##### Check the installation

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

### 9.2 Commands that regenerate each figure and table 

```bash
python -m evaluation.run_demo                # everything: 4 cases on odometry and on the EKF, repeated trials, comparison
python -m evaluation.run_demo centre --ekf   # one case on the EKF -> results/demo_centre_ekf.png
python -m evaluation.random_trials --n 20    # 20 random scenarios on the EKF -> results/random_trials.png/.json
python -m evaluation.random_trials --n 20 --odometry   # the same 20 on odometry only
python -m evaluation.sweep_r --n 20          # R x10, x1, /10: NEES, NIS, error -> results/sweep_r.png/.json
python -m evaluation.marker_layout           # where the lab landmarks go: markers in view + EKF replay -> results/marker_layout.*
python -m evaluation.consistency_sweep       # Qd x Rn grid on recorded lab runs: ANEES, ANIS, 2-sigma coverage -> results/consistency_*.*
python -m evaluation.plot_like_a3          # Assessment 3 style: EKF error +-2 sigma, 2x2 summary, sim and real clips -> results/ekf_*, real_anim_*
python -m evaluation.observability          # rank / Gramian per marker case, P growth with no marker, rank along the path -> results/observability.*
python -m evaluation.sweep_lqr --n 20       # LQR weight R x10, x1, /10 on the EKF: tracking error vs effort -> results/sweep_lqr.*
python -m evaluation.model_limits --n 20    # fixed K: |eig| vs speed, linear vs nonlinear, mission at 0.10/0.20/0.26 m/s -> results/model_limits.*
```

Figures and metrics go to `results/` (`demo_*.png`, `demo_metrics*.json`, `demo_trials_*.json`,
`demo_compare.*`). An EKF run takes about 2 min on a machine without a GPU, because rendering the
camera dominates; odometry-only runs take about 20 s. The trial scripts run `--workers 4` sims at a time
(`evaluation/__init__.py`): one per core renders 20 cameras at once and freezes a laptop.

### 9.3 Ground-truth rule: truth is read only in evaluation/

The controller must use the estimated state, never the ground truth. Ground truth can only be read in evaluation/


## 10. Team and Declarations 
### 10.1 Group number and team members

- Addison Possamai (25840164)
- Nguyen Kim Uyen Phan (25596519)
- Nhat Thien Mai (25161391)

### 10.2 Individual contribution statement

#### Contribution table
|   Task                             | Person          |
| :--------------------------------- | :-------------- |
|    Overview                        | Addison + Kim   | 
|    Modelling (A)                   | Kim             | 
|    Sensing (B)                     | Thien           | 
|    Estimation (C)                  | Addison         | 
|    Control (D)                     | Kim             | 
|    Integration and Robustness (E)  | Kim             | 
|    Real Robot                      | Thien           |
|    Limitations and Lessons         | Addison         | 
|    Reproducibility                 | Addison         | 

#### Addison's  statement
I was assigned the estimation section of the project for the robot. With Kim helping me, as part of her Control section. After that, I was responsible for organising the project page. Getting everything up. As well as helping to organise hardware access for the team.

#### Kim's statement
I was responsible for designing the robot controller and contributed to system integration and assisted with real-robot implementation. I also contributed to the Overview, Modelling, Control, and Integration sections of the project page. For the final video, I prepared my own slides and script and was responsible for editing and finalising the video. Additionally, I conducted a final quality check of the group's assignment to ensure its accuracy, consistency, and completeness.

#### Thien's statement


### 10.3 Generative AI declaration

**Claude (Anthropic)** was used to scaffold the simulation environment, the sensor models and their checks.

Claude was also utilised to help guide the development of the project page. By recommending what to cover for each section, and where to get the information from. Acting only as a push in the right direction. 

Additionally, Claude was utilised to turn strings of data outputs into tables. Especially important for scripts such as ([run_demo](evaluation/run_demo.py)). As the text produced was too long, and confusing to read, to belong on the project page.

Claude was also used to convert the team's Word documents into Markdown for the project page.

### 10.4 Acknowledgements

- Robot model and meshes: [ROBOTIS robotis_mujoco_menagerie](https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie),
  Apache-2.0 (`models/robotis_tb3/LICENSE`); sensor poses from ROBOTIS `turtlebot3_waffle_pi.urdf`.
- Code adapted from the 41014 Week 2-9 tutorial solutions (cited in each module's docstring).
- Real-robot link: [UTS-RI/turtlebot_rosbridge](https://github.com/UTS-RI/turtlebot_rosbridge) (`rosbridge_api`,
  MIT), used unmodified; `src/robot/real_robot.py` hooks its message callbacks to keep arrival times.



