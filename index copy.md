# Track A project page structure (final)

## GROUP NUMBER / TEAM MEMBERS
**TEAM 12**
- Addison Possamai (25840164)
- Nguyen Kim Uyen Phan (25596519)
- Nhat Thien Mai (25161391)

## 1. Overview
### 1.1 Task, platform, the rule that the controller uses the estimated state only, plus repo and video links [D]
### 1.2 Pipeline diagram, simulation and real-robot photos [D]
### 1.3 What is new beyond the coursework (Assessment 3 covered path-error LQR/LQI and a range-only EKF): range+bearing ArUco EKF, LiDAR avoidance, IBVS docking, randomised trials, real robot [D]
### 1.4 Headline results: simulation vs real robot [HD]
### 1.5 Marker's guide: where each criterion is evidenced, plus video timestamps for the task completed, a disturbed trial, and the estimate with uncertainty vs ground truth [D]

## 2. Modelling (A): state-space model, controllability and observability analysis
### 2.1 State-space model: derived for the TurtleBot3, with stated assumptions [D]
**Error dynamics:** Path following regulates the error to the path. With cross-track error e<sub>y</sub>, heading error e<sub>θ</sub> and curvature κ, the error dynamics are (A3 A.1)

$$\dot e_y = v\sin e_\theta, \qquad \dot e_\theta = \omega - \frac{v\kappa\cos e_\theta}{1-\kappa e_y}$$

**Linearisation:** Substituting ω = vκ + δω and writing the error dynamics above as ė = f(e, δω), A and B are the Jacobians of f at the operating point e = 0, δω = 0 (W5):

$$A = \left.\frac{\partial f}{\partial e}\right|_{e=0}, \qquad B = \left.\frac{\partial f}{\partial\,\delta\omega}\right|_{e=0}$$

$$\dot e = \begin{bmatrix}0 & v\\ -v\kappa^2 & 0\end{bmatrix} e + \begin{bmatrix}0\\ 1\end{bmatrix}\delta\omega$$

On the lab's straight segments κ = 0, so A = [[0, v], [0, 0]] and B = [0, 1]: v sits inside A, and at v = 0 the robot cannot steer out a lateral error.

**Discretisation:** The controller runs every T<sub>s</sub> = 0.05 s and holds each command until the next, so the model is discretised with a zero-order hold:

$$A_d = e^{AT_s}, \qquad B_d = \int_0^{T_s} e^{A\tau}B\,d\tau$$

Because A² = 0 on a straight, the series e<sup>AT<sub>s</sub></sup> = I + AT<sub>s</sub> + (AT<sub>s</sub>)²/2 + … stops after two terms, so the result is exact:

$$A_d = I + AT_s = \begin{bmatrix}1 & vT_s\\ 0 & 1\end{bmatrix}, \qquad e^{A\tau}B = \begin{bmatrix}v\tau\\ 1\end{bmatrix} \;\Rightarrow\; B_d = \begin{bmatrix}vT_s^2/2\\ T_s\end{bmatrix}$$

At v = 0.2 m/s:

$$A_d = \begin{bmatrix}1 & 0.01\\ 0 & 1\end{bmatrix}, \qquad B_d = \begin{bmatrix}0.00025\\ 0.05\end{bmatrix}, \qquad e[k+1] = A_d\,e[k] + B_d\,\delta\omega[k]$$

Steering changes e<sub>θ</sub> directly (T<sub>s</sub> = 0.05 per unit δω), but moves e<sub>y</sub> only by turning the heading first, with coupling vT<sub>s</sub> = 0.01, so the authority over e<sub>y</sub> shrinks with speed and vanishes at v = 0.

**Assumptions:** Kinematics only (no slip, no motor lag), constant v, small errors (checked in 2.4), nominal track width 288 mm (UMBmark: 292.6 mm). The controller sees only the EKF estimate.

**Open-loop behaviour:** On a straight, the open loop has a repeated eigenvalue at z = 1, so it is unstable: a constant heading error makes e<sub>y</sub> grow linearly. Our lab path is straight segments (κ = 0) with corners turned on the spot, but to check the model on a curve we also test an arc of R = 0.5 m (κ = 2 /m). There the eigenvalues are s = ±j vκ = ±0.4j rad/s (on the unit circle in z), so it is marginally stable but not asymptotically stable. Feedback is needed in both cases, and because v sits inside A, everything below depends on speed.

### 2.2 Controllability analysis [D]
With A = [[0, v], [0, 0]] and B = [0, 1] from 2.1 (on a straight), C = [B AB] = [[0, v], [1, 0]], so det C = −v.

| Speed | Rank of C | det C |
|---|---|---|
| 0.20 m/s (cruise) | 2 | −0.20 |
| 0.05 m/s (docking) | 2 | −0.05 |
| 0 (turn on the spot) | **1** | 0 |

**Interpretation:** Controllable at every speed except zero: at v = 0, the only path from steering to e<sub>y</sub> (ė<sub>y</sub> = v e<sub>θ</sub>) is cut, so rank C drops to 1. This is the nonholonomic constraint: the robot cannot move sideways.

The rank test only says whether the system is controllable, not how easy it is to control. The singular values of C are 1 and v (calculated in the code). Therefore, slow driving is weakly controllable, not uncontrollable. So, the LQR is used only when the robot is driving on a straight. At the corners, the robot stops and turns on the spot with a P-controller, as e<sub>θ</sub> stays controllable at v = 0.

### 2.3 Observability analysis, including marker lost [D] MISSING
**Measurement:** The EKF measures range and bearing to each marker centre, not the full ArUco pose, so each marker in view adds two rows to the measurement Jacobian H<sub>k</sub>. The pose has three states, (x, y, θ).

**Rank test:** A linear model with sensor matrix C (y = Cx) is observable when the observability matrix has full rank:

$$\mathcal{O} = \begin{bmatrix}C\\ CA\\ \vdots\\ CA^{n-1}\end{bmatrix}, \qquad \mathrm{rank}(\mathcal{O}) = n$$

In the EKF, the measurement Jacobian H<sub>k</sub> and the motion Jacobian F<sub>k</sub> change every frame, so the same idea is written frame by frame. Each frame's measurement is predicted from the initial pose x<sub>0</sub>:

$$z_0 = H_0x_0, \qquad z_1 = H_1F_0\,x_0, \qquad z_2 = H_2F_1F_0\,x_0, \ \dots$$

Stacking these gives

$$\begin{bmatrix}z_0\\ z_1\\ z_2\\ \vdots\end{bmatrix} = \mathcal{O}_N\,x_0, \qquad \mathcal{O}_N = \begin{bmatrix}H_0\\ H_1F_0\\ H_2F_1F_0\\ \vdots\end{bmatrix}$$

If rank(O<sub>N</sub>) = 3, x<sub>0</sub> can be solved from the measurements.

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

With no marker, nothing is measured and the error drifts instead of decaying, so the pose is not even detectable. Driving blind would let the 3σ bound pass the 3 cm tolerance after about 3.5 s, so docking never drives without a marker. It stops at once, waits for the marker, and declares failure if none returns within 3 s ([docking.py](src/control/docking.py), `marker_lost_timeout` = 3 s in [control.json](config/control.json)).

### 2.4 Model correctness: where the model breaks down and what that costs, e.g. closed-loop eigenvalues vs speed (adapt Assessment 3 D.4) [HD] (A has no hardware credit, so this is A's HD)
The robot was run at three speeds without redesigning K (20 runs each, [model_limits.py](evaluation/model_limits.py)). For δω = −(k<sub>y</sub> e<sub>y</sub> + k<sub>θ</sub> e<sub>θ</sub>) on a straight, the closed-loop poles satisfy:

$$\lambda^2 + k_\theta\lambda + k_y v = 0$$

so they depend on v. The decay time shortens 3.5 times from 0.10 to 0.26 m/s (3.2 to 0.92 s), but the distance over which an error decays changes much less, from 0.32 to 0.24 m.

| Speed | Slow pole \|z\|, straight / test arc (R = 0.5 m) | S1 cross-track RMS | NEES ≤ 7.81 | Docked |
|---|---|---|---|---|
| 0.10 m/s | 0.985 / 0.984 | 1.65 cm | **89 %** | 20 / 20 |
| 0.20 m/s (design) | 0.965 / 0.960 | 1.73 cm | 96 % | 20 / 20 |
| 0.26 m/s (top speed) | 0.947 / 0.933 | 1.80 cm | 99 % | 20 / 20 |

<p align="center">
  <img src="results/model_limits.png" width="90%" alt="Model limits">
  <br><em>Closed-loop poles, linear vs nonlinear response, and tracking and NEES against speed.</em>
</p>

**Linear vs nonlinear:** Both models start from the same lateral offset at 0.2 m/s on a straight, with the same K. The nonlinear model uses sin e<sub>θ</sub> and clips ω at the robot's 1.82 rad/s (same script; values in [model_limits.json](results/model_limits.json), "validity").

| Initial offset | Peak command | Peak heading error | Linear vs nonlinear gap |
|---|---|---|---|
| 5 cm (maximum in the full runs) | 0.46 rad/s | 6° | 0.13 % |
| 20 cm | 1.83 rad/s (limit reached) | 26° | 1.2 % |
| 50 cm | 4.6 rad/s (clipped) | 64° | 12 % |

In the full runs, the cross-track error peaks at about 5 cm (5.2 cm in the worst of 60 runs), well inside the range where the linear model holds; it breaks down only for large offsets, where the command is clipped at 1.82 rad/s and the heading error is too large for sin e<sub>θ</sub> ≈ e<sub>θ</sub>.

**Interpretation:**

- **Controller: robust to speed.** An error is always corrected over roughly the same distance, so tracking RMS rises only 9 % from 0.10 to 0.26 m/s. The same K works at every tested speed without redesigning.
- **Estimator: degrades first, at low speed.** We follow the Assessment 3 specification S4: samples with NEES inside the 95 % χ²<sub>3</sub> bound (7.81) ≥ 90 %. Position RMSE does not change with speed (0.74, 0.79, 0.69 cm), but at 0.10 m/s NEES coverage falls to 89 %, below A3's 90 %, so the filter is overconfident. A likely cause, not yet tested on its own, is that Q<sub>d</sub> (process noise covariance) was tuned around the design speed and does not grow enough at low speed, so the filter becomes overconfident.
- **What the model leaves out.** The linear model ignores the per-wheel speed limit (saturation), camera latency (lag) and a +0.6 cm steady offset (persistent error). An LQI could remove an offset caused by an outside disturbance. However, if the EKF estimate itself is biased, the LQI cannot remove that offset, as it integrates the estimated e<sub>y</sub> and sees no error to correct.

## 3. Sensing (B): calibration, noise handling and frame management
The robot in this project uses three sensors including: wheel envodeer, a camera, and a LiDAR. Both the [simulator](src/robot/sim_robot.py) and the [real robot](src/robot/real_robot.py) read sensor data through the same three functions: read_encoders(), read_image() and read_scan(), with the real robot sends its data through rosbridge. These functions are used in both the simulator and the real robot to be processed using the same perception and estimation logic.  

- Wheel encoders (20Hz): provide wheel rotation measurements for odometry calulation, which then use for EKF to predict the robot's pose

- Pi camera (640 x 480 pixels, 20Hz): measures range and bearing to visible ArUco markers for EKF updates. Our track uses three markers from the DICT_4x4_50 dictionary, with a 100 mm side length. 

- LDS-01 LiDAR (360 beams, 1° interval, 5Hz): measures the free distance ahead, obstacle's edges and side clearances for obstacle avoidane
### 3.1 Calibration: camera, plus LiDAR validated against ray-cast truth 
Camera intrinsics: We calibrated both caemeras using a 9 x 6 checkerboard with 25 mm squares, ([calibrate_camera.py](experiments/calibrate_camera.py)). 

**Simulated camera**: In simulation, the board was palced at 20 random poses in front of the camera
<p align="center">
  <img src="results/camera_calibration.png" width="80%" alt="Simulated camera calibration">
  <br><em>Simulated camera: board found in 20 of 20 frames.</em>
</p>
*Simulated amera: Board found in 20 of 20 frames*.

**Real camera**: For the real camera, the board was held by hand, trying different posision and angles in front of it. A set of 40 captured images were collected for calibration.
<p align="center">
  <img src="results/page/camera_calibration_real.png" width="80%" alt="Real camera calibration">
  <br><em>Real camera: board found in 40 of 51 captured images; those 40 were used for calibration.</em>
</p>

*Real camera: Board found in 40 of 51 captured images, with thoes 40 images used for calibration*

These collected data was then used to estimate the camera matrix K and lens distortion. To assessed the calibration quality, reprojection error was used, with the table below compares the results from both cameras. 

| Parameter | Simulation | Real camera |
|---|---|---|
| Frames used | 20 / 20 | 40 / 51 |
| Reprojection RMS | 0.046 px | 0.41 px |
| Focal lengths f<sub>x</sub>, f<sub>y</sub> | 692.07, 692.01 px | 691.5, 692.3 px |
| Principal point (c<sub>x</sub>, c<sub>y</sub>) | (302.2, 245.2) px | (302.2, 245.2) px |

The estimated simulated camera were then also compared with the true value from Mujoco camera. The estimated focal lengths matched the true value (692.00 px) within 0.010 %.

**Camera mounting pose**: Both the simulated and real cameras use the URDF mounting position: 0.076 m forward and 0.103 m above the robot's base origin. For the real camera, we also estimated pose of the camera ([camera_pitch.py](experiments/camera_pitch.py)). Returning an estimation a downward tilt of 6.3 degree from 50 AruCO images taken at different distances with a fixed marker height ([camera tilt images data](results/hardware)). The simulation camera was then reconfigured with the same 6.3° tilt to match the real one. 

**Real-camera range correction**: To verify the camera distances measurement, we compared the raw AruCo range estimates with the reference distances ([aruco_real_noise.py](experiments/aruco_real_noise.py)). The camera overestimated rage by approximately 0.6mm +3.2% of the reference distance, therefore a correction for this offset and scale error are used by [aruco.measure()](src/perception/aruco.py) to correct each range estimate before it is passed to the EKF for real robot

**Simulated LiDAR validation**: To check whether our LiDar measurement, noise and dropout behaviour matched the configured model, we compared its range with reference distances calcualted by a seperate 2D ray caster using the same map geometry, at five poses on the robot route (([check_sensor_lab.py](evaluation/check_sensors_lab.py)), ray caster in [grouth_truth.py](evaluation/ground_truth.py)).
The result show that, tange-noise standard deviation was 9.95 mm, close to the configured 10 mm. The beam dropout rate was 1.01%, compared with the configured 1%
<p align="center">
  <img src="results/page/ray_cast_test.png" width="65%" alt="Simulated LiDAR vs ray caster">
</p>

### 3.2 Noise handling: measured noise characterisation 


**Planar-pose ambiguity**:
We use ArUco range and bearing for the EKF correction measurement. This is because, for a flat marker viewed nearly straight on, IPPE can return two similar pose solution with different marker orientation. Therefore, we exclude the marker yaw and evaluate the range and bearing below. 
**ArUco measurement errors in simulation.**: 
We tested how ArUco range and bearing errors change with distance to check the EKF’s range-bias correction and measurement-noise settings in R<sub>n</sub>. This is done by using 20 frames with different robot poses then compare it with simulation grouth truth

The result showing that simulated camera tended to overestimate marker distance, with mean range errors of around 65 mm at 3 m and 131 mm at 4.5 m. Bearing bias remained below 0.02° at the tested poses. 
Because similar measurement accross consecutive frames can still share the same bias. consistent reading are not always accurate. To solve this, EKF first subtrach the know bias (4.5 mmr^2) and its range noise is set larger than the frame to frame spread to reduce it overcondident on repeated readings.  
<p align="center">
  <img src="results/page/lab_aruco.png" width="90%" alt="ArUco errors on the lab track">
</p>

**Real-camera range measurements**:
We tested the camera range reading at seven different distance, keeping the robot and marker stationary between each distance. Between these trial, the range standard deviation across repeated image was 0.05-3.4 mm. We then used tape-measured distances as reference value and fitted a linear correction to the raw ArUco ranges using ([aruco_real_noise.py](experiments/aruco_real_noise.py)). This removes a fixed offset and adjust the range scale. After appying it, the mean diffeerence was within +- 3mm for the datasets used to fit the EKF correction. 
<p align="center">
  <img src="results/page/aruco_real_simple.png" width="60%" alt="Real camera ArUco range">
</p>

**Wheel-encoder evaluation**:
To evaluate the robot wheel odometry, we used [ubmmark.py](experiments/umbmark.py) to assess it errors. The robot drove 0.5 m squres three times in clockwise and counter-clockwise. With each run we measure the final position and heading errors. 

The result showing that a right to left wheel size ratio is 1.006 and a wheel seperation of 292.6 mm, compared with the nomina 288 mm. The wheel distance noise estimate was 5.2%, to include manual measurement error, we kepth k_wheel=6%. We kept the nominal wheel parameters and used Aruco updates to reduce pose drift. 
### 3.3 Noise handling: outlier rejection and marker-loss handling [D]

### 3.4 Frame management: frame chain from sensor measurement to controller input [D]

### 3.5 Simulated vs real sensor noise, and why they differ [HD-hw]

## 4. Estimation (C): EKF design, tuning and reported performance
### 4.1 EKF design [D]
### 4.2 Tuning: Q, R and P0, and where each value comes from [D]
### 4.3 Reported performance: RMSE, NEES, NIS, ±3σ bounds [D]
### 4.4 Tuning trade-off: consistency vs responsiveness (R sweep) [D]
### 4.5 Comparison with a second estimator: EKF vs odometry only, same seeds [HD] (C has no hardware credit, so this is C's HD; re-run, current files identical)

## 5. Control (D): controller design, tuning and tracking results
### 5.1 Controller design: LQR [D]
### 5.2 Tuning: weights, why, and what limits the gains [D]
### 5.3 Tracking results in sim [D] (real-robot numbers: see 7.2)
### 5.4 Tuning trade-off: control effort vs tracking error (weight sweep, 3 values over a factor of 10; adapt Assessment 3 D.3) [D] MISSING
### 5.5 (Optional) Controller extension compared against baseline: IBVS vs PBVS docking [HD]

## 6. Integration and Robustness (E): performance under disturbance and noise
### 6.1 Performance under disturbance and noise: trials per condition, completion rate, docking success rate, collision rate, in sim [D] (real runs: see 7.2)
### 6.2 Failure analysis traced to model, sensing, estimator or controller [D]
### 6.3 (Optional) Robustness study beyond the brief, compared against nominal: sensor noise sweep (x1, x2, x4) MISSING, second obstacle in the narrow corridor (leg1-box), hidden placement and wheel-scale errors [HD]

## 7. Real Robot
### 7.1 Setup and differences from simulation [HD-hw]
### 7.2 Real-robot results with the same metrics: tracking (D), docking (D), number of runs under real noise (E) [HD-hw]
### 7.3 Quantitative simulation vs real comparison [HD-hw]

## 8. Independent Contribution Summary (only list what has results) Core (needed):
### 8.1 Real-robot implementation (section 7): HD evidence for B, D, E [HD-hw]
### 8.2 Model limits (2.4) and limitations (9.1): HD evidence for A [HD]
### 8.3 Second estimator comparison (4.5): HD evidence for C. Cheapest option: the odometry-only mode already exists, it just needs a proper re-run [HD]

## 9. Limitations and Lessons
### 9.1 What the results do and do not support [HD]
### 9.2 What failed during development, and how it was diagnosed [D]

## 10. Reproducibility
### 10.1 Install and run from a fresh environment [D]
### 10.2 Commands that regenerate each figure and table [D]
### 10.3 Ground-truth rule: truth is read only in evaluation/ [D]

## 11. Team and Declarations (required; a missing item caps the affected criteria at Pass)
### 11.1 Group number and team members
### 11.2 Individual contribution statement
### 11.3 Generative AI declaration: tools and purpose
### 11.4 Acknowledgements: tutorial code, Assessment 2/3 code reused, ROBOTIS model, MuJoCo examples
Every results subsection ends with one sentence explaining why it looks that way.



