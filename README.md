# SCMS Track A Project (Team 12)

## 1. Overview
**Task**

This is Track A, a mobile robot. This task has a TurtleBot navigating a room of obstacles to reach its ArUco-marked charging station. This task has been performed both in MuJoCo and in real life. Utilising: 
- 2D LiDAR
- Camera
- Wheel-encoder odometry
This task’s constraint is that the controller must use the estimated state, never the ground truth

**pipeline diagram**

I.O.U

**simulation**

![Your Alt Text](./Overview_Simulation.png)

**hardware appliance**

![Your Alt Text](./Overview_Real.png)


**what is included in coursework**

**Results**

**Marker's Guide**


**repo:**
(https://github.com/Atomic-add/SCMS_IntegratedProject.github.io.git)

**video link**
INSERT VIDEO



## 2. Modelling (A): state-space model, controllability and observability analysis


## 3. Sensing (B): calibration, noise handling and frame management


## 4. Estimation (C): EKF design, tuning and reported performance

### 4.1 EKF design
The estimation section is adapted from **Week 7 C.3 / C.4**, and **Assessment 3, Task B.3**, borrowing the following variables:
- ekf_predict
- ekf_update
- F_uni
- h_rb
- H_rb
- Qd reasoning

The design has 2 main functions, predict, and update.

predict is generated once per new encoder sample. Utilised to move the estimate, with the wheel increments, and grow P by the wheel noise. 

Update is generated once per new camera frame. This fuses one marker's range and bearing, and returns its NIS.

### 4.2 Tuning: Q, R and P0, and where each value comes from [D]

- **Q (process noise)**: This refers to noise generated from the robot's environment. Some include the roughness of the floor, and the radi of the wheels.

- **R (sensor noise)**: These are random, unwanted variations of the sensor's output. Affecting accuracy and reliability of measurements

- **P0 (initial/reference position)**: This is the starting position of the robot.

The table below is an expansion upon these 3 variable groups. Illustrating what each variable belongs to, and what it does.

|  Group   |    Variable     |                    What it is                    |
| :------- | :-------------- |  :---------------------------------------------- |
|   Q      | k_wheel         | Wheel error, which grows with distance travelled |
|          | q_floor         | noise, which is added to Qd every step. Keeps P  |    |          |                 | positive                                         |
|   R      | range_sd [a,b]  | range noise = a + b*r^2                          |
|          | bearing_sd_deg  | bearing noise                                    |
|          | range_bias_r2   | Known camera rnage bias.                         |
|   P0     | P0_sd           | Starting uncertainty: x, y, heading              |    

All variables come from ([estimation.json](config/estimation.json))

### 4.3 Reported performance

To begin, below is a simple pass/fail check from ([estimation_test.py](src/estimation/estimation_test.py)). This serves to be a quick and efficient method of checking if it is able to pass everything or not.

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

0 failed - all checks passed

------------------------------------------------------------------------------------

Expanding beyond this, sweep_lqr g


### 4.4 Tuning trade-off

This runs a test to compare Consistency to Responsiveness. It scales the EKF's measurement noise (R). Watching NEES, NIS, and the error.

These test results are from ([sweep_r](evaluation/sweep_r.py))


### 4.5 Comparison with 2nd estimator
This compares the EKF to odometry through 20 random tests. Observing through simulation if it could dock successfully

| Run | Docked | Contacts | Notes |
|---|---|---|---|
| EKF | 20 / 20 | 0 | 42.0 +- 1.9 s, dock lateral 0.23 +- 0.44 cm, yaw -0.37 +- 0.76 deg, ANEES 3.8; seed 5 (box beside the lane) used to stop `too_close` on its narrow side; the pass side is now the one needing the smaller sideways offset (`avoidance.py`) |
| Odometry only | 9 / 20 | 2 | the hidden placement and wheel errors are never corrected (estimate off by up to 74 cm) |

## 5. Control (D): controller design, tuning and tracking results



## 6. Integration and Robustness (E): performance under disturbance and noise



## 7. Real Robot


## 8. Independent Contribution Summary (only list what has results) Core (needed):


## 9. Limitations and Lessons
**What do and dont the results support?**

**What failed during development? How was it diagnosed**

## 10. Reproducibility


## 11. Team and Declarations 
### Group Number and Team members
**TEAM 12**
- Addison Possamai (25840164)
- Nguyen Kim Uyen Phan (25596519)
- Nhat Thien Mai (25161391)

### individual contributions

|   Task                             | Person          |
| :--------------------------------- | :-------------- |
|    Overview                        | Addison         | 
|    Modelling (A)                   | Kim             | 
|    Sensing (B)                     | Thien           | 
|    Estimation (C)                  | Addison         | 
|    Control (D)                     | Kim             | 
|    Integration and Robustness (E)  | Kim             | 
|    Real Robot                      | Thien           |
|    Independent Contribution Summary| Kim             | 
|    Limitations and Lessons         | Addison         | 
|    Reproducibility                 | Addison         | 



### Generative AI delaration
**Claude (Anthropic)** was used to scaffold the simulation environment, the sensor
models and their checks; see the project page's GenAI declaration.

### Acknowledgements
- Robot model and meshes: [ROBOTIS robotis_mujoco_menagerie](https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie),
  Apache-2.0 (`models/robotis_tb3/LICENSE`); sensor poses from ROBOTIS `turtlebot3_waffle_pi.urdf`.
- Code adapted from the 41014 Week 2-9 tutorial solutions (cited in each module's docstring).
- Real-robot link: [UTS-RI/turtlebot_rosbridge](https://github.com/UTS-RI/turtlebot_rosbridge) (`rosbridge_api`,
  MIT), used unmodified; `src/robot/real_robot.py` hooks its message callbacks to keep arrival times.
