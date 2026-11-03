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

### EKF design
The estimation section is adapted from **Week 7 C.3 / C.4**, and **Assessment 3, Task B.3**, borrowing the following variables:
- ekf_predict
- ekf_update
- F_uni
- h_rb
- H_rb
- Qd reasoning

#### state

State x = [x, y, theta]

xh, P = ekf.step(encoders, marker_frame = None)

#### inputs

#### motion model (predict)

#### measurement model (update)

#### safeguards

With the 


### Tuning

### Reported performance
Below is the testing for the estimation.

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


### Tuning trade-off

### Comparison with 2nd estimator


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
