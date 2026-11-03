# SCMS Track A Project (Team 12)

## 1. Overview
**Task**
This is Track A, a mobile robot. This task has a TurtleBot navigating a room of obstacles to reach its ArUco-marked charging station. This task has been performed both in MuJoCo and in real life. Utilising: 
- 2D LiDAR
- Camera
- Wheel-encoder odometry
This task’s constraint is that the controller must use the estimated state, never the ground truth

INSERT PIPELINE DIAGRAM

![Your Alt Text](./images/banner.png)

![Your Alt Text](./images/banner.png)


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
