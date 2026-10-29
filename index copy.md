# Track A project page structure (final)



## 1. Overview
### 1.1 Task, platform, the rule that the controller uses the estimated state only, plus repo and video links [D]
### 1.2 Pipeline diagram, simulation and real-robot photos [D]
### 1.3 What is new beyond the coursework (Assessment 3 covered path-error LQR/LQI and a range-only EKF): range+bearing ArUco EKF, LiDAR avoidance, IBVS docking, randomised trials, real robot [D]
### 1.4 Headline results: simulation vs real robot [HD]
### 1.5 Marker's guide: where each criterion is evidenced, plus video timestamps for the task completed, a disturbed trial, and the estimate with uncertainty vs ground truth [D]

## 2. Modelling (A): state-space model, controllability and observability analysis
### 2.1 State-space model: derived for the TurtleBot3, with stated assumptions [D]
### 2.2 Controllability analysis [D]
### 2.3 Observability analysis, including marker lost [D] MISSING
### 2.4 Model correctness: where the model breaks down and what that costs, e.g. closed-loop eigenvalues vs speed (adapt Assessment 3 D.4) [HD] (A has no hardware credit, so this is A's HD)

## 3. Sensing (B): calibration, noise handling and frame management
### 3.1 Calibration: camera, plus LiDAR validated against ray-cast truth [D]
### 3.2 Noise handling: measured noise characterisation [D]
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



