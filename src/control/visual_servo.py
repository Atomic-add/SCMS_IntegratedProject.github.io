"""Image-based visual servoing (IBVS) for the last part of docking, on the Waffle Pi's own camera.

Adapted from 41014 Week 9 (Visual Servoing SOLUTIONS):
  A1/A3  the point image Jacobian (interaction matrix) J, p_dot = J nu, stacked over the four tag
         corners (8 x 6), checked against a finite difference (visjac_p, self_checks);
  B1b    the IBVS law  nu = lam * J^+ (p* - p), error stacked [u1, v1, u2, v2, ...] (flatten(order="F"));
  B2     the desired features p* by projecting the tag from the desired relative pose (desired_features);
  B3     a saturated command (vmax), as any real robot needs (docking.py);
  B4     depth: we use the corner depths from the tag pose PnP gives in every frame, the "first PnP
         depth, or updated online" choice; never the simulator's true depth;
  D4     the reduced Jacobian of a unicycle, J_mr = [J[:, 2], -J[:, 4]]: two columns, for (v, omega);
  D3/D4  the lesson that servoing alone leaves a unicycle FACING the tag but not on its normal. So
         docking.py first brings the robot onto the approach line with the LQR (EKF heading), and only
         then hands the last part to IBVS (self_checks shows the lateral offset IBVS cannot remove).

Added for the Waffle Pi:
  - the camera is NOT on the axle: it sits 7.6 cm ahead of the wheel axis (T_BC from the URDF), so a
    yaw rate also slides the camera sideways. robot_to_camera(T_BC) builds the 6 x 2 map from (v, omega)
    to the camera twist; with the camera on the axle it reduces to Week 9's [J[:, 2], -J[:, 4]];
  - fx, fy and the principal point come from OUR calibration (config/camera_sim.json), and the corners
    are undistorted with the calibrated distortion before they enter J;
  - p* comes from the docking geometry in the config (stop range, marker height, T_BC), so the IBVS
    goal and the range-based docking checks describe the same docked pose.

Frames: {B} base_footprint (x forward, y left, z up), {C} OpenCV camera (x right, y down, z forward),
{M} marker (x right and y up as the robot sees it, z out of the marker towards the robot). T_AB is the
pose of B in A. Corners in ArUco order TL, TR, BR, BL (src/perception/aruco.py).

Nothing here reads simulator ground truth: inputs are pixels, the PnP pose and the calibration.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from spatialmath import SE3

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # project root, so "src" imports work even
                                                                # when this file is run directly (VS Code Run)

from src.perception import aruco  # noqa: E402


# ---------------------------------------------------------------- the image Jacobian (Week 9 Part A)
def visjac_p(uv, Z, K):
    """Image Jacobian of point features in PIXELS (Week 9 A1/A3, MVTB CentralCamera.visjac_p).

    uv: 2 x N pixel coordinates, Z: depth of each point [m] (one number = the same for all), K: 3 x 3.
    Returns J (2N x 6) with p_dot = J nu, where nu = (v_x, v_y, v_z, w_x, w_y, w_z) is the camera's
    velocity in its own frame {C}. Rows are [u1, v1, u2, v2, ...], the order of (p* - p).flatten("F").
    With normalised coordinates x = (u - cx) / fx, y = (v - cy) / fy (Chaumette and Hutchinson 2006):
        L = [[-1/Z,  0,   x/Z,  x y,    -(1 + x^2),  y],
             [ 0,   -1/Z, y/Z,  1 + y^2, -x y,       -x]],      J = diag(fx, fy) L
    """
    uv = np.asarray(uv, float).reshape(2, -1)
    Z = np.broadcast_to(np.asarray(Z, float), (uv.shape[1],))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    J = np.zeros((2 * uv.shape[1], 6))
    for i, ((u, v), z) in enumerate(zip(uv.T, Z)):
        x, y = (u - cx) / fx, (v - cy) / fy
        L = np.array([[-1 / z, 0.0, x / z, x * y, -(1 + x * x), y],
                      [0.0, -1 / z, y / z, 1 + y * y, -x * y, -x]])
        J[2 * i:2 * i + 2] = np.diag([fx, fy]) @ L
    return J


def robot_to_camera(T_BC=aruco.T_BC):
    """6 x 2 matrix G with nu_C = G @ [v, omega] for a differential-drive base (Week 9 D4, generalised).

    The base drives at v along x_B and turns at omega about z_B (the wheel axis is the base origin).
    The camera origin p_BC is carried along: its velocity is v x_B + omega z_B x p_BC, and it turns
    with omega z_B. In {C}:   v_C = R_CB (v x_B + omega z_B x p_BC),   w_C = R_CB (omega z_B).
    Waffle Pi (p_BC = 0.076 m ahead): nu_C = (-0.076 omega, 0, v, 0, -omega, 0).
    Camera on the axle (p_BC = 0): nu_C = (0, 0, v, 0, -omega, 0), so J @ G = [J[:, 2], -J[:, 4]] (Week 9).
    """
    R_CB, p = T_BC.R.T, T_BC.t
    ex, ez = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    G = np.zeros((6, 2))
    G[:3, 0] = R_CB @ ex                    # forward speed
    G[:3, 1] = R_CB @ np.cross(ez, p)       # yaw rate slides the offset camera sideways
    G[3:, 1] = R_CB @ ez                    # yaw rate turns the camera
    return G


# ---------------------------------------------------------------- goal and measurements
def desired_pose(stop_range, marker_height, T_BC=aruco.T_BC):
    """T_CM*: the marker as the camera sees it from the docked pose (Week 9 B2 T_star).

    Docked = the marker centre straight ahead of the base origin at stop_range [m], at marker_height [m]
    above the floor, seen square on: x_M = -y_B, y_M = z_B, z_M = -x_B. Then T_CM* = T_BC^-1 T_BM*.
    For the Waffle Pi this is the marker 0.280 m in front of the camera, 1.7 cm above its axis, R = Rx(pi).
    """
    R_BM = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    return T_BC.inv() * SE3.Rt(R_BM, [stop_range, 0.0, marker_height])


def project(K, T_CM, side=aruco.MARKER_SIDE):
    """Pixel coordinates (2 x 4) of the marker corners for the marker pose T_CM (pinhole, no distortion)."""
    P_C = np.asarray(T_CM * aruco.marker_corners(side).T.astype(float))   # 3 x 4 in {C}
    return (K @ (P_C / P_C[2]))[:2]


def desired_features(K, stop_range, marker_height, side=aruco.MARKER_SIDE, T_BC=aruco.T_BC):
    """p* (2 x 4): where the four corners must appear in the image when the robot is docked (Week 9 B2)."""
    return project(K, desired_pose(stop_range, marker_height, T_BC), side)


def undistort(corners, K, dist):
    """Detected corners (4 x 2, as aruco.detect returns them) -> undistorted pixels, 2 x 4.

    J and p* use the pinhole model, so the lens distortion is removed first (calibrated dist).
    """
    pts = cv2.undistortPoints(np.asarray(corners, np.float64).reshape(-1, 1, 2), K, dist, P=K)
    return pts.reshape(-1, 2).T


def corner_depths(T_CM, side=aruco.MARKER_SIDE):
    """Depth Z of each corner in {C} from the PnP pose of this frame (Week 9 B4: depth updated online)."""
    return np.asarray(T_CM * aruco.marker_corners(side).T.astype(float))[2]


# ---------------------------------------------------------------- the control law (Week 9 B1b, D4)
def ibvs_command(uv, Z, p_star, K, G, lam):
    """One IBVS step for the base: (v, omega) = lam * J_mr^+ (p* - p).

    uv: current corners (2 x 4, undistorted pixels), Z: their depths, p_star: desired corners (2 x 4),
    G: robot_to_camera(T_BC), lam: gain [1/s]. Returns v [m/s], omega [rad/s], the feature error e (8,)
    and J_mr (8 x 2). Near the goal the error decays like e_dot = -lam e in the directions the base
    can produce; what is left (the part of e outside the columns of J_mr) the base cannot remove.
    """
    J_mr = visjac_p(uv, Z, K) @ G
    e = (np.asarray(p_star, float) - np.asarray(uv, float)).flatten(order="F")
    v, w = lam * np.linalg.pinv(J_mr) @ e
    return float(v), float(w), e, J_mr


# ---------------------------------------------------------------- checks (no simulator)
def _camera_pose(x, T_BC=aruco.T_BC):
    """T_WC for a base pose x = (x, y, theta) on the floor."""
    return SE3(x[0], x[1], 0.0) * SE3.Rz(x[2]) * T_BC


def _pixels(x, T_WM, K, side=aruco.MARKER_SIDE):
    """Marker corners in pixels and their depths, seen from base pose x (pure geometry)."""
    T_CM = _camera_pose(x).inv() * T_WM
    return project(K, T_CM, side), corner_depths(T_CM, side)


def _unicycle(x, v, w, dt):
    """Midpoint unicycle step (same model as src/estimation/odometry.move)."""
    m = x[2] + w * dt / 2
    return np.array([x[0] + v * dt * np.cos(m), x[1] + v * dt * np.sin(m), x[2] + w * dt])


def self_checks(K=None, stop_range=0.356, marker_height=0.12):
    """Week 9-style checks, printed PASS / FAIL; returns the number of failures."""
    if K is None:
        from src.perception import camera
        from src.robot.sim_robot import PROJECT_ROOT
        K, _, _ = camera.load_intrinsics(PROJECT_ROOT / "config" / "camera_sim.json")
    fails = []

    def check(name, ok, detail=""):
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"   ({detail})" if detail else ""))
        if not ok:
            fails.append(name)

    # 1. Week 9 A1: J against a finite difference of a small camera motion
    P_C = np.array([[0.05, -0.04, 0.06, -0.03], [-0.02, 0.03, 0.04, -0.05], [0.40, 0.45, 0.50, 0.42]])
    uv = (K @ (P_C / P_C[2]))[:2]
    nu = np.array([0.03, -0.02, 0.05, 0.1, -0.2, 0.15])
    dt = 1e-6
    P_new = np.asarray(SE3.Delta(nu * dt).inv() * P_C)          # the points seen from the moved camera
    fd = (((K @ (P_new / P_new[2]))[:2] - uv) / dt).flatten(order="F")
    J = visjac_p(uv, P_C[2], K)
    check("Week 9 A1: image Jacobian matches a finite difference", np.allclose(J @ nu, fd, rtol=1e-4, atol=1e-3),
          f"max diff {np.abs(J @ nu - fd).max():.2e} px/s")

    # 2. Added: the reduced Jacobian with the REAL camera mount, against moving the base itself
    T_WM = SE3.Rt(np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), [0.6, 0.02, marker_height])
    x0 = np.array([0.0, 0.0, 0.05])
    uv0, Z0 = _pixels(x0, T_WM, K)
    G = robot_to_camera()
    J_mr = visjac_p(uv0, Z0, K) @ G
    for v, w in [(0.08, 0.0), (0.0, 0.5), (0.05, -0.3)]:
        uv1, _ = _pixels(_unicycle(x0, v, w, dt), T_WM, K)
        fd = ((uv1 - uv0) / dt).flatten(order="F")
        check(f"added: J_mr = J G matches driving the base (v = {v}, omega = {w})",
              np.allclose(J_mr @ [v, w], fd, rtol=1e-4, atol=1e-3), f"max diff {np.abs(J_mr @ [v, w] - fd).max():.2e} px/s")

    # 3. With the camera on the axle, G reduces to Week 9 D4's J_mr = [J[:, 2], -J[:, 4]]
    T_axle = SE3.Rt(aruco.T_BC.R, [0.0, 0.0, aruco.T_BC.t[2]])
    J_full = visjac_p(uv0, Z0, K)
    check("Week 9 D4: camera on the axle gives J_mr = [J[:, 2], -J[:, 4]]",
          np.allclose(J_full @ robot_to_camera(T_axle), np.c_[J_full[:, 2], -J_full[:, 4]]))
    check("added: the Waffle Pi mount adds -0.076 J[:, 0] to the omega column",
          np.allclose(J_full @ G, np.c_[J_full[:, 2], -0.076 * J_full[:, 0] - J_full[:, 4]]))

    # 4. p* is the docked view: the marker centred, at the stop range, 0.28 m from the camera
    p_star = desired_features(K, stop_range, marker_height)
    T_s = desired_pose(stop_range, marker_height)
    size = np.mean(np.linalg.norm(p_star - np.roll(p_star, 1, axis=1), axis=0))
    check("Week 9 B2: p* = the marker 0.280 m ahead of the camera, square on, centred left-right",
          abs(T_s.t[2] - (stop_range - aruco.T_BC.t[0])) < 1e-9 and abs(p_star[0].mean() - K[0, 2]) < 1e-6,
          f"marker {size:.0f} px wide, centre ({p_star[0].mean():.1f}, {p_star[1].mean():.1f}) px")
    J_star = visjac_p(p_star, corner_depths(T_s), K) @ G
    check("J_mr at the goal has full column rank 2", np.linalg.matrix_rank(J_star) == 2,
          f"condition number {np.linalg.cond(J_star):.1f}")

    # 5. The closed loop on pure geometry (Week 9 D4 loop with our camera): from the approach line
    #    0.55 m out and 3 deg off, IBVS reaches p*; from 5 cm sideways it ends facing the marker but
    #    still sideways (Week 9 D3/D4), which is why docking.py aligns with the LQR first.
    goal = np.array([0.6 - stop_range, 0.02, 0.0])          # base pose with the marker straight ahead

    def loop(x, lam=0.8, dt=0.05, n=600):
        for _ in range(n):
            uv_k, Z_k = _pixels(x, T_WM, K)
            v, w, e, _ = ibvs_command(uv_k, Z_k, p_star, K, G, lam)
            x = _unicycle(x, v, w, dt)
        return x, np.sqrt(np.mean(e ** 2))

    x_end, e_rms = loop(np.array([0.6 - 0.55, 0.02, np.deg2rad(3.0)]))
    # (it drives a little sideways while the 3 deg heading error is being turned out, and IBVS cannot
    #  take that sideways part back: the same D3/D4 effect, small here)
    check("IBVS loop: on the line, 0.55 m out, 3 deg off -> reaches p*, within 1 cm of the goal",
          e_rms < 0.5 and np.hypot(*(x_end[:2] - goal[:2])) < 0.01,
          f"error {e_rms:.2f} px rms, {1000 * np.hypot(*(x_end[:2] - goal[:2])):.1f} mm from the goal, "
          f"heading {np.degrees(x_end[2] - goal[2]):+.2f} deg")
    x_end, e_rms = loop(np.array([0.6 - 0.55, 0.02 + 0.05, 0.0]))
    check("Week 9 D3/D4: from 5 cm sideways IBVS alone leaves a lateral offset (so align first)",
          abs(x_end[1] - goal[1]) > 0.02, f"{100 * (x_end[1] - goal[1]):+.1f} cm sideways, {e_rms:.1f} px rms left")
    print(f"\n{len(fails)} failed" + (": " + ", ".join(fails) if fails else " - all checks passed"))
    return len(fails)


if __name__ == "__main__":
    # Run from the project root:  python -m src.control.visual_servo
    sys.exit(self_checks())
