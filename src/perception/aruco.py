"""ArUco marker detection, marker pose and the range-bearing measurement for the EKF.

Adapted from 41014 Week 3 Part E: marker_image (marker with a white quiet zone), the corner order
TL, TR, BR, BL at (+-s/2, +-s/2, 0), PnP giving C_T_M (the marker in the camera frame), and the
planar-pose ambiguity demonstration; plus the Week 3 wrap-up sigma_Z ~ Z^2 sigma_px / (f_px s).
Additions: both IPPE solutions are kept so an ambiguous pose can be flagged, and the measurement is
expressed in the robot base frame through the camera extrinsic T_BC.

Frames: {B} base_footprint (x forward, y left, z up), {C} camera optical frame (OpenCV: x right,
y down, z forward), {M} marker (x right and y up as the robot sees it, z out of the marker towards
the robot). T_AB is the pose of frame B in frame A.
"""
import cv2
import numpy as np
from spatialmath import SE3

# PLACEHOLDERS: must match the real docking station. W3 uses the same dictionary.
ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_ID = 0
MARKER_SIDE = 0.10          # m, side of the black square

# Pose of the camera optical frame {C} in the base {B}. Source: turtlebot3_waffle_pi.urdf:
# camera_joint (0.073, -0.011, 0.084) + camera_rgb_joint (0.003, 0.011, 0.009) on base_link, which is
# 0.010 above base_footprint -> (0.076, 0, 0.103); camera_rgb_optical_joint rpy (-pi/2, 0, -pi/2)
# (written -1.57 in the URDF) turns base axes into OpenCV axes: z_C = x_B, x_C = -y_B, y_C = -z_B.
T_BC = SE3(0.076, 0.0, 0.103) * SE3.RPY(-np.pi / 2, 0.0, -np.pi / 2)


def marker_image(marker_id=MARKER_ID, px=600, pad=60, dictionary=ARUCO_DICT):
    """An ArUco marker as a greyscale image with a white quiet zone (W3 Part E marker_image)."""
    tag = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(dictionary), marker_id, px)
    return cv2.copyMakeBorder(tag, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)


def marker_corners(side=MARKER_SIDE):
    """Marker corners in {M}, 4 x 3: top-left, top-right, bottom-right, bottom-left (W3 Part E)."""
    s = side / 2
    return np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float32)


def detect(rgb, marker_id=MARKER_ID, dictionary=ARUCO_DICT):
    """Corners (4 x 2, TL TR BR BL, pixels) of the marker with marker_id, or None.

    Sub-pixel corner refinement (cornerSubPix, as for the W3 checkerboard) is on. It still leaves the
    corners 0.1-0.4 px inside the true ones, because blur rounds an L-shaped corner, so the range reads
    slightly long: a few mm at 1 m and tens of mm beyond 2 m in the sim (results/aruco_noise.csv). A real
    lens blurs too, so expect the same sign on the robot. In the sim, OpenCV's other refinements were
    no better: none and contour gave larger inward errors, and apriltag gave uneven corner errors that
    tilt the IPPE pose and pull the range short instead.
    """
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dictionary), params)
    corners, ids, _ = det.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    if ids is None or marker_id not in ids.ravel():
        return None
    return corners[list(ids.ravel()).index(marker_id)].reshape(4, 2)


def marker_pose(corners, K, dist, side=MARKER_SIDE):
    """Both IPPE solutions for C_T_M, best first, as a list of (T_CM, reprojection rms in px).

    A nearly face-on marker seen from far away has two poses that explain its four corners almost
    equally well (W3 Part E). solvePnPGeneric with SOLVEPNP_IPPE_SQUARE returns both.
    """
    _, rvecs, tvecs, errs = cv2.solvePnPGeneric(marker_corners(side), corners.astype(np.float32), K, dist,
                                                flags=cv2.SOLVEPNP_IPPE_SQUARE)
    sols = sorted(zip(np.ravel(errs), rvecs, tvecs), key=lambda s: s[0])
    return [(SE3.Rt(cv2.Rodrigues(r)[0], t.ravel()), float(e)) for e, r, t in sols]


def measure(rgb, K, dist, T_BC=T_BC, side=MARKER_SIDE, marker_id=MARKER_ID, ambiguity_ratio=0.1):
    """The marker measurement for the EKF, or None when the marker is not seen.

    z = [range, bearing]: from the base origin to the marker centre, in the floor plane of {B}.
    When the pose is ambiguous both IPPE solutions put the marker centre in almost the same place
    (sim: within 0.4 mm and 0.04 deg), so z survives the ambiguity.
    yaw: direction of the marker normal (its z axis) in {B}; pi when the robot faces the marker square
    on. With the marker's world heading theta_M, the EKF model is yaw = wrap(theta_M - theta).
    It comes from the small perspective differences between the marker edges, so it is only trusted
    when the second solution explains the corners clearly worse: ambiguous if err1 > ambiguity_ratio * err2.
    With 0.1, trusted yaws were within 0.5 deg (95th percentile) and flagged ones off by up to 16 deg
    in the sweep of evaluation/check_camera.py. Seen square on, it was flagged at most distances beyond
    0.25 m; seen 30 deg off the normal, it was trusted at every distance tested (0.3-1.4 m).
    """
    corners = detect(rgb, marker_id)
    if corners is None:
        return None
    sols = marker_pose(corners, K, dist, side)
    T_CM, e1 = sols[0]
    e2 = sols[1][1] if len(sols) > 1 else np.inf
    T_BM = T_BC * T_CM
    p, n = T_BM.t, T_BM.R[:, 2]
    return dict(z=np.array([np.hypot(p[0], p[1]), np.arctan2(p[1], p[0])]),
                yaw=float(np.arctan2(n[1], n[0])),
                ambiguous=bool(e1 > ambiguity_ratio * e2),
                T_BM=T_BM, T_CM=T_CM, T_CM_alt=sols[1][0] if len(sols) > 1 else None,
                reproj_px=(e1, e2), corners=corners,
                size_px=float(np.mean(np.linalg.norm(corners - np.roll(corners, 1, axis=0), axis=1))))


def sigma_range_model(Z, f_px, side=MARKER_SIDE, sigma_px=0.3):
    """Range sd predicted from corner noise: sigma_Z ~ Z^2 sigma_px / (f_px s) (W3 wrap-up)."""
    return np.asarray(Z) ** 2 * sigma_px / (f_px * side)


def _label(img, text, org, scale=0.6):
    """White text with a dark outline, readable on the light walls and the dark dock alike."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def draw_markers(rgb, K, dist, marker_ids, T_BC=T_BC, side=MARKER_SIDE):
    """Copy of the frame with every listed marker that is seen: its outline, its ID and the
    z = [range, bearing] in {B} that measure() hands to the EKF. Display only (main.py camera window)."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    lines = []
    for mid in marker_ids:
        meas = measure(rgb, K, dist, T_BC=T_BC, side=side, marker_id=mid)
        if meas is None:
            continue
        corners = meas["corners"]
        cv2.polylines(bgr, [corners.astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 2)
        _label(bgr, f"ID {mid}", (int(corners[:, 0].min()), max(15, int(corners[:, 1].min()) - 8)), 0.5)
        r, b = meas["z"]
        lines.append(f"ID {mid}: range {r:.2f} m, bearing {np.degrees(b):+.1f} deg")
    for i, text in enumerate(lines or ["no marker in view"]):
        _label(bgr, text, (10, 25 + 24 * i))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def draw_overlay(rgb, meas, K, dist, side=MARKER_SIDE):
    """Copy of the frame with the detected marker, its axes and the measurement written on it."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if meas is None:
        cv2.putText(bgr, "marker not detected", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.polylines(bgr, [meas["corners"].astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 2)
        rvec, _ = cv2.Rodrigues(meas["T_CM"].R)
        cv2.drawFrameAxes(bgr, K, dist, rvec, meas["T_CM"].t, side * 0.6, 2)
        r, b = meas["z"]
        cv2.putText(bgr, f"range {r:.3f} m   bearing {np.degrees(b):+.1f} deg", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)
        if meas["ambiguous"]:
            cv2.putText(bgr, "yaw ambiguous: not used", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        else:
            cv2.putText(bgr, f"marker yaw in base {np.degrees(meas['yaw']):+.1f} deg", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)