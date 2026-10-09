"""2D LiDAR processing: invalid-beam filtering, the frame chain, obstacles for avoidance.

Adapted from 41014 Week 2 Part C (h_lidar and invert: the frame chain p^W = T^WB T^BS p^S with SE2)
and Week 4 Part 5.1 (unit beam directions, polar to Cartesian P_S = dirs_S[ok] * rho[ok, None]).
Everything here takes plain arrays, so it runs unchanged on /scan from the real robot:
pass msg.ranges, msg.range_min and msg.range_max.

Frames: {W} world, {B} base_footprint (x forward, y left), {S} base_scan (the LiDAR).
T_AB is the pose of frame B in frame A.
"""
import numpy as np
from spatialmath import SE2

# Pose of the LiDAR {S} in the base {B}. Source: turtlebot3_waffle_pi.urdf, scan_joint (-0.064, 0, 0.122)
# on base_link, which is 0.010 above base_footprint. Only the planar part enters a 2D chain: the scan
# plane is horizontal, so its 0.132 m height changes no x-y coordinate.
T_BS = SE2(-0.064, 0.0, 0.0)

# Waffle Pi footprint in {B}, for the avoidance helpers. Source: turtlebot3_waffle_pi.urdf.
ROBOT_HALF_WIDTH = 0.153   # m, wheel outer face: joint at 0.144 plus half the 0.018 tyre width
ROBOT_FRONT = 0.086        # m, front face of the camera housing (camera_link collision box)


def beam_angles(n_beams=360):
    """Bearing of each beam in {S} [rad], counter-clockwise from the sensor x axis."""
    return np.arange(n_beams) * 2 * np.pi / n_beams


def valid_beams(rho, range_min, range_max):
    """True where a beam carries a usable return.

    Drivers mark 'no return' as 0, inf or NaN, and returns outside [range_min, range_max] are not
    trustworthy, so all of these are rejected.
    """
    rho = np.asarray(rho, dtype=float)
    with np.errstate(invalid="ignore"):
        return np.isfinite(rho) & (rho >= range_min) & (rho <= range_max)


def scan_to_points(rho, alpha, ok):
    """Valid returns as points in {S}, 2 x N (W4 Part 5.1, planar case)."""
    return np.vstack([rho[ok] * np.cos(alpha[ok]), rho[ok] * np.sin(alpha[ok])])


def to_base(P_S, T_BS=T_BS):
    """p^B = T^BS p^S for a 2 x N array of points."""
    return T_BS * P_S


def to_world(P_S, x, T_BS=T_BS):
    """p^W = T^WB T^BS p^S for robot pose x = [x, y, theta] (W2 Part C invert, for many points).

    Use the ESTIMATED pose at run time. Leaving T_BS out shifts every point by the 6.4 cm mounting
    offset; W2 C.2 measures that kind of damage.
    """
    return (SE2(x[0], x[1], x[2]) * T_BS) * P_S


def h_lidar(x, p_W, T_BS=T_BS):
    """Range and bearing in {S} to a known world point p_W, noise free (W2 Part C h_lidar).

    The measurement model for using a mapped feature as an EKF update later.
    """
    T_WS = SE2(x[0], x[1], x[2]) * T_BS
    p_S = (T_WS.inv() * np.asarray(p_W, dtype=float).reshape(2, 1)).flatten()
    return np.array([np.hypot(*p_S), np.arctan2(p_S[1], p_S[0])])


def nearest_obstacle(P_B):
    """Distance and bearing in {B} of the closest return; (inf, nan) when there is none."""
    if P_B.shape[1] == 0:
        return np.inf, np.nan
    d = np.hypot(P_B[0], P_B[1])
    i = int(np.argmin(d))
    return float(d[i]), float(np.arctan2(P_B[1, i], P_B[0, i]))


def free_distance_ahead(P_B, margin=0.05, half_width=ROBOT_HALF_WIDTH, front=ROBOT_FRONT):
    """How far the robot can drive straight ahead before a return enters its swept corridor [m].

    The corridor is the robot's width plus a safety margin on each side, starting at its front face.
    Returns inf when the corridor is clear.
    """
    in_corridor = (np.abs(P_B[1]) <= half_width + margin) & (P_B[0] > front)
    return float(np.min(P_B[0, in_corridor]) - front) if np.any(in_corridor) else np.inf