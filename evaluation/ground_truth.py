"""Ground truth from the simulator, for evaluation and plotting only.

This is the only package allowed to read the true robot pose. Nothing in src/estimation or
src/control may import it.
"""
import mujoco
import numpy as np
from spatialmath import SE2, SE3

R_MJCAM_TO_CV = np.diag([1.0, -1.0, -1.0])   # MuJoCo camera (-z forward, +y up) -> OpenCV (+z forward, +y down)


def true_pose_se3(robot):
    """True 3D pose T_WB of base_footprint."""
    mujoco.mj_kinematics(robot.m, robot.d)
    bid = robot.m.body("base").id
    return SE3.Rt(robot.d.xmat[bid].reshape(3, 3), robot.d.xpos[bid], check=False)


def true_camera_pose(robot, cam="rgb"):
    """True pose T_WC of the camera's OpenCV optical frame, read from the MuJoCo camera."""
    mujoco.mj_kinematics(robot.m, robot.d)
    mujoco.mj_camlight(robot.m, robot.d)
    cid = robot.m.camera(cam).id
    return SE3.Rt(robot.d.cam_xmat[cid].reshape(3, 3) @ R_MJCAM_TO_CV, robot.d.cam_xpos[cid], check=False)


def true_marker_pose(robot, geom="aruco_marker"):
    """True pose T_WM of the marker: the frame of the textured plane (x right, y up, z out)."""
    mujoco.mj_kinematics(robot.m, robot.d)
    gid = robot.m.geom(geom).id
    return SE3.Rt(robot.d.geom_xmat[gid].reshape(3, 3), robot.d.geom_xpos[gid], check=False)


def true_intrinsics(robot, cam="rgb"):
    """The sim camera's exact K, from the MJCF fovy and resolution (pinhole, no distortion).

    OpenCV puts pixel centres at integer coordinates, so the principal point is ((W-1)/2, (H-1)/2).
    """
    cid = robot.m.camera(cam).id
    w, h = robot.m.cam_resolution[cid]
    f = (h / 2) / np.tan(np.deg2rad(robot.m.cam_fovy[cid]) / 2)
    return np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1.0]])


def true_pose(robot):
    """True pose [x, y, theta] of base_footprint in the world."""
    q = robot.d.qpos[3:7]
    yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    return np.array([robot.d.qpos[0], robot.d.qpos[1], yaw])


def true_sensor_pose(robot, site="base_scan"):
    """True planar pose T_WS of the LiDAR, read straight from the MuJoCo site (not via T_BS)."""
    mujoco.mj_kinematics(robot.m, robot.d)
    sid = robot.m.site(site).id
    p, R = robot.d.site_xpos[sid], robot.d.site_xmat[sid].reshape(3, 3)
    return SE2(p[0], p[1], np.arctan2(R[1, 0], R[0, 0]))


def room_shapes(robot, z_scan=0.132):
    """Static geometry cut by the scan plane: axis-aligned boxes and vertical cylinders.

    Planes (floor, marker faces) and the robot's own geoms are skipped.
    """
    m, d = robot.m, robot.d
    mujoco.mj_kinematics(m, d)
    robot_root = m.body("base").id
    shapes = []
    for g in range(m.ngeom):
        if m.body_rootid[m.geom_bodyid[g]] == robot_root:
            continue
        typ, c, size = m.geom_type[g], d.geom_xpos[g], m.geom_size[g]
        if typ == mujoco.mjtGeom.mjGEOM_BOX:
            assert np.allclose(d.geom_xmat[g].reshape(3, 3), np.eye(3)), "boxes must be axis aligned"
            if abs(z_scan - c[2]) <= size[2]:
                shapes.append(dict(kind="box", name=m.geom(g).name, lo=c[:2] - size[:2], hi=c[:2] + size[:2]))
        elif typ == mujoco.mjtGeom.mjGEOM_CYLINDER:
            if abs(z_scan - c[2]) <= size[1]:
                shapes.append(dict(kind="circle", name=m.geom(g).name, c=c[:2].copy(), r=size[0]))
    return shapes


def _box(o, d, lo, hi):
    """Slab-method ray/box distance (W4 Part 2 _box, planar)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (lo - o) / d, (hi - o) / d
    tmin = np.nanmax(np.minimum(t1, t2), axis=1)
    tmax = np.nanmin(np.maximum(t1, t2), axis=1)
    hit = (tmax > 0) & (tmin <= tmax)
    return np.where(hit, np.where(tmin > 0, tmin, tmax), np.inf)


def _circle(o, d, c, r):
    """Ray/circle distance (W4 Part 2 _sphere, planar)."""
    oc = o - c
    a, b = np.sum(d * d, axis=1), 2 * (d @ oc)
    disc = b ** 2 - 4 * a * (oc @ oc - r ** 2)
    sq = np.sqrt(np.where(disc >= 0, disc, 0))
    t = np.where((-b - sq) > 0, -b - sq, -b + sq) / (2 * a)
    return np.where((disc >= 0) & (t > 0), t, np.inf)


def raycast_2d(T_WS, alpha, shapes):
    """Exact range of every beam from sensor pose T_WS against the shapes (W4 Part 2 raycast)."""
    o = T_WS.t
    ang = T_WS.theta() + alpha
    d = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    best = np.full(len(alpha), np.inf)
    for s in shapes:
        t = _box(o, d, s["lo"], s["hi"]) if s["kind"] == "box" else _circle(o, d, s["c"], s["r"])
        best = np.minimum(best, t)
    return best