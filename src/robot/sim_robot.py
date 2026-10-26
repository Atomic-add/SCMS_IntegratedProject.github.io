"""Simulated TurtleBot3 Waffle Pi: the MuJoCo side of the hardware abstraction layer.

Interface shared with the real robot (src/robot/real_robot.py, ROS 2 topics over rosbridge):

    read_encoders() -> (t, phi_L, phi_R)   wheel angles [rad]       real robot: /joint_states
    read_scan()     -> (t, rho)            n_beams ranges [m]       real robot: /scan
    read_image()    -> (t, rgb)            H x W x 3 uint8, RGB     real robot: camera image topic
    send_cmd(v, w)                         body velocity command    real robot: /cmd_vel
    step(dt)                               advance the physics by dt (real robot: wait until dt has passed)

The camera intrinsics are deliberately not exposed: perception uses the calibrated K from config/,
exactly as it will with the real camera (experiments/calibrate_camera.py).

This class never returns the robot's pose. Ground truth is read only in evaluation/.
"""
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENE = PROJECT_ROOT / "models" / "robotis_tb3" / "scene_room.xml"

# Waffle Pi parameters and their sources
WHEEL_RADIUS = 0.033       # m, turtlebot3_waffle_pi.urdf (wheel collision cylinder)
WHEEL_SEPARATION = 0.288   # m, turtlebot3_waffle_pi.urdf (wheel joints at y = +/-0.144)
MAX_V = 0.26               # m/s, ROBOTIS e-Manual, Waffle Pi
MAX_W = 1.82               # rad/s, ROBOTIS e-Manual, Waffle Pi
ENCODER_CPR = 4096         # counts per wheel revolution, XM430-W210 datasheet


@dataclass
class LidarSpec:
    """Planar LiDAR model. Defaults follow the ROBOTIS LDS-02 (an LDS-01 is 0.12-3.5 m).

    sigma and dropout are placeholders until they are measured on the lab robot.
    """
    n_beams: int = 360         # 1 deg resolution, beam i at i deg CCW from the sensor x axis
    rate_hz: float = 5.0       # scans per second
    range_min: float = 0.16    # m
    range_max: float = 8.0     # m
    sigma: float = 0.01        # m, constant range noise (W4 Part 5: lidar noise does not grow with range)
    dropout: float = 0.01      # fraction of beams that return nothing
    no_return: float = 0.0     # value reported when a beam returns nothing (check /scan on the real robot)


@dataclass
class CameraSpec:
    """RGB camera model: the lab robot's camera (Raspberry Pi Camera v2 on the Waffle Pi).

    width, height, focal length and tilt live in the MJCF <camera> (measured on 23 Oct: f 692 px,
    49.7 x 38.2 deg, 6.3 deg down); pixel_noise is a placeholder until it is measured on the lab robot.
    """
    name: str = "rgb"          # camera name in the MJCF
    width: int = 640
    height: int = 480
    rate_hz: float = 30.0      # frames per second reaching the computer (lab robot, check_robot.py)
    pixel_noise: float = 2.0   # sd of additive grey-level noise per pixel channel


# The sim camera's calibration, written by experiments/calibrate_camera.py (the robot's software uses the
# file, never the MJCF). The real robot uses config/camera_real.json instead.
SIM_INTRINSICS = PROJECT_ROOT / "config" / "camera_sim.json"


class SimRobot:
    """TurtleBot3 Waffle Pi in MuJoCo, with wheel encoders, a 2D LiDAR and an RGB camera."""

    # The LiDAR ignores geom group 2 (visual-only decorations such as the marker sheet on the dock).
    LIDAR_GEOMGROUP = np.array([1, 1, 0, 1, 1, 1], dtype=np.uint8)

    def __init__(self, scene=DEFAULT_SCENE, lidar=None, camera=None, seed=0):
        self.m = mujoco.MjModel.from_xml_path(str(scene))
        self.d = mujoco.MjData(self.m)
        self.lidar = lidar if lidar is not None else LidarSpec()
        self.camera = camera if camera is not None else CameraSpec()
        self.rng = np.random.default_rng(seed)

        self._wheel_qadr = [self.m.jnt_qposadr[self.m.joint(n).id] for n in ("wheel_left", "wheel_right")]
        self._wheel_max = self.m.actuator_ctrlrange[0, 1]          # rad/s, from the MJCF actuator
        self._scan_site = self.m.site("base_scan").id
        self._robot_body = self.m.body("base").id                  # rays ignore the robot's own chassis

        # Unit beam directions in the sensor frame {S}, N x 3 (W4 Part 5.1 dirs_S with zero elevation)
        n = self.lidar.n_beams
        self.scan_angles = np.arange(n) * 2 * np.pi / n
        self._dirs_S = np.stack([np.cos(self.scan_angles), np.sin(self.scan_angles), np.zeros(n)], axis=1)
        self._geomid = np.zeros(n, dtype=np.int32)
        self._dist = np.zeros(n)
        self._scan_t, self._scan = -np.inf, None

        cam_id = self.m.camera(self.camera.name).id
        if tuple(self.m.cam_resolution[cam_id]) != (self.camera.width, self.camera.height):
            raise ValueError(f"CameraSpec {self.camera.width}x{self.camera.height} does not match the MJCF "
                             f"camera resolution {tuple(self.m.cam_resolution[cam_id])}")
        self._renderer = None
        self._frame_t, self._frame = -np.inf, None
        mujoco.mj_forward(self.m, self.d)

    def close(self):
        """Release the offscreen renderer (call when done, or use SimRobot as a context manager)."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------ simulation control
    def reset(self, x=0.0, y=0.0, yaw=0.0, settle=0.2):
        """Sim only: put the robot at (x, y, yaw) in the world and let it settle on the floor."""
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[0:3] = [x, y, 0.0]
        self.d.qpos[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        mujoco.mj_forward(self.m, self.d)
        self._scan_t = self._frame_t = -np.inf
        self.step(settle)

    def step(self, dt):
        """Sim only: advance the physics by dt seconds (MuJoCo timestep 2 ms)."""
        mujoco.mj_step(self.m, self.d, max(1, int(round(dt / self.m.opt.timestep))))

    def next_image(self):
        """Sim only: advance the physics to the camera's next frame and return it (t_frame, rgb).

        For scripts that move something and need a fresh frame (calibration, checks). A control
        loop does not need it: it calls step() and read_image() and looks at t_frame.
        """
        while not self._due(self._frame_t, self.camera.rate_hz):
            self.step(self.m.opt.timestep)
        return self.read_image()

    @property
    def t(self):
        return self.d.time

    def _due(self, t_last, rate_hz):
        """True when a sensor sampled at t_last has a new sample: the sim clock has crossed the next
        multiple of 1/rate_hz. Samples stay on that fixed grid, so the average rate is exact even
        when the period is not a whole number of 2 ms timesteps (15 Hz = 66.7 ms)."""
        return np.floor(self.t * rate_hz + 1e-6) > np.floor(t_last * rate_hz + 1e-6)

    # ------------------------------------------------------------------ actuation
    def send_cmd(self, v, w):
        """Body velocity command, like /cmd_vel: saturate to the Waffle Pi limits, then wheel speeds.

        If a wheel would exceed its speed limit, both wheels are scaled down together so the
        turning radius v/w is kept. Acceleration limiting is the controller's job.
        """
        v = float(np.clip(v, -MAX_V, MAX_V))
        w = float(np.clip(w, -MAX_W, MAX_W))
        wl = (v - w * WHEEL_SEPARATION / 2) / WHEEL_RADIUS
        wr = (v + w * WHEEL_SEPARATION / 2) / WHEEL_RADIUS
        s = max(1.0, abs(wl) / self._wheel_max, abs(wr) / self._wheel_max)
        self.d.ctrl[:] = [wl / s, wr / s]

    # ------------------------------------------------------------------ sensors
    def read_encoders(self):
        """Wheel angles [rad] quantised to the encoder resolution 2*pi/ENCODER_CPR (W2 Part A)."""
        res = 2 * np.pi / ENCODER_CPR
        phi = np.round(self.d.qpos[self._wheel_qadr] / res) * res
        return self.t, float(phi[0]), float(phi[1])

    def read_scan(self):
        """Latest scan (t_scan, rho), refreshed at rate_hz like a /scan topic.

        Call it as often as you like: a new scan appears every 1/rate_hz seconds, and t_scan tells
        you whether it is new. rho[i] is the range of beam i at scan_angles[i]; beams with no
        return carry lidar.no_return.
        """
        if self._scan is None or self._due(self._scan_t, self.lidar.rate_hz):
            self._scan = self._cast_scan()
            self._scan_t = self.t
        return self._scan_t, self._scan.copy()

    def read_image(self):
        """Latest camera frame (t_frame, rgb), refreshed at camera.rate_hz like an image topic.

        rgb is H x W x 3 uint8 in RGB order (convert with cv2.COLOR_RGB2BGR for OpenCV drawing).
        A new t_frame means a new frame.
        """
        if self._frame is None or self._due(self._frame_t, self.camera.rate_hz):
            self._frame = self._render()
            self._frame_t = self.t
        return self._frame_t, self._frame.copy()

    def _render(self):
        """Render the MJCF camera for the current state and add pixel noise."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.m, self.camera.height, self.camera.width)
        mujoco.mj_kinematics(self.m, self.d)                       # body and camera poses for the
        mujoco.mj_camlight(self.m, self.d)                         # current qpos (and mocap)
        self._renderer.update_scene(self.d, camera=self.camera.name)
        img = self._renderer.render().astype(np.float64)
        if self.camera.pixel_noise > 0:                            # sensor noise: Gaussian grey levels per pixel
            img += self.rng.normal(0.0, self.camera.pixel_noise, img.shape)
        return np.clip(np.round(img), 0, 255).astype(np.uint8)

    def _cast_scan(self):
        """One planar scan: cast n_beams rays from the base_scan site and apply the sensor model."""
        L = self.lidar
        mujoco.mj_kinematics(self.m, self.d)                       # site pose for the current qpos
        R_WS = self.d.site_xmat[self._scan_site].reshape(3, 3)     # sensor orientation in the world
        vec = (self._dirs_S @ R_WS.T).ravel()                      # beam directions in the world
        mujoco.mj_multiRay(self.m, self.d, self.d.site_xpos[self._scan_site].copy(), vec, self.LIDAR_GEOMGROUP, 1,
                           self._robot_body, self._geomid, self._dist, None, L.n_beams, L.range_max)
        hit = self._dist >= 0                                      # -1 means nothing within range_max
        # Sensor model: Gaussian range noise (sigma), random lost beams (dropout), then the range limits
        rho = np.where(hit, self._dist, np.inf) + self.rng.normal(0.0, L.sigma, L.n_beams)
        lost = ~hit | (self.rng.random(L.n_beams) < L.dropout) | (rho < L.range_min) | (rho > L.range_max)
        return np.where(lost, L.no_return, rho)
