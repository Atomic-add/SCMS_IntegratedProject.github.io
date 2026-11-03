"""Real TurtleBot3 Waffle Pi: the hardware side of the abstraction layer, through rosbridge.

Same interface as SimRobot, so the mission code runs unchanged:

    read_encoders() -> (t, phi_L, phi_R)   wheel angles [rad]       /joint_states, picked by joint name
    read_scan()     -> (t, rho)            360 ranges [m]           /scan, resampled to beam i at i deg
    read_image()    -> (t, rgb)            H x W x 3 uint8, RGB     /camera/image_raw/compressed (BGR -> RGB)
    send_cmd(v, w)                         body velocity command    /cmd_vel
    step(dt)                               waits until dt after the previous step (the sim advances physics)
    t, lidar.range_min, lidar.range_max

The link is UTS-RI/turtlebot_rosbridge (rosbridge_api 0.1.0, roslibpy over WebSocket port 9090).
Install it without its dependencies: it asks for opencv-python, which breaks opencv-contrib-python.
The pinned commit and roslibpy version are in requirements-real.txt:

    python -m pip install -r requirements-real.txt
    python -m pip install --no-deps "git+https://github.com/UTS-RI/turtlebot_rosbridge.git@8a6c481c506d1b817937d865af0d794ea0d45d51"

Clock: every t is the time the message reached this computer (time.monotonic, 0 at construction).
rosbridge_api drops the ROS header of images and scans, and the robot's clock is not synchronised
with the laptop, so the arrival time is the one clock all three sensors share. It is later than the
measurement by the wifi and JPEG latency; measure that on the robot before tightening max_frame_age.

Settings (topics, joint names, speed caps) are in config/robot_real.json. Like SimRobot, this class
never returns the robot's pose, and the camera intrinsics come from config/camera_real.json.

Sources: the hardware abstraction layer of the Module 3 slides (one robot interface, sim or real);
encoder angles as in 41014 Week 2 Part A WheelEncoder (the wheel angle is what the counter gives; r
and L are assumed, Week 2 Part B, so they are calibrated separately); the 360-beam scan layout of
Week 2 Part C; the speed limits of the ROBOTIS e-Manual. The rosbridge link is UTS-RI/turtlebot_rosbridge.
Offline checks with a fake link: python -m src.robot.real_robot_test

Adapted from 41014 Week 2 A (wheel angles from the encoders: /joint_states, never /odom).
Additions: the rosbridge adapter (UTS-RI/turtlebot_rosbridge), arrival-time stamps, scan resampling, speed caps,
and the stops on stale sensors, a lost link, a second robot or Ctrl+C.
"""
import json
import threading
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from src.perception.lidar import beam_angles
from src.robot.sim_robot import (MAX_V, MAX_W, PROJECT_ROOT, WHEEL_RADIUS, WHEEL_SEPARATION, CameraSpec,
                                 LidarSpec)

CONFIG = PROJECT_ROOT / "config" / "robot_real.json"
WHEEL_MAX = MAX_V / WHEEL_RADIUS   # rad/s, the wheel speed at the Waffle Pi's top speed
# rad between two /joint_states messages that no wheel can turn: at 20 Hz the top speed is 0.4 rad per
# message. A bigger step means the wheel angles came from another robot (ROS 2 robots on one wifi with
# the same ROS_DOMAIN_ID see each other's topics, seen in the lab on 23 Oct) or the robot restarted.
JOINT_JUMP = 2.0


def load_config(path=CONFIG):
    return json.loads(Path(path).read_text())


def make_bridge(host, cfg, clock):
    """rosbridge_api.RosbridgeRobot that also keeps the arrival time of each message.

    The package keeps only the latest decoded sample of each topic. The subclass stores
    (t_arrival, sample) as one pair under one lock, so a reader never gets a new sample with an
    old time. It hooks the package's private callbacks (_on_image, ...): check them if rosbridge_api
    is updated past 0.1.0.
    """
    try:
        from rosbridge_api import RosbridgeRobot       # imported here: the simulation does not need it
    except ImportError as e:
        raise ImportError("rosbridge_api is not installed; see the install lines at the top of "
                          "src/robot/real_robot.py (use --no-deps)") from e

    class StampedBridge(RosbridgeRobot):
        def __init__(self):
            topics = cfg["topics"]
            super().__init__(host, port=cfg["port"], cmd_vel_topic=topics["cmd_vel"],
                             image_topic=topics["image"], laser_scan_topic=topics["scan"],
                             joint_state_topic=topics["joint_states"],
                             connection_timeout=cfg["connect_timeout"])
            self._lock = threading.Lock()
            self._samples = {}
            self.counts = {"joint_states": 0, "scan": 0, "image": 0}
            self.joint_jumps = 0                        # wheel-angle steps no wheel can make (JOINT_JUMP)
            self._last_wheels = None

        def sample(self, key):
            """(t_arrival, sample) of the latest message on 'joint_states', 'scan' or 'image', or None."""
            with self._lock:
                return self._samples.get(key)

        def _store(self, key, t, value):
            if value is None:
                return
            with self._lock:
                self._samples[key] = (t, value)
                self.counts[key] += 1

        # roslibpy runs these one at a time on its own thread; the parent decodes the message
        def _on_joint_state(self, message):
            t = clock()
            super()._on_joint_state(message)
            try:
                names, pos = list(message["name"]), message["position"]
                wheels = np.array([pos[names.index(j)] for j in cfg["wheel_joints"]], dtype=float)
                if self._last_wheels is not None and np.max(np.abs(wheels - self._last_wheels)) > JOINT_JUMP:
                    self.joint_jumps += 1
                self._last_wheels = wheels
            except (KeyError, ValueError, TypeError):
                pass                                    # connect() reports missing wheel joints
            self._store("joint_states", t, message)

        def _on_laser_scan(self, message):
            t, before = clock(), self._latest_laser_scan
            super()._on_laser_scan(message)
            if self._latest_laser_scan is not before:   # unchanged when the message did not decode
                self._store("scan", t, self._latest_laser_scan)

        def _on_image(self, message):
            t, before = clock(), self._latest_image
            super()._on_image(message)
            if self._latest_image is not before:
                self._store("image", t, self._latest_image)

    return StampedBridge()


class RealRobot:
    """TurtleBot3 Waffle Pi over rosbridge, with the SimRobot interface.

    Use it as a context manager: entering connects and waits for the first message on every sensor
    topic, leaving (also on an exception or Ctrl+C) sends a zero velocity and disconnects.

        with RealRobot("192.168.1.213") as robot:
            t, phi_l, phi_r = robot.read_encoders()

    bridge: anything with setup(), cleanup(), send_command(v, w), is_connected and sample(key);
    default the rosbridge link above (tests pass a fake).
    strict: refuse to drive when the wheel angles jump (JOINT_JUMP), which means a second robot is
    publishing on the same topics or the robot restarted; check_robot.py passes False to report it.
    """

    SENSORS = ("joint_states", "scan", "image")

    def __init__(self, host=None, config=CONFIG, bridge=None, strict=True):
        self.cfg = load_config(config) if not isinstance(config, dict) else config
        self.host = host or self.cfg["host"]
        if bridge is None and not self.host:
            raise ValueError("no robot address: pass host (192.168.1.2XX) or set it in config/robot_real.json")
        self._t0 = time.monotonic()
        self._bridge = bridge if bridge is not None else make_bridge(self.host, self.cfg, lambda: self.t)
        self.v_max = min(self.cfg["v_max"], MAX_V)
        self.w_max = min(self.cfg["w_max"], MAX_W)
        self.wheel_joints = list(self.cfg["wheel_joints"])
        self.wheel_sign = np.array(self.cfg["wheel_sign"], dtype=float)
        self.lidar = LidarSpec()                       # n_beams and no_return as in the sim; the rest
        self.camera = CameraSpec(name="real")          # is replaced from the first messages in connect()
        self.scan_angles = beam_angles(self.lidar.n_beams)
        self._tick = None
        self.overruns = 0                              # steps that started late (the loop took > dt)
        self._connected = False
        self.strict = strict
        self._jumps_at_connect = 0

    # ------------------------------------------------------------------ connection
    def connect(self):
        """Connect, wait for every sensor topic and read the sensor specs from the first messages."""
        if self._connected:
            return self
        self._bridge.setup()
        try:
            deadline = time.monotonic() + self.cfg["connect_timeout"]
            while any(self._bridge.sample(k) is None for k in self.SENSORS):
                if time.monotonic() > deadline:
                    missing = [self.cfg["topics"][k] for k in self.SENSORS if self._bridge.sample(k) is None]
                    raise TimeoutError(f"no message on {', '.join(missing)} within {self.cfg['connect_timeout']} s; "
                                       "check the topic names (ros2 topic list) in config/robot_real.json")
                time.sleep(0.05)

            names = list(self._bridge.sample("joint_states")[1]["name"])
            if not all(j in names for j in self.wheel_joints):
                raise ValueError(f"/joint_states has joints {names}, not {self.wheel_joints}: "
                                 "fix wheel_joints in config/robot_real.json")
            scan = self._bridge.sample("scan")[1]
            self.lidar = replace(self.lidar, range_min=scan.range_min, range_max=scan.range_max,
                                 rate_hz=1.0 / scan.scan_time if scan.scan_time > 0 else float("nan"),
                                 sigma=float("nan"), dropout=float("nan"))   # sim noise model, unknown here
            h, w = self._bridge.sample("image")[1].shape[:2]
            self.camera = replace(self.camera, width=w, height=h, rate_hz=float("nan"), pixel_noise=float("nan"))
            time.sleep(1.0)                            # about 20 joint states: enough to see a second robot
            if self.strict and self.joint_jumps():
                raise RuntimeError(self._jump_message())
            self._jumps_at_connect = self.joint_jumps()
        except BaseException:
            self._bridge.cleanup()
            raise
        self._connected = True
        return self

    def close(self):
        """Stop the robot and disconnect (rosbridge_api sends the zero velocity in cleanup)."""
        self._connected = False
        self._bridge.cleanup()

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------ time
    @property
    def t(self):
        return time.monotonic() - self._t0

    def step(self, dt):
        """Wait until dt after the previous step, so the control loop runs at 1/dt.

        A loop that took longer than dt is not caught up (that would send commands in a burst): the
        next step counts from now and overruns goes up. Also the safety check of the loop: if the
        link is down or /joint_states or /scan has gone quiet for max_sensor_age, the robot is
        stopped and ConnectionError is raised, which ends the mission through the context manager.
        """
        now = self.t
        if self._tick is not None and self._tick + dt > now:
            time.sleep(self._tick + dt - now)
            self._tick += dt
        else:
            self.overruns += self._tick is not None
            self._tick = now
        self._check_link()

    def joint_jumps(self):
        """Wheel-angle steps between consecutive /joint_states that no wheel can make."""
        return getattr(self._bridge, "joint_jumps", 0)

    def _jump_message(self):
        return (f"/joint_states jumped by more than {JOINT_JUMP} rad {self.joint_jumps()} times: another robot "
                "is publishing on the same topics (same ROS_DOMAIN_ID on the lab wifi), or the robot restarted. "
                "Not driving: /cmd_vel would reach the other robot too. Ask for a separate ROS_DOMAIN_ID per robot "
                "(or ROS_LOCALHOST_ONLY=1 on the robot) and run experiments/check_robot.py again.")

    def _check_link(self):
        problem = None
        if not self._bridge.is_connected:
            problem = "rosbridge connection lost"
        elif self.strict and self.joint_jumps() > self._jumps_at_connect:
            problem = self._jump_message()
        else:
            limits = {"joint_states": self.cfg["max_sensor_age"],
                      "scan": self.cfg.get("max_scan_silence", self.cfg["max_sensor_age"])}
            for key, limit in limits.items():
                age = self.t - self._bridge.sample(key)[0]
                if age > limit:
                    problem = f"no {self.cfg['topics'][key]} for {age:.1f} s"
        if problem:
            try:
                self._bridge.send_command(0.0, 0.0)
            except Exception:
                pass                                   # the link is the problem; cleanup tries again
            raise ConnectionError(problem)

    # ------------------------------------------------------------------ actuation
    def send_cmd(self, v, w):
        """Body velocity to /cmd_vel: the SimRobot saturation, then the safety caps from the config.

        As in the sim, v and w are clipped to the Waffle Pi limits and both wheels are scaled down
        together if one would exceed its speed. The caps v_max, w_max then scale v and w by the same
        factor, so the turning radius v/w is kept.
        """
        v = float(np.clip(v, -MAX_V, MAX_V))
        w = float(np.clip(w, -MAX_W, MAX_W))
        wl = (v - w * WHEEL_SEPARATION / 2) / WHEEL_RADIUS
        wr = (v + w * WHEEL_SEPARATION / 2) / WHEEL_RADIUS
        s = max(1.0, abs(wl) / WHEEL_MAX, abs(wr) / WHEEL_MAX, abs(v) / self.v_max, abs(w) / self.w_max)
        self._bridge.send_command(v / s, w / s)

    # ------------------------------------------------------------------ sensors
    def read_encoders(self):
        """Latest wheel angles [rad] from /joint_states, left and right picked by joint name."""
        t, msg = self._bridge.sample("joint_states")
        names, pos = list(msg["name"]), msg["position"]
        phi = self.wheel_sign * np.array([pos[names.index(j)] for j in self.wheel_joints], dtype=float)
        return t, float(phi[0]), float(phi[1])

    def read_scan(self):
        """Latest scan (t_scan, rho) in the sim layout: rho[i] is the range at scan_angles[i] = i deg.

        Each output beam takes the /scan beam nearest in angle, so a scan with another start angle,
        direction or beam count still lines up. Beams the scan does not cover, and returns that are
        0, inf, NaN or outside [range_min, range_max], carry lidar.no_return.
        """
        t, scan = self._bridge.sample("scan")
        n, inc = scan.ranges.size, scan.angle_increment
        k = np.rint(np.mod((self.scan_angles - scan.angle_min) * np.sign(inc), 2 * np.pi) / abs(inc)).astype(int)
        if n * abs(inc) >= 2 * np.pi - abs(inc) / 2:   # full circle: the last beam is next to the first
            k %= n
        covered = k < n
        rho = np.full(self.lidar.n_beams, self.lidar.no_return)
        rho[covered] = scan.ranges[k[covered]]
        with np.errstate(invalid="ignore"):
            bad = ~np.isfinite(rho) | (rho < scan.range_min) | (rho > scan.range_max)
        rho[bad] = self.lidar.no_return
        return t, rho

    def read_image(self):
        """Latest camera frame (t_frame, rgb), H x W x 3 uint8 in RGB order like SimRobot.read_image()."""
        t, bgr = self._bridge.sample("image")
        return t, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------------------ diagnostics (check_robot.py)
    def latest(self, key):
        """(t_arrival, sample) as rosbridge_api decoded it: the joint_states dict, its LaserScan, or the BGR image."""
        return self._bridge.sample(key)

    def message_counts(self):
        """Messages received per sensor topic since connecting."""
        return dict(getattr(self._bridge, "counts", {}))
