"""Quick pass/fail checks for src/robot/real_robot.py without a robot (a fake rosbridge link).

Run from the project root:   python -m src.robot.real_robot_test      (about 5 s)

RealRobot must hand the mission the same data layout as SimRobot whatever the robot sends, and it must
stop the robot whenever something is wrong. Each check feeds made-up messages through a fake link and
compares with what SimRobot would return: beam i at i deg, no return = 0, wheels picked by name, RGB
images; speed caps that keep the turning radius; a zero command and an exception when the link or a
sensor goes quiet or a second robot shows up. If rosbridge_api is installed, the same messages also go
through its real decoding callbacks (the private hooks real_robot.py relies on). The exit code is the
number of failures (0 = all passed), as in src/control/control_test.py.
"""
import base64
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # project root, so "src" imports work

from src.robot import real_robot  # noqa: E402
from src.robot.real_robot import JOINT_JUMP, RealRobot, load_config  # noqa: E402
from src.robot.sim_robot import MAX_V, MAX_W, WHEEL_RADIUS, WHEEL_SEPARATION  # noqa: E402

FAILS = []
CFG = {**load_config(), "connect_timeout": 2.0, "max_sensor_age": 0.3, "max_scan_silence": 0.3}


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"   ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def scan_msg(angle_min, inc, n, r_of_deg, range_min=0.12, range_max=3.5):
    """A decoded /scan with beam k at angle_min + k inc and range r_of_deg(angle in deg, 0-360)."""
    ang = angle_min + inc * np.arange(n)
    return SimpleNamespace(ranges=np.array([r_of_deg(np.rad2deg(a) % 360) for a in ang], np.float32),
                           angle_min=angle_min, angle_increment=inc, range_min=range_min, range_max=range_max,
                           scan_time=0.2)


def joint_msg(names, pos):
    return {"name": list(names), "position": list(pos)}


class FakeLink:
    """Stands in for the rosbridge link: samples are set by the test, commands are recorded."""

    def __init__(self, scan, joints=None, image=None):
        self.is_connected, self.joint_jumps, self.commands, self.cleaned = False, 0, [], 0
        self.counts = {"joint_states": 0, "scan": 0, "image": 0}
        self._s = {"joint_states": (0.0, joints or joint_msg(["wheel_left_joint", "wheel_right_joint"], [0, 0])),
                   "scan": (0.0, scan),
                   "image": (0.0, image if image is not None else np.zeros((480, 640, 3), np.uint8))}

    def setup(self):
        self.is_connected = True

    def cleanup(self):
        if self.is_connected:
            self.commands.append((0.0, 0.0))                  # rosbridge_api sends a stop in cleanup()
        self.is_connected = False
        self.cleaned += 1

    def send_command(self, v, w):
        self.commands.append((v, w))

    def sample(self, key):
        return self._s.get(key)

    def put(self, key, t, value):
        self._s[key] = (t, value)


def connected(link, cfg=CFG, **kw):
    robot = RealRobot("fake", config=cfg, bridge=link, **kw)
    robot.connect()
    return robot


def r_of(deg):
    return 1.0 + 0.001 * deg                                  # a unique range per angle: 1.000 ... 1.359 m


# ---------------------------------------------------------------- scan layout (as SimRobot: beam i at i deg)
lds01 = scan_msg(0.0, np.deg2rad(1.0), 360, r_of)
lds01.ranges[[10, 200]] = 0.0                                 # LDS-01 codes a missing return as 0
robot = connected(FakeLink(lds01))
t, rho = robot.read_scan()
want = r_of(np.arange(360.0))
want[[10, 200]] = 0.0
check("LDS-01 layout (0 deg start, CCW, 360 beams) passes through unchanged", np.allclose(rho, want, atol=1e-6))
check("connect() takes range limits and rate from the first scan",
      np.isclose(robot.lidar.range_min, 0.12) and np.isclose(robot.lidar.range_max, 3.5)
      and np.isclose(robot.lidar.rate_hz, 5.0))
robot.close()

cases = {
    "clockwise scan starting at 180 deg": scan_msg(np.pi, -np.deg2rad(1.0), 360, r_of),
    "720 beams at 0.5 deg starting at -180 deg": scan_msg(-np.pi, np.deg2rad(0.5), 720, r_of),
}
for name, msg in cases.items():
    robot = connected(FakeLink(msg))
    _, rho = robot.read_scan()
    err = np.max(np.abs(rho - r_of(np.arange(360.0))))
    check(f"{name} is resampled to beam i at i deg", err < 1e-5, f"max error {1000 * err:.3f} mm")
    robot.close()

half = scan_msg(-np.pi / 2, np.deg2rad(1.0), 181, r_of)       # front half only (-90 ... +90 deg)
robot = connected(FakeLink(half))
_, rho = robot.read_scan()
deg = np.arange(360)
front = (deg <= 90) | (deg >= 270)
check("beams a partial scan does not cover read 'no return' (0)",
      np.all(rho[~front] == 0.0) and np.allclose(rho[front], r_of(deg[front].astype(float)), atol=1e-6))
robot.close()

bad = scan_msg(0.0, np.deg2rad(1.0), 360, lambda d: 1.0)
bad.ranges[[0, 1, 2, 3]] = [np.inf, np.nan, 0.05, 9.0]       # inf, NaN, below range_min, above range_max
robot = connected(FakeLink(bad))
_, rho = robot.read_scan()
check("inf, NaN and out-of-range returns become 'no return' (0)", np.all(rho[:4] == 0.0) and np.all(rho[4:] == 1.0))
robot.close()

# ---------------------------------------------------------------- encoders and image
joints = joint_msg(["wheel_right_joint", "caster", "wheel_left_joint"], [5.0, 9.9, -2.0])
cfg_sign = {**CFG, "wheel_sign": [1, -1]}
robot = connected(FakeLink(lds01, joints=joints), cfg=cfg_sign)
_, phi_l, phi_r = robot.read_encoders()
check("wheels are picked by name (any order in /joint_states) and wheel_sign is applied",
      phi_l == -2.0 and phi_r == -5.0, f"phi_L {phi_l}, phi_R {phi_r}")
robot.close()

bgr = np.zeros((480, 640, 3), np.uint8)
bgr[..., 2] = 255                                             # pure red in OpenCV's BGR order
robot = connected(FakeLink(lds01, image=bgr))
_, rgb = robot.read_image()
check("images come out RGB like SimRobot.read_image()", tuple(rgb[0, 0]) == (255, 0, 0)
      and (robot.camera.width, robot.camera.height) == (640, 480))

# ---------------------------------------------------------------- commands
link = robot._bridge
robot.v_max, robot.w_max = 0.10, 1.0
robot.send_cmd(0.20, 0.50)
v, w = link.commands[-1]
check("a command over the caps is scaled down with v/w kept", np.isclose(v, 0.10) and np.isclose(v / w, 0.4),
      f"sent ({v:.3f}, {w:.3f}) for (0.20, 0.50)")
robot.send_cmd(0.0, 5.0)
check("turn rate is clipped to the Waffle Pi and the cap", np.isclose(link.commands[-1][1], min(MAX_W, 1.0)))
robot.v_max, robot.w_max = MAX_V, MAX_W
robot.send_cmd(MAX_V, MAX_W)
v, w = link.commands[-1]
wheel = max(abs(v - w * WHEEL_SEPARATION / 2), abs(v + w * WHEEL_SEPARATION / 2)) / WHEEL_RADIUS
check("no wheel is asked for more than the top wheel speed", wheel <= real_robot.WHEEL_MAX + 1e-9,
      f"{wheel:.2f} rad/s <= {real_robot.WHEEL_MAX:.2f}")

# ---------------------------------------------------------------- loop timing and safety
robot._tick = None
for key in ("joint_states", "scan"):
    link.put(key, robot.t, link.sample(key)[1])
t0 = time.monotonic()
robot.step(0.05)
robot.step(0.05)
robot.step(0.05)
elapsed = time.monotonic() - t0                               # Windows + Python 3.10 sleeps in ~16 ms ticks
check("step(dt) keeps the loop at 1/dt", 0.095 <= elapsed <= 0.13, f"{elapsed:.3f} s for 2 periods of 0.05 s")

link.put("scan", robot.t - 1.0, link.sample("scan")[1])       # /scan has gone quiet for 1 s
n_cmd = len(link.commands)
try:
    robot.step(0.05)
    check("a quiet /scan stops the robot and raises", False)
except ConnectionError as e:
    check("a quiet /scan stops the robot and raises", link.commands[n_cmd:] == [(0.0, 0.0)], str(e))
link.put("scan", robot.t, link.sample("scan")[1])
link.is_connected = False
try:
    robot.step(0.05)
    check("a lost rosbridge link raises", False)
except ConnectionError as e:
    check("a lost rosbridge link raises", "lost" in str(e), str(e))
link.is_connected = True
robot.close()
check("close() leaves the robot with a zero command", link.commands[-1] == (0.0, 0.0) and link.cleaned == 1)

link = FakeLink(lds01)
link.joint_jumps = 3                                          # a second robot on the same ROS_DOMAIN_ID
try:
    connected(link)
    check("strict: wheel-angle jumps at connect refuse to drive", False)
except RuntimeError:
    check("strict: wheel-angle jumps at connect refuse to drive", link.cleaned == 1 and not link.is_connected)
check("check_robot mode (strict=False) still connects to report it", connected(FakeLink(lds01), strict=False)._connected)

try:
    with RealRobot("fake", config=CFG, bridge=FakeLink(lds01)) as robot:
        raise KeyboardInterrupt                               # Ctrl+C in the middle of a mission
except KeyboardInterrupt:
    pass
check("Ctrl+C inside the with block still stops and disconnects", robot._bridge.commands[-1] == (0.0, 0.0)
      and robot._bridge.cleaned == 1)

# ---------------------------------------------------------------- the real decoding callbacks (optional)
try:
    import rosbridge_api  # noqa: F401
except ImportError:
    print("SKIP  rosbridge_api callbacks (not installed; see the install lines in real_robot.py)")
else:
    bridge = real_robot.make_bridge("fake", CFG, time.monotonic)   # no connection is made before setup()
    bridge._on_joint_state(joint_msg(["wheel_left_joint", "wheel_right_joint"], [0.0, 0.1]))
    bridge._on_joint_state(joint_msg(["wheel_left_joint", "wheel_right_joint"], [0.3, 0.4]))
    bridge._on_laser_scan(dict(ranges=[1.0] * 360, angle_min=0.0, angle_max=np.deg2rad(359),
                               angle_increment=np.deg2rad(1.0), range_min=0.12, range_max=3.5, scan_time=0.2))
    ok, jpg = cv2.imencode(".jpg", bgr)
    bridge._on_image({"data": base64.b64encode(jpg.tobytes()).decode()})
    got = {k: bridge.sample(k) for k in ("joint_states", "scan", "image")}
    check("rosbridge_api callbacks are hooked: every sample is stored with its arrival time",
          all(v is not None for v in got.values()) and got["scan"][1].ranges.size == 360
          and got["image"][1].shape == (480, 640, 3) and bridge.counts["joint_states"] == 2)
    bridge._on_joint_state(joint_msg(["wheel_left_joint", "wheel_right_joint"], [0.3 + JOINT_JUMP + 0.5, 0.4]))
    check("a wheel-angle step no wheel can make is counted", bridge.joint_jumps == 1)

print(f"\n{len(FAILS)} failed" + (": " + ", ".join(FAILS) if FAILS else " - all checks passed"))
sys.exit(len(FAILS))
