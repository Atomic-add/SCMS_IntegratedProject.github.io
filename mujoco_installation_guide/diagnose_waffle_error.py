"""Diagnose why the ROBOTIS Waffle Pi model does not move like the real robot.

Run from the project root:  python mujoco_installation_guide/diagnose_waffle.py

Prints (1) which geoms touch the floor and with which friction, and
(2) how the robot really moves when commanded, for the original model and with each fix applied.
A correct differential-drive model should give r_eff ~ 0.033 m and L_eff ~ 0.288 m
(turtlebot3_waffle_pi.urdf: wheel radius 0.033, wheels at y = +/-0.144).
"""
from pathlib import Path

import mujoco
import numpy as np

SCENE = Path(__file__).resolve().parents[1] / "models" / "robotis_tb3" / "scene_turtlebot3_waffle_pi.xml"
R_NOM, L_NOM = 0.033, 0.288


def load(caster_priority=False, no_base_mesh=False, kv=None):
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    base = m.body("base").id
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != base:
            continue
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE and caster_priority:
            m.geom_priority[g] = 1                      # casters: use THEIR friction/condim
        if (no_base_mesh and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_contype[g]
                and m.mesh(m.geom_dataid[g]).name == "waffle_pi_base"):
            m.geom_contype[g] = m.geom_conaffinity[g] = 0  # chassis mesh no longer touches the floor
    if kv is not None:
        m.actuator_gainprm[:, 0] = kv
        m.actuator_biasprm[:, 2] = -kv
        m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    return m


def floor_contacts(m):
    d = mujoco.MjData(m)
    mujoco.mj_step(m, d, 500)
    out = []
    for c in d.contact[:d.ncon]:
        g = c.geom2 if m.geom(c.geom1).name == "floor" else c.geom1
        kind = mujoco.mjtGeom(m.geom_type[g]).name.replace("mjGEOM_", "").lower()
        label = m.mesh(m.geom_dataid[g]).name if kind == "mesh" else kind
        out.append(f"{m.body(m.geom_bodyid[g]).name}/{label} at x={c.pos[0]:+.3f} y={c.pos[1]:+.3f}  mu={c.friction[0]:.4g} condim={c.dim}")
    return sorted(set(out))


def drive(m, ctrl, T=4.0):
    """Hold wheel commands; return mean forward speed, yaw rate and wheel speeds over the last 1.5 s."""
    d = mujoco.MjData(m)
    d.ctrl[:] = ctrl
    dl, dr = m.jnt_dofadr[m.joint("wheel_left").id], m.jnt_dofadr[m.joint("wheel_right").id]
    log = []
    for i in range(int(T / m.opt.timestep)):
        mujoco.mj_step(m, d)
        if i * m.opt.timestep > T - 1.5:
            q = d.qpos[3:7]
            yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
            v = d.qvel[0] * np.cos(yaw) + d.qvel[1] * np.sin(yaw)
            log.append([v, d.qvel[5], d.qvel[dl], d.qvel[dr]])
    return np.mean(log, axis=0)


def report(name, m):
    v, _, wl, wr = drive(m, [5.0, 5.0])
    _, w, wl2, wr2 = drive(m, [-3.0, 3.0])
    r_eff = v / ((wl + wr) / 2)
    L_eff = R_NOM * (wr2 - wl2) / w
    print(f"{name:<38} wheels {wl:4.2f} rad/s for cmd 5 | v {v:.3f} m/s (ideal {5 * R_NOM:.3f}) | "
          f"r_eff {r_eff:.4f} | yaw {w:.3f} rad/s (ideal {R_NOM * 6 / L_NOM:.3f}) | L_eff {L_eff:.3f}")


print("Floor contacts, original model:")
for line in floor_contacts(load()):
    print("   ", line)
print("Floor contacts, chassis mesh off + caster priority:")
for line in floor_contacts(load(caster_priority=True, no_base_mesh=True)):
    print("   ", line)
print()
report("original", load())
report("+ kv 5 only", load(kv=5))
report("+ kv 5, caster priority", load(caster_priority=True, kv=5))
report("+ kv 5, chassis mesh off", load(no_base_mesh=True, kv=5))
report("+ kv 5, both contact fixes (= fixed)", load(caster_priority=True, no_base_mesh=True, kv=5))