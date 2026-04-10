# german_rocker_switch_demo.py
#
# Run with:
#   ./python.sh german_rocker_switch_demo.py
#
# What it creates:
#   /World/Switch
#       base      (rigid body fixed to world)
#       rocker    (rigid body on revolute joint around X axis)
#       world_fix (fixed joint world -> base)
#       hinge     (revolute joint base -> rocker)
#   /World/Lamp/light
#   /World/Finger (kinematic cube that presses top/bottom periodically)
#
# Why this is RL-friendly later:
#   - Single revolute DOF
#   - Clear state: joint position, joint velocity, light_on
#   - Proper fixed-base articulation root
#
# Notes:
#   - Units are SI-ish with stage units in meters.
#   - The bistable "German rocker" feel is approximated by snapping the drive target
#     to one of two stable angles after the joint crosses a threshold.
#   - The demo finger is only for visualization. Later, replace it with your robot.

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import math
import omni
import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, PhysxSchema

# Isaac Sim world helper
from omni.isaac.core import World


# -----------------------------------------------------------------------------
# Small USD helper utilities
# -----------------------------------------------------------------------------

def set_translate(xformable, xyz):
    op = None
    for candidate in xformable.GetOrderedXformOps():
        if candidate.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op = candidate
            break
    if op is None:
        op = xformable.AddTranslateOp()
    op.Set(Gf.Vec3f(*xyz))
    return op


def set_orient_quat(xformable, quat_wxyz):
    """quat_wxyz = (w, x, y, z)"""
    w, x, y, z = quat_wxyz
    q = Gf.Quatf(w, Gf.Vec3f(x, y, z))
    op = None
    for candidate in xformable.GetOrderedXformOps():
        if candidate.GetOpType() == UsdGeom.XformOp.TypeOrient:
            op = candidate
            break
    if op is None:
        op = xformable.AddOrientOp()
    op.Set(q)
    return op


def create_box_body(
    stage,
    body_path: str,
    size_xyz,
    position_xyz,
    color_rgb=(0.7, 0.7, 0.7),
    mass=0.2,
    kinematic=False,
):
    """Create an Xform rigid body with a Cube child used as collider/render."""
    body_xf = UsdGeom.Xform.Define(stage, body_path)
    set_translate(body_xf, position_xyz)

    cube_path = f"{body_path}/geom"
    cube = UsdGeom.Cube.Define(stage, cube_path)
    cube.CreateSizeAttr(1.0)
    # Scale the unit cube to desired size
    cube_xf = UsdGeom.Xformable(cube.GetPrim())
    scale_op = cube_xf.AddScaleOp()
    scale_op.Set(Gf.Vec3f(*size_xyz))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color_rgb)])

    # Physics on parent body
    rb = UsdPhysics.RigidBodyAPI.Apply(body_xf.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(body_xf.GetPrim())
    mass_api.CreateMassAttr(float(mass))

    physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(body_xf.GetPrim())
    physx_rb.CreateAngularDampingAttr(2.0)
    physx_rb.CreateLinearDampingAttr(0.2)
    physx_rb.CreateMaxAngularVelocityAttr(720.0)

    if kinematic:
        rb.CreateKinematicEnabledAttr(True)

    # Collider on geometry child
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    physx_col = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
    physx_col.CreateRestOffsetAttr(0.0)
    physx_col.CreateContactOffsetAttr(0.005)

    return body_xf.GetPrim(), cube.GetPrim()


def create_static_box(stage, path, size_xyz, position_xyz, color_rgb=(0.2, 0.2, 0.2), orient_quat_wxyz=None):
    xf = UsdGeom.Xform.Define(stage, path)
    set_translate(xf, position_xyz)
    if orient_quat_wxyz is not None:
        set_orient_quat(xf, orient_quat_wxyz)

    cube = UsdGeom.Cube.Define(stage, f"{path}/geom")
    cube.CreateSizeAttr(1.0)
    cube_xf = UsdGeom.Xformable(cube.GetPrim())
    cube_xf.AddScaleOp().Set(Gf.Vec3f(*size_xyz))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color_rgb)])

    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())

    return xf.GetPrim(), cube.GetPrim()


def create_fixed_joint_to_world(stage, joint_path: str, body_path: str):
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    # body0 omitted => world/static frame
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body_path)])
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr(Gf.Quatf(1.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
    return joint


def create_revolute_joint(
    stage,
    joint_path: str,
    body0_path: str,
    body1_path: str,
    local_pos0,
    local_pos1,
    axis="X",
    lower_deg=-15.0,
    upper_deg=15.0,
):
    joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])

    joint.CreateLocalPos0Attr(Gf.Vec3f(*local_pos0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(*local_pos1))
    joint.CreateLocalRot0Attr(Gf.Quatf(1.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
    joint.CreateAxisAttr(axis)
    joint.CreateLowerLimitAttr(float(lower_deg))
    joint.CreateUpperLimitAttr(float(upper_deg))

    # Drive = damped spring around target angle
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), UsdPhysics.Tokens.angular)
    drive.CreateTypeAttr("force")
    drive.CreateTargetPositionAttr(0.0)
    drive.CreateTargetVelocityAttr(0.0)
    drive.CreateStiffnessAttr(25.0)
    drive.CreateDampingAttr(6.0)
    drive.CreateMaxForceAttr(500.0)

    # Joint state API gives us direct access to joint position/velocity
    joint_state = PhysxSchema.JointStateAPI.Apply(joint.GetPrim(), UsdPhysics.Tokens.angular)
    joint_state.CreatePositionAttr(0.0)
    joint_state.CreateVelocityAttr(0.0)

    return joint, drive, joint_state


# -----------------------------------------------------------------------------
# Build the scene
# -----------------------------------------------------------------------------

world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 120.0, rendering_dt=1.0 / 60.0)
world.scene.add_default_ground_plane()

stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

# Root
switch_root = UsdGeom.Xform.Define(stage, "/World/Switch")
set_translate(switch_root, (0.0, 0.0, 1.0))

# Mark as articulation root so Isaac Lab can load it later as an articulation asset
UsdPhysics.ArticulationRootAPI.Apply(switch_root.GetPrim())
PhysxSchema.PhysxArticulationAPI.Apply(switch_root.GetPrim())

# Wall behind the switch
create_static_box(
    stage,
    "/World/Wall",
    size_xyz=(0.05, 0.40, 0.40),
    position_xyz=(0.0, -0.036, 1.0),
    color_rgb=(0.85, 0.85, 0.88),
    orient_quat_wxyz=(math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)),
)

# Switch base: fixed rigid body
base_prim, _ = create_box_body(
    stage,
    "/World/Switch/base",
    size_xyz=(0.08, 0.02, 0.12),   # x, y, z
    position_xyz=(0.0, 0.0, 0.0),  # local under /World/Switch
    color_rgb=(0.95, 0.95, 0.95),
    mass=0.5,
    kinematic=False,
)

# Rocker: dynamic rigid body
rocker_prim, _ = create_box_body(
    stage,
    "/World/Switch/rocker",
    size_xyz=(0.07, 0.015, 0.11),
    position_xyz=(0.0, 0.018, 0.0),
    color_rgb=(0.92, 0.92, 0.92),
    mass=0.08,
    kinematic=False,
)

# Fix the base to world
create_fixed_joint_to_world(stage, "/World/Switch/world_fix", "/World/Switch/base")

# Hinge across the horizontal axis (X) through the center of the rocker.
# Pressing top/bottom rotates around X.
hinge, hinge_drive, hinge_state = create_revolute_joint(
    stage=stage,
    joint_path="/World/Switch/hinge",
    body0_path="/World/Switch/base",
    body1_path="/World/Switch/rocker",
    local_pos0=(0.0, 0.015, 0.0),   # near front face of base
    local_pos1=(0.0, -0.003, 0.0),  # slightly inside rocker thickness
    axis="X",
    lower_deg=-15.0,
    upper_deg=15.0,
)

# Lamp
lamp_xf = UsdGeom.Xform.Define(stage, "/World/Lamp")
set_translate(lamp_xf, (0.35, 0.2, 1.4))
light = UsdLux.SphereLight.Define(stage, "/World/Lamp/light")
light.CreateRadiusAttr(0.04)
light.CreateIntensityAttr(0.0)
light.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.85))
set_translate(UsdGeom.Xformable(light.GetPrim()), (0.0, 0.0, 0.0))

# Demo finger: kinematic box that moves to press the upper/lower half
finger_prim, _ = create_box_body(
    stage,
    "/World/Finger",
    size_xyz=(0.02, 0.04, 0.02),
    position_xyz=(0.0, 0.10, 1.04),
    color_rgb=(0.25, 0.55, 0.95),
    mass=0.1,
    kinematic=True,
)

finger_xf = UsdGeom.Xformable(finger_prim)

# -----------------------------------------------------------------------------
# Bistable rocker controller
# -----------------------------------------------------------------------------

ON_ANGLE_DEG = 11.0
OFF_ANGLE_DEG = -11.0
SWITCH_THRESHOLD_DEG = 3.0

LIGHT_ON_INTENSITY = 25000.0
LIGHT_OFF_INTENSITY = 0.0

# Start "off" on the top-pressed side
target_angle_deg = OFF_ANGLE_DEG
hinge_drive.GetTargetPositionAttr().Set(target_angle_deg)
light.GetIntensityAttr().Set(LIGHT_OFF_INTENSITY)
light_on = False


def set_light(on: bool):
    light.GetIntensityAttr().Set(LIGHT_ON_INTENSITY if on else LIGHT_OFF_INTENSITY)


def get_switch_obs():
    """Small RL-ready observation dictionary."""
    q_deg = float(hinge_state.GetPositionAttr().Get())
    dq_deg_s = float(hinge_state.GetVelocityAttr().Get())
    return {
        "joint_pos_deg": q_deg,
        "joint_vel_deg_s": dq_deg_s,
        "light_on": float(light_on),
    }


# -----------------------------------------------------------------------------
# Sim loop
# -----------------------------------------------------------------------------

world.reset()

# Give physics a moment to settle
for _ in range(30):
    world.step(render=True)

sim_t = 0.0
print("Running German rocker switch demo.")
print("Close the Isaac Sim window to stop.\n")

while simulation_app.is_running():
    # Demo finger motion:
    #   - first half-cycle presses upper half
    #   - second half-cycle presses lower half
    # Finger is kinematic and sweeps inward/outward in Y while changing Z target.
    period = 4.0
    phase = (sim_t % period) / period  # [0, 1)

    if phase < 0.5:
        # press upper half
        z = 1.04
        # approach -> press -> retreat
        local = phase / 0.5
    else:
        # press lower half
        z = 0.96
        local = (phase - 0.5) / 0.5

    # triangular press profile in Y
    if local < 0.25:
        y = 0.10 - 0.05 * (local / 0.25)
    elif local < 0.75:
        y = 0.05
    else:
        y = 0.05 + 0.05 * ((local - 0.75) / 0.25)

    # X is centered on the switch
    set_translate(finger_xf, (0.0, y, z))

    # Read joint state
    joint_angle_deg = float(hinge_state.GetPositionAttr().Get())
    joint_vel_deg_s = float(hinge_state.GetVelocityAttr().Get())

    # Bistable snap logic:
    # If the rocker is pushed far enough to one side, make that side the new stable target.
    global_target = target_angle_deg
    global_light = light_on

    if joint_angle_deg > SWITCH_THRESHOLD_DEG and target_angle_deg != ON_ANGLE_DEG:
        target_angle_deg = ON_ANGLE_DEG
        hinge_drive.GetTargetPositionAttr().Set(target_angle_deg)
        light_on = True
        set_light(light_on)
        print(f"[{sim_t:6.2f}s] SWITCH -> ON   q={joint_angle_deg:6.2f} deg  dq={joint_vel_deg_s:7.2f} deg/s")
    elif joint_angle_deg < -SWITCH_THRESHOLD_DEG and target_angle_deg != OFF_ANGLE_DEG:
        target_angle_deg = OFF_ANGLE_DEG
        hinge_drive.GetTargetPositionAttr().Set(target_angle_deg)
        light_on = False
        set_light(light_on)
        print(f"[{sim_t:6.2f}s] SWITCH -> OFF  q={joint_angle_deg:6.2f} deg  dq={joint_vel_deg_s:7.2f} deg/s")

    # Step
    world.step(render=True)
    sim_t += world.get_physics_dt()

# Clean shutdown
simulation_app.close()
