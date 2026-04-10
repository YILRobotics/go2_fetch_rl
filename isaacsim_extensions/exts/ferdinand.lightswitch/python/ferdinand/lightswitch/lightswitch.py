import math

import omni.usd
from isaacsim.core.api import World
from isaacsim.examples.interactive.base_sample import BaseSample
from pxr import Gf, Sdf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics


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
    body_xf = UsdGeom.Xform.Define(stage, body_path)
    set_translate(body_xf, position_xyz)

    cube_path = f"{body_path}/geom"
    cube = UsdGeom.Cube.Define(stage, cube_path)
    cube.CreateSizeAttr(1.0)

    cube_xf = UsdGeom.Xformable(cube.GetPrim())
    cube_xf.AddScaleOp().Set(Gf.Vec3f(*size_xyz))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color_rgb)])

    rb = UsdPhysics.RigidBodyAPI.Apply(body_xf.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(body_xf.GetPrim())
    mass_api.CreateMassAttr(float(mass))

    physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(body_xf.GetPrim())
    physx_rb.CreateAngularDampingAttr(2.0)
    physx_rb.CreateLinearDampingAttr(0.2)
    physx_rb.CreateMaxAngularVelocityAttr(720.0)

    if kinematic:
        rb.CreateKinematicEnabledAttr(True)

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

    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), UsdPhysics.Tokens.angular)
    drive.CreateTypeAttr("force")
    drive.CreateTargetPositionAttr(0.0)
    drive.CreateTargetVelocityAttr(0.0)
    drive.CreateStiffnessAttr(25.0)
    drive.CreateDampingAttr(6.0)
    drive.CreateMaxForceAttr(500.0)

    joint_state = PhysxSchema.JointStateAPI.Apply(joint.GetPrim(), UsdPhysics.Tokens.angular)
    joint_state.CreatePositionAttr(0.0)
    joint_state.CreateVelocityAttr(0.0)

    return joint, drive, joint_state


class LightSwitchSample(BaseSample):
    ON_ANGLE_DEG = 11.0
    OFF_ANGLE_DEG = -11.0
    SWITCH_THRESHOLD_DEG = 3.0

    LIGHT_ON_INTENSITY = 25000.0
    LIGHT_OFF_INTENSITY = 0.0

    def __init__(self) -> None:
        super().__init__()
        self._physics_callback_name = "lightswitch_step"
        self._world = None
        self._sim_t = 0.0

        self._hinge_drive = None
        self._hinge_state = None
        self._light = None
        self._finger_xf = None

        self._target_angle_deg = self.OFF_ANGLE_DEG
        self._light_on = False

    def setup_scene(self):
        world = World.instance()
        world.scene.add_default_ground_plane()

        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

        switch_root = UsdGeom.Xform.Define(stage, "/World/Switch")
        set_translate(switch_root, (0.0, 0.0, 1.0))

        UsdPhysics.ArticulationRootAPI.Apply(switch_root.GetPrim())
        PhysxSchema.PhysxArticulationAPI.Apply(switch_root.GetPrim())

        create_static_box(
            stage,
            "/World/Wall",
            size_xyz=(0.05, 0.40, 0.40),
            position_xyz=(0.0, -0.036, 1.0),
            color_rgb=(0.85, 0.85, 0.88),
            orient_quat_wxyz=(math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)),
        )

        create_box_body(
            stage,
            "/World/Switch/base",
            size_xyz=(0.08, 0.02, 0.12),
            position_xyz=(0.0, 0.0, 0.0),
            color_rgb=(0.95, 0.95, 0.95),
            mass=0.5,
            kinematic=False,
        )

        create_box_body(
            stage,
            "/World/Switch/rocker",
            size_xyz=(0.07, 0.015, 0.11),
            position_xyz=(0.0, 0.018, 0.0),
            color_rgb=(0.92, 0.92, 0.92),
            mass=0.08,
            kinematic=False,
        )

        create_fixed_joint_to_world(stage, "/World/Switch/world_fix", "/World/Switch/base")

        _, self._hinge_drive, self._hinge_state = create_revolute_joint(
            stage=stage,
            joint_path="/World/Switch/hinge",
            body0_path="/World/Switch/base",
            body1_path="/World/Switch/rocker",
            local_pos0=(0.0, 0.015, 0.0),
            local_pos1=(0.0, -0.003, 0.0),
            axis="X",
            lower_deg=-15.0,
            upper_deg=15.0,
        )

        lamp_xf = UsdGeom.Xform.Define(stage, "/World/Lamp")
        set_translate(lamp_xf, (0.35, 0.2, 1.4))
        self._light = UsdLux.SphereLight.Define(stage, "/World/Lamp/light")
        self._light.CreateRadiusAttr(0.04)
        self._light.CreateIntensityAttr(0.0)
        self._light.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.85))
        set_translate(UsdGeom.Xformable(self._light.GetPrim()), (0.0, 0.0, 0.0))

        finger_prim, _ = create_box_body(
            stage,
            "/World/Finger",
            size_xyz=(0.02, 0.04, 0.02),
            position_xyz=(0.0, 0.10, 1.04),
            color_rgb=(0.25, 0.55, 0.95),
            mass=0.1,
            kinematic=True,
        )
        self._finger_xf = UsdGeom.Xformable(finger_prim)

        self._target_angle_deg = self.OFF_ANGLE_DEG
        self._hinge_drive.GetTargetPositionAttr().Set(self._target_angle_deg)
        self._light_on = False
        self._set_light(False)
        self._sim_t = 0.0

    async def setup_post_load(self):
        self._world = self.get_world()
        self._remove_physics_callback()
        self._world.add_physics_callback(self._physics_callback_name, self._on_physics_step)

    async def setup_pre_reset(self):
        self._remove_physics_callback()

    async def setup_post_reset(self):
        self._target_angle_deg = self.OFF_ANGLE_DEG
        self._light_on = False
        self._sim_t = 0.0

        if self._hinge_drive is not None:
            self._hinge_drive.GetTargetPositionAttr().Set(self._target_angle_deg)
        self._set_light(False)

        if self._world is not None:
            self._world.add_physics_callback(self._physics_callback_name, self._on_physics_step)

    def world_cleanup(self):
        self._remove_physics_callback()

    def _remove_physics_callback(self):
        if self._world is None:
            return
        try:
            self._world.remove_physics_callback(self._physics_callback_name)
        except Exception:
            pass

    def _set_light(self, on: bool):
        if self._light is None:
            return
        self._light.GetIntensityAttr().Set(self.LIGHT_ON_INTENSITY if on else self.LIGHT_OFF_INTENSITY)

    def get_switch_obs(self):
        if self._hinge_state is None:
            return {"joint_pos_deg": 0.0, "joint_vel_deg_s": 0.0, "light_on": 0.0}
        q_deg = float(self._hinge_state.GetPositionAttr().Get())
        dq_deg_s = float(self._hinge_state.GetVelocityAttr().Get())
        return {
            "joint_pos_deg": q_deg,
            "joint_vel_deg_s": dq_deg_s,
            "light_on": float(self._light_on),
        }

    def _on_physics_step(self, step_size):
        if self._finger_xf is None or self._hinge_state is None or self._hinge_drive is None:
            return

        period = 4.0
        phase = (self._sim_t % period) / period

        if phase < 0.5:
            z = 1.04
            local = phase / 0.5
        else:
            z = 0.96
            local = (phase - 0.5) / 0.5

        if local < 0.25:
            y = 0.10 - 0.05 * (local / 0.25)
        elif local < 0.75:
            y = 0.05
        else:
            y = 0.05 + 0.05 * ((local - 0.75) / 0.25)

        set_translate(self._finger_xf, (0.0, y, z))

        joint_angle_deg = float(self._hinge_state.GetPositionAttr().Get())
        joint_vel_deg_s = float(self._hinge_state.GetVelocityAttr().Get())

        if joint_angle_deg > self.SWITCH_THRESHOLD_DEG and self._target_angle_deg != self.ON_ANGLE_DEG:
            self._target_angle_deg = self.ON_ANGLE_DEG
            self._hinge_drive.GetTargetPositionAttr().Set(self._target_angle_deg)
            self._light_on = True
            self._set_light(True)
            print(
                f"[{self._sim_t:6.2f}s] SWITCH -> ON   q={joint_angle_deg:6.2f} deg  dq={joint_vel_deg_s:7.2f} deg/s"
            )
        elif joint_angle_deg < -self.SWITCH_THRESHOLD_DEG and self._target_angle_deg != self.OFF_ANGLE_DEG:
            self._target_angle_deg = self.OFF_ANGLE_DEG
            self._hinge_drive.GetTargetPositionAttr().Set(self._target_angle_deg)
            self._light_on = False
            self._set_light(False)
            print(
                f"[{self._sim_t:6.2f}s] SWITCH -> OFF  q={joint_angle_deg:6.2f} deg  dq={joint_vel_deg_s:7.2f} deg/s"
            )

        self._sim_t += float(step_size)
