import math
import os
import re
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import omni.kit.commands
import omni.usd
from isaacsim.asset.importer.urdf import _urdf
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.examples.interactive.base_sample import BaseSample
from pxr import Gf, Sdf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics

UNITREE_ROS_DIR = "/home/ferdinand/fetchrobot/unitree_ros"
DEFAULT_GO2_URDF_PATH = f"{UNITREE_ROS_DIR}/robots/go2_description/urdf/go2_description.urdf"


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


def set_scale(xformable, xyz):
    op = None
    for candidate in xformable.GetOrderedXformOps():
        if candidate.GetOpType() == UsdGeom.XformOp.TypeScale:
            op = candidate
            break
    if op is None:
        op = xformable.AddScaleOp()
    op.Set(Gf.Vec3f(*xyz))
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
    set_scale(cube_xf, size_xyz)
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
    set_scale(cube_xf, size_xyz)
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
    drive.CreateStiffnessAttr(14.0) # spring gain
    drive.CreateDampingAttr(2.0) # damping gain
    drive.CreateMaxForceAttr(60.0) # force limit 

    joint_state = PhysxSchema.JointStateAPI.Apply(joint.GetPrim(), UsdPhysics.Tokens.angular)
    joint_state.CreatePositionAttr(0.0)
    joint_state.CreateVelocityAttr(0.0)

    return joint, drive, joint_state


def _sanitize_usd_identifier(name: str, fallback: str = "material") -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", name)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    if not sanitized:
        sanitized = fallback
    if not re.match(r"[A-Za-z_]", sanitized[0]):
        sanitized = f"a_{sanitized}"
    return sanitized


def _to_float_or_none(value: str | None):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sanitize_urdf_copy(source_urdf: str, target_urdf: str):
    tree = ET.parse(source_urdf)
    root = tree.getroot()

    # URDF importer expects one material binding per visual.
    # Some Unitree URDF visuals contain multiple material tags; keep the first.
    for visual in root.iter("visual"):
        materials = [child for child in list(visual) if child.tag == "material"]
        for extra_material in materials[1:]:
            visual.remove(extra_material)

    used_material_names = {}
    for material in root.iter("material"):
        name = material.get("name")
        if not name:
            continue
        base_name = _sanitize_usd_identifier(name, fallback="material")
        suffix = used_material_names.get(base_name, 0)
        unique_name = base_name if suffix == 0 else f"{base_name}_{suffix}"
        used_material_names[base_name] = suffix + 1
        material.set("name", unique_name)

    for link in root.iter("link"):
        if link.find("visual") is not None:
            pass
        else:
            visual = ET.SubElement(link, "visual")
            ET.SubElement(visual, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
            geometry = ET.SubElement(visual, "geometry")
            ET.SubElement(geometry, "sphere", {"radius": "0.001"})
            ET.SubElement(visual, "material", {"name": "auto_visual_marker"})

        # Ensure every link has valid inertial data (positive mass + inertia tensor).
        inertial = link.find("inertial")
        if inertial is None:
            inertial = ET.SubElement(link, "inertial")
            ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
            ET.SubElement(inertial, "mass", {"value": "1e-6"})
            ET.SubElement(
                inertial,
                "inertia",
                {"ixx": "1e-8", "iyy": "1e-8", "izz": "1e-8", "ixy": "0", "ixz": "0", "iyz": "0"},
            )
        else:
            if inertial.find("origin") is None:
                ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})

            mass_el = inertial.find("mass")
            mass_ok = False
            if mass_el is not None:
                mass_v = _to_float_or_none(mass_el.get("value"))
                mass_ok = mass_v is not None and mass_v > 0.0
            if not mass_ok:
                if mass_el is None:
                    mass_el = ET.SubElement(inertial, "mass")
                mass_el.set("value", "1e-6")

            inertia_el = inertial.find("inertia")
            inertia_ok = False
            if inertia_el is not None:
                ixx = _to_float_or_none(inertia_el.get("ixx"))
                iyy = _to_float_or_none(inertia_el.get("iyy"))
                izz = _to_float_or_none(inertia_el.get("izz"))
                inertia_ok = ixx is not None and iyy is not None and izz is not None and ixx > 0.0 and iyy > 0.0 and izz > 0.0
            if not inertia_ok:
                if inertia_el is None:
                    inertia_el = ET.SubElement(inertial, "inertia")
                inertia_el.set("ixx", "1e-8")
                inertia_el.set("iyy", "1e-8")
                inertia_el.set("izz", "1e-8")
                inertia_el.set("ixy", "0")
                inertia_el.set("ixz", "0")
                inertia_el.set("iyz", "0")

        # Ensure every link has at least one collider so mass properties can be computed robustly.
        has_collision_geom = False
        for col in link.findall("collision"):
            if col.find("geometry") is not None:
                has_collision_geom = True
                break
        if not has_collision_geom:
            collision = ET.SubElement(link, "collision")
            ET.SubElement(collision, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
            geometry = ET.SubElement(collision, "geometry")
            ET.SubElement(geometry, "sphere", {"radius": "0.001"})

    tree.write(target_urdf, encoding="utf-8", xml_declaration=True)


def _prepare_sanitized_urdf_asset(source_urdf: str) -> str:
    urdf_abs = os.path.abspath(source_urdf)
    urdf_dir = os.path.dirname(urdf_abs)
    package_dir = os.path.dirname(urdf_dir)
    package_name = os.path.basename(package_dir)

    tmp_root = "/tmp/IsaacLab/unitree_rl_lab/urdf_sanitized"
    tmp_package_dir = os.path.join(tmp_root, package_name)
    tmp_urdf_dir = os.path.join(tmp_package_dir, "urdf")
    tmp_urdf_path = os.path.join(tmp_urdf_dir, os.path.basename(urdf_abs))

    if os.path.exists(tmp_package_dir):
        if os.path.islink(tmp_package_dir) or os.path.isfile(tmp_package_dir):
            os.remove(tmp_package_dir)
        else:
            shutil.rmtree(tmp_package_dir)

    os.makedirs(tmp_urdf_dir, exist_ok=True)

    for entry in os.listdir(package_dir):
        if entry == "urdf":
            continue
        src = os.path.join(package_dir, entry)
        dst = os.path.join(tmp_package_dir, entry)
        os.symlink(src, dst)

    _sanitize_urdf_copy(urdf_abs, tmp_urdf_path)
    return tmp_urdf_path


class LightSwitchSample(BaseSample):
    SWITCH_WORLD_X = 0.45
    SWITCH_WORLD_Y = 0.0
    SWITCH_WORLD_Z = 1.0

    ON_ANGLE_DEG = 9.0
    OFF_ANGLE_DEG = -9.0
    SWITCH_ON_THRESHOLD_DEG = 1.0
    SWITCH_OFF_THRESHOLD_DEG = -1.0
    SNAP_BOOST_VEL_DEG_S = 60.0
    SNAP_BOOST_TIME_S = 0.10

    LIGHT_ON_INTENSITY = 25000.0
    LIGHT_OFF_INTENSITY = 0.0

    def __init__(self) -> None:
        super().__init__()
        self._physics_callback_name = f"lightswitch_step_{id(self)}"
        self._physics_callback_registered = False
        self._world = None
        self._sim_t = 0.0

        self._hinge_drive = None
        self._hinge_state = None
        self._light = None
        self._finger_xf = None
        self._go2 = None
        self._go2_stance_positions = None
        self._go2_stance_velocities = None
        self._go2_pose_reapply_steps = 0

        self._target_angle_deg = self.OFF_ANGLE_DEG
        self._light_on = False
        self._snap_boost_time_left_s = 0.0

        # Disable the demo finger by default so the interactive force tool can push freely.
        self._demo_finger_enabled = os.environ.get("LIGHTSWITCH_DEMO_FINGER", "1") == "1"

    def _compute_go2_default_pose(self):
        if self._go2 is None:
            return None, None

        dof_names = list(self._go2.dof_names)
        if not dof_names:
            return None, None

        positions = np.zeros((len(dof_names),), dtype=np.float32)
        velocities = np.zeros((len(dof_names),), dtype=np.float32)

        for idx, name in enumerate(dof_names):
            if re.match(r".*R_hip_joint$", name):
                positions[idx] = -0.1
            elif re.match(r".*L_hip_joint$", name):
                positions[idx] = 0.1
            elif re.match(r"F[L,R]_thigh_joint$", name):
                positions[idx] = 0.8
            elif re.match(r"R[L,R]_thigh_joint$", name):
                positions[idx] = 1.0
            elif re.match(r".*_calf_joint$", name):
                positions[idx] = -1.5

        return positions, velocities

    def _apply_go2_default_pose(self):
        positions, velocities = self._compute_go2_default_pose()
        if positions is None:
            return

        self._go2_stance_positions = positions
        self._go2_stance_velocities = velocities

        try:
            # Keep reset behavior deterministic across Stop/Play.
            self._go2.set_joints_default_state(positions=positions, velocities=velocities)
        except Exception as exc:
            print(f"WARNING: Failed to set GO2 default joint state: {exc}")

        try:
            if not self._go2.handles_initialized:
                self._go2.initialize()
        except Exception:
            return

        self._apply_go2_pose_targets()

    def _apply_go2_pose_targets(self):
        if self._go2 is None or self._go2_stance_positions is None:
            return
        try:
            if not self._go2.handles_initialized:
                return

            self._go2.set_joint_positions(self._go2_stance_positions)
            self._go2.set_joint_velocities(self._go2_stance_velocities)
            self._go2.apply_action(
                ArticulationAction(
                    joint_positions=self._go2_stance_positions,
                    joint_velocities=self._go2_stance_velocities,
                )
            )
        except Exception:
            return

    def _load_go2_from_urdf(self, world):
        urdf_source_path = os.environ.get("GO2_URDF_PATH", DEFAULT_GO2_URDF_PATH)
        if not os.path.isfile(urdf_source_path):
            print(f"ERROR: GO2 URDF not found: {urdf_source_path}")
            return

        urdf_path = _prepare_sanitized_urdf_asset(urdf_source_path)

        import_config = _urdf.ImportConfig()
        import_config.fix_base = False
        import_config.make_default_prim = True
        import_config.merge_fixed_joints = False
        import_config.import_inertia_tensor = True

        parse_ok, robot_model = omni.kit.commands.execute(
            "URDFParseFile",
            urdf_path=urdf_path,
            import_config=import_config,
        )
        if not parse_ok or robot_model is None:
            print(f"ERROR: Failed to parse GO2 URDF: {urdf_path}")
            return

        import_ok, prim_path = omni.kit.commands.execute(
            "URDFImportRobot",
            urdf_path=urdf_path,
            urdf_robot=robot_model,
            import_config=import_config,
        )
        if not import_ok or not prim_path:
            print("ERROR: Failed to import GO2 URDF robot into stage")
            return

        self._go2 = world.scene.add(
            SingleArticulation(
                prim_path=prim_path,
                name="go2_robot",
                position=np.array([0.0, 0.0, 0.4]),
            )
        )
        print(f"GO2 robot loaded from URDF: {urdf_source_path}")

    def setup_scene(self):
        world = World.instance()
        world.scene.add_default_ground_plane()

        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

        switch_root = UsdGeom.Xform.Define(stage, "/World/Switch")
        set_translate(switch_root, (self.SWITCH_WORLD_X, self.SWITCH_WORLD_Y, self.SWITCH_WORLD_Z))
        # Rotate switch 90 deg about Z so the front faces toward the robot.
        set_orient_quat(switch_root, (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)))

        UsdPhysics.ArticulationRootAPI.Apply(switch_root.GetPrim())
        PhysxSchema.PhysxArticulationAPI.Apply(switch_root.GetPrim())

        create_static_box(
            stage,
            "/World/Wall",
            size_xyz=(0.05, 0.40, 0.40),
            position_xyz=(self.SWITCH_WORLD_X + 0.036, self.SWITCH_WORLD_Y, self.SWITCH_WORLD_Z),
            color_rgb=(0.85, 0.85, 0.88),
        )

        create_box_body(
            stage,
            "/World/Switch/base",
            size_xyz=(0.05, 0.010, 0.05),
            position_xyz=(0.0, 0.0, 0.0),
            color_rgb=(0.95, 0.95, 0.95),
            mass=0.5,
            kinematic=False,
        )

        create_box_body(
            stage,
            "/World/Switch/rocker",
            size_xyz=(0.044, 0.007, 0.044),
            position_xyz=(0.0, 0.009, 0.0),
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
            local_pos0=(0.0, 0.0055, 0.0),
            local_pos1=(0.0, -0.0035, 0.0),
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
            position_xyz=(self.SWITCH_WORLD_X - 0.12, self.SWITCH_WORLD_Y, self.SWITCH_WORLD_Z + 0.012),
            color_rgb=(0.25, 0.55, 0.95),
            mass=0.1,
            kinematic=True,
        )
        self._finger_xf = UsdGeom.Xformable(finger_prim)

        self._load_go2_from_urdf(world)

        self._target_angle_deg = self.OFF_ANGLE_DEG
        self._hinge_drive.GetTargetPositionAttr().Set(self._target_angle_deg)
        self._hinge_drive.GetTargetVelocityAttr().Set(0.0)
        self._light_on = False
        self._set_light(False)
        self._sim_t = 0.0
        self._snap_boost_time_left_s = 0.0

    async def setup_post_load(self):
        self._world = self.get_world()
        try:
            self._go2 = self._world.scene.get_object("go2_robot")
        except Exception:
            pass
        self._apply_go2_default_pose()
        self._go2_pose_reapply_steps = 240
        self._register_physics_callback()

    async def setup_pre_reset(self):
        self._remove_physics_callback()

    async def setup_post_reset(self):
        self._target_angle_deg = self.OFF_ANGLE_DEG
        self._light_on = False
        self._sim_t = 0.0

        if self._hinge_drive is not None:
            self._hinge_drive.GetTargetPositionAttr().Set(self._target_angle_deg)
            self._hinge_drive.GetTargetVelocityAttr().Set(0.0)
        self._set_light(False)
        self._apply_go2_default_pose()
        self._go2_pose_reapply_steps = 240
        self._snap_boost_time_left_s = 0.0

        self._register_physics_callback()

    def world_cleanup(self):
        self._remove_physics_callback()

    def _register_physics_callback(self):
        if self._world is None or self._physics_callback_registered:
            return
        self._world.add_physics_callback(self._physics_callback_name, self._on_physics_step)
        self._physics_callback_registered = True

    def _remove_physics_callback(self):
        if self._world is None or not self._physics_callback_registered:
            return
        try:
            self._world.remove_physics_callback(self._physics_callback_name)
        except Exception:
            pass
        self._physics_callback_registered = False

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

        if self._go2_pose_reapply_steps > 0:
            self._apply_go2_pose_targets()
            self._go2_pose_reapply_steps -= 1

        if self._demo_finger_enabled:
            period = 4.0
            phase = (self._sim_t % period) / period

            if phase < 0.5:
                z = self.SWITCH_WORLD_Z + 0.012
                local = phase / 0.5
            else:
                z = self.SWITCH_WORLD_Z - 0.012
                local = (phase - 0.5) / 0.5

            # Finger motion profile in X (switch was rotated 90 deg about Z):
            # - approach from `approach_x` toward the switch,
            # - hold at `press_x` to maintain contact,
            # - retreat back to `approach_x`.
            approach_x = self.SWITCH_WORLD_X - 0.15
            press_x = self.SWITCH_WORLD_X - 0.005

            if local < 0.25:
                # Approach phase: linearly move inward.
                x = approach_x + (press_x - approach_x) * (local / 0.25)
            elif local < 0.75:
                # Hold phase: keep full press depth.
                x = press_x
            else:
                # Retreat phase: linearly move back out.
                x = press_x + (approach_x - press_x) * ((local - 0.75) / 0.25)

            set_translate(self._finger_xf, (x, self.SWITCH_WORLD_Y, z))
        else:
            # Park finger away from the switch when using interactive force tool.
            set_translate(self._finger_xf, (self.SWITCH_WORLD_X - 0.18, self.SWITCH_WORLD_Y, self.SWITCH_WORLD_Z + 0.12))

        joint_angle_deg = float(self._hinge_state.GetPositionAttr().Get())
        joint_vel_deg_s = float(self._hinge_state.GetVelocityAttr().Get())

        # Bistable latch: switch changes state only after crossing signed thresholds.
        if (not self._light_on) and joint_angle_deg > self.SWITCH_ON_THRESHOLD_DEG:
            self._target_angle_deg = self.ON_ANGLE_DEG
            self._hinge_drive.GetTargetPositionAttr().Set(self._target_angle_deg)
            self._hinge_drive.GetTargetVelocityAttr().Set(self.SNAP_BOOST_VEL_DEG_S)
            self._snap_boost_time_left_s = self.SNAP_BOOST_TIME_S
            self._light_on = True
            self._set_light(True)
            print(
                f"[{self._sim_t:6.2f}s] SWITCH -> ON   q={joint_angle_deg:6.2f} deg  dq={joint_vel_deg_s:7.2f} deg/s"
            )
        elif self._light_on and joint_angle_deg < self.SWITCH_OFF_THRESHOLD_DEG:
            self._target_angle_deg = self.OFF_ANGLE_DEG
            self._hinge_drive.GetTargetPositionAttr().Set(self._target_angle_deg)
            self._hinge_drive.GetTargetVelocityAttr().Set(-self.SNAP_BOOST_VEL_DEG_S)
            self._snap_boost_time_left_s = self.SNAP_BOOST_TIME_S
            self._light_on = False
            self._set_light(False)
            print(
                f"[{self._sim_t:6.2f}s] SWITCH -> OFF  q={joint_angle_deg:6.2f} deg  dq={joint_vel_deg_s:7.2f} deg/s"
            )

        if self._snap_boost_time_left_s > 0.0:
            self._snap_boost_time_left_s -= float(step_size)
            if self._snap_boost_time_left_s <= 0.0:
                self._hinge_drive.GetTargetVelocityAttr().Set(0.0)

        self._sim_t += float(step_size)
