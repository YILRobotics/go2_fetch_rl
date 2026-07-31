from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.sim.utils.stage import get_current_stage

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Switch behavior constants mirrored from the extension.
ON_ANGLE_DEG = 9.0
OFF_ANGLE_DEG = -9.0
SWITCH_ON_THRESHOLD_DEG = 1.0
SWITCH_OFF_THRESHOLD_DEG = -1.0
SNAP_BOOST_VEL_DEG_S = 60.0
SNAP_BOOST_TIME_S = 0.10
PRESS_HOLD_TIME_S = 1.0
PRESS_CONTACT_GRACE_TIME_S = 0.10
STAND_PHASE_DURATION_S = 0.50
CROUCH_PHASE_TIMEOUT_S = 0.50
CROUCH_HOLD_TIME_S = 0.04
JUMP_READY_HOLD_TIME_S = 0.02
JUMP_PHASE_DURATION_S = 2.7
LAND_PHASE_HOLD_TIME_S = 0.75

# Ordered behavior phases.  The policy receives these as a one-hot observation.
PHASE_STAND = 0
PHASE_LIFT = 1  # four-foot preload crouch
PHASE_JUMP = 2
PHASE_LAND = 3  # recovery after either a completed press hold or reach timeout
PHASE_SUCCESS = 4
# Compatibility alias for older analysis utilities; new episodes never have a
# separate press phase because completing the hold immediately starts recovery.
PHASE_PRESS = PHASE_LAND
NUM_BEHAVIOR_PHASES = 5

# The wall is independent of the randomized switch height.  It starts at the
# floor and remains tall enough to contain the complete switch-height range.
WALL_HEIGHT = 1.0
WALL_CENTER_Z = WALL_HEIGHT * 0.5


# General helpers and curriculum metrics


def _to_env_ids(env: ManagerBasedRLEnv, env_ids: torch.Tensor | slice | list[int] | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=env.device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


def _scene_env_origins_xy(env: ManagerBasedRLEnv) -> torch.Tensor:
    default_env_origins = getattr(env.scene, "_default_env_origins", None)
    if default_env_origins is not None:
        return default_env_origins[:, :2]
    return env.scene.env_origins[:, :2]


def _env_step_time_s(env: ManagerBasedRLEnv) -> float:
    step_dt = getattr(env, "step_dt", None)
    if step_dt is not None:
        return max(float(step_dt), 1e-6)
    cfg = getattr(env, "cfg", None)
    if cfg is not None and getattr(cfg, "sim", None) is not None and hasattr(cfg, "decimation"):
        return max(float(cfg.sim.dt) * float(cfg.decimation), 1e-6)
    return 1.0


def _duration_steps(duration_s: float, step_dt: float) -> int:
    """Convert a duration to at least one complete environment step."""
    return max(1, int(math.ceil(float(duration_s) / step_dt)))


def curriculum_common_step_counter(env, env_ids):
    del env_ids
    return float(env.common_step_counter)


def curriculum_phase_fraction(env, env_ids, phase: int):
    """Fraction of parallel environments currently occupying one behavior phase."""
    del env_ids
    _ensure_switch_buffers(env)
    return float((env._ls_phase == int(phase)).float().mean().item())


def curriculum_press_success_fraction(env, env_ids):
    """Fraction of the episodes being reset that successfully pressed the rocker."""
    _ensure_switch_buffers(env)
    ids = _to_env_ids(env, env_ids)
    if ids.numel() == 0:
        return 0.0
    return float(env._ls_success[ids].float().mean().item())


def curriculum_behavior_difficulty(
    env,
    env_ids,
    jump_rise_start: float = 0.18,
    jump_rise_target: float = 0.28,
    foot_lift_start: float = 0.12,
    foot_lift_target: float = 0.28,
    press_hold_start_s: float = 0.20,
    press_hold_target_s: float = 1.0,
    success_rate_start: float = 0.20,
    success_rate_full: float = 0.50,
    ema_rate: float = 0.20,
):
    """Raise lift, jump, and press-hold difficulty after reliable task success."""
    _ensure_switch_buffers(env)
    ids = _to_env_ids(env, env_ids)
    completed = env.episode_length_buf[ids] > 0
    if torch.any(completed):
        jump_rate = env._ls_success[ids[completed]].float().mean()
        rate = float(ema_rate)
        env._ls_jump_success_ema = (1.0 - rate) * env._ls_jump_success_ema + rate * jump_rate

    denominator = max(float(success_rate_full) - float(success_rate_start), 1e-6)
    jump_alpha = torch.clamp(
        (env._ls_jump_success_ema - float(success_rate_start)) / denominator, 0.0, 1.0
    )
    env._ls_adaptive_jump_rise = float(jump_rise_start) + jump_alpha * (
        float(jump_rise_target) - float(jump_rise_start)
    )
    env._ls_adaptive_foot_lift = float(foot_lift_start) + jump_alpha * (
        float(foot_lift_target) - float(foot_lift_start)
    )
    env._ls_adaptive_press_hold_s = float(press_hold_start_s) + jump_alpha * (
        float(press_hold_target_s) - float(press_hold_start_s)
    )
    env._ls_behavior_difficulty = jump_alpha
    return {
        "difficulty": jump_alpha,
        "jump_success_ema": env._ls_jump_success_ema,
        "jump_rise": env._ls_adaptive_jump_rise,
        "foot_lift": env._ls_adaptive_foot_lift,
        "press_hold_s": env._ls_adaptive_press_hold_s,
    }


def curriculum_maneuver_diagnostics(env, env_ids):
    """Expose physical progress metrics needed to distinguish reward and power failures."""
    del env_ids
    _ensure_switch_buffers(env)
    robot: Articulation = env.scene["robot"]
    effort = torch.abs(robot.data.applied_torque)
    return {
        "episode_max_crouch_depth": env._ls_episode_max_crouch.mean(),
        "episode_max_base_rise": env._ls_episode_max_rise.mean(),
        "episode_max_front_lift": env._ls_episode_max_front_lift.mean(),
        "episode_max_left_foot_forward_extension": env._ls_episode_max_foot_forward.mean(),
        "episode_max_rear_step": env._ls_episode_max_rear_step.mean(),
        "episode_min_body_wall_distance": env._ls_episode_min_body_wall_distance.mean(),
        "episode_min_left_foot_distance": env._ls_episode_min_foot_distance.mean(),
        "episode_min_left_foot_horizontal_error": env._ls_episode_min_foot_horizontal_error.mean(),
        "episode_min_left_foot_vertical_error": env._ls_episode_min_foot_vertical_error.mean(),
        "episode_max_press_hold_s": env._ls_episode_max_press_hold_s.mean(),
        "episode_first_contact_time_s": env._ls_episode_first_contact_time_s.mean(),
        "episode_contact_dropouts": env._ls_episode_contact_dropouts.float().mean(),
        "first_contact_fraction": env._ls_first_contact_seen.float().mean(),
        "valid_crouch_fraction": env._ls_valid_crouch.float().mean(),
        "crouch_depth_pass_fraction": env._ls_crouch_depth_ok.float().mean(),
        "crouch_ground_pass_fraction": env._ls_crouch_ground_ok.float().mean(),
        "crouch_motion_pass_fraction": env._ls_crouch_motion_ok.float().mean(),
        "crouch_shift_pass_fraction": env._ls_crouch_shift_ok.float().mean(),
        "rear_support_fraction": env._ls_rear_ground.float().mean(),
        "touch_fraction": env._ls_success.float().mean(),
        "effort_saturation_fraction": (effort >= 19.5).float().mean(),
    }


# Episode state


def _ensure_switch_buffers(env: ManagerBasedRLEnv):
    if not hasattr(env, "_ls_switch_center_w") or env._ls_switch_center_w.shape[0] != env.num_envs:
        env._ls_switch_center_w = torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)

        env._ls_current_state = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env._ls_target_state = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        env._ls_success = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env._ls_contact = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env._ls_rocker_contact_force = torch.zeros(env.num_envs, device=env.device)
        env._ls_first_contact_seen = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env._ls_first_contact_trigger = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env._ls_contact_was_active = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env._ls_toggle_trigger = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env._ls_press_in_progress = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env._ls_joint_pos_deg = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        env._ls_joint_vel_deg_s = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        env._ls_snap_boost_time_left_s = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        env._ls_switch_ready = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

        env._ls_phase = torch.full(
            (env.num_envs,), PHASE_STAND, device=env.device, dtype=torch.long
        )
        env._ls_phase_entered = torch.full(
            (env.num_envs,), -1, device=env.device, dtype=torch.long
        )
        env._ls_lift_hold_counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        env._ls_land_hold_counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        env._ls_jump_hold_counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        env._ls_settled_base_z = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        env._ls_settled_base_xy = torch.zeros((env.num_envs, 2), device=env.device)
        env._ls_wall_near_face_x = torch.zeros(env.num_envs, device=env.device)
        env._ls_initial_body_wall_distance = torch.zeros(env.num_envs, device=env.device)
        env._ls_crouched_base_z = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        env._ls_settled_foot_z = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        env._ls_jump_front_start_w = torch.zeros((env.num_envs, 2, 3), device=env.device)
        env._ls_jump_rear_start_w = torch.zeros((env.num_envs, 2, 3), device=env.device)
        env._ls_phase_step_counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        env._ls_rear_ground = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        env._ls_left_front_air = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env._ls_jump_ready = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env._ls_jump_ready_seen = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env._ls_jump_reach_active = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env._ls_crouch_ready_trigger = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env._ls_valid_crouch = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env._ls_crouch_depth_ok = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env._ls_crouch_ground_ok = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env._ls_crouch_motion_ok = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env._ls_crouch_shift_ok = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env._ls_episode_max_crouch = torch.zeros(env.num_envs, device=env.device)
        env._ls_episode_max_rise = torch.zeros(env.num_envs, device=env.device)
        env._ls_episode_max_front_lift = torch.zeros(env.num_envs, device=env.device)
        env._ls_episode_max_foot_forward = torch.zeros(env.num_envs, device=env.device)
        env._ls_episode_max_rear_step = torch.zeros(env.num_envs, device=env.device)
        env._ls_episode_min_body_wall_distance = torch.full(
            (env.num_envs,), 2.0, device=env.device
        )
        env._ls_episode_min_foot_distance = torch.full((env.num_envs,), 2.0, device=env.device)
        env._ls_episode_min_foot_horizontal_error = torch.full(
            (env.num_envs,), 2.0, device=env.device
        )
        env._ls_episode_min_foot_vertical_error = torch.full(
            (env.num_envs,), 2.0, device=env.device
        )
        env._ls_episode_max_press_hold_s = torch.zeros(env.num_envs, device=env.device)
        env._ls_episode_first_contact_time_s = torch.full(
            (env.num_envs,), float(JUMP_PHASE_DURATION_S), device=env.device
        )
        env._ls_episode_contact_dropouts = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.long
        )
        env._ls_phase_last_step = -1
        env._ls_jump_success_ema = torch.tensor(0.0, device=env.device)
        env._ls_adaptive_jump_rise = torch.tensor(0.18, device=env.device)
        env._ls_adaptive_foot_lift = torch.tensor(0.12, device=env.device)
        env._ls_adaptive_press_hold_s = torch.tensor(0.20, device=env.device)
        env._ls_behavior_difficulty = torch.tensor(0.0, device=env.device)

        env._ls_joint_pos_attrs = [None] * env.num_envs
        env._ls_joint_vel_attrs = [None] * env.num_envs
        env._ls_drive_target_pos_attrs = [None] * env.num_envs
        env._ls_drive_target_vel_attrs = [None] * env.num_envs

        env._ls_interaction_last_step = -1
        env._ls_success_hold_counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        env._ls_contact_dropout_counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)


# USD stage construction


def _set_translate(prim, xyz: tuple[float, float, float]):
    xformable = UsdGeom.Xformable(prim)
    op = None
    for candidate in xformable.GetOrderedXformOps():
        if candidate.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op = candidate
            break
    if op is None:
        op = xformable.AddTranslateOp()
    if op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
        op.Set(Gf.Vec3f(float(xyz[0]), float(xyz[1]), float(xyz[2])))
    else:
        op.Set(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))


def _set_orient_quat(prim, quat_wxyz: tuple[float, float, float, float]):
    w, x, y, z = quat_wxyz
    xformable = UsdGeom.Xformable(prim)
    op = None
    for candidate in xformable.GetOrderedXformOps():
        if candidate.GetOpType() == UsdGeom.XformOp.TypeOrient:
            op = candidate
            break
    if op is None:
        op = xformable.AddOrientOp()
    if op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
        op.Set(Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z))))
    else:
        op.Set(Gf.Quatd(float(w), Gf.Vec3d(float(x), float(y), float(z))))


def _set_scale(prim, xyz: tuple[float, float, float]):
    xformable = UsdGeom.Xformable(prim)
    op = None
    for candidate in xformable.GetOrderedXformOps():
        if candidate.GetOpType() == UsdGeom.XformOp.TypeScale:
            op = candidate
            break
    if op is None:
        op = xformable.AddScaleOp()
    if op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
        op.Set(Gf.Vec3f(float(xyz[0]), float(xyz[1]), float(xyz[2])))
    else:
        op.Set(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))


def _create_box_body(
    stage,
    body_path: str,
    size_xyz: tuple[float, float, float],
    position_xyz: tuple[float, float, float],
    color_rgb: tuple[float, float, float],
    mass: float,
    kinematic: bool = False,
):
    body_xf = UsdGeom.Xform.Define(stage, body_path)
    _set_translate(body_xf.GetPrim(), position_xyz)

    cube = UsdGeom.Cube.Define(stage, f"{body_path}/geom")
    cube.CreateSizeAttr(1.0)
    _set_scale(cube.GetPrim(), size_xyz)
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


def _create_static_box(
    stage,
    path: str,
    size_xyz: tuple[float, float, float],
    position_xyz: tuple[float, float, float],
    color_rgb: tuple[float, float, float],
):
    xf = UsdGeom.Xform.Define(stage, path)
    _set_translate(xf.GetPrim(), position_xyz)

    cube = UsdGeom.Cube.Define(stage, f"{path}/geom")
    cube.CreateSizeAttr(1.0)
    _set_scale(cube.GetPrim(), size_xyz)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color_rgb)])

    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())


def _create_fixed_joint_to_world(stage, joint_path: str, body_path: str):
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body_path)])
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr(Gf.Quatf(1.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0))


def _create_revolute_joint(
    stage,
    joint_path: str,
    body0_path: str,
    body1_path: str,
    local_pos0: tuple[float, float, float],
    local_pos1: tuple[float, float, float],
):
    joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*local_pos0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(*local_pos1))
    joint.CreateLocalRot0Attr(Gf.Quatf(1.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
    joint.CreateAxisAttr("X")
    joint.CreateLowerLimitAttr(-15.0)
    joint.CreateUpperLimitAttr(15.0)

    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), UsdPhysics.Tokens.angular)
    drive.CreateTypeAttr("force")
    drive.CreateTargetPositionAttr(0.0)
    drive.CreateTargetVelocityAttr(0.0)
    drive.CreateStiffnessAttr(14.0)
    drive.CreateDampingAttr(2.0)
    drive.CreateMaxForceAttr(60.0)

    joint_state = PhysxSchema.JointStateAPI.Apply(joint.GetPrim(), UsdPhysics.Tokens.angular)
    joint_state.CreatePositionAttr(0.0)
    joint_state.CreateVelocityAttr(0.0)


def _cache_switch_handles_for_env(env: ManagerBasedRLEnv, env_id: int, switch_root_path: str):
    stage = get_current_stage()
    if stage is None:
        return
    hinge_prim = stage.GetPrimAtPath(f"{switch_root_path}/hinge")
    if not hinge_prim or not hinge_prim.IsValid():
        return

    drive = UsdPhysics.DriveAPI.Get(hinge_prim, UsdPhysics.Tokens.angular)
    if not drive:
        drive = UsdPhysics.DriveAPI.Apply(hinge_prim, UsdPhysics.Tokens.angular)

    pos_attr = drive.GetTargetPositionAttr()
    if not pos_attr:
        pos_attr = drive.CreateTargetPositionAttr(0.0)
    vel_attr = drive.GetTargetVelocityAttr()
    if not vel_attr:
        vel_attr = drive.CreateTargetVelocityAttr(0.0)

    joint_state = PhysxSchema.JointStateAPI.Get(hinge_prim, UsdPhysics.Tokens.angular)
    if not joint_state:
        joint_state = PhysxSchema.JointStateAPI.Apply(hinge_prim, UsdPhysics.Tokens.angular)

    joint_pos_attr = joint_state.GetPositionAttr()
    if not joint_pos_attr:
        joint_pos_attr = joint_state.CreatePositionAttr(0.0)
    joint_vel_attr = joint_state.GetVelocityAttr()
    if not joint_vel_attr:
        joint_vel_attr = joint_state.CreateVelocityAttr(0.0)

    env._ls_drive_target_pos_attrs[env_id] = pos_attr
    env._ls_drive_target_vel_attrs[env_id] = vel_attr
    env._ls_joint_pos_attrs[env_id] = joint_pos_attr
    env._ls_joint_vel_attrs[env_id] = joint_vel_attr
    env._ls_switch_ready[env_id] = True


def _set_switch_root_wall_transforms_for_envs(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    switch_center_local: torch.Tensor,
    switch_yaw: float,
    wall_x_offset: float = 0.036,
):
    stage = get_current_stage()
    if stage is None:
        return

    q_w = math.cos(float(switch_yaw) * 0.5)
    q_z = math.sin(float(switch_yaw) * 0.5)
    quat_wxyz = (q_w, 0.0, 0.0, q_z)

    env_ids_cpu = env_ids.to(device="cpu", dtype=torch.long).tolist()
    centers_cpu = switch_center_local.to(device="cpu")
    for local_i, env_i in enumerate(env_ids_cpu):
        cx = float(centers_cpu[local_i, 0].item())
        cy = float(centers_cpu[local_i, 1].item())
        cz = float(centers_cpu[local_i, 2].item())

        root_path = f"{env.scene.env_ns}/env_{env_i}/Switch"
        wall_path = f"{env.scene.env_ns}/env_{env_i}/Wall"

        root_prim = stage.GetPrimAtPath(root_path)
        if root_prim and root_prim.IsValid():
            _set_translate(root_prim, (cx, cy, cz))
            _set_orient_quat(root_prim, quat_wxyz)

        wall_prim = stage.GetPrimAtPath(wall_path)
        if wall_prim and wall_prim.IsValid():
            _set_translate(wall_prim, (cx + float(wall_x_offset), cy, WALL_CENTER_Z))


def setup_lightswitch_stage(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | slice | list[int] | None,
    switch_center_xy: tuple[float, float] = (0.45, 0.0),
    switch_x_range: tuple[float, float] = (0.42, 0.48),
    switch_y_range: tuple[float, float] = (-0.04, 0.04),
    switch_default_height: float = 1.0,
    switch_height_range: tuple[float, float] = (0.8, 1.1),
    switch_yaw: float = math.pi * 0.5,
    wall_x_offset: float = 0.036,
    wall_thickness: float = 0.05,
):
    env_ids = _to_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return
    _ensure_switch_buffers(env)

    stage = get_current_stage()
    if stage is None:
        return

    env_ids_cpu = env_ids.to(device="cpu", dtype=torch.long).tolist()
    for env_i in env_ids_cpu:
        env_root_path = f"{env.scene.env_ns}/env_{env_i}"
        switch_root_path = f"{env_root_path}/Switch"
        wall_path = f"{env_root_path}/Wall"

        switch_root_prim = stage.GetPrimAtPath(switch_root_path)
        if not switch_root_prim or not switch_root_prim.IsValid():
            switch_root = UsdGeom.Xform.Define(stage, switch_root_path)
            _set_translate(switch_root.GetPrim(), (0.0, 0.0, float(switch_default_height)))
            _set_orient_quat(switch_root.GetPrim(), (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)))
            UsdPhysics.ArticulationRootAPI.Apply(switch_root.GetPrim())
            PhysxSchema.PhysxArticulationAPI.Apply(switch_root.GetPrim())

            _create_static_box(
                stage=stage,
                path=wall_path,
                size_xyz=(float(wall_thickness), 0.80, WALL_HEIGHT),
                position_xyz=(
                    float(switch_center_xy[0]) + float(wall_x_offset),
                    float(switch_center_xy[1]),
                    WALL_CENTER_Z,
                ),
                color_rgb=(0.85, 0.85, 0.88),
            )

            _create_box_body(
                stage=stage,
                body_path=f"{switch_root_path}/base",
                size_xyz=(0.05, 0.010, 0.05),
                position_xyz=(0.0, 0.0, 0.0),
                color_rgb=(0.95, 0.95, 0.95),
                mass=0.5,
                kinematic=False,
            )
            _create_box_body(
                stage=stage,
                body_path=f"{switch_root_path}/rocker",
                size_xyz=(0.044, 0.007, 0.044),
                position_xyz=(0.0, 0.009, 0.0),
                color_rgb=(0.92, 0.92, 0.92),
                mass=0.08,
                kinematic=False,
            )
            _create_fixed_joint_to_world(stage, f"{switch_root_path}/world_fix", f"{switch_root_path}/base")
            _create_revolute_joint(
                stage=stage,
                joint_path=f"{switch_root_path}/hinge",
                body0_path=f"{switch_root_path}/base",
                body1_path=f"{switch_root_path}/rocker",
                local_pos0=(0.0, 0.0055, 0.0),
                local_pos1=(0.0, -0.0035, 0.0),
            )

        _cache_switch_handles_for_env(env, env_i, switch_root_path)

    missing_ids = env_ids[~env._ls_switch_ready[env_ids]]
    if missing_ids.numel() > 0:
        raise RuntimeError(f"Light-switch setup failed for environment IDs: {missing_ids.cpu().tolist()}")

    env_origins_xy = _scene_env_origins_xy(env)[env_ids]
    local_x = math_utils.sample_uniform(
        float(switch_x_range[0]), float(switch_x_range[1]), (env_ids.shape[0],), device=env.device
    )
    local_y = math_utils.sample_uniform(
        float(switch_y_range[0]), float(switch_y_range[1]), (env_ids.shape[0],), device=env.device
    )
    center_z = math_utils.sample_uniform(
        float(switch_height_range[0]),
        float(switch_height_range[1]),
        (env_ids.shape[0],),
        device=env.device,
    )

    switch_center_local = torch.stack((local_x, local_y, center_z), dim=1)
    switch_center_w = torch.stack((env_origins_xy[:, 0] + local_x, env_origins_xy[:, 1] + local_y, center_z), dim=1)
    env._ls_switch_center_w[env_ids] = switch_center_w
    _set_switch_root_wall_transforms_for_envs(
        env=env,
        env_ids=env_ids,
        switch_center_local=switch_center_local,
        switch_yaw=switch_yaw,
        wall_x_offset=wall_x_offset,
    )


# Robot geometry and episode reset


def _left_foot_pos_w(
    env: ManagerBasedRLEnv,
    foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="FL_foot.*"),
) -> torch.Tensor:
    robot: Articulation = env.scene[foot_cfg.name]
    return robot.data.body_pos_w[:, foot_cfg.body_ids, :].mean(dim=1)


def _left_foot_press_target_w(env: ManagerBasedRLEnv, press_depth: float = 0.03) -> torch.Tensor:
    """Target slightly through the rocker so Cartesian shaping produces contact force."""
    target = env._ls_switch_center_w.clone()
    target[:, 0] += float(press_depth)
    return target


def _front_foot_pos_w(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Front-left and front-right foot positions in deterministic order."""
    robot: Articulation = env.scene[robot_cfg.name]
    if not hasattr(env, "_ls_front_waypoint_body_ids"):
        env._ls_front_waypoint_body_ids = robot.find_bodies(
            ["FL_foot.*", "FR_foot.*"], preserve_order=True
        )[0]
    return robot.data.body_pos_w[:, env._ls_front_waypoint_body_ids, :]


def _reset_switch_episode_state(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    initial_on: torch.Tensor,
    target_on: torch.Tensor,
):
    env._ls_current_state[env_ids] = initial_on
    env._ls_target_state[env_ids] = target_on
    env._ls_success[env_ids] = False
    env._ls_contact[env_ids] = False
    env._ls_rocker_contact_force[env_ids] = 0.0
    env._ls_first_contact_seen[env_ids] = False
    env._ls_first_contact_trigger[env_ids] = False
    env._ls_contact_was_active[env_ids] = False
    env._ls_toggle_trigger[env_ids] = False
    env._ls_press_in_progress[env_ids] = False
    env._ls_snap_boost_time_left_s[env_ids] = 0.0
    env._ls_joint_pos_deg[env_ids] = torch.where(
        initial_on,
        torch.full_like(env._ls_joint_pos_deg[env_ids], float(ON_ANGLE_DEG)),
        torch.full_like(env._ls_joint_pos_deg[env_ids], float(OFF_ANGLE_DEG)),
    )
    env._ls_joint_vel_deg_s[env_ids] = 0.0

    env._ls_interaction_last_step = -1
    env._ls_success_hold_counter[env_ids] = 0
    env._ls_contact_dropout_counter[env_ids] = 0
    env._ls_phase[env_ids] = PHASE_STAND
    env._ls_phase_entered[env_ids] = -1
    env._ls_lift_hold_counter[env_ids] = 0
    env._ls_land_hold_counter[env_ids] = 0
    env._ls_jump_hold_counter[env_ids] = 0
    env._ls_settled_base_z[env_ids] = 0.0
    env._ls_settled_base_xy[env_ids] = 0.0
    env._ls_initial_body_wall_distance[env_ids] = 0.0
    env._ls_crouched_base_z[env_ids] = 0.0
    env._ls_settled_foot_z[env_ids] = 0.0
    env._ls_jump_front_start_w[env_ids] = 0.0
    env._ls_jump_rear_start_w[env_ids] = 0.0
    env._ls_phase_step_counter[env_ids] = 0
    env._ls_rear_ground[env_ids] = True
    env._ls_left_front_air[env_ids] = False
    env._ls_jump_ready[env_ids] = False
    env._ls_jump_ready_seen[env_ids] = False
    env._ls_jump_reach_active[env_ids] = False
    env._ls_crouch_ready_trigger[env_ids] = False
    env._ls_valid_crouch[env_ids] = False
    env._ls_crouch_depth_ok[env_ids] = False
    env._ls_crouch_ground_ok[env_ids] = False
    env._ls_crouch_motion_ok[env_ids] = False
    env._ls_crouch_shift_ok[env_ids] = False
    env._ls_episode_max_crouch[env_ids] = 0.0
    env._ls_episode_max_rise[env_ids] = 0.0
    env._ls_episode_max_front_lift[env_ids] = 0.0
    env._ls_episode_max_foot_forward[env_ids] = 0.0
    env._ls_episode_max_rear_step[env_ids] = 0.0
    env._ls_episode_min_body_wall_distance[env_ids] = 2.0
    env._ls_episode_min_foot_distance[env_ids] = 2.0
    env._ls_episode_min_foot_horizontal_error[env_ids] = 2.0
    env._ls_episode_min_foot_vertical_error[env_ids] = 2.0
    env._ls_episode_max_press_hold_s[env_ids] = 0.0
    env._ls_episode_first_contact_time_s[env_ids] = float(JUMP_PHASE_DURATION_S)
    env._ls_episode_contact_dropouts[env_ids] = 0
    env._ls_phase_last_step = -1

def reset_robot_and_lightswitch(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    robot_forward_range: tuple[float, float] = (0.15, 0.35),
    robot_forward_range_full: tuple[float, float] | None = None,
    robot_lateral_range: tuple[float, float] = (-0.2, 0.2),
    robot_lateral_range_full: tuple[float, float] | None = None,
    robot_yaw_range: tuple[float, float] = (-0.2, 0.2),
    wall_x_offset: float = 0.036,
    wall_thickness: float = 0.05,
    head_center_x: float = 0.285,
    head_collision_radius: float = 0.05,
    wall_spawn_clearance: float = 0.015,
    robot_velocity_range: dict[str, tuple[float, float]] | None = None,
):
    if robot_forward_range_full is None:
        robot_forward_range_full = robot_forward_range
    if robot_lateral_range_full is None:
        robot_lateral_range_full = robot_lateral_range
    if robot_velocity_range is None:
        robot_velocity_range = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }

    env_ids = _to_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return
    _ensure_switch_buffers(env)

    # In direct-GPU mode, do not author USD articulation/joint state during runtime resets.
    # Stage creation and transforms are handled in prestartup only.

    robot: Articulation = env.scene[robot_cfg.name]
    num_resets = int(env_ids.shape[0])
    device = env.device

    # Use the authoritative physical switch pose sampled during prestartup.
    center_x = env._ls_switch_center_w[env_ids, 0].clone()
    center_y = env._ls_switch_center_w[env_ids, 1].clone()
    missing_mask = ~env._ls_switch_ready[env_ids]
    if torch.any(missing_mask):
        raise RuntimeError(f"Reset requested without switches for environment IDs: {env_ids[missing_mask].cpu().tolist()}")

    # Random initial ON/OFF and opposite target side.
    initial_on = torch.rand(num_resets, device=device) > 0.5
    target_on = ~initial_on
    _reset_switch_episode_state(env=env, env_ids=env_ids, initial_on=initial_on, target_on=target_on)

    # Begin close and centered, then widen reset randomization as completed press
    # holds become reliable. Switch height remains fully randomized at startup.
    difficulty = torch.clamp(env._ls_behavior_difficulty, 0.0, 1.0)
    forward_min = float(robot_forward_range[0]) + difficulty * (
        float(robot_forward_range_full[0]) - float(robot_forward_range[0])
    )
    forward_max = float(robot_forward_range[1]) + difficulty * (
        float(robot_forward_range_full[1]) - float(robot_forward_range[1])
    )
    lateral_min = float(robot_lateral_range[0]) + difficulty * (
        float(robot_lateral_range_full[0]) - float(robot_lateral_range[0])
    )
    lateral_max = float(robot_lateral_range[1]) + difficulty * (
        float(robot_lateral_range_full[1]) - float(robot_lateral_range[1])
    )
    robot_yaw = math_utils.sample_uniform(
        float(robot_yaw_range[0]), float(robot_yaw_range[1]), (num_resets,), device=device
    )
    forward_dist = math_utils.sample_uniform(
        forward_min, forward_max, (num_resets,), device=device
    )
    # The Go2 head collision is centered well ahead of the base. Clamp every
    # sampled distance against its yaw-dependent forward projection so reset
    # poses retain a physical gap from the wall instead of starting overlapped.
    wall_near_face_offset = float(wall_x_offset) - 0.5 * float(wall_thickness)
    head_forward_extent = (
        abs(float(head_center_x)) * torch.abs(torch.cos(robot_yaw))
        + float(head_collision_radius)
    )
    minimum_forward_dist = (
        head_forward_extent - wall_near_face_offset + float(wall_spawn_clearance)
    )
    forward_dist = torch.maximum(forward_dist, minimum_forward_dist)
    lateral = math_utils.sample_uniform(
        lateral_min, lateral_max, (num_resets,), device=device
    )

    robot_x = center_x - forward_dist
    # Keep the robot near the environment center rather than recentering it on
    # the switch. This makes sampled switch Y a real left/right task variation.
    robot_y = _scene_env_origins_xy(env)[env_ids, 1] + lateral
    robot_root_state = robot.data.default_root_state[env_ids].clone()
    robot_pos = torch.cat((robot_x.unsqueeze(1), robot_y.unsqueeze(1), robot_root_state[:, 2:3]), dim=1)
    env._ls_wall_near_face_x[env_ids] = center_x + wall_near_face_offset
    env._ls_initial_body_wall_distance[env_ids] = torch.abs(
        env._ls_wall_near_face_x[env_ids] - robot_x
    )

    robot_quat = math_utils.quat_from_euler_xyz(
        torch.zeros(num_resets, device=device),
        torch.zeros(num_resets, device=device),
        robot_yaw,
    )

    range_list = [robot_velocity_range.get(k, (0.0, 0.0)) for k in ["x", "y", "z", "roll", "pitch", "yaw"]]
    vel_ranges = torch.tensor(range_list, device=device, dtype=torch.float32)
    robot_vel = math_utils.sample_uniform(
        vel_ranges[:, 0],
        vel_ranges[:, 1],
        (num_resets, 6),
        device=device,
    )

    robot.write_root_pose_to_sim(torch.cat((robot_pos, robot_quat), dim=1), env_ids=env_ids)
    robot.write_root_velocity_to_sim(robot_vel, env_ids=env_ids)


# Switch contact and latch state


def _update_switch_latch_and_interaction(
    env: ManagerBasedRLEnv,
    foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="FL_foot.*"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("fl_rocker_contact", body_names="FL_foot.*"),
    force_threshold: float = 2.0,
):
    _ensure_switch_buffers(env)
    current_step = int(env.common_step_counter)
    if getattr(env, "_ls_interaction_last_step", -1) == current_step:
        return
    dt = _env_step_time_s(env)
    env._ls_first_contact_trigger.fill_(False)

    # The dedicated sensor supplies only FL-foot/rocker contact.
    foot_pos_w = _left_foot_pos_w(env, foot_cfg=foot_cfg)
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    # This force matrix is filtered to the rocker, excluding floor/wall forces.
    if sensor.data.force_matrix_w is None:
        raise RuntimeError("The light-switch contact sensor must filter against the rocker prim.")
    foot_force = torch.linalg.norm(
        sensor.data.force_matrix_w[:, sensor_cfg.body_ids, :, :], dim=3
    ).amax(dim=(1, 2))
    env._ls_rocker_contact_force = foot_force
    rel = foot_pos_w - env._ls_switch_center_w
    dz = rel[:, 2]
    # The sensor is filtered to the rocker, so measured force is stronger evidence
    # than a body-origin proximity box (which can reject valid edge contacts).
    physical_contact = foot_force >= float(force_threshold)
    top_touch = physical_contact & (dz > 0.0)

    # V1 accepts either rocker half, but only after the ordered jump milestone.
    # The touched half chooses the visual rocker direction; it is not a task command.
    valid_phase = (
        (env._ls_phase == PHASE_JUMP)
        & env._ls_rear_ground
        & env._ls_jump_ready_seen
    )
    holding_contact = physical_contact & valid_phase & (~env._ls_success)
    first_contact = holding_contact & (~env._ls_first_contact_seen)
    env._ls_first_contact_trigger[first_contact] = True
    first_contact_time_s = env._ls_phase_step_counter.float() * dt
    env._ls_episode_first_contact_time_s = torch.where(
        first_contact,
        first_contact_time_s,
        env._ls_episode_first_contact_time_s,
    )
    env._ls_first_contact_seen = env._ls_first_contact_seen | holding_contact

    active_attempt = valid_phase & (~env._ls_success)
    lost_contact = active_attempt & (~physical_contact)
    new_dropout = lost_contact & env._ls_contact_was_active
    env._ls_episode_contact_dropouts += new_dropout.long()
    env._ls_contact_was_active = holding_contact
    env._ls_contact_dropout_counter = torch.where(
        holding_contact,
        torch.zeros_like(env._ls_contact_dropout_counter),
        torch.where(
            lost_contact,
            env._ls_contact_dropout_counter + 1,
            torch.zeros_like(env._ls_contact_dropout_counter),
        ),
    )
    grace_steps = _duration_steps(PRESS_CONTACT_GRACE_TIME_S, dt)
    grace_expired = env._ls_contact_dropout_counter > grace_steps
    env._ls_success_hold_counter = torch.where(
        holding_contact,
        env._ls_success_hold_counter + 1,
        torch.where(
            active_attempt & (~grace_expired),
            env._ls_success_hold_counter,
            torch.zeros_like(env._ls_success_hold_counter),
        ),
    )
    measured_hold_s = torch.clamp(
        env._ls_success_hold_counter.float() * dt,
        max=float(PRESS_HOLD_TIME_S),
    )
    hold_target_s = torch.clamp(
        env._ls_adaptive_press_hold_s,
        min=dt,
        max=float(PRESS_HOLD_TIME_S),
    )
    completed_hold = holding_contact & (measured_hold_s >= hold_target_s)
    env._ls_episode_max_press_hold_s = torch.maximum(
        env._ls_episode_max_press_hold_s,
        measured_hold_s,
    )

    if torch.any(holding_contact):
        touched_state = top_touch[holding_contact]
        env._ls_target_state[holding_contact] = touched_state
        env._ls_press_in_progress[holding_contact] = True
        env._ls_snap_boost_time_left_s[holding_contact] = float(SNAP_BOOST_TIME_S)

    # Brief filtered-contact dropouts pause the timer; a dropout longer than the
    # grace window resets it.  Only measured contact steps advance the timer.
    env._ls_success[completed_hold] = True

    step_dt = float(dt)
    commanded_state = torch.where(env._ls_press_in_progress, env._ls_target_state, env._ls_current_state)
    target_angle = torch.where(
        commanded_state,
        torch.full_like(env._ls_joint_pos_deg, float(ON_ANGLE_DEG)),
        torch.full_like(env._ls_joint_pos_deg, float(OFF_ANGLE_DEG)),
    )
    prev_angle = env._ls_joint_pos_deg.clone()
    fast_rate = float(SNAP_BOOST_VEL_DEG_S)
    slow_rate = 30.0
    max_rate = torch.where(
        env._ls_snap_boost_time_left_s > 0.0,
        torch.full_like(env._ls_joint_pos_deg, fast_rate),
        torch.full_like(env._ls_joint_pos_deg, slow_rate),
    )
    max_delta = max_rate * step_dt
    angle_error = target_angle - prev_angle
    angle_delta = torch.clamp(angle_error, min=-max_delta, max=max_delta)
    env._ls_joint_pos_deg = prev_angle + angle_delta
    env._ls_joint_vel_deg_s = angle_delta / max(step_dt, 1e-6)
    env._ls_snap_boost_time_left_s = torch.clamp(env._ls_snap_boost_time_left_s - step_dt, min=0.0)

    crossed_threshold = torch.where(
        env._ls_target_state,
        env._ls_joint_pos_deg >= float(SWITCH_ON_THRESHOLD_DEG),
        env._ls_joint_pos_deg <= float(SWITCH_OFF_THRESHOLD_DEG),
    )
    reached_target = env._ls_press_in_progress & crossed_threshold
    env._ls_current_state[reached_target] = env._ls_target_state[reached_target]
    env._ls_press_in_progress[reached_target] = False
    env._ls_toggle_trigger = completed_hold

    env._ls_contact = physical_contact
    env._ls_interaction_last_step = current_step


# Observations and terminations


def switch_center_position_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Switch center relative to the base, expressed in the robot body frame."""
    _ensure_switch_buffers(env)
    robot: Articulation = env.scene[robot_cfg.name]
    return quat_apply_inverse(robot.data.root_quat_w, env._ls_switch_center_w - robot.data.root_pos_w)


def left_foot_to_switch_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="FL_foot.*"),
) -> torch.Tensor:
    """Switch-minus-front-left-foot vector expressed in the robot body frame."""
    _ensure_switch_buffers(env)
    robot: Articulation = env.scene[robot_cfg.name]
    foot_to_switch_w = env._ls_switch_center_w - _left_foot_pos_w(env, foot_cfg=foot_cfg)
    return quat_apply_inverse(robot.data.root_quat_w, foot_to_switch_w)


def privileged_switch_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Simulator-only rocker state for the asymmetric critic."""
    _ensure_switch_buffers(env)
    return torch.stack(
        (
            env._ls_joint_pos_deg,
            env._ls_joint_vel_deg_s,
            env._ls_contact.float(),
            env._ls_success.float(),
        ),
        dim=1,
    )


def switch_contact_proxy_obs(
    env: ManagerBasedRLEnv,
    foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="FL_foot.*"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("fl_rocker_contact", body_names="FL_foot.*"),
    force_threshold: float = 2.0,
) -> torch.Tensor:
    _update_switch_latch_and_interaction(
        env=env,
        foot_cfg=foot_cfg,
        sensor_cfg=sensor_cfg,
        force_threshold=force_threshold,
    )
    return env._ls_contact.float().unsqueeze(1)


def behavior_phase_one_hot(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Ordered task phase for the memoryless actor and critic."""
    _ensure_switch_buffers(env)
    return torch.nn.functional.one_hot(env._ls_phase, num_classes=NUM_BEHAVIOR_PHASES).float()


def behavior_phase_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Normalized clock within the current maneuver phase for waypoint tracking."""
    _ensure_switch_buffers(env)
    dt = _env_step_time_s(env)
    duration = torch.ones(env.num_envs, device=env.device)
    duration = torch.where(
        env._ls_phase == PHASE_STAND,
        torch.full_like(duration, float(STAND_PHASE_DURATION_S)),
        duration,
    )
    duration = torch.where(
        env._ls_phase == PHASE_LIFT,
        torch.full_like(duration, float(CROUCH_PHASE_TIMEOUT_S)),
        duration,
    )
    duration = torch.where(
        env._ls_phase == PHASE_JUMP,
        torch.full_like(duration, float(JUMP_PHASE_DURATION_S)),
        duration,
    )
    duration = torch.where(
        env._ls_phase == PHASE_LAND,
        torch.full_like(duration, float(LAND_PHASE_HOLD_TIME_S)),
        duration,
    )
    progress = torch.clamp(env._ls_phase_step_counter.float() * dt / duration, 0.0, 1.0)
    return progress.unsqueeze(1)


def phase_aware_bad_orientation(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    nominal_limit_angle: float = 1.0,
    maneuver_pitch_limit_angle: float = 1.40,
    maneuver_roll_limit_angle: float = 0.80,
) -> torch.Tensor:
    """Allow extra sideways roll during the jump, press, and landing maneuver."""
    _ensure_switch_buffers(env)
    robot: Articulation = env.scene[robot_cfg.name]
    tilt_angle = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0)).abs()
    roll, pitch, _ = math_utils.euler_xyz_from_quat(robot.data.root_quat_w)
    maneuvering = (
        (env._ls_phase == PHASE_JUMP)
        | (env._ls_phase == PHASE_LAND)
    )
    maneuver_bad = (
        (torch.abs(roll) > float(maneuver_roll_limit_angle))
        | (torch.abs(pitch) > float(maneuver_pitch_limit_angle))
    )
    nominal_bad = tilt_angle > float(nominal_limit_angle)
    return torch.where(maneuvering, maneuver_bad, nominal_bad)


def _feet_grounded(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    min_vertical_force: float = 1.0,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    vertical_force = torch.abs(sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    return vertical_force >= float(min_vertical_force)


# Ordered maneuver state machine


def _update_behavior_phase(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="FL_foot.*"),
    other_feet_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["RL_foot.*", "RR_foot.*"]
    ),
    foot_sensor_cfg: SceneEntityCfg = SceneEntityCfg("fl_rocker_contact", body_names="FL_foot.*"),
    other_feet_sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "contact_forces", body_names=["RL_foot.*", "RR_foot.*"]
    ),
    front_feet_sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "contact_forces", body_names=["FL_foot.*", "FR_foot.*"]
    ),
    all_feet_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot.*"),
    settle_time_s: float = 1.0,
    lift_height: float = 0.30,
    lift_hold_time_s: float = 0.10,
    jump_base_rise: float = 0.15,
    land_hold_time_s: float = 0.50,
):
    """Advance one rear-supported stand, crouch, reach, press-hold, and recovery attempt."""
    _ensure_switch_buffers(env)
    current_step = int(env.common_step_counter)
    if getattr(env, "_ls_phase_last_step", -1) == current_step:
        return

    robot: Articulation = env.scene[robot_cfg.name]
    old_phase = env._ls_phase.clone()
    env._ls_phase_entered.fill_(-1)
    env._ls_crouch_ready_trigger.fill_(False)
    dt = _env_step_time_s(env)
    env._ls_phase_step_counter += 1

    planar_speed = torch.linalg.norm(robot.data.root_lin_vel_w[:, :2], dim=1)
    vertical_speed = torch.abs(robot.data.root_lin_vel_w[:, 2])
    angular_speed = torch.linalg.norm(robot.data.root_ang_vel_w, dim=1)
    tilt = torch.linalg.norm(robot.data.projected_gravity_b[:, :2], dim=1)

    front_ground = _feet_grounded(env, front_feet_sensor_cfg)
    left_front_air = ~front_ground[:, 0]
    # One planted rear foot is enough support. Requiring both feet prevented the
    # policy from taking a short alternating step toward the wall.
    rear_ground = _feet_grounded(env, other_feet_sensor_cfg).any(dim=1)
    env._ls_rear_ground = rear_ground
    env._ls_left_front_air = left_front_air
    all_ground = _feet_grounded(env, all_feet_sensor_cfg).all(dim=1)
    left_foot_z = _left_foot_pos_w(env, foot_cfg=foot_cfg)[:, 2]

    # The action term holds the exact default pose during this window.  Transition
    # on elapsed time so exploration cannot lock every environment in STAND.
    stand_mask = old_phase == PHASE_STAND
    settle_steps = _duration_steps(settle_time_s, dt)
    enter_crouch = stand_mask & (env._ls_phase_step_counter >= settle_steps)
    if torch.any(enter_crouch):
        env._ls_phase[enter_crouch] = PHASE_LIFT
        env._ls_phase_entered[enter_crouch] = PHASE_LIFT
        env._ls_phase_step_counter[enter_crouch] = 0
        env._ls_settled_base_z[enter_crouch] = robot.data.root_pos_w[enter_crouch, 2]
        env._ls_settled_base_xy[enter_crouch] = robot.data.root_pos_w[enter_crouch, :2]
        env._ls_settled_foot_z[enter_crouch] = left_foot_z[enter_crouch]

    # Preload all four legs.  A useful crouch advances early; the bounded timeout
    # guarantees that imperfect early policies still see rise/reach rewards.
    crouch_mask = old_phase == PHASE_LIFT
    crouch_depth = env._ls_settled_base_z - robot.data.root_pos_w[:, 2]
    switch_direction_xy = env._ls_switch_center_w[:, :2] - env._ls_settled_base_xy
    switch_direction_xy = switch_direction_xy / torch.clamp(
        torch.linalg.norm(switch_direction_xy, dim=1, keepdim=True), min=1e-6
    )
    crouch_forward_shift = torch.sum(
        (robot.data.root_pos_w[:, :2] - env._ls_settled_base_xy) * switch_direction_xy,
        dim=1,
    )
    env._ls_episode_max_crouch = torch.where(
        crouch_mask,
        torch.maximum(env._ls_episode_max_crouch, torch.clamp(crouch_depth, min=0.0)),
        env._ls_episode_max_crouch,
    )
    crouch_depth_ok = (crouch_depth >= 0.045) & (crouch_depth <= 0.095)
    crouch_ground_ok = all_ground
    crouch_motion_ok = (
        (planar_speed <= 0.20)
        & (vertical_speed <= 0.20)
        & (angular_speed <= 0.65)
    )
    crouch_shift_ok = (crouch_forward_shift >= -0.03) & (crouch_forward_shift <= 0.05)
    # Only the preload depth and grounded stance gate progression.  Motion and
    # fore-aft shift remain diagnostics/reward signals, not hard prerequisites.
    quiet_crouch = crouch_mask & crouch_depth_ok & crouch_ground_ok
    env._ls_lift_hold_counter = torch.where(
        quiet_crouch,
        env._ls_lift_hold_counter + 1,
        torch.where(crouch_mask, torch.zeros_like(env._ls_lift_hold_counter), env._ls_lift_hold_counter),
    )
    crouch_hold_steps = _duration_steps(CROUCH_HOLD_TIME_S, dt)
    crouch_ready = crouch_mask & (env._ls_lift_hold_counter >= crouch_hold_steps)
    crouch_timeout_steps = _duration_steps(lift_hold_time_s, dt)
    enter_jump = crouch_mask & (crouch_ready | (env._ls_phase_step_counter >= crouch_timeout_steps))
    if torch.any(enter_jump):
        env._ls_crouch_ready_trigger[enter_jump] = crouch_ready[enter_jump]
        env._ls_valid_crouch[enter_jump] = crouch_ready[enter_jump]
        env._ls_crouch_depth_ok[enter_jump] = crouch_depth_ok[enter_jump]
        env._ls_crouch_ground_ok[enter_jump] = crouch_ground_ok[enter_jump]
        env._ls_crouch_motion_ok[enter_jump] = crouch_motion_ok[enter_jump]
        env._ls_crouch_shift_ok[enter_jump] = crouch_shift_ok[enter_jump]
        env._ls_phase[enter_jump] = PHASE_JUMP
        env._ls_phase_entered[enter_jump] = PHASE_JUMP
        env._ls_phase_step_counter[enter_jump] = 0
        env._ls_crouched_base_z[enter_jump] = robot.data.root_pos_w[enter_jump, 2]
        env._ls_settled_foot_z[enter_jump] = left_foot_z[enter_jump]
        env._ls_jump_front_start_w[enter_jump] = _front_foot_pos_w(env, robot_cfg)[enter_jump]
        env._ls_jump_rear_start_w[enter_jump] = robot.data.body_pos_w[
            :, other_feet_cfg.body_ids, :
        ][enter_jump]

    jump_mask = old_phase == PHASE_JUMP
    base_rise = robot.data.root_pos_w[:, 2] - env._ls_crouched_base_z
    left_foot_lift = left_foot_z - env._ls_settled_foot_z
    front_lift = left_foot_lift
    env._ls_episode_max_rise = torch.where(
        jump_mask,
        torch.maximum(env._ls_episode_max_rise, torch.clamp(base_rise, min=0.0)),
        env._ls_episode_max_rise,
    )
    env._ls_episode_max_front_lift = torch.where(
        jump_mask,
        torch.maximum(env._ls_episode_max_front_lift, torch.clamp(front_lift, min=0.0)),
        env._ls_episode_max_front_lift,
    )
    left_foot_pos_w = _left_foot_pos_w(env, foot_cfg=foot_cfg)
    foot_error_w = left_foot_pos_w - env._ls_switch_center_w
    foot_distance = torch.linalg.norm(foot_error_w, dim=1)
    foot_horizontal_error = torch.linalg.norm(foot_error_w[:, :2], dim=1)
    foot_vertical_error = torch.abs(foot_error_w[:, 2])
    foot_forward = torch.sum(
        (left_foot_pos_w[:, :2] - env._ls_jump_front_start_w[:, 0, :2])
        * switch_direction_xy,
        dim=1,
    )
    rear_pos_w = robot.data.body_pos_w[:, other_feet_cfg.body_ids, :]
    rear_forward = torch.sum(
        (rear_pos_w[:, :, :2] - env._ls_jump_rear_start_w[:, :, :2])
        * switch_direction_xy.unsqueeze(1),
        dim=2,
    ).amin(dim=1)
    env._ls_episode_min_foot_distance = torch.minimum(env._ls_episode_min_foot_distance, foot_distance)
    env._ls_episode_min_foot_horizontal_error = torch.minimum(
        env._ls_episode_min_foot_horizontal_error, foot_horizontal_error
    )
    env._ls_episode_min_foot_vertical_error = torch.minimum(
        env._ls_episode_min_foot_vertical_error, foot_vertical_error
    )
    env._ls_episode_max_foot_forward = torch.where(
        jump_mask,
        torch.maximum(env._ls_episode_max_foot_forward, torch.clamp(foot_forward, min=0.0)),
        env._ls_episode_max_foot_forward,
    )
    env._ls_episode_max_rear_step = torch.where(
        jump_mask,
        torch.maximum(env._ls_episode_max_rear_step, torch.clamp(rear_forward, min=0.0)),
        env._ls_episode_max_rear_step,
    )
    supported_left_air = left_front_air & rear_ground
    jump_threshold = torch.clamp(env._ls_adaptive_jump_rise, max=float(jump_base_rise))
    lift_threshold = torch.clamp(env._ls_adaptive_foot_lift, max=float(lift_height))
    valid_supported_jump = (
        jump_mask
        & supported_left_air
        & (base_rise >= jump_threshold)
        & (left_foot_lift >= lift_threshold)
    )
    env._ls_jump_hold_counter = torch.where(
        valid_supported_jump,
        env._ls_jump_hold_counter + 1,
        torch.where(jump_mask, torch.zeros_like(env._ls_jump_hold_counter), env._ls_jump_hold_counter),
    )
    jump_hold_steps = _duration_steps(JUMP_READY_HOLD_TIME_S, dt)
    env._ls_jump_ready = valid_supported_jump & (env._ls_jump_hold_counter >= jump_hold_steps)
    env._ls_jump_ready_seen = env._ls_jump_ready_seen | env._ls_jump_ready
    env._ls_jump_reach_active = (
        jump_mask
        & supported_left_air
        & env._ls_jump_ready_seen
    )

    # Approach and contact now happen during the high, rear-supported jump.
    _update_switch_latch_and_interaction(env=env, foot_cfg=foot_cfg, sensor_cfg=foot_sensor_cfg)
    # Always finish the reach on the observable phase clock.  Contact still
    # determines task success, but cannot act as hidden information that tells
    # the policy when to land (the real robot has no switch-contact signal).
    jump_timeout_steps = _duration_steps(JUMP_PHASE_DURATION_S, dt)
    enter_land = jump_mask & (env._ls_phase_step_counter >= jump_timeout_steps)
    env._ls_phase[enter_land] = PHASE_LAND
    env._ls_phase_entered[enter_land] = PHASE_LAND
    env._ls_phase_step_counter[enter_land] = 0

    base_height_error = torch.abs(robot.data.root_pos_w[:, 2] - env._ls_settled_base_z)
    stable_landing = (
        all_ground
        & (planar_speed <= 0.10)
        & (vertical_speed <= 0.10)
        & (angular_speed <= 0.35)
        & (tilt <= 0.20)
        & (base_height_error <= 0.10)
    )
    land_mask = old_phase == PHASE_LAND
    env._ls_land_hold_counter = torch.where(
        land_mask & stable_landing,
        env._ls_land_hold_counter + 1,
        torch.where(land_mask, torch.zeros_like(env._ls_land_hold_counter), env._ls_land_hold_counter),
    )
    land_steps = _duration_steps(land_hold_time_s, dt)
    enter_success = land_mask & (env._ls_land_hold_counter >= land_steps)
    env._ls_phase[enter_success] = PHASE_SUCCESS
    env._ls_phase_entered[enter_success] = PHASE_SUCCESS
    env._ls_phase_step_counter[enter_success] = 0
    env._ls_phase_last_step = current_step


# Rewards


def phase_milestone_reward(
    env: ManagerBasedRLEnv,
    target_phase: int,
    robot_cfg: SceneEntityCfg,
    foot_cfg: SceneEntityCfg,
    other_feet_cfg: SceneEntityCfg,
    foot_sensor_cfg: SceneEntityCfg,
    other_feet_sensor_cfg: SceneEntityCfg,
    front_feet_sensor_cfg: SceneEntityCfg,
    all_feet_sensor_cfg: SceneEntityCfg,
    settle_time_s: float,
    lift_height: float,
    lift_hold_time_s: float,
    jump_base_rise: float,
    land_hold_time_s: float,
) -> torch.Tensor:
    _update_behavior_phase(
        env, robot_cfg, foot_cfg, other_feet_cfg, foot_sensor_cfg, other_feet_sensor_cfg,
        front_feet_sensor_cfg, all_feet_sensor_cfg, settle_time_s, lift_height,
        lift_hold_time_s, jump_base_rise, land_hold_time_s
    )
    return (env._ls_phase_entered == int(target_phase)).float()


def jump_base_progress_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    foot_cfg: SceneEntityCfg,
    other_feet_cfg: SceneEntityCfg,
    foot_sensor_cfg: SceneEntityCfg,
    other_feet_sensor_cfg: SceneEntityCfg,
    front_feet_sensor_cfg: SceneEntityCfg,
    all_feet_sensor_cfg: SceneEntityCfg,
    settle_time_s: float,
    lift_height: float,
    lift_hold_time_s: float,
    jump_base_rise: float,
    land_hold_time_s: float,
) -> torch.Tensor:
    _update_behavior_phase(
        env, robot_cfg, foot_cfg, other_feet_cfg, foot_sensor_cfg, other_feet_sensor_cfg,
        front_feet_sensor_cfg, all_feet_sensor_cfg, settle_time_s, lift_height,
        lift_hold_time_s, jump_base_rise, land_hold_time_s
    )
    robot: Articulation = env.scene[robot_cfg.name]
    rise = robot.data.root_pos_w[:, 2] - env._ls_crouched_base_z
    target_rise = torch.clamp(env._ls_adaptive_jump_rise, min=0.18)
    rise_fraction = torch.clamp(rise / target_rise, 0.0, 1.0)
    if not hasattr(env, "_ls_prev_base_rise"):
        env._ls_prev_base_rise = rise_fraction.clone()
    reset = (env.episode_length_buf == 0) | (env._ls_phase_entered == PHASE_JUMP)
    env._ls_prev_base_rise[reset] = rise_fraction[reset]
    progress = torch.clamp(rise_fraction - env._ls_prev_base_rise, min=-0.25, max=0.25)
    env._ls_prev_base_rise[:] = rise_fraction
    active = (env._ls_phase == PHASE_JUMP) & env._ls_rear_ground
    return active.float() * progress


def early_jump_rise_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reward_window_s: float = 0.80,
) -> torch.Tensor:
    """Reward achieving the required base rise early in the jump phase."""
    _ensure_switch_buffers(env)
    robot: Articulation = env.scene[robot_cfg.name]
    rise = robot.data.root_pos_w[:, 2] - env._ls_crouched_base_z
    target_rise = torch.clamp(env._ls_adaptive_jump_rise, min=0.18)
    rise_fraction = torch.clamp(rise / target_rise, 0.0, 1.0)
    elapsed_s = env._ls_phase_step_counter.float() * _env_step_time_s(env)
    early_window = elapsed_s <= float(reward_window_s)
    active = (env._ls_phase == PHASE_JUMP) & env._ls_rear_ground & early_window
    return active.float() * rise_fraction


def raised_posture_hold_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_std: float = 0.04,
) -> torch.Tensor:
    """Reward holding the required raised base height instead of bouncing through it."""
    _ensure_switch_buffers(env)
    robot: Articulation = env.scene[robot_cfg.name]
    rise = robot.data.root_pos_w[:, 2] - env._ls_crouched_base_z
    target_rise = torch.clamp(env._ls_adaptive_jump_rise, min=0.18, max=0.28)
    score = torch.exp(-torch.square((rise - target_rise) / float(height_std)))
    vertical_score = torch.exp(-torch.square(robot.data.root_lin_vel_w[:, 2] / 0.12))
    active = (env._ls_phase == PHASE_JUMP) & env._ls_rear_ground
    return active.float() * score * vertical_score


def front_body_raise_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_angle_rad: float = 0.50,
) -> torch.Tensor:
    """Reward raising the torso front while also achieving the required base rise."""
    _ensure_switch_buffers(env)
    robot: Articulation = env.scene[robot_cfg.name]
    forward_axis_b = torch.zeros((env.num_envs, 3), device=env.device)
    forward_axis_b[:, 0] = 1.0
    forward_axis_w = math_utils.quat_apply(robot.data.root_quat_w, forward_axis_b)
    target_forward_z = max(math.sin(float(target_angle_rad)), 1e-6)
    front_up = torch.clamp(forward_axis_w[:, 2] / target_forward_z, 0.0, 1.0)
    rise = robot.data.root_pos_w[:, 2] - env._ls_crouched_base_z
    rise_fraction = torch.clamp(
        rise / torch.clamp(env._ls_adaptive_jump_rise, min=0.18), 0.0, 1.0
    )
    vertical_score = torch.exp(-torch.square(robot.data.root_lin_vel_w[:, 2] / 0.25))
    angular_score = torch.exp(
        -torch.square(torch.linalg.norm(robot.data.root_ang_vel_w, dim=1) / 0.80)
    )
    active = (env._ls_phase == PHASE_JUMP) & env._ls_rear_ground
    return active.float() * front_up * rise_fraction * vertical_score * angular_score


def crouch_progress_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    foot_cfg: SceneEntityCfg,
    other_feet_cfg: SceneEntityCfg,
    foot_sensor_cfg: SceneEntityCfg,
    other_feet_sensor_cfg: SceneEntityCfg,
    front_feet_sensor_cfg: SceneEntityCfg,
    all_feet_sensor_cfg: SceneEntityCfg,
    settle_time_s: float,
    lift_height: float,
    lift_hold_time_s: float,
    jump_base_rise: float,
    land_hold_time_s: float,
) -> torch.Tensor:
    """Potential-based reward for a 6 cm four-foot preload crouch."""
    _update_behavior_phase(
        env, robot_cfg, foot_cfg, other_feet_cfg, foot_sensor_cfg, other_feet_sensor_cfg,
        front_feet_sensor_cfg, all_feet_sensor_cfg, settle_time_s, lift_height,
        lift_hold_time_s, jump_base_rise, land_hold_time_s
    )
    robot: Articulation = env.scene[robot_cfg.name]
    depth = torch.clamp((env._ls_settled_base_z - robot.data.root_pos_w[:, 2]) / 0.06, 0.0, 1.0)
    if not hasattr(env, "_ls_prev_crouch_depth"):
        env._ls_prev_crouch_depth = depth.clone()
    reset = (env.episode_length_buf == 0) | (env._ls_phase_entered == PHASE_LIFT)
    env._ls_prev_crouch_depth[reset] = depth[reset]
    progress = depth - env._ls_prev_crouch_depth
    env._ls_prev_crouch_depth[:] = depth
    grounded = _feet_grounded(env, all_feet_sensor_cfg).all(dim=1)
    return (env._ls_phase == PHASE_LIFT).float() * grounded.float() * progress


def crouch_pose_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    foot_cfg: SceneEntityCfg,
    other_feet_cfg: SceneEntityCfg,
    foot_sensor_cfg: SceneEntityCfg,
    other_feet_sensor_cfg: SceneEntityCfg,
    front_feet_sensor_cfg: SceneEntityCfg,
    all_feet_sensor_cfg: SceneEntityCfg,
    settle_time_s: float,
    lift_height: float,
    lift_hold_time_s: float,
    jump_base_rise: float,
    land_hold_time_s: float,
) -> torch.Tensor:
    """Reward a quiet 6 cm preload rather than fast downward motion."""
    _update_behavior_phase(
        env, robot_cfg, foot_cfg, other_feet_cfg, foot_sensor_cfg, other_feet_sensor_cfg,
        front_feet_sensor_cfg, all_feet_sensor_cfg, settle_time_s, lift_height,
        lift_hold_time_s, jump_base_rise, land_hold_time_s
    )
    robot: Articulation = env.scene[robot_cfg.name]
    depth = env._ls_settled_base_z - robot.data.root_pos_w[:, 2]
    depth_score = torch.exp(-torch.square((depth - 0.06) / 0.018))
    vertical_score = torch.exp(-torch.square(robot.data.root_lin_vel_w[:, 2] / 0.15))
    angular_score = torch.exp(-torch.square(torch.linalg.norm(robot.data.root_ang_vel_w, dim=1) / 0.50))
    all_ground = _feet_grounded(env, all_feet_sensor_cfg).all(dim=1)
    return (
        (env._ls_phase == PHASE_LIFT).float()
        * all_ground.float()
        * depth_score
        * vertical_score
        * angular_score
    )


def crouch_forward_position_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    foot_cfg: SceneEntityCfg,
    other_feet_cfg: SceneEntityCfg,
    foot_sensor_cfg: SceneEntityCfg,
    other_feet_sensor_cfg: SceneEntityCfg,
    front_feet_sensor_cfg: SceneEntityCfg,
    all_feet_sensor_cfg: SceneEntityCfg,
    settle_time_s: float,
    lift_height: float,
    lift_hold_time_s: float,
    jump_base_rise: float,
    land_hold_time_s: float,
) -> torch.Tensor:
    """Prefer a stationary-to-slightly-forward base during the preload crouch."""
    _update_behavior_phase(
        env, robot_cfg, foot_cfg, other_feet_cfg, foot_sensor_cfg, other_feet_sensor_cfg,
        front_feet_sensor_cfg, all_feet_sensor_cfg, settle_time_s, lift_height,
        lift_hold_time_s, jump_base_rise, land_hold_time_s
    )
    robot: Articulation = env.scene[robot_cfg.name]
    direction_xy = env._ls_switch_center_w[:, :2] - env._ls_settled_base_xy
    direction_xy = direction_xy / torch.clamp(
        torch.linalg.norm(direction_xy, dim=1, keepdim=True), min=1e-6
    )
    forward_shift = torch.sum(
        (robot.data.root_pos_w[:, :2] - env._ls_settled_base_xy) * direction_xy,
        dim=1,
    )
    # A 3 cm target makes the desired 2--4 cm band nearly optimal while still
    # giving partial credit for holding the original forward position.
    position_score = torch.exp(-torch.square((forward_shift - 0.03) / 0.03))
    all_ground = _feet_grounded(env, all_feet_sensor_cfg).all(dim=1)
    return (env._ls_phase == PHASE_LIFT).float() * all_ground.float() * position_score


def crouch_backward_drift_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    foot_cfg: SceneEntityCfg,
    other_feet_cfg: SceneEntityCfg,
    foot_sensor_cfg: SceneEntityCfg,
    other_feet_sensor_cfg: SceneEntityCfg,
    front_feet_sensor_cfg: SceneEntityCfg,
    all_feet_sensor_cfg: SceneEntityCfg,
    settle_time_s: float,
    lift_height: float,
    lift_hold_time_s: float,
    jump_base_rise: float,
    land_hold_time_s: float,
) -> torch.Tensor:
    """Penalize only the portion of crouch displacement beyond 3 cm backward."""
    _update_behavior_phase(
        env, robot_cfg, foot_cfg, other_feet_cfg, foot_sensor_cfg, other_feet_sensor_cfg,
        front_feet_sensor_cfg, all_feet_sensor_cfg, settle_time_s, lift_height,
        lift_hold_time_s, jump_base_rise, land_hold_time_s
    )
    robot: Articulation = env.scene[robot_cfg.name]
    direction_xy = env._ls_switch_center_w[:, :2] - env._ls_settled_base_xy
    direction_xy = direction_xy / torch.clamp(
        torch.linalg.norm(direction_xy, dim=1, keepdim=True), min=1e-6
    )
    forward_shift = torch.sum(
        (robot.data.root_pos_w[:, :2] - env._ls_settled_base_xy) * direction_xy,
        dim=1,
    )
    excess_backward = torch.clamp((-forward_shift - 0.03) / 0.05, min=0.0, max=2.0)
    return (env._ls_phase == PHASE_LIFT).float() * torch.square(excess_backward)


def crouch_overshoot_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    foot_cfg: SceneEntityCfg,
    other_feet_cfg: SceneEntityCfg,
    foot_sensor_cfg: SceneEntityCfg,
    other_feet_sensor_cfg: SceneEntityCfg,
    front_feet_sensor_cfg: SceneEntityCfg,
    all_feet_sensor_cfg: SceneEntityCfg,
    settle_time_s: float,
    lift_height: float,
    lift_hold_time_s: float,
    jump_base_rise: float,
    land_hold_time_s: float,
) -> torch.Tensor:
    """Prevent the deep, high-energy crouch seen in run 3_12."""
    _update_behavior_phase(
        env, robot_cfg, foot_cfg, other_feet_cfg, foot_sensor_cfg, other_feet_sensor_cfg,
        front_feet_sensor_cfg, all_feet_sensor_cfg, settle_time_s, lift_height,
        lift_hold_time_s, jump_base_rise, land_hold_time_s
    )
    robot: Articulation = env.scene[robot_cfg.name]
    depth = env._ls_settled_base_z - robot.data.root_pos_w[:, 2]
    overshoot = torch.clamp((depth - 0.08) / 0.05, min=0.0, max=2.0)
    return (env._ls_phase == PHASE_LIFT).float() * torch.square(overshoot)


def left_foot_clearance_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    foot_cfg: SceneEntityCfg,
    other_feet_cfg: SceneEntityCfg,
    foot_sensor_cfg: SceneEntityCfg,
    other_feet_sensor_cfg: SceneEntityCfg,
    front_feet_sensor_cfg: SceneEntityCfg,
    all_feet_sensor_cfg: SceneEntityCfg,
    settle_time_s: float,
    lift_height: float,
    lift_hold_time_s: float,
    jump_base_rise: float,
    land_hold_time_s: float,
) -> torch.Tensor:
    """Absolute clearance reward for the reaching front-left foot."""
    _update_behavior_phase(
        env, robot_cfg, foot_cfg, other_feet_cfg, foot_sensor_cfg, other_feet_sensor_cfg,
        front_feet_sensor_cfg, all_feet_sensor_cfg, settle_time_s, lift_height,
        lift_hold_time_s, jump_base_rise, land_hold_time_s
    )
    left_z = _front_foot_pos_w(env, robot_cfg)[:, 0, 2]
    left_lift = left_z - env._ls_jump_front_start_w[:, 0, 2]
    clearance = torch.clamp(left_lift / max(float(lift_height), 1e-6), 0.0, 1.0)
    left_air = ~_feet_grounded(env, front_feet_sensor_cfg)[:, 0]
    active = (
        (env._ls_phase == PHASE_JUMP)
        & left_air
        & env._ls_rear_ground
    )
    return active.float() * clearance


def left_foot_waypoint_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    foot_cfg: SceneEntityCfg,
    other_feet_cfg: SceneEntityCfg,
    foot_sensor_cfg: SceneEntityCfg,
    other_feet_sensor_cfg: SceneEntityCfg,
    front_feet_sensor_cfg: SceneEntityCfg,
    all_feet_sensor_cfg: SceneEntityCfg,
    settle_time_s: float,
    lift_height: float,
    lift_hold_time_s: float,
    jump_base_rise: float,
    land_hold_time_s: float,
) -> torch.Tensor:
    """Track a lift-first Cartesian path and then close the final rocker gap."""
    _update_behavior_phase(
        env, robot_cfg, foot_cfg, other_feet_cfg, foot_sensor_cfg, other_feet_sensor_cfg,
        front_feet_sensor_cfg, all_feet_sensor_cfg, settle_time_s, lift_height,
        lift_hold_time_s, jump_base_rise, land_hold_time_s
    )
    front_pos = _front_foot_pos_w(env, robot_cfg)
    start = env._ls_jump_front_start_w[:, 0, :]
    lift = start.clone()
    lift[:, 2] += float(lift_height)
    precontact = env._ls_switch_center_w.clone()
    precontact[:, 0] -= 0.05
    t = env._ls_phase_step_counter.float() * _env_step_time_s(env)
    lift_alpha = torch.clamp(t / 0.35, 0.0, 1.0).unsqueeze(1)
    approach_alpha = torch.clamp((t - 0.35) / 0.75, 0.0, 1.0).unsqueeze(1)
    contact_alpha = torch.clamp((t - 1.10) / 0.30, 0.0, 1.0).unsqueeze(1)
    # Smoothstep keeps target velocity continuous at each waypoint boundary.
    lift_alpha = lift_alpha * lift_alpha * (3.0 - 2.0 * lift_alpha)
    approach_alpha = approach_alpha * approach_alpha * (3.0 - 2.0 * approach_alpha)
    contact_alpha = contact_alpha * contact_alpha * (3.0 - 2.0 * contact_alpha)
    reach_ready = env._ls_jump_ready_seen.float().unsqueeze(1)
    contact_alpha = contact_alpha * reach_ready
    waypoint = start + lift_alpha * (lift - start)
    waypoint = waypoint + approach_alpha * (precontact - waypoint)
    press_target = _left_foot_press_target_w(env, press_depth=0.03)
    waypoint = waypoint + contact_alpha * (press_target - waypoint)
    left_error = torch.linalg.norm(front_pos[:, 0, :] - waypoint, dim=1)
    # Keep a broad exploration basin, but make precise final extension much more
    # valuable than stopping 15--25 cm short of the rocker.
    broad_score = torch.exp(-torch.square(left_error / 0.25))
    precise_score = torch.exp(-torch.square(left_error / 0.03))
    ready = env._ls_jump_ready_seen.float()
    left_score = (1.0 - ready) * (0.35 * broad_score + 0.65 * precise_score) + ready * precise_score
    active = (
        (env._ls_phase == PHASE_JUMP)
        & env._ls_rear_ground
    )
    return active.float() * left_score


def left_foot_height_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    foot_cfg: SceneEntityCfg,
    other_feet_cfg: SceneEntityCfg,
    foot_sensor_cfg: SceneEntityCfg,
    other_feet_sensor_cfg: SceneEntityCfg,
    front_feet_sensor_cfg: SceneEntityCfg,
    all_feet_sensor_cfg: SceneEntityCfg,
    settle_time_s: float,
    lift_height: float,
    lift_hold_time_s: float,
    jump_base_rise: float,
    land_hold_time_s: float,
) -> torch.Tensor:
    """Dense absolute reward for raising FL from takeoff height to switch height."""
    _update_behavior_phase(
        env, robot_cfg, foot_cfg, other_feet_cfg, foot_sensor_cfg, other_feet_sensor_cfg,
        front_feet_sensor_cfg, all_feet_sensor_cfg, settle_time_s, lift_height,
        lift_hold_time_s, jump_base_rise, land_hold_time_s
    )
    left_z = _front_foot_pos_w(env, robot_cfg)[:, 0, 2]
    start_z = env._ls_jump_front_start_w[:, 0, 2]
    target_z = env._ls_switch_center_w[:, 2]
    height_fraction = torch.clamp(
        (left_z - start_z) / torch.clamp(target_z - start_z, min=0.10), 0.0, 1.0
    )
    left_air = ~_feet_grounded(env, front_feet_sensor_cfg)[:, 0]
    active = (
        (env._ls_phase == PHASE_JUMP)
        & left_air
        & env._ls_rear_ground
    )
    return active.float() * height_fraction


def front_ground_contact_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    foot_cfg: SceneEntityCfg,
    other_feet_cfg: SceneEntityCfg,
    foot_sensor_cfg: SceneEntityCfg,
    other_feet_sensor_cfg: SceneEntityCfg,
    front_feet_sensor_cfg: SceneEntityCfg,
    all_feet_sensor_cfg: SceneEntityCfg,
    settle_time_s: float,
    lift_height: float,
    lift_hold_time_s: float,
    jump_base_rise: float,
    land_hold_time_s: float,
) -> torch.Tensor:
    """Make repeated front-foot ground strikes costly after the takeoff grace period."""
    _update_behavior_phase(
        env, robot_cfg, foot_cfg, other_feet_cfg, foot_sensor_cfg, other_feet_sensor_cfg,
        front_feet_sensor_cfg, all_feet_sensor_cfg, settle_time_s, lift_height,
        lift_hold_time_s, jump_base_rise, land_hold_time_s
    )
    contacts = _feet_grounded(env, front_feet_sensor_cfg).float().mean(dim=1)
    elapsed = env._ls_phase_step_counter.float() * _env_step_time_s(env)
    active = (env._ls_phase == PHASE_JUMP) & (elapsed > 0.08)
    return active.float() * contacts


def front_ground_impact_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    foot_cfg: SceneEntityCfg,
    other_feet_cfg: SceneEntityCfg,
    foot_sensor_cfg: SceneEntityCfg,
    other_feet_sensor_cfg: SceneEntityCfg,
    front_feet_sensor_cfg: SceneEntityCfg,
    all_feet_sensor_cfg: SceneEntityCfg,
    settle_time_s: float,
    lift_height: float,
    lift_hold_time_s: float,
    jump_base_rise: float,
    land_hold_time_s: float,
) -> torch.Tensor:
    """Penalize downward speed specifically when a front foot strikes the floor."""
    _update_behavior_phase(
        env, robot_cfg, foot_cfg, other_feet_cfg, foot_sensor_cfg, other_feet_sensor_cfg,
        front_feet_sensor_cfg, all_feet_sensor_cfg, settle_time_s, lift_height,
        lift_hold_time_s, jump_base_rise, land_hold_time_s
    )
    robot: Articulation = env.scene[robot_cfg.name]
    if not hasattr(env, "_ls_front_impact_body_ids"):
        env._ls_front_impact_body_ids = robot.find_bodies(
            ["FL_foot.*", "FR_foot.*"], preserve_order=True
        )[0]
    downward_speed = torch.clamp(-robot.data.body_lin_vel_w[:, env._ls_front_impact_body_ids, 2], min=0.0)
    contacts = _feet_grounded(env, front_feet_sensor_cfg).float()
    impact = torch.square(torch.clamp(downward_speed / 1.0, max=3.0)) * contacts
    return (env._ls_phase == PHASE_JUMP).float() * impact.mean(dim=1)


def crouch_complete_milestone_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    foot_cfg: SceneEntityCfg,
    other_feet_cfg: SceneEntityCfg,
    foot_sensor_cfg: SceneEntityCfg,
    other_feet_sensor_cfg: SceneEntityCfg,
    front_feet_sensor_cfg: SceneEntityCfg,
    all_feet_sensor_cfg: SceneEntityCfg,
    settle_time_s: float,
    lift_height: float,
    lift_hold_time_s: float,
    jump_base_rise: float,
    land_hold_time_s: float,
) -> torch.Tensor:
    _update_behavior_phase(
        env, robot_cfg, foot_cfg, other_feet_cfg, foot_sensor_cfg, other_feet_sensor_cfg,
        front_feet_sensor_cfg, all_feet_sensor_cfg, settle_time_s, lift_height,
        lift_hold_time_s, jump_base_rise, land_hold_time_s
    )
    return env._ls_crouch_ready_trigger.float()


def uncrouched_jump_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    foot_cfg: SceneEntityCfg,
    other_feet_cfg: SceneEntityCfg,
    foot_sensor_cfg: SceneEntityCfg,
    other_feet_sensor_cfg: SceneEntityCfg,
    front_feet_sensor_cfg: SceneEntityCfg,
    all_feet_sensor_cfg: SceneEntityCfg,
    settle_time_s: float,
    lift_height: float,
    lift_hold_time_s: float,
    jump_base_rise: float,
    land_hold_time_s: float,
) -> torch.Tensor:
    """One-shot penalty when the crouch timeout advances without a valid preload."""
    _update_behavior_phase(
        env, robot_cfg, foot_cfg, other_feet_cfg, foot_sensor_cfg, other_feet_sensor_cfg,
        front_feet_sensor_cfg, all_feet_sensor_cfg, settle_time_s, lift_height,
        lift_hold_time_s, jump_base_rise, land_hold_time_s
    )
    invalid_entry = (env._ls_phase_entered == PHASE_JUMP) & (~env._ls_valid_crouch)
    return invalid_entry.float()


def touch_milestone_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    foot_cfg: SceneEntityCfg,
    other_feet_cfg: SceneEntityCfg,
    foot_sensor_cfg: SceneEntityCfg,
    other_feet_sensor_cfg: SceneEntityCfg,
    front_feet_sensor_cfg: SceneEntityCfg,
    all_feet_sensor_cfg: SceneEntityCfg,
    settle_time_s: float,
    lift_height: float,
    lift_hold_time_s: float,
    jump_base_rise: float,
    land_hold_time_s: float,
) -> torch.Tensor:
    """One-shot reward after holding filtered rocker contact for one second."""
    _update_behavior_phase(
        env, robot_cfg, foot_cfg, other_feet_cfg, foot_sensor_cfg, other_feet_sensor_cfg,
        front_feet_sensor_cfg, all_feet_sensor_cfg, settle_time_s, lift_height,
        lift_hold_time_s, jump_base_rise, land_hold_time_s
    )
    return env._ls_toggle_trigger.float()


def press_hold_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    foot_cfg: SceneEntityCfg,
    other_feet_cfg: SceneEntityCfg,
    foot_sensor_cfg: SceneEntityCfg,
    other_feet_sensor_cfg: SceneEntityCfg,
    front_feet_sensor_cfg: SceneEntityCfg,
    all_feet_sensor_cfg: SceneEntityCfg,
    settle_time_s: float,
    lift_height: float,
    lift_hold_time_s: float,
    jump_base_rise: float,
    land_hold_time_s: float,
) -> torch.Tensor:
    """Dense feedback for maintaining left-front-foot contact during the hold."""
    _update_behavior_phase(
        env, robot_cfg, foot_cfg, other_feet_cfg, foot_sensor_cfg, other_feet_sensor_cfg,
        front_feet_sensor_cfg, all_feet_sensor_cfg, settle_time_s, lift_height,
        lift_hold_time_s, jump_base_rise, land_hold_time_s
    )
    hold_target_s = torch.clamp(
        env._ls_adaptive_press_hold_s,
        min=_env_step_time_s(env),
        max=float(PRESS_HOLD_TIME_S),
    )
    hold_fraction = torch.clamp(
        env._ls_success_hold_counter.float() * _env_step_time_s(env) / hold_target_s,
        0.0,
        1.0,
    )
    # Increasing value over the uninterrupted hold makes maintaining contact
    # more useful than collecting the same number of disconnected taps.
    hold_score = 0.25 + 0.75 * hold_fraction
    return env._ls_jump_reach_active.float() * env._ls_contact.float() * hold_score


def press_contact_stability_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    foot_cfg: SceneEntityCfg,
    other_feet_cfg: SceneEntityCfg,
    foot_sensor_cfg: SceneEntityCfg,
    other_feet_sensor_cfg: SceneEntityCfg,
    front_feet_sensor_cfg: SceneEntityCfg,
    all_feet_sensor_cfg: SceneEntityCfg,
    settle_time_s: float,
    lift_height: float,
    lift_hold_time_s: float,
    jump_base_rise: float,
    land_hold_time_s: float,
    target_force: float = 12.0,
    force_std: float = 10.0,
    speed_std: float = 0.20,
) -> torch.Tensor:
    """Reward a quiet foot and moderate force while physical rocker contact exists."""
    _update_behavior_phase(
        env, robot_cfg, foot_cfg, other_feet_cfg, foot_sensor_cfg, other_feet_sensor_cfg,
        front_feet_sensor_cfg, all_feet_sensor_cfg, settle_time_s, lift_height,
        lift_hold_time_s, jump_base_rise, land_hold_time_s
    )
    robot: Articulation = env.scene[robot_cfg.name]
    foot_speed = torch.linalg.norm(robot.data.body_lin_vel_w[:, foot_cfg.body_ids, :], dim=2).mean(dim=1)
    speed_score = torch.exp(-torch.square(foot_speed / float(speed_std)))
    force_score = torch.exp(
        -torch.square((env._ls_rocker_contact_force - float(target_force)) / float(force_std))
    )
    active = env._ls_jump_reach_active & env._ls_contact
    return active.float() * speed_score * force_score


def early_press_start_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    foot_cfg: SceneEntityCfg,
    other_feet_cfg: SceneEntityCfg,
    foot_sensor_cfg: SceneEntityCfg,
    other_feet_sensor_cfg: SceneEntityCfg,
    front_feet_sensor_cfg: SceneEntityCfg,
    all_feet_sensor_cfg: SceneEntityCfg,
    settle_time_s: float,
    lift_height: float,
    lift_hold_time_s: float,
    jump_base_rise: float,
    land_hold_time_s: float,
) -> torch.Tensor:
    """Small one-shot bonus that is larger when rocker contact begins earlier."""
    _update_behavior_phase(
        env, robot_cfg, foot_cfg, other_feet_cfg, foot_sensor_cfg, other_feet_sensor_cfg,
        front_feet_sensor_cfg, all_feet_sensor_cfg, settle_time_s, lift_height,
        lift_hold_time_s, jump_base_rise, land_hold_time_s
    )
    elapsed_s = env._ls_phase_step_counter.float() * _env_step_time_s(env)
    remaining_fraction = torch.clamp(
        1.0 - elapsed_s / float(JUMP_PHASE_DURATION_S),
        0.0,
        1.0,
    )
    return env._ls_first_contact_trigger.float() * remaining_fraction


def touch_and_stable_milestone_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    foot_cfg: SceneEntityCfg,
    other_feet_cfg: SceneEntityCfg,
    foot_sensor_cfg: SceneEntityCfg,
    other_feet_sensor_cfg: SceneEntityCfg,
    front_feet_sensor_cfg: SceneEntityCfg,
    all_feet_sensor_cfg: SceneEntityCfg,
    settle_time_s: float,
    lift_height: float,
    lift_hold_time_s: float,
    jump_base_rise: float,
    land_hold_time_s: float,
) -> torch.Tensor:
    """Final task reward only when a completed press hold is followed by a stable landing."""
    _update_behavior_phase(
        env, robot_cfg, foot_cfg, other_feet_cfg, foot_sensor_cfg, other_feet_sensor_cfg,
        front_feet_sensor_cfg, all_feet_sensor_cfg, settle_time_s, lift_height,
        lift_hold_time_s, jump_base_rise, land_hold_time_s
    )
    return ((env._ls_phase_entered == PHASE_SUCCESS) & env._ls_success).float()


def jump_heading_error_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize turning the body away from the switch during the jump."""
    _ensure_switch_buffers(env)
    robot: Articulation = env.scene[robot_cfg.name]
    switch_rel_b = quat_apply_inverse(
        robot.data.root_quat_w, env._ls_switch_center_w - robot.data.root_pos_w
    )
    heading_error = torch.atan2(switch_rel_b[:, 1], switch_rel_b[:, 0])
    return (env._ls_phase == PHASE_JUMP).float() * torch.square(heading_error)


def jump_yaw_rate_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Suppress slow or continuous on-the-spot rotation during the jump."""
    _ensure_switch_buffers(env)
    robot: Articulation = env.scene[robot_cfg.name]
    return (env._ls_phase == PHASE_JUMP).float() * torch.square(robot.data.root_ang_vel_b[:, 2])


def maneuver_roll_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Keep the body laterally upright without suppressing the required pitch."""
    _ensure_switch_buffers(env)
    robot: Articulation = env.scene[robot_cfg.name]
    roll, _, _ = math_utils.euler_xyz_from_quat(robot.data.root_quat_w)
    active = (env._ls_phase == PHASE_LIFT) | (env._ls_phase == PHASE_JUMP) | (env._ls_phase == PHASE_LAND)
    return active.float() * torch.square(roll)


def rear_support_loss_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    foot_cfg: SceneEntityCfg,
    other_feet_cfg: SceneEntityCfg,
    foot_sensor_cfg: SceneEntityCfg,
    other_feet_sensor_cfg: SceneEntityCfg,
    front_feet_sensor_cfg: SceneEntityCfg,
    all_feet_sensor_cfg: SceneEntityCfg,
    settle_time_s: float,
    lift_height: float,
    lift_hold_time_s: float,
    jump_base_rise: float,
    land_hold_time_s: float,
) -> torch.Tensor:
    """Penalize losing both rear supports while allowing an alternating rear step."""
    _update_behavior_phase(
        env, robot_cfg, foot_cfg, other_feet_cfg, foot_sensor_cfg, other_feet_sensor_cfg,
        front_feet_sensor_cfg, all_feet_sensor_cfg, settle_time_s, lift_height,
        lift_hold_time_s, jump_base_rise, land_hold_time_s
    )
    rear_contacts = _feet_grounded(env, other_feet_sensor_cfg).float().sum(dim=1)
    active = env._ls_phase == PHASE_JUMP
    return active.float() * torch.clamp(1.0 - rear_contacts, min=0.0)


def base_forward_progress_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Signed reward for moving the rear-supported base toward the switch."""
    _ensure_switch_buffers(env)
    robot: Articulation = env.scene[robot_cfg.name]
    distance = torch.linalg.norm(
        env._ls_switch_center_w[:, :2] - robot.data.root_pos_w[:, :2], dim=1
    )
    if not hasattr(env, "_ls_prev_base_switch_distance"):
        env._ls_prev_base_switch_distance = distance.clone()
    reset = (env.episode_length_buf == 0) | (env._ls_phase_entered == PHASE_JUMP)
    env._ls_prev_base_switch_distance[reset] = distance[reset]
    progress = torch.clamp(
        env._ls_prev_base_switch_distance - distance,
        min=-0.01,
        max=0.01,
    )
    env._ls_prev_base_switch_distance[:] = distance
    active = (env._ls_phase == PHASE_JUMP) & env._ls_rear_ground
    return active.float() * progress


def body_wall_half_distance_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    distance_std: float = 0.14,
) -> torch.Tensor:
    """Reward an absolute base-to-wall distance equal to half the reset distance."""
    _ensure_switch_buffers(env)
    robot: Articulation = env.scene[robot_cfg.name]
    distance = torch.abs(env._ls_wall_near_face_x - robot.data.root_pos_w[:, 0])
    target_distance = 0.5 * env._ls_initial_body_wall_distance
    # Give a useful gradient from the reset pose and stop increasing once the
    # base reaches the target, rather than rewarding motion through the wall.
    excess_distance = torch.clamp(distance - target_distance, min=0.0)
    score = torch.exp(-torch.square(excess_distance / float(distance_std)))
    env._ls_episode_min_body_wall_distance = torch.minimum(
        env._ls_episode_min_body_wall_distance, distance
    )
    active = (env._ls_phase == PHASE_JUMP) & env._ls_rear_ground
    return active.float() * score


def rear_foot_hard_impact_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    other_feet_cfg: SceneEntityCfg,
    other_feet_sensor_cfg: SceneEntityCfg,
    speed_threshold: float = 0.8,
) -> torch.Tensor:
    """Penalize only hard rear-foot touchdowns, leaving normal stepping unopposed."""
    _ensure_switch_buffers(env)
    robot: Articulation = env.scene[robot_cfg.name]
    downward_speed = torch.clamp(
        -robot.data.body_lin_vel_w[:, other_feet_cfg.body_ids, 2]
        - float(speed_threshold),
        min=0.0,
    )
    contacts = _feet_grounded(env, other_feet_sensor_cfg).float()
    impact = torch.square(torch.clamp(downward_speed, max=2.0)) * contacts
    return (env._ls_phase == PHASE_JUMP).float() * impact.mean(dim=1)


def reach_vertical_velocity_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Damp body bouncing after the required jump height has been reached."""
    _ensure_switch_buffers(env)
    robot: Articulation = env.scene[robot_cfg.name]
    normalized_speed = torch.clamp(robot.data.root_lin_vel_w[:, 2] / 0.25, min=-3.0, max=3.0)
    elapsed_s = env._ls_phase_step_counter.float() * _env_step_time_s(env)
    # Leave the initial takeoff unopposed, then demand a held raised posture even
    # if a weak policy has not yet crossed the full jump-ready curriculum gate.
    damping_window = env._ls_jump_ready_seen | (elapsed_s >= 0.65)
    active = (env._ls_phase == PHASE_JUMP) & damping_window
    return active.float() * torch.square(normalized_speed)


def rear_lower_leg_ground_contact_penalty(
    env: ManagerBasedRLEnv,
    rear_legs_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Prevent supporting the raised posture on the rear calves or thighs."""
    _ensure_switch_buffers(env)
    rear_leg_contacts = _feet_grounded(env, rear_legs_sensor_cfg).float().sum(dim=1)
    return (env._ls_phase == PHASE_JUMP).float() * rear_leg_contacts


def front_foot_excess_speed_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    foot_cfg: SceneEntityCfg,
    other_feet_cfg: SceneEntityCfg,
    foot_sensor_cfg: SceneEntityCfg,
    other_feet_sensor_cfg: SceneEntityCfg,
    front_feet_sensor_cfg: SceneEntityCfg,
    all_feet_sensor_cfg: SceneEntityCfg,
    settle_time_s: float,
    lift_height: float,
    lift_hold_time_s: float,
    jump_base_rise: float,
    land_hold_time_s: float,
) -> torch.Tensor:
    """Suppress fast front-foot flicking while leaving room for one deliberate reach."""
    _update_behavior_phase(
        env, robot_cfg, foot_cfg, other_feet_cfg, foot_sensor_cfg, other_feet_sensor_cfg,
        front_feet_sensor_cfg, all_feet_sensor_cfg, settle_time_s, lift_height,
        lift_hold_time_s, jump_base_rise, land_hold_time_s
    )
    robot: Articulation = env.scene[robot_cfg.name]
    if not hasattr(env, "_ls_front_foot_body_ids"):
        env._ls_front_foot_body_ids = robot.find_bodies(
            ["FL_foot.*", "FR_foot.*"], preserve_order=True
        )[0]
    speeds = torch.linalg.norm(robot.data.body_lin_vel_w[:, env._ls_front_foot_body_ids, :], dim=2)
    jump_penalty = torch.square(torch.clamp(speeds - 0.8, min=0.0)).sum(dim=1)
    return torch.where(env._ls_phase == PHASE_JUMP, jump_penalty, torch.zeros_like(jump_penalty))


def press_approach_progress_reward(
    env: ManagerBasedRLEnv,
    foot_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    other_feet_cfg: SceneEntityCfg,
    foot_sensor_cfg: SceneEntityCfg,
    other_feet_sensor_cfg: SceneEntityCfg,
    front_feet_sensor_cfg: SceneEntityCfg,
    all_feet_sensor_cfg: SceneEntityCfg,
    settle_time_s: float,
    lift_height: float,
    lift_hold_time_s: float,
    jump_base_rise: float,
    land_hold_time_s: float,
) -> torch.Tensor:
    _update_behavior_phase(
        env, robot_cfg, foot_cfg, other_feet_cfg, foot_sensor_cfg, other_feet_sensor_cfg,
        front_feet_sensor_cfg, all_feet_sensor_cfg, settle_time_s, lift_height,
        lift_hold_time_s, jump_base_rise, land_hold_time_s
    )
    press_target = _left_foot_press_target_w(env, press_depth=0.03)
    distance = torch.linalg.norm(_left_foot_pos_w(env, foot_cfg=foot_cfg) - press_target, dim=1)
    normalized_distance = distance / 0.30
    if not hasattr(env, "_ls_prev_press_dist"):
        env._ls_prev_press_dist = normalized_distance.clone()
    reset = (env.episode_length_buf == 0) | (env._ls_phase_entered == PHASE_JUMP)
    env._ls_prev_press_dist[reset] = normalized_distance[reset]
    progress = torch.clamp(
        env._ls_prev_press_dist - normalized_distance, min=-0.10, max=0.10
    )
    env._ls_prev_press_dist[:] = normalized_distance
    active = (
        (env._ls_phase == PHASE_JUMP)
        & env._ls_rear_ground
        & (~env._ls_first_contact_seen)
    )
    return active.float() * progress


def close_reach_reward(
    env: ManagerBasedRLEnv,
    foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="FL_foot.*"),
    distance_std: float = 0.08,
    press_depth: float = 0.03,
) -> torch.Tensor:
    """Absolute final-proximity reward once the supported jump is ready to reach."""
    _ensure_switch_buffers(env)
    press_target = _left_foot_press_target_w(env, press_depth=press_depth)
    distance = torch.linalg.norm(_left_foot_pos_w(env, foot_cfg=foot_cfg) - press_target, dim=1)
    score = torch.exp(-torch.square(distance / float(distance_std)))
    return env._ls_jump_reach_active.float() * score


def physical_press_force_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    foot_cfg: SceneEntityCfg,
    other_feet_cfg: SceneEntityCfg,
    foot_sensor_cfg: SceneEntityCfg,
    other_feet_sensor_cfg: SceneEntityCfg,
    front_feet_sensor_cfg: SceneEntityCfg,
    all_feet_sensor_cfg: SceneEntityCfg,
    settle_time_s: float,
    lift_height: float,
    lift_hold_time_s: float,
    jump_base_rise: float,
    land_hold_time_s: float,
    target_force: float = 18.0,
) -> torch.Tensor:
    """Reward measured FL-rocker force, giving no credit for hovering nearby."""
    _update_behavior_phase(
        env, robot_cfg, foot_cfg, other_feet_cfg, foot_sensor_cfg, other_feet_sensor_cfg,
        front_feet_sensor_cfg, all_feet_sensor_cfg, settle_time_s, lift_height,
        lift_hold_time_s, jump_base_rise, land_hold_time_s
    )
    force_fraction = torch.clamp(
        env._ls_rocker_contact_force / max(float(target_force), 1e-6), 0.0, 1.0
    )
    active = env._ls_jump_reach_active & env._ls_contact
    return active.float() * force_fraction


def phase_stability_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    foot_cfg: SceneEntityCfg,
    other_feet_cfg: SceneEntityCfg,
    foot_sensor_cfg: SceneEntityCfg,
    other_feet_sensor_cfg: SceneEntityCfg,
    front_feet_sensor_cfg: SceneEntityCfg,
    all_feet_sensor_cfg: SceneEntityCfg,
    settle_time_s: float,
    lift_height: float,
    lift_hold_time_s: float,
    jump_base_rise: float,
    land_hold_time_s: float,
) -> torch.Tensor:
    _update_behavior_phase(
        env, robot_cfg, foot_cfg, other_feet_cfg, foot_sensor_cfg, other_feet_sensor_cfg,
        front_feet_sensor_cfg, all_feet_sensor_cfg, settle_time_s, lift_height,
        lift_hold_time_s, jump_base_rise, land_hold_time_s
    )
    score = stability_reward(env=env, robot_cfg=robot_cfg)
    active = (env._ls_phase == PHASE_STAND) | (env._ls_phase == PHASE_LAND)
    return active.float() * score


def _initial_stand_window(env: ManagerBasedRLEnv, duration_s: float) -> torch.Tensor:
    """Full stand shaping initially, followed by a short decay until stance is verified."""
    elapsed_s = env.episode_length_buf * _env_step_time_s(env)
    overrun_s = torch.clamp(elapsed_s - float(duration_s), min=0.0)
    decay = torch.exp(-overrun_s / 0.35)
    return (env._ls_phase == PHASE_STAND).float() * decay


def initial_stand_pose_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_std: float = 0.25,
    duration_s: float = 1.0,
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    error = torch.square(robot.data.joint_pos - robot.data.default_joint_pos).mean(dim=1)
    return _initial_stand_window(env, duration_s) * torch.exp(
        -error / (float(joint_std) * float(joint_std))
    )


def initial_stand_base_height_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_std: float = 0.04,
    duration_s: float = 1.0,
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    target_height = robot.data.default_root_state[:, 2]
    error = torch.square(robot.data.root_pos_w[:, 2] - target_height)
    return _initial_stand_window(env, duration_s) * torch.exp(
        -error / (float(height_std) * float(height_std))
    )


def initial_stand_contact_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot.*"),
    duration_s: float = 1.0,
) -> torch.Tensor:
    contact_fraction = _feet_grounded(env, sensor_cfg).float().mean(dim=1)
    return _initial_stand_window(env, duration_s) * contact_fraction


def initial_stand_stability_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    duration_s: float = 1.0,
) -> torch.Tensor:
    score = stability_reward(env=env, robot_cfg=robot_cfg, lin_vel_std=0.15, tilt_std=0.25)
    return _initial_stand_window(env, duration_s) * score


def landing_recovery_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    all_feet_sensor_cfg: SceneEntityCfg,
    foot_cfg: SceneEntityCfg,
    other_feet_cfg: SceneEntityCfg,
    foot_sensor_cfg: SceneEntityCfg,
    other_feet_sensor_cfg: SceneEntityCfg,
    front_feet_sensor_cfg: SceneEntityCfg,
    settle_time_s: float,
    lift_height: float,
    lift_hold_time_s: float,
    jump_base_rise: float,
    land_hold_time_s: float,
) -> torch.Tensor:
    _update_behavior_phase(
        env, robot_cfg, foot_cfg, other_feet_cfg, foot_sensor_cfg, other_feet_sensor_cfg,
        front_feet_sensor_cfg, all_feet_sensor_cfg, settle_time_s, lift_height,
        lift_hold_time_s, jump_base_rise, land_hold_time_s
    )
    robot: Articulation = env.scene[robot_cfg.name]
    contact_fraction = _feet_grounded(env, all_feet_sensor_cfg).float().mean(dim=1)
    speed = torch.linalg.norm(robot.data.root_lin_vel_w, dim=1)
    angular_speed = torch.linalg.norm(robot.data.root_ang_vel_w, dim=1)
    tilt = torch.linalg.norm(robot.data.projected_gravity_b[:, :2], dim=1)
    recovery = contact_fraction * torch.exp(-torch.square(speed / 0.25) - torch.square(angular_speed / 0.5) - torch.square(tilt / 0.25))
    return (env._ls_phase == PHASE_LAND).float() * recovery


def phase_vertical_velocity_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    foot_cfg: SceneEntityCfg,
    other_feet_cfg: SceneEntityCfg,
    foot_sensor_cfg: SceneEntityCfg,
    other_feet_sensor_cfg: SceneEntityCfg,
    front_feet_sensor_cfg: SceneEntityCfg,
    all_feet_sensor_cfg: SceneEntityCfg,
    settle_time_s: float,
    lift_height: float,
    lift_hold_time_s: float,
    jump_base_rise: float,
    land_hold_time_s: float,
) -> torch.Tensor:
    _update_behavior_phase(
        env, robot_cfg, foot_cfg, other_feet_cfg, foot_sensor_cfg, other_feet_sensor_cfg,
        front_feet_sensor_cfg, all_feet_sensor_cfg, settle_time_s, lift_height,
        lift_hold_time_s, jump_base_rise, land_hold_time_s
    )
    robot: Articulation = env.scene[robot_cfg.name]
    active = (env._ls_phase == PHASE_STAND) | (env._ls_phase == PHASE_LAND)
    return active.float() * torch.square(robot.data.root_lin_vel_w[:, 2])


def landing_impact_penalty(
    env: ManagerBasedRLEnv,
    all_feet_sensor_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    foot_cfg: SceneEntityCfg,
    other_feet_cfg: SceneEntityCfg,
    foot_sensor_cfg: SceneEntityCfg,
    other_feet_sensor_cfg: SceneEntityCfg,
    front_feet_sensor_cfg: SceneEntityCfg,
    settle_time_s: float,
    lift_height: float,
    lift_hold_time_s: float,
    jump_base_rise: float,
    land_hold_time_s: float,
    force_threshold: float = 250.0,
) -> torch.Tensor:
    _update_behavior_phase(
        env, robot_cfg, foot_cfg, other_feet_cfg, foot_sensor_cfg, other_feet_sensor_cfg,
        front_feet_sensor_cfg, all_feet_sensor_cfg, settle_time_s, lift_height,
        lift_hold_time_s, jump_base_rise, land_hold_time_s
    )
    sensor: ContactSensor = env.scene[all_feet_sensor_cfg.name]
    force_z = torch.abs(sensor.data.net_forces_w[:, all_feet_sensor_cfg.body_ids, 2]).amax(dim=1)
    excess = torch.clamp((force_z - float(force_threshold)) / float(force_threshold), min=0.0, max=2.0)
    return (env._ls_phase == PHASE_LAND).float() * torch.square(excess)


def stability_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lin_vel_std: float = 0.15,
    tilt_std: float = 0.35,
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    planar_speed = torch.linalg.norm(robot.data.root_lin_vel_w[:, :2], dim=1)
    tilt = torch.linalg.norm(robot.data.projected_gravity_b[:, :2], dim=1)
    return torch.exp(
        -torch.square(planar_speed) / (lin_vel_std * lin_vel_std)
        - torch.square(tilt) / (tilt_std * tilt_std)
    )


def lightswitch_goal_reached(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    foot_cfg: SceneEntityCfg,
    other_feet_cfg: SceneEntityCfg,
    foot_sensor_cfg: SceneEntityCfg,
    other_feet_sensor_cfg: SceneEntityCfg,
    front_feet_sensor_cfg: SceneEntityCfg,
    all_feet_sensor_cfg: SceneEntityCfg,
    settle_time_s: float,
    lift_height: float,
    lift_hold_time_s: float,
    jump_base_rise: float,
    land_hold_time_s: float,
):
    _update_behavior_phase(
        env, robot_cfg, foot_cfg, other_feet_cfg, foot_sensor_cfg, other_feet_sensor_cfg,
        front_feet_sensor_cfg, all_feet_sensor_cfg, settle_time_s, lift_height,
        lift_hold_time_s, jump_base_rise, land_hold_time_s
    )
    return (env._ls_phase == PHASE_SUCCESS) & env._ls_success
