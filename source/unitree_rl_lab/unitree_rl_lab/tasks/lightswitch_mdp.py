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

# Ordered behavior phases.  The policy receives these as a one-hot observation.
PHASE_STAND = 0
PHASE_LIFT = 1
PHASE_JUMP = 2
PHASE_PRESS = 3
PHASE_LAND = 4
PHASE_SUCCESS = 5
NUM_BEHAVIOR_PHASES = 6


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


def curriculum_common_step_counter(env, env_ids):
    del env_ids
    return float(env.common_step_counter)


def curriculum_phase_fraction(env, env_ids, phase: int):
    """Fraction of parallel environments currently occupying one behavior phase."""
    del env_ids
    _ensure_switch_buffers(env)
    return float((env._ls_phase == int(phase)).float().mean().item())


def curriculum_behavior_difficulty(
    env,
    env_ids,
    jump_rise_start: float = 0.08,
    jump_rise_target: float = 0.12,
    success_rate_start: float = 0.40,
    success_rate_full: float = 0.70,
    ema_rate: float = 0.10,
):
    """Raise lift/jump thresholds only after recent episodes reliably clear each milestone."""
    _ensure_switch_buffers(env)
    ids = _to_env_ids(env, env_ids)
    completed = env.episode_length_buf[ids] > 0
    if torch.any(completed):
        phases = env._ls_phase[ids[completed]]
        jump_rate = (phases >= PHASE_PRESS).float().mean()
        rate = float(ema_rate)
        env._ls_jump_success_ema = (1.0 - rate) * env._ls_jump_success_ema + rate * jump_rate

    denominator = max(float(success_rate_full) - float(success_rate_start), 1e-6)
    jump_alpha = torch.clamp((env._ls_jump_success_ema - float(success_rate_start)) / denominator, 0.0, 1.0)
    env._ls_adaptive_jump_rise = float(jump_rise_start) + jump_alpha * (
        float(jump_rise_target) - float(jump_rise_start)
    )
    return {
        "difficulty": jump_alpha,
        "jump_success_ema": env._ls_jump_success_ema,
        "jump_rise": env._ls_adaptive_jump_rise,
    }


def _ensure_switch_buffers(env: ManagerBasedRLEnv):
    if not hasattr(env, "_ls_switch_center_w") or env._ls_switch_center_w.shape[0] != env.num_envs:
        env._ls_switch_center_w = torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)
        env._ls_switch_center_local = torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)

        env._ls_current_state = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env._ls_target_state = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        env._ls_target_side_sign = torch.ones(env.num_envs, device=env.device, dtype=torch.float32)
        env._ls_success = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env._ls_contact = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env._ls_correct_touch = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env._ls_wrong_touch = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
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
        env._ls_settle_hold_counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        env._ls_lift_hold_counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        env._ls_land_hold_counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        env._ls_jump_hold_counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        env._ls_settled_base_z = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        env._ls_front_air_seen = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env._ls_front_air_trigger = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env._ls_rear_ground = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        env._ls_phase_last_step = -1
        env._ls_jump_success_ema = torch.tensor(0.0, device=env.device)
        env._ls_adaptive_jump_rise = torch.tensor(0.08, device=env.device)

        env._ls_joint_pos_attrs = [None] * env.num_envs
        env._ls_joint_vel_attrs = [None] * env.num_envs
        env._ls_drive_target_pos_attrs = [None] * env.num_envs
        env._ls_drive_target_vel_attrs = [None] * env.num_envs

        env._ls_interaction_last_step = -1
        env._ls_success_hold_counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        env._ls_success_hold_last_step = -1


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
            _set_translate(wall_prim, (cx + float(wall_x_offset), cy, cz))


def setup_lightswitch_stage(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | slice | list[int] | None,
    switch_center_xy: tuple[float, float] = (0.45, 0.0),
    switch_x_range: tuple[float, float] = (0.42, 0.48),
    switch_y_range: tuple[float, float] = (-0.04, 0.04),
    switch_default_height: float = 1.0,
    switch_height_range: tuple[float, float] = (0.8, 1.1),
    switch_yaw: float = math.pi * 0.5,
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
                size_xyz=(0.05, 0.40, 0.40),
                position_xyz=(float(switch_center_xy[0]) + 0.036, float(switch_center_xy[1]), float(switch_default_height)),
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
    env._ls_switch_center_local[env_ids] = switch_center_local
    _set_switch_root_wall_transforms_for_envs(
        env=env,
        env_ids=env_ids,
        switch_center_local=switch_center_local,
        switch_yaw=switch_yaw,
        wall_x_offset=0.036,
    )


def _left_foot_pos_w(
    env: ManagerBasedRLEnv,
    foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="FL_foot.*"),
) -> torch.Tensor:
    robot: Articulation = env.scene[foot_cfg.name]
    return robot.data.body_pos_w[:, foot_cfg.body_ids, :].mean(dim=1)


def _one_shot_bool_trigger(
    env: ManagerBasedRLEnv,
    mask: torch.Tensor,
    state_prefix: str,
) -> torch.Tensor:
    prev_name = f"{state_prefix}_prev"
    flag_name = f"{state_prefix}_flag"
    last_step_name = f"{state_prefix}_last_step"

    if not hasattr(env, prev_name) or getattr(env, prev_name).shape[0] != env.num_envs:
        setattr(env, prev_name, torch.zeros(env.num_envs, device=env.device, dtype=torch.bool))
        setattr(env, flag_name, torch.zeros(env.num_envs, device=env.device, dtype=torch.bool))
        setattr(env, last_step_name, -1)

    current_step = int(env.common_step_counter)
    if getattr(env, last_step_name, -1) != current_step:
        prev_mask = getattr(env, prev_name)
        reset_mask = env.episode_length_buf == 0
        prev_mask[reset_mask] = False
        trigger = torch.logical_and(~prev_mask, mask)
        setattr(env, flag_name, trigger)
        setattr(env, prev_name, mask.clone())
        setattr(env, last_step_name, current_step)
    return getattr(env, flag_name)


def _reset_switch_episode_state(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    initial_on: torch.Tensor,
    target_on: torch.Tensor,
):
    env._ls_current_state[env_ids] = initial_on
    env._ls_target_state[env_ids] = target_on
    env._ls_target_side_sign[env_ids] = torch.where(
        target_on,
        torch.ones_like(initial_on, dtype=torch.float32, device=env.device),
        -torch.ones_like(initial_on, dtype=torch.float32, device=env.device),
    )
    env._ls_success[env_ids] = False
    env._ls_contact[env_ids] = False
    env._ls_correct_touch[env_ids] = False
    env._ls_wrong_touch[env_ids] = False
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
    env._ls_success_hold_last_step = -1
    env._ls_success_hold_counter[env_ids] = 0
    env._ls_phase[env_ids] = PHASE_STAND
    env._ls_phase_entered[env_ids] = -1
    env._ls_settle_hold_counter[env_ids] = 0
    env._ls_lift_hold_counter[env_ids] = 0
    env._ls_land_hold_counter[env_ids] = 0
    env._ls_jump_hold_counter[env_ids] = 0
    env._ls_settled_base_z[env_ids] = 0.0
    env._ls_front_air_seen[env_ids] = False
    env._ls_front_air_trigger[env_ids] = False
    env._ls_rear_ground[env_ids] = True
    env._ls_phase_last_step = -1

    if hasattr(env, "_ls_switch_success_prev"):
        env._ls_switch_success_prev[env_ids] = False
    if hasattr(env, "_ls_switch_success_flag"):
        env._ls_switch_success_flag[env_ids] = False
    if hasattr(env, "_ls_switch_success_last_step"):
        env._ls_switch_success_last_step = -1
    if hasattr(env, "_ls_finish_prev"):
        env._ls_finish_prev[env_ids] = False
    if hasattr(env, "_ls_finish_flag"):
        env._ls_finish_flag[env_ids] = False
    if hasattr(env, "_ls_finish_last_step"):
        env._ls_finish_last_step = -1


def reset_robot_and_lightswitch(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    switch_center_xy: tuple[float, float] = (0.45, 0.0),
    switch_height_range: tuple[float, float] = (0.8, 1.1),
    switch_yaw: float = math.pi * 0.5,
    wall_x_offset: float = 0.036,
    robot_forward_range: tuple[float, float] = (0.15, 0.35),
    robot_lateral_range: tuple[float, float] = (-0.2, 0.2),
    robot_yaw_range: tuple[float, float] = (-0.2, 0.2),
    robot_velocity_range: dict[str, tuple[float, float]] | None = None,
):
    del switch_center_xy, switch_yaw, wall_x_offset, switch_height_range
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

    # Robot spawn in front of switch with lateral randomization.
    forward_dist = math_utils.sample_uniform(
        float(robot_forward_range[0]), float(robot_forward_range[1]), (num_resets,), device=device
    )
    lateral = math_utils.sample_uniform(
        float(robot_lateral_range[0]), float(robot_lateral_range[1]), (num_resets,), device=device
    )

    robot_x = center_x - forward_dist
    robot_y = center_y + lateral
    robot_root_state = robot.data.default_root_state[env_ids].clone()
    robot_pos = torch.cat((robot_x.unsqueeze(1), robot_y.unsqueeze(1), robot_root_state[:, 2:3]), dim=1)

    robot_yaw = math_utils.sample_uniform(
        float(robot_yaw_range[0]), float(robot_yaw_range[1]), (num_resets,), device=device
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


def _update_switch_latch_and_interaction(
    env: ManagerBasedRLEnv,
    foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="FL_foot.*"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names="FL_foot.*"),
    contact_x_threshold: float = 0.10,
    contact_y_threshold: float = 0.08,
    contact_z_threshold: float = 0.08,
):
    _ensure_switch_buffers(env)
    current_step = int(env.common_step_counter)
    if getattr(env, "_ls_interaction_last_step", -1) == current_step:
        return

    # Spatial classification plus measured foot contact. Position alone cannot toggle.
    foot_pos_w = _left_foot_pos_w(env, foot_cfg=foot_cfg)
    robot: Articulation = env.scene[foot_cfg.name]
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    foot_force = torch.linalg.norm(sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=2).amax(dim=1)
    rel = foot_pos_w - env._ls_switch_center_w
    dx = rel[:, 0]
    dy = rel[:, 1]
    dz = rel[:, 2]
    in_contact_box = (
        (torch.abs(dx) <= float(contact_x_threshold))
        & (torch.abs(dy) <= float(contact_y_threshold))
        & (torch.abs(dz) <= float(contact_z_threshold))
    )
    physical_contact = in_contact_box & (foot_force >= 2.0)
    top_touch = physical_contact & (dz > 0.0)
    bottom_touch = physical_contact & (dz <= 0.0)
    correct_touch = torch.where(env._ls_target_side_sign > 0.0, top_touch, bottom_touch)
    wrong_touch = physical_contact & (~correct_touch)

    # V1 accepts either rocker half, but only after the ordered jump milestone.
    # The touched half chooses the visual rocker direction; it is not a task command.
    valid_phase = (env._ls_phase == PHASE_PRESS) & env._ls_rear_ground
    # Contact force and location already prove a real press.  Requiring positive
    # X velocity on the exact contact frame made valid impacts get discarded.
    can_press = physical_contact & valid_phase & (~env._ls_success)
    if torch.any(can_press):
        touched_state = top_touch[can_press]
        env._ls_target_state[can_press] = touched_state
        env._ls_target_side_sign[can_press] = torch.where(
            touched_state,
            torch.ones_like(touched_state, dtype=torch.float32),
            -torch.ones_like(touched_state, dtype=torch.float32),
        )
        env._ls_press_in_progress[can_press] = True
        env._ls_snap_boost_time_left_s[can_press] = float(SNAP_BOOST_TIME_S)

    step_dt = float(_env_step_time_s(env))
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
    reached_target = env._ls_press_in_progress & crossed_threshold & (~env._ls_success)
    env._ls_current_state[reached_target] = env._ls_target_state[reached_target]
    env._ls_press_in_progress[reached_target] = False
    trigger = _one_shot_bool_trigger(env=env, mask=reached_target, state_prefix="_ls_switch_success")
    env._ls_success = torch.logical_or(env._ls_success, trigger)
    env._ls_toggle_trigger = trigger

    env._ls_contact = physical_contact
    env._ls_correct_touch = correct_touch
    env._ls_wrong_touch = wrong_touch
    env._ls_interaction_last_step = current_step


def switch_center_position_local(env: ManagerBasedRLEnv) -> torch.Tensor:
    _ensure_switch_buffers(env)
    return env._ls_switch_center_local


def switch_center_position_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Switch center relative to the base, expressed in the robot body frame."""
    _ensure_switch_buffers(env)
    robot: Articulation = env.scene[robot_cfg.name]
    return quat_apply_inverse(robot.data.root_quat_w, env._ls_switch_center_w - robot.data.root_pos_w)


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
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names="FL_foot.*"),
    contact_x_threshold: float = 0.10,
    contact_y_threshold: float = 0.08,
    contact_z_threshold: float = 0.08,
) -> torch.Tensor:
    _update_switch_latch_and_interaction(
        env=env,
        foot_cfg=foot_cfg,
        sensor_cfg=sensor_cfg,
        contact_x_threshold=contact_x_threshold,
        contact_y_threshold=contact_y_threshold,
        contact_z_threshold=contact_z_threshold,
    )
    return env._ls_contact.float().unsqueeze(1)


def behavior_phase_one_hot(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Ordered task phase for the memoryless actor and critic."""
    _ensure_switch_buffers(env)
    return torch.nn.functional.one_hot(env._ls_phase, num_classes=NUM_BEHAVIOR_PHASES).float()


def _feet_grounded(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    min_vertical_force: float = 1.0,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    vertical_force = torch.abs(sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    return vertical_force >= float(min_vertical_force)


def _update_behavior_phase(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="FL_foot.*"),
    other_feet_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["RL_foot.*", "RR_foot.*"]
    ),
    foot_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names="FL_foot.*"),
    other_feet_sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "contact_forces", body_names=["RL_foot.*", "RR_foot.*"]
    ),
    front_feet_sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "contact_forces", body_names=["FL_foot.*", "FR_foot.*"]
    ),
    all_feet_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot.*"),
    settle_time_s: float = 0.25,
    lift_height: float = 0.30,
    lift_hold_time_s: float = 0.10,
    jump_base_rise: float = 0.12,
    land_hold_time_s: float = 0.50,
):
    """Advance each environment through stand, jump, press, and land exactly once."""
    _ensure_switch_buffers(env)
    current_step = int(env.common_step_counter)
    if getattr(env, "_ls_phase_last_step", -1) == current_step:
        return

    robot: Articulation = env.scene[robot_cfg.name]
    old_phase = env._ls_phase.clone()
    env._ls_phase_entered.fill_(-1)
    env._ls_front_air_trigger.fill_(False)
    dt = _env_step_time_s(env)

    planar_speed = torch.linalg.norm(robot.data.root_lin_vel_w[:, :2], dim=1)
    vertical_speed = torch.abs(robot.data.root_lin_vel_w[:, 2])
    angular_speed = torch.linalg.norm(robot.data.root_ang_vel_w, dim=1)
    tilt = torch.linalg.norm(robot.data.projected_gravity_b[:, :2], dim=1)

    front_air = (~_feet_grounded(env, front_feet_sensor_cfg)).all(dim=1)
    rear_ground = _feet_grounded(env, other_feet_sensor_cfg).all(dim=1)
    env._ls_rear_ground = rear_ground
    all_ground = _feet_grounded(env, all_feet_sensor_cfg).all(dim=1)

    # Presses are accepted only while both rear feet support the robot.
    _update_switch_latch_and_interaction(env=env, foot_cfg=foot_cfg, sensor_cfg=foot_sensor_cfg)

    # Reset already places the robot in a nominal stance.  This gate only verifies a
    # short visible settling period; tight force/velocity limits previously trapped
    # more than 99% of environments in STAND forever.
    grounded_count = _feet_grounded(env, all_feet_sensor_cfg).sum(dim=1)
    stable_stand = (
        (grounded_count >= 3)
        & (planar_speed <= 0.30)
        & (vertical_speed <= 0.20)
        & (angular_speed <= 0.80)
        & (tilt <= 0.35)
    )
    stand_mask = old_phase == PHASE_STAND
    env._ls_settle_hold_counter = torch.where(
        stand_mask & stable_stand,
        env._ls_settle_hold_counter + 1,
        torch.where(stand_mask, torch.zeros_like(env._ls_settle_hold_counter), env._ls_settle_hold_counter),
    )
    settle_steps = max(1, int(math.ceil(float(settle_time_s) / dt)))
    settle_fallback_steps = max(settle_steps, int(math.ceil(0.50 / dt)))
    settle_fallback = (env.episode_length_buf >= settle_fallback_steps) & (tilt <= 0.50)
    enter_jump_from_stand = stand_mask & (
        (env._ls_settle_hold_counter >= settle_steps) | settle_fallback
    )
    if torch.any(enter_jump_from_stand):
        # Go directly into the jump.  The manipulation leg does not need to be
        # raised separately before takeoff.
        env._ls_phase[enter_jump_from_stand] = PHASE_JUMP
        env._ls_phase_entered[enter_jump_from_stand] = PHASE_JUMP
        env._ls_settled_base_z[enter_jump_from_stand] = robot.data.root_pos_w[enter_jump_from_stand, 2]

    jump_mask = old_phase == PHASE_JUMP
    base_rise = robot.data.root_pos_w[:, 2] - env._ls_settled_base_z
    supported_front_air = front_air & rear_ground
    new_front_air = jump_mask & supported_front_air & (~env._ls_front_air_seen)
    env._ls_front_air_trigger[new_front_air] = True
    env._ls_front_air_seen = env._ls_front_air_seen | (jump_mask & supported_front_air)
    jump_threshold = torch.clamp(env._ls_adaptive_jump_rise, max=float(jump_base_rise))
    valid_supported_jump = jump_mask & supported_front_air & (base_rise >= jump_threshold)
    env._ls_jump_hold_counter = torch.where(
        valid_supported_jump,
        env._ls_jump_hold_counter + 1,
        torch.where(jump_mask, torch.zeros_like(env._ls_jump_hold_counter), env._ls_jump_hold_counter),
    )
    jump_hold_steps = max(1, int(math.ceil(0.12 / dt)))
    enter_press = jump_mask & (env._ls_jump_hold_counter >= jump_hold_steps)
    env._ls_phase[enter_press] = PHASE_PRESS
    env._ls_phase_entered[enter_press] = PHASE_PRESS

    enter_land = (old_phase == PHASE_PRESS) & env._ls_success
    env._ls_phase[enter_land] = PHASE_LAND
    env._ls_phase_entered[enter_land] = PHASE_LAND

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
    land_steps = max(1, int(math.ceil(float(land_hold_time_s) / dt)))
    enter_success = land_mask & (env._ls_land_hold_counter >= land_steps)
    env._ls_phase[enter_success] = PHASE_SUCCESS
    env._ls_phase_entered[enter_success] = PHASE_SUCCESS
    env._ls_phase_last_step = current_step


def front_feet_air_milestone_reward(
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
    return env._ls_front_air_trigger.float()


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
    rise = robot.data.root_pos_w[:, 2] - env._ls_settled_base_z
    if not hasattr(env, "_ls_prev_base_rise"):
        env._ls_prev_base_rise = rise.clone()
    reset = (env.episode_length_buf == 0) | (env._ls_phase_entered == PHASE_JUMP)
    env._ls_prev_base_rise[reset] = rise[reset]
    progress = torch.clamp(rise - env._ls_prev_base_rise, min=-0.02, max=0.02)
    env._ls_prev_base_rise[:] = rise
    return (env._ls_phase == PHASE_JUMP).float() * env._ls_rear_ground.float() * progress


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
    _update_behavior_phase(
        env, robot_cfg, foot_cfg, other_feet_cfg, foot_sensor_cfg, other_feet_sensor_cfg,
        front_feet_sensor_cfg, all_feet_sensor_cfg, settle_time_s, lift_height,
        lift_hold_time_s, jump_base_rise, land_hold_time_s
    )
    rear_contacts = _feet_grounded(env, other_feet_sensor_cfg).float().sum(dim=1)
    active = (env._ls_phase == PHASE_JUMP) | (env._ls_phase == PHASE_PRESS)
    return active.float() * (2.0 - rear_contacts)


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
    jump_penalty = torch.square(torch.clamp(speeds - 1.0, min=0.0)).sum(dim=1)
    press_fl = torch.square(torch.clamp(speeds[:, 0] - 1.0, min=0.0))
    press_fr = 2.0 * torch.square(torch.clamp(speeds[:, 1] - 0.35, min=0.0))
    return torch.where(
        env._ls_phase == PHASE_JUMP,
        jump_penalty,
        torch.where(env._ls_phase == PHASE_PRESS, press_fl + press_fr, torch.zeros_like(jump_penalty)),
    )


def jump_crouch_penalty(
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
    """Allow a small preload crouch, but prevent staying low on the rear legs."""
    _update_behavior_phase(
        env, robot_cfg, foot_cfg, other_feet_cfg, foot_sensor_cfg, other_feet_sensor_cfg,
        front_feet_sensor_cfg, all_feet_sensor_cfg, settle_time_s, lift_height,
        lift_hold_time_s, jump_base_rise, land_hold_time_s
    )
    robot: Articulation = env.scene[robot_cfg.name]
    excessive_drop = torch.clamp(
        (env._ls_settled_base_z - robot.data.root_pos_w[:, 2] - 0.03) / 0.10,
        min=0.0,
        max=2.0,
    )
    return (env._ls_phase == PHASE_JUMP).float() * torch.square(excessive_drop)


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
    distance = torch.linalg.norm(_left_foot_pos_w(env, foot_cfg=foot_cfg) - env._ls_switch_center_w, dim=1)
    if not hasattr(env, "_ls_prev_press_dist"):
        env._ls_prev_press_dist = distance.clone()
    reset = (env.episode_length_buf == 0) | (env._ls_phase_entered == PHASE_PRESS)
    env._ls_prev_press_dist[reset] = distance[reset]
    progress = torch.clamp(env._ls_prev_press_dist - distance, min=-0.02, max=0.02)
    env._ls_prev_press_dist[:] = distance
    return (env._ls_phase == PHASE_PRESS).float() * progress


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
    return env._ls_phase == PHASE_SUCCESS
