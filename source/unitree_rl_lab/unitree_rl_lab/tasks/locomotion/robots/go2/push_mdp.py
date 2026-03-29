from __future__ import annotations

import colorsys
import random
from typing import TYPE_CHECKING

import torch
from pxr import Gf, Sdf, UsdPhysics, UsdShade

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def randomize_rigid_body_mass_simple(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    mass_distribution_params: tuple[float, float],
    operation: str = "add",
    min_mass: float = 1e-6,
):
    """Function-style mass randomization compatible with EventManager direct calls.

    This mirrors the common startup use-case of IsaacLab's mass randomization term.
    """
    asset = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    masses = asset.root_physx_view.get_masses()
    default_mass = asset.data.default_mass

    # Reset selected entries to defaults before applying new randomization.
    masses[env_ids[:, None], body_ids] = default_mass[env_ids[:, None], body_ids].clone()

    low, high = mass_distribution_params
    samples = math_utils.sample_uniform(low, high, (len(env_ids), len(body_ids)), device="cpu")

    if operation == "add":
        masses[env_ids[:, None], body_ids] = masses[env_ids[:, None], body_ids] + samples
    elif operation == "scale":
        masses[env_ids[:, None], body_ids] = masses[env_ids[:, None], body_ids] * samples
    elif operation == "abs":
        masses[env_ids[:, None], body_ids] = samples
    else:
        raise ValueError(f"Unsupported mass randomization operation: {operation}")

    masses = torch.clamp(masses, min=min_mass)
    asset.root_physx_view.set_masses(masses, env_ids)


def randomize_floor_friction_per_reset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    static_friction_range: tuple[float, float] = (0.55, 1.25),
    dynamic_friction_range: tuple[float, float] = (0.45, 1.05),
    restitution_range: tuple[float, float] = (0.0, 0.02),
    terrain_material_prim_path: str = "/World/ground/terrain/physicsMaterial",
):
    """Randomize global terrain friction each reset call.

    Note: In this setup the terrain physics material is shared across environments, so
    the sampled values apply globally for the simulation step after each reset.
    """
    del env_ids

    stage = get_current_stage()
    prim = stage.GetPrimAtPath(terrain_material_prim_path)
    if not prim or not prim.IsValid():
        return

    material_api = UsdPhysics.MaterialAPI(prim)
    if not material_api:
        material_api = UsdPhysics.MaterialAPI.Apply(prim)

    static_friction = float(
        math_utils.sample_uniform(
            static_friction_range[0],
            static_friction_range[1],
            (1,),
            device=env.device,
        )[0].item()
    )
    dynamic_friction = float(
        math_utils.sample_uniform(
            dynamic_friction_range[0],
            dynamic_friction_range[1],
            (1,),
            device=env.device,
        )[0].item()
    )
    restitution = float(
        math_utils.sample_uniform(
            restitution_range[0],
            restitution_range[1],
            (1,),
            device=env.device,
        )[0].item()
    )

    dynamic_friction = min(dynamic_friction, static_friction)

    material_api.CreateStaticFrictionAttr().Set(static_friction)
    material_api.CreateDynamicFrictionAttr().Set(dynamic_friction)
    material_api.CreateRestitutionAttr().Set(restitution)


def set_cube_and_goal_matching_env_colors(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    palette_size: int = 32,
    brightness_range: tuple[float, float] = (0.92, 1.08),
    saturation_range: tuple[float, float] = (0.90, 1.10),
    random_seed: int | None = None,
):
    """Assign one color per environment to both cube and goal marker.

    Base hue comes from a deterministic palette. Brightness/saturation are jittered
    per environment (small ranges) to increase visual separation between nearby hues.
    """
    stage = get_current_stage()
    if stage is None:
        return

    num_envs = int(env.scene.num_envs)
    if num_envs <= 0:
        return

    if env_ids is None:
        env_indices = torch.arange(num_envs, dtype=torch.long, device="cpu")
    else:
        env_indices = env_ids.to(device="cpu", dtype=torch.long).flatten()

    palette_size = max(1, min(int(palette_size), num_envs))
    palette = [colorsys.hsv_to_rgb(i / palette_size, 0.7, 0.9) for i in range(palette_size)]

    for env_idx in env_indices.tolist():
        base_color = palette[env_idx % palette_size]
        rng = random.Random((int(random_seed) + env_idx) if random_seed is not None else None)

        # Small per-env saturation/brightness jitter in HSV space.
        h, s, v = colorsys.rgb_to_hsv(*base_color)
        sat = max(0.0, min(1.0, s * rng.uniform(saturation_range[0], saturation_range[1])))
        val = max(0.0, min(1.0, v * rng.uniform(brightness_range[0], brightness_range[1])))
        r, g, b = colorsys.hsv_to_rgb(h, sat, val)

        color = (r, g, b)
        color_vec = Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))
        for prim_name in ("Cube", "GoalArea"):
            shader_prim = stage.GetPrimAtPath(f"{env.scene.env_ns}/env_{env_idx}/{prim_name}/geometry/material/Shader")
            if not shader_prim or not shader_prim.IsValid():
                continue
            shader = UsdShade.Shader(shader_prim)
            diffuse_input = shader.GetInput("diffuseColor")
            if not diffuse_input:
                diffuse_input = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
            diffuse_input.Set(color_vec)


def _goal_xy_world(env: ManagerBasedRLEnv, env_ids: torch.Tensor, goal_xy: tuple[float, float]) -> torch.Tensor:
    goal_xy_tensor = torch.tensor(goal_xy, device=env.device, dtype=torch.float32)
    return _scene_env_origins_xy(env)[env_ids, :] + goal_xy_tensor.unsqueeze(0)


def _scene_env_origins_xy(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return per-env XY origins matching cloned env Xforms used by {ENV_REGEX_NS} assets.

    With replicate_physics=False, InteractiveScene keeps cloned-grid origins in
    `_default_env_origins`, while `scene.env_origins` may come from terrain
    patch assignment. Push task assets (robot/cube/goal marker) are cloned under
    env Xforms, so we anchor them to `_default_env_origins` when available.
    """
    default_env_origins = getattr(env.scene, "_default_env_origins", None)
    if default_env_origins is not None:
        return default_env_origins[:, :2]
    return env.scene.env_origins[:, :2]


def _cube_goal_distance(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg,
    goal_xy: tuple[float, float],
) -> torch.Tensor:
    cube: RigidObject = env.scene[cube_cfg.name]
    env_ids = torch.arange(env.num_envs, device=env.device)
    goal_xy_w = _goal_xy_world(env, env_ids, goal_xy)
    cube_xy_w = cube.data.root_pos_w[:, :2]
    return torch.linalg.norm(cube_xy_w - goal_xy_w, dim=1)


def _curriculum_alpha(env: ManagerBasedRLEnv, transition_steps: int) -> torch.Tensor:
    if transition_steps <= 0:
        value = 1.0
    else:
        value = min(1.0, max(0.0, float(env.common_step_counter) / float(transition_steps)))
    return torch.full((env.num_envs,), value, device=env.device)


def _symmetric_curriculum_range(limit_range: tuple[float, float], initial_abs: float, alpha: float) -> list[float]:
    # Keep command ranges symmetric around zero while ramping from a small start envelope.
    limit_abs = min(abs(float(limit_range[0])), abs(float(limit_range[1])))
    start_abs = max(0.0, min(float(initial_abs), limit_abs))
    current_abs = start_abs + (limit_abs - start_abs) * float(alpha)
    return [-current_abs, current_abs]


def command_velocity_envelope_stepwise_curriculum(
    env,
    env_ids,
    step_size: int = 5000,
    lin_vel_increment: float = 0.05,
    ang_vel_increment: float = 0.02,
    initial_lin_vel_abs: float = 0.05,
    initial_ang_vel_abs: float = 0.02,
    limit_lin_vel_x: float = 0.6,
    limit_lin_vel_y: float = 0.5,
    limit_ang_vel_z: float = 0.3,
):
    """Increase command velocity limits in steps after every step_size steps."""
    del env_ids
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    # Number of increments so far
    n_increments = int(env.common_step_counter // step_size)
    # Compute new abs limits
    lin_vel_abs = min(initial_lin_vel_abs + n_increments * lin_vel_increment, limit_lin_vel_x)
    ang_vel_abs = min(initial_ang_vel_abs + n_increments * ang_vel_increment, limit_ang_vel_z)

    # Set symmetric ranges
    ranges.lin_vel_x = [-lin_vel_abs, lin_vel_abs]
    ranges.lin_vel_y = [-lin_vel_abs, lin_vel_abs]  # Use same increment for y
    ranges.ang_vel_z = [-ang_vel_abs, ang_vel_abs]

    # For logging: return the current increment
    return lin_vel_abs


def cube_position_xy(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    cube: RigidObject = env.scene[cube_cfg.name]
    return cube.data.root_pos_w[:, :2] - _scene_env_origins_xy(env)


def cube_linear_velocity_xy(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    cube: RigidObject = env.scene[cube_cfg.name]
    return cube.data.root_lin_vel_w[:, :2]


def goal_position_xy(env: ManagerBasedRLEnv, goal_xy: tuple[float, float] = (0.0, 0.0)) -> torch.Tensor:
    goal_xy_tensor = torch.tensor(goal_xy, device=env.device, dtype=torch.float32)
    return goal_xy_tensor.unsqueeze(0).repeat(env.num_envs, 1)


def goal_radius_obs(env: ManagerBasedRLEnv, goal_radius: float) -> torch.Tensor:
    return torch.full((env.num_envs, 1), goal_radius, device=env.device)


def cube_to_goal_vector_xy(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    goal_xy: tuple[float, float] = (0.0, 0.0),
) -> torch.Tensor:
    return goal_position_xy(env, goal_xy=goal_xy) - cube_position_xy(env, cube_cfg=cube_cfg)


def left_front_foot_to_cube_vector_xy(
    env: ManagerBasedRLEnv,
    foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="FL_foot.*"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    robot: Articulation = env.scene[foot_cfg.name]
    cube: RigidObject = env.scene[cube_cfg.name]
    foot_xy = robot.data.body_pos_w[:, foot_cfg.body_ids, :2].mean(dim=1)
    cube_xy = cube.data.root_pos_w[:, :2]
    return cube_xy - foot_xy


def robot_position_xy(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    return robot.data.root_pos_w[:, :2] - _scene_env_origins_xy(env)


def robot_linear_velocity_xy(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    return robot.data.root_lin_vel_w[:, :2]


def cube_to_goal_progress_reward(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    goal_xy: tuple[float, float] = (0.0, 0.0),
    transition_steps: int = 250_000,
) -> torch.Tensor:
    dist = _cube_goal_distance(env, cube_cfg=cube_cfg, goal_xy=goal_xy)
    if not hasattr(env, "_push_prev_cube_goal_dist"):
        env._push_prev_cube_goal_dist = dist.clone()

    reset_mask = env.episode_length_buf == 0
    env._push_prev_cube_goal_dist[reset_mask] = dist[reset_mask]

    progress = env._push_prev_cube_goal_dist - dist
    env._push_prev_cube_goal_dist[:] = dist

    return _curriculum_alpha(env, transition_steps) * progress


def success_bonus_reward(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    goal_xy: tuple[float, float] = (0.0, 0.0),
    goal_radius: float = 0.35,
    cube_speed_threshold: float = 0.2,
    transition_steps: int = 250_000,
) -> torch.Tensor:
    cube: RigidObject = env.scene[cube_cfg.name]
    dist = _cube_goal_distance(env, cube_cfg=cube_cfg, goal_xy=goal_xy)
    speed = torch.linalg.norm(cube.data.root_lin_vel_w[:, :2], dim=1)
    is_success = torch.logical_and(dist <= goal_radius, speed <= cube_speed_threshold)
    return _curriculum_alpha(env, transition_steps) * is_success.float()


def cube_settled_in_goal_reward(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    goal_xy: tuple[float, float] = (0.0, 0.0),
    goal_radius: float = 0.35,
    vel_std: float = 0.12,
    transition_steps: int = 250_000,
) -> torch.Tensor:
    cube: RigidObject = env.scene[cube_cfg.name]
    dist = _cube_goal_distance(env, cube_cfg=cube_cfg, goal_xy=goal_xy)
    speed = torch.linalg.norm(cube.data.root_lin_vel_w[:, :2], dim=1)
    in_goal = dist <= goal_radius
    settle_score = torch.exp(-torch.square(speed) / (vel_std * vel_std))
    return _curriculum_alpha(env, transition_steps) * in_goal.float() * settle_score


def robot_to_cube_approach_progress_reward(
    env: ManagerBasedRLEnv,
    foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="FL_foot.*"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    goal_xy: tuple[float, float] = (0.0, 0.0),
    cube_far_distance: float = 0.7,
    transition_steps: int = 250_000,
) -> torch.Tensor:
    vec = left_front_foot_to_cube_vector_xy(env, foot_cfg=foot_cfg, cube_cfg=cube_cfg)
    dist = torch.linalg.norm(vec, dim=1)

    if not hasattr(env, "_push_prev_foot_cube_dist"):
        env._push_prev_foot_cube_dist = dist.clone()

    reset_mask = env.episode_length_buf == 0
    env._push_prev_foot_cube_dist[reset_mask] = dist[reset_mask]

    progress = env._push_prev_foot_cube_dist - dist
    env._push_prev_foot_cube_dist[:] = dist

    cube_goal_dist = _cube_goal_distance(env, cube_cfg=cube_cfg, goal_xy=goal_xy)
    gate = (cube_goal_dist > cube_far_distance).float()
    return (1.0 - _curriculum_alpha(env, transition_steps)) * gate * progress


def left_front_foot_cube_contact_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("cube_contact_forces", body_names="FL_foot.*"),
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if sensor.data.force_matrix_w is not None:
        # Sum filtered contact magnitudes between left-front foot and cube.
        force_mag = torch.linalg.norm(sensor.data.force_matrix_w[:, sensor_cfg.body_ids, :, :], dim=-1)
        return (torch.sum(force_mag, dim=(1, 2)) > 1e-3).float()
    in_contact = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    return torch.any(in_contact, dim=1).float()


def push_direction_reward(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    goal_xy: tuple[float, float] = (0.0, 0.0),
    transition_steps: int = 250_000,
) -> torch.Tensor:
    cube: RigidObject = env.scene[cube_cfg.name]
    cube_goal_vec = cube_to_goal_vector_xy(env, cube_cfg=cube_cfg, goal_xy=goal_xy)
    cube_goal_dir = cube_goal_vec / (torch.linalg.norm(cube_goal_vec, dim=1, keepdim=True) + 1e-6)
    cube_vel_xy = cube.data.root_lin_vel_w[:, :2]
    toward_goal_speed = torch.sum(cube_vel_xy * cube_goal_dir, dim=1)
    return _curriculum_alpha(env, transition_steps) * torch.clamp(toward_goal_speed, min=0.0)


def cube_goal_reached(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    goal_xy: tuple[float, float] = (0.0, 0.0),
    goal_radius: float = 0.35,
    cube_speed_threshold: float = 0.2,
) -> torch.Tensor:
    cube: RigidObject = env.scene[cube_cfg.name]
    dist = _cube_goal_distance(env, cube_cfg=cube_cfg, goal_xy=goal_xy)
    speed = torch.linalg.norm(cube.data.root_lin_vel_w[:, :2], dim=1)
    return torch.logical_and(dist <= goal_radius, speed <= cube_speed_threshold)


def reset_robot_and_cube_uniform_around_goal(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_xy: tuple[float, float] = (0.0, 0.0),
    goal_radius: float = 0.35,
    cube_spawn_radius_range: tuple[float, float] = (0.8, 2.0),
    cube_goal_clearance: float = 0.1,
    cube_height: float = 0.10,
    cube_yaw_range: tuple[float, float] = (-3.14159, 3.14159),
    robot_spawn_radius_range: tuple[float, float] = (0.4, 1.2),
    robot_min_distance_to_cube: float = 0.4,
    robot_yaw_range: tuple[float, float] = (-3.14159, 3.14159),
    robot_velocity_range: dict[str, tuple[float, float]] | None = None,
):
    if robot_velocity_range is None:
        robot_velocity_range = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }

    robot: Articulation = env.scene[robot_cfg.name]
    cube: RigidObject = env.scene[cube_cfg.name]

    num_resets = len(env_ids)
    device = env.device

    goal_xy_w = _goal_xy_world(env, env_ids, goal_xy)

    # Randomize cube in an annulus around the goal and keep it outside the goal region.
    cube_min_radius = max(cube_spawn_radius_range[0], goal_radius + cube_goal_clearance)
    cube_max_radius = max(cube_spawn_radius_range[1], cube_min_radius + 1e-3)
    cube_radius = math_utils.sample_uniform(cube_min_radius, cube_max_radius, (num_resets,), device=device)
    cube_angle = math_utils.sample_uniform(-torch.pi, torch.pi, (num_resets,), device=device)
    cube_xy = goal_xy_w + torch.stack((cube_radius * torch.cos(cube_angle), cube_radius * torch.sin(cube_angle)), dim=1)

    cube_yaw = math_utils.sample_uniform(cube_yaw_range[0], cube_yaw_range[1], (num_resets,), device=device)
    cube_quat = math_utils.quat_from_euler_xyz(
        torch.zeros(num_resets, device=device),
        torch.zeros(num_resets, device=device),
        cube_yaw,
    )
    cube_pos = torch.cat((cube_xy, torch.full((num_resets, 1), cube_height, device=device)), dim=1)
    cube_vel = torch.zeros((num_resets, 6), device=device)
    cube.write_root_pose_to_sim(torch.cat((cube_pos, cube_quat), dim=1), env_ids=env_ids)
    cube.write_root_velocity_to_sim(cube_vel, env_ids=env_ids)

    # Randomize robot pose around the cube and keep a minimum spacing to avoid overlaps.
    robot_min_radius = max(robot_spawn_radius_range[0], robot_min_distance_to_cube)
    robot_max_radius = max(robot_spawn_radius_range[1], robot_min_radius + 1e-3)
    robot_radius = math_utils.sample_uniform(robot_min_radius, robot_max_radius, (num_resets,), device=device)
    robot_angle = math_utils.sample_uniform(-torch.pi, torch.pi, (num_resets,), device=device)
    robot_xy = cube_xy + torch.stack((robot_radius * torch.cos(robot_angle), robot_radius * torch.sin(robot_angle)), dim=1)

    robot_root_state = robot.data.default_root_state[env_ids].clone()
    robot_yaw = math_utils.sample_uniform(robot_yaw_range[0], robot_yaw_range[1], (num_resets,), device=device)
    robot_quat = math_utils.quat_from_euler_xyz(
        torch.zeros(num_resets, device=device),
        torch.zeros(num_resets, device=device),
        robot_yaw,
    )

    robot_pos = torch.cat((robot_xy, robot_root_state[:, 2:3]), dim=1)

    range_list = [robot_velocity_range.get(k, (0.0, 0.0)) for k in ["x", "y", "z", "roll", "pitch", "yaw"]]
    vel_ranges = torch.tensor(range_list, device=device)
    robot_vel = math_utils.sample_uniform(
        vel_ranges[:, 0],
        vel_ranges[:, 1],
        (num_resets, 6),
        device=device,
    )

    robot.write_root_pose_to_sim(torch.cat((robot_pos, robot_quat), dim=1), env_ids=env_ids)
    robot.write_root_velocity_to_sim(robot_vel, env_ids=env_ids)
