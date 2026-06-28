from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


# def foot_force(
#     env: ManagerBasedRLEnv,
#     sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# ) -> torch.Tensor:
#     """Return the original 3-D contact-force magnitude for each foot."""
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     foot_forces_w = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]

#     raw = torch.linalg.norm(foot_forces_w, dim=2)
#     # Contact forces can briefly become non-finite during unstable PhysX contacts.
#     # Keep invalid sensor samples out of the policy observation.
#     raw = torch.nan_to_num(raw, nan=0.0, posinf=200.0, neginf=0.0)

#     step = env.common_step_counter
#     should_print = (
#         getattr(env, "print_foot_force", False)
#         and step % 1 == 0 # %1 is at simulation frequency = 50hz
#         and getattr(env, "last_foot_force_print_step", None) != step
#     )
#     should_record = (
#         getattr(env, "record_foot_force", False)
#         and getattr(env, "last_foot_force_record_step", None) != step
#     )

#     if should_print or should_record:
#         vertical_force = torch.clamp(foot_forces_w[:, :, 2], min=0.0)
#         asset: Articulation = env.unwrapped.scene[asset_cfg.name].data
#         joint_wrench_x = asset.body_incoming_joint_wrench_b[:, 15:19, 0] / 100

#     if should_record:
#         env.last_foot_force_record_step = step
#         record_path = Path(getattr(env, "foot_force_record_path", "foot_force_recording.csv"))
#         record_path.parent.mkdir(parents=True, exist_ok=True)
#         write_header = not record_path.exists()

#         row = [step, step * env.step_dt]
#         row.extend(raw[0].detach().cpu().tolist())
#         row.extend(vertical_force[0].detach().cpu().tolist())
#         row.extend(joint_wrench_x[0].detach().cpu().tolist())
#         command = env.command_manager.get_command("base_velocity")
#         row.extend(command[0, :3].detach().cpu().tolist())

#         with record_path.open("a", newline="") as csv_file:
#             writer = csv.writer(csv_file)
#             if write_header:
#                 writer.writerow(
#                     ["step", "time_s"]
#                     + [f"magnitude_foot_{index}" for index in range(4)]
#                     + [f"vertical_foot_{index}" for index in range(4)]
#                     + [f"joint_wrench_x_div100_foot_{index}" for index in range(4)]
#                     + ["cmd_lin_vel_x", "cmd_lin_vel_y", "cmd_ang_vel_z"]
#                 )
#             writer.writerow(row)

#     return raw


def foot_force(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return compressive contact force along each Go2 foot's local sensor axis."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    foot_forces_w = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]

    # The Go2 foot-end sensor measures compression along the foot/calf axis, not
    # the complete 3-D contact-force magnitude. In the Go2 URDF, local +Z points
    # from each foot toward its calf, so project the world-frame contact force
    # onto that axis and discard tensile force.
    asset: Articulation = env.unwrapped.scene[asset_cfg.name]
    sensor_body_ids = sensor_cfg.body_ids
    if isinstance(sensor_body_ids, slice):
        sensor_body_key = (sensor_body_ids.start, sensor_body_ids.stop, sensor_body_ids.step)
    elif isinstance(sensor_body_ids, int):
        sensor_body_key = (sensor_body_ids,)
    else:
        sensor_body_key = tuple(sensor_body_ids)

    cache_key = (asset_cfg.name, sensor_cfg.name, sensor_body_key)
    body_id_cache = getattr(env, "_foot_force_body_id_cache", None)
    if body_id_cache is None:
        body_id_cache = {}
        env._foot_force_body_id_cache = body_id_cache

    foot_body_ids = body_id_cache.get(cache_key)
    if foot_body_ids is None:
        if isinstance(sensor_body_ids, slice):
            foot_body_names = contact_sensor.body_names[sensor_body_ids]
        elif isinstance(sensor_body_ids, int):
            foot_body_names = [contact_sensor.body_names[sensor_body_ids]]
        else:
            foot_body_names = [contact_sensor.body_names[index] for index in sensor_body_ids]
        resolved_body_ids, _ = asset.find_bodies(foot_body_names, preserve_order=True)
        foot_body_ids = torch.as_tensor(resolved_body_ids, dtype=torch.long, device=foot_forces_w.device)
        body_id_cache[cache_key] = foot_body_ids

    foot_quat_w = asset.data.body_quat_w.index_select(1, foot_body_ids)

    # Third column of the quaternion rotation matrix: local +Z in world frame.
    qw, qx, qy, qz = foot_quat_w.unbind(dim=-1)
    force_x, force_y, force_z = foot_forces_w.unbind(dim=-1)
    normal_x = 2.0 * (qx * qz + qw * qy)
    normal_y = 2.0 * (qy * qz - qw * qx)
    normal_z = 1.0 - 2.0 * (qx.square() + qy.square())
    raw = (force_x * normal_x + force_y * normal_y + force_z * normal_z).clamp_min(0.0)
    # Contact forces can briefly become non-finite during unstable PhysX contacts.
    # Keep invalid sensor samples out of the policy observation.
    raw = torch.nan_to_num(raw, nan=0.0, posinf=200.0, neginf=0.0)

    step = env.common_step_counter
    should_print = (
        getattr(env, "print_foot_force", False)
        and step % 1 == 0 # %1 is at simulation frequency = 50hz
        and getattr(env, "last_foot_force_print_step", None) != step
    )
    should_record = (
        getattr(env, "record_foot_force", False)
        and getattr(env, "last_foot_force_record_step", None) != step
    )

    if should_print:
        env.last_foot_force_print_step = step
        raw_values = raw[0].detach().cpu().tolist()
        scaled_values = (raw[0] * 0.01).detach().cpu().tolist()
        print(
            "[FOOT FORCE NORMAL] raw:",
            [f"{value:.3f}" for value in raw_values],
            "scaled (x0.01):",
            [f"{value:.3f}" for value in scaled_values],
        )

    if should_record:
        env.last_foot_force_record_step = step
        record_path = Path(getattr(env, "foot_force_record_path", "foot_force_recording.csv"))
        record_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not record_path.exists()

        row = [step, step * env.step_dt]
        row.extend(raw[0].detach().cpu().tolist())

        with record_path.open("a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            if write_header:
                writer.writerow(
                    ["step", "time_s"]
                    + [f"normal_force_foot_{index}" for index in range(4)]
                )
            writer.writerow(row)

    return raw


# From french people
# def feet_contact_force(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
#     asset: Articulation = env.unwrapped.scene[asset_cfg.name].data
#     # height scan: height = sensor_height - hit_point_z - offset
#     feet=(asset.body_incoming_joint_wrench_b[:, 15:19, 0])/100
#     #print("feet: ")
#     #print(feet)
#     return feet


def foot_contact_obs(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return (contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0).float()
