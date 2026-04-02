from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def foot_contact_too_long(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    max_contact_time_s: float,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    return torch.any(contact_time > max_contact_time_s, dim=-1)
