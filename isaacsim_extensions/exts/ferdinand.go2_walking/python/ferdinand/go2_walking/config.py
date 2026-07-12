"""Runtime configuration for the bundled Go2 velocity policy."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
DEFAULT_POLICY_PATH = DATA_DIR / "policy.pt"
DEFAULT_DEPLOY_CONFIG_PATH = DATA_DIR / "deploy.yaml"

GO2_SDK_JOINT_NAMES = (
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
)
GO2_FOOT_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")


@dataclass(frozen=True)
class DeployConfig:
    """Validated subset of the Isaac Lab deployment configuration."""

    joint_names: tuple[str, ...]
    step_dt: float
    stiffness: tuple[float, ...]
    damping: tuple[float, ...]
    default_joint_pos: tuple[float, ...]
    action_scale: tuple[float, ...]
    action_offset: tuple[float, ...]
    command_min: tuple[float, float, float]
    command_max: tuple[float, float, float]


def load_deploy_config(path: str | os.PathLike[str] = DEFAULT_DEPLOY_CONFIG_PATH) -> DeployConfig:
    """Load and validate the deployment values used by the selected policy."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Go2 deployment config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)

    joint_ids_map = tuple(int(index) for index in data["joint_ids_map"])
    joint_names = tuple(GO2_SDK_JOINT_NAMES[index] for index in joint_ids_map)
    action = data["actions"]["JointPositionAction"]
    ranges = data["commands"]["base_velocity"]["ranges"]
    command_keys = ("lin_vel_x", "lin_vel_y", "ang_vel_z")

    cfg = DeployConfig(
        joint_names=joint_names,
        step_dt=float(data["step_dt"]),
        stiffness=tuple(float(value) for value in data["stiffness"]),
        damping=tuple(float(value) for value in data["damping"]),
        default_joint_pos=tuple(float(value) for value in data["default_joint_pos"]),
        action_scale=tuple(float(value) for value in action["scale"]),
        action_offset=tuple(float(value) for value in action["offset"]),
        command_min=tuple(float(ranges[key][0]) for key in command_keys),
        command_max=tuple(float(ranges[key][1]) for key in command_keys),
    )
    vector_lengths = {
        len(cfg.joint_names),
        len(cfg.stiffness),
        len(cfg.damping),
        len(cfg.default_joint_pos),
        len(cfg.action_scale),
        len(cfg.action_offset),
    }
    if vector_lengths != {12}:
        raise ValueError(f"Go2 deployment vectors must all contain 12 values, got {sorted(vector_lengths)}")
    if cfg.step_dt <= 0:
        raise ValueError(f"Go2 policy step_dt must be positive, got {cfg.step_dt}")
    return cfg


def resolve_go2_usd(usd_path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the training-model Go2 USD without embedding a user-specific path."""
    candidates: list[Path] = []
    if usd_path:
        candidates.append(Path(usd_path).expanduser())
    env_path = os.environ.get("GO2_MODEL_USD", "").strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())

    for parent in PACKAGE_DIR.parents:
        if (parent / "isaacsim_extensions").is_dir() and (parent / "source").is_dir():
            candidates.append(parent.parent.parent / "unitree_model" / "Go2" / "usd" / "go2.usd")
            break

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    rendered = "\n  - ".join(str(candidate) for candidate in candidates) or "<none>"
    raise FileNotFoundError(
        "Training-model Go2 USD was not found. Pass usd_path or set GO2_MODEL_USD. "
        f"Checked:\n  - {rendered}"
    )
