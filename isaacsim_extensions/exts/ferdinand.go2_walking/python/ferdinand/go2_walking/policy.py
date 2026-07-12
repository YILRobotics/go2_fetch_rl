"""PhysX controller for the selected Go2 4L foot-force velocity policy."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import carb
import isaacsim.core.experimental.utils.transform as transform_utils
import omni
import omni.physics.tensors
import torch
import warp as wp
from isaacsim.core.experimental.prims import Articulation
from isaacsim.core.experimental.utils.prim import get_prim_at_path
from isaacsim.core.experimental.utils.stage import define_prim
from isaacsim.core.simulation_manager import SimulationManager
from omni.physics.core import get_physics_simulation_interface
from pxr import PhysxSchema, Usd

from .config import (
    DEFAULT_DEPLOY_CONFIG_PATH,
    DEFAULT_POLICY_PATH,
    GO2_FOOT_NAMES,
    GO2_SDK_JOINT_NAMES,
    DeployConfig,
    load_deploy_config,
    resolve_go2_usd,
)


class Go2VelocityPolicy:
    """Run Ferdinand's 49-observation Go2 4L velocity policy in PhysX."""

    OBSERVATION_SIZE = 49
    ACTION_SIZE = 12

    def __init__(
        self,
        prim_path: str,
        *,
        usd_path: str | None = None,
        position: Sequence[float] | None = None,
        orientation: Sequence[float] | None = None,
        policy_path: str | None = None,
        deploy_config_path: str | None = None,
    ) -> None:
        self._prim_path = prim_path
        self._usd_path = resolve_go2_usd(usd_path)
        self._config: DeployConfig = load_deploy_config(deploy_config_path or DEFAULT_DEPLOY_CONFIG_PATH)

        prim = get_prim_at_path(prim_path)
        if not prim.IsValid():
            prim = define_prim(prim_path, "Xform")
            prim.GetReferences().AddReference(str(self._usd_path))

        self._foot_paths = self._find_and_prepare_feet(prim)
        self.robot = Articulation(
            paths=prim_path,
            positions=list(position) if position is not None else None,
            orientations=list(orientation) if orientation is not None else None,
            reset_xform_op_properties=True,
        )

        self._torch = torch
        self._device = torch.device(str(self.robot._device))
        selected_policy = Path(policy_path or DEFAULT_POLICY_PATH).expanduser().resolve()
        if not selected_policy.is_file():
            raise FileNotFoundError(f"Go2 policy not found: {selected_policy}")
        self.policy = torch.jit.load(str(selected_policy), map_location=self._device).eval()
        self._validate_policy_contract()

        self._policy_to_asset: list[int] | None = None
        self._default_policy = torch.tensor(self._config.default_joint_pos, device=self._device)
        self._action_scale = torch.tensor(self._config.action_scale, device=self._device)
        self._action_offset = torch.tensor(self._config.action_offset, device=self._device)
        self._command_min = torch.tensor(self._config.command_min, device=self._device)
        self._command_max = torch.tensor(self._config.command_max, device=self._device)
        self._previous_action = torch.zeros((1, self.ACTION_SIZE), device=self._device)
        self._current_action = torch.zeros_like(self._previous_action)
        self._contact_sim_view = None
        self._contact_view = None
        self._time_since_policy = self._config.step_dt

    def _find_and_prepare_feet(self, root_prim: Usd.Prim) -> tuple[str, ...]:
        found: dict[str, list[str]] = {name: [] for name in GO2_FOOT_NAMES}
        for prim in Usd.PrimRange(root_prim):
            name = prim.GetName()
            if name in found:
                found[name].append(str(prim.GetPath()))
        invalid = {name: paths for name, paths in found.items() if len(paths) != 1}
        if invalid:
            raise RuntimeError(f"Expected exactly one rigid prim for each Go2 foot, got: {invalid}")
        paths = tuple(found[name][0] for name in GO2_FOOT_NAMES)
        stage = omni.usd.get_context().get_stage()
        for path in paths:
            foot_prim = stage.GetPrimAtPath(path)
            report_api = PhysxSchema.PhysxContactReportAPI.Apply(foot_prim)
            report_api.CreateThresholdAttr().Set(0.0)
        return paths

    def _validate_policy_contract(self) -> None:
        with self._torch.inference_mode():
            output = self.policy(self._torch.zeros((1, self.OBSERVATION_SIZE), device=self._device))
        if tuple(output.shape) != (1, self.ACTION_SIZE):
            raise ValueError(
                f"Expected Go2 policy output shape (1, {self.ACTION_SIZE}), got {tuple(output.shape)}"
            )

    def _resolve_joint_order(self) -> None:
        asset_names = tuple(self.robot.dof_names)
        if len(asset_names) != self.ACTION_SIZE or set(asset_names) != set(GO2_SDK_JOINT_NAMES):
            raise RuntimeError(
                f"Go2 asset must expose exactly these 12 DOFs: {GO2_SDK_JOINT_NAMES}; got {asset_names}"
            )
        self._policy_to_asset = [asset_names.index(name) for name in self._config.joint_names]

    def _policy_vector_to_asset(self, policy_values: "torch.Tensor") -> "torch.Tensor":
        if self._policy_to_asset is None:
            raise RuntimeError("Go2 joint order is unavailable before initialize()")
        output_shape = (*policy_values.shape[:-1], self.ACTION_SIZE)
        asset_values = self._torch.empty(output_shape, dtype=policy_values.dtype, device=policy_values.device)
        asset_values[..., self._policy_to_asset] = policy_values
        return asset_values

    def _sdk_vector_to_asset(self, sdk_values: Sequence[float]) -> "torch.Tensor":
        asset_names = tuple(self.robot.dof_names)
        values_by_name = dict(zip(GO2_SDK_JOINT_NAMES, sdk_values))
        return self._torch.tensor([values_by_name[name] for name in asset_names], device=self._device)

    def initialize(self) -> None:
        """Initialize articulation state, PD gains, and the four-foot contact view."""
        get_engine = getattr(SimulationManager, "get_active_physics_engine", None)
        if get_engine is not None:
            engine = (get_engine() or "").lower()
            if engine != "physx":
                raise RuntimeError(f"Go2VelocityPolicy supports PhysX only; active engine is {engine!r}")
        if not self.robot.is_physics_tensor_entity_valid():
            raise RuntimeError("Go2 articulation physics tensors are not ready")

        self._resolve_joint_order()
        self.robot.set_dof_drive_types("force")
        get_physics_simulation_interface().flush_changes()
        self.robot.switch_dof_control_mode("position")

        default_asset = self._policy_vector_to_asset(self._default_policy)
        stiffness_asset = self._sdk_vector_to_asset(self._config.stiffness)
        damping_asset = self._sdk_vector_to_asset(self._config.damping)
        zeros_asset = self._torch.zeros_like(default_asset)
        self.robot.set_dof_positions(wp.from_torch(default_asset))
        self.robot.set_dof_velocities(wp.from_torch(zeros_asset))
        self.robot.set_dof_gains(wp.from_torch(stiffness_asset), wp.from_torch(damping_asset))
        self.robot.set_dof_friction_properties(static_frictions=0.01, dynamic_frictions=0.01)
        self.robot.set_default_state(
            dof_positions=wp.from_torch(default_asset),
            dof_velocities=wp.from_torch(zeros_asset),
        )

        self._contact_sim_view = SimulationManager.get_physics_sim_view()
        if self._contact_sim_view is None:
            raise RuntimeError("PhysX simulation view is not ready for Go2 foot-force sensing")
        self._contact_view = self._contact_sim_view.create_rigid_contact_view(list(self._foot_paths))
        if self._contact_view.sensor_count != len(GO2_FOOT_NAMES):
            raise RuntimeError(
                f"Expected four Go2 contact sensors, got {self._contact_view.sensor_count}"
            )
        self.reset_policy_state()

    def invalidate_physics_handles(self) -> None:
        """Release contact handles invalidated by a timeline stop."""
        self._contact_view = None
        self._contact_sim_view = None
        self.reset_policy_state()

    def reset_policy_state(self) -> None:
        """Reset recurrent input state and schedule an immediate policy update."""
        self._previous_action.zero_()
        self._current_action.zero_()
        self._time_since_policy = self._config.step_dt

    def post_reset(self) -> None:
        """Restore the configured standing pose and clear policy history."""
        self.robot.reset_to_default_state()
        self.reset_policy_state()

    def _foot_force_observation(self) -> "torch.Tensor":
        if self._contact_view is None:
            raise RuntimeError("Go2 contact view is not initialized")
        forces = self._contact_view.get_net_contact_forces(self._config.step_dt / 4.0)
        if not isinstance(forces, self._torch.Tensor):
            forces = self._torch.as_tensor(forces, device=self._device)
        forces = forces.to(self._device).reshape(len(GO2_FOOT_NAMES), 3)
        magnitudes = self._torch.linalg.vector_norm(forces, dim=-1)
        magnitudes = self._torch.nan_to_num(magnitudes, nan=0.0, posinf=150.0, neginf=0.0)
        return magnitudes.clamp_(0.0, 150.0).mul_(0.01).unsqueeze(0)

    def compute_observation(self, command: Sequence[float] | "torch.Tensor") -> "torch.Tensor":
        """Build the exact 49-value observation used during 4L training."""
        _, angular_velocity_w = self.robot.get_velocities()
        _, orientation_w = self.robot.get_world_poses()
        rotation_wb = wp.to_torch(transform_utils.quaternion_to_rotation_matrix(orientation_w))
        rotation_bw = rotation_wb.transpose(1, 2)
        angular_velocity_b = self._torch.bmm(rotation_bw, wp.to_torch(angular_velocity_w).unsqueeze(-1)).squeeze(-1)
        gravity_w = self._torch.tensor([0.0, 0.0, -1.0], device=self._device).expand(1, 3)
        gravity_b = self._torch.bmm(rotation_bw, gravity_w.unsqueeze(-1)).squeeze(-1)

        command_tensor = self._torch.as_tensor(command, dtype=self._torch.float32, device=self._device).reshape(1, 3)
        command_tensor = self._torch.maximum(self._torch.minimum(command_tensor, self._command_max), self._command_min)
        joint_pos_asset = wp.to_torch(self.robot.get_dof_positions())
        joint_vel_asset = wp.to_torch(self.robot.get_dof_velocities())
        joint_pos_policy = joint_pos_asset[:, self._policy_to_asset]
        joint_vel_policy = joint_vel_asset[:, self._policy_to_asset]

        observation = self._torch.cat(
            (
                angular_velocity_b * 0.2,
                gravity_b,
                command_tensor,
                joint_pos_policy - self._default_policy,
                joint_vel_policy * 0.05,
                self._foot_force_observation(),
                self._previous_action,
            ),
            dim=-1,
        )
        if tuple(observation.shape) != (1, self.OBSERVATION_SIZE) or not self._torch.isfinite(observation).all():
            raise RuntimeError(f"Invalid Go2 policy observation: shape={tuple(observation.shape)}")
        return observation

    def forward(self, dt: float, command: Sequence[float] | "torch.Tensor") -> None:
        """Advance policy inference while holding joint targets between 50 Hz updates."""
        self._time_since_policy += float(dt)
        if self._time_since_policy + 1e-9 < self._config.step_dt:
            return
        self._time_since_policy %= self._config.step_dt
        observation = self.compute_observation(command)
        with self._torch.inference_mode():
            action = self.policy(observation).reshape(1, self.ACTION_SIZE)
        if not self._torch.isfinite(action).all():
            raise RuntimeError("Go2 policy produced a non-finite action")
        self._current_action.copy_(action)
        self._previous_action.copy_(action)
        target_policy = self._action_offset + action.squeeze(0) * self._action_scale
        target_asset = self._policy_vector_to_asset(target_policy).unsqueeze(0)
        self.robot.set_dof_position_targets(wp.from_torch(target_asset))
