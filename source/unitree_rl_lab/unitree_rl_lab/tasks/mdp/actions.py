from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers import ActionTerm
from isaaclab.utils import configclass
from isaaclab.utils.buffers import DelayBuffer
from isaaclab_tasks.manager_based.navigation.mdp.pre_trained_policy_action import (
    PreTrainedPolicyAction,
    PreTrainedPolicyActionCfg,
)


class DelayedJointPositionAction(JointPositionAction):
    """Joint-position action with randomized per-environment policy-step delay."""

    cfg: DelayedJointPositionActionCfg

    def __init__(self, cfg: DelayedJointPositionActionCfg, env):
        if cfg.min_delay < 0 or cfg.max_delay < cfg.min_delay:
            raise ValueError(
                f"Invalid action delay range [{cfg.min_delay}, {cfg.max_delay}]; "
                "expected 0 <= min_delay <= max_delay."
            )
        super().__init__(cfg, env)
        self._delay_buffer = DelayBuffer(cfg.max_delay, self.num_envs, self.device)

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        self._processed_actions = self._delay_buffer.compute(self._processed_actions)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None or env_ids == slice(None):
            env_ids = None
            num_envs = self.num_envs
        else:
            num_envs = len(env_ids)
        delays = torch.randint(
            self.cfg.min_delay,
            self.cfg.max_delay + 1,
            (num_envs,),
            device=self.device,
            dtype=torch.int,
        )
        self._delay_buffer.set_time_lag(delays, env_ids)
        self._delay_buffer.reset(env_ids)


@configclass
class DelayedJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for policy-step-delayed joint-position actions."""

    class_type: type[ActionTerm] = DelayedJointPositionAction
    min_delay: int = 0
    max_delay: int = 2


class ResettablePreTrainedPolicyAction(PreTrainedPolicyAction):
    """Bounded pre-trained policy command that resets its nested low-level action term."""

    cfg: ResettablePreTrainedPolicyActionCfg

    def __init__(self, cfg: ResettablePreTrainedPolicyActionCfg, env):
        super().__init__(cfg, env)
        if len(cfg.command_limits) != self.action_dim or any(limit <= 0.0 for limit in cfg.command_limits):
            raise ValueError(f"command_limits must contain {self.action_dim} positive values")
        if len(cfg.initial_command_limits) != self.action_dim or any(
            limit <= 0.0 for limit in cfg.initial_command_limits
        ):
            raise ValueError(f"initial_command_limits must contain {self.action_dim} positive values")
        self._command_limits = torch.tensor(cfg.command_limits, device=self.device, dtype=torch.float32)
        self._initial_command_limits = torch.tensor(
            cfg.initial_command_limits, device=self.device, dtype=torch.float32
        )

    def process_actions(self, actions: torch.Tensor) -> None:
        """Smoothly bound commands while increasing their limits during early training."""
        if self.cfg.command_limit_curriculum_steps <= 0:
            alpha = 1.0
        else:
            alpha = min(1.0, self._env.common_step_counter / self.cfg.command_limit_curriculum_steps)
        active_limits = self._initial_command_limits + alpha * (
            self._command_limits - self._initial_command_limits
        )
        # self._raw_actions[:] = torch.tanh(actions) * active_limits
        self._raw_actions[:] = torch.clamp(
            actions,
            min=-active_limits,
            max=active_limits,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self.low_level_actions[env_ids] = 0.0
        self._low_level_action_term.reset(env_ids)


@configclass
class ResettablePreTrainedPolicyActionCfg(PreTrainedPolicyActionCfg):
    """Configuration for a bounded, reset-aware hierarchical policy action."""

    class_type: type[ActionTerm] = ResettablePreTrainedPolicyAction
    command_limits: tuple[float, float, float] = (1.0, 1.0, 1.0)
    initial_command_limits: tuple[float, float, float] = (0.1, 0.1, 0.1)
    command_limit_curriculum_steps: int = 0
