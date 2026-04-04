import gymnasium as gym


def _safe_register(env_id: str, kwargs: dict):
    if env_id in gym.registry:
        return
    gym.register(
        id=env_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs=kwargs,
    )


_safe_register(
    "Unitree-Go2-Velocity-3L",
    {
        "env_cfg_entry_point": f"{__name__}.velocity_3l_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_3l_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

_safe_register(
    "Unitree-Go2-Velocity-4L",
    {
        "env_cfg_entry_point": f"{__name__}.velocity_4l_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_4l_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

_safe_register(
    "Unitree-Go2-PushCube-4L",
    {
        "env_cfg_entry_point": f"{__name__}.push_env_cfg:RobotPushEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.push_env_cfg:RobotPushPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.agents.rsl_rl_push_ppo_cfg:PushPPORunnerCfg",
    },
)
