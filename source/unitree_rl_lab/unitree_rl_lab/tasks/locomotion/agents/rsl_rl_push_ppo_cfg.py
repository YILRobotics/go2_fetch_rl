# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class PushPPORunnerCfg(BasePPORunnerCfg):
    """Dedicated PPO runner settings for the Go2 push task.

    Keeps locomotion defaults but allows independent tuning from velocity tasks.
    """

    # Slightly longer horizon and training budget are usually helpful for sparse-ish push objectives.
    num_steps_per_env = 32
    max_iterations = 2000
    save_interval = 100
    experiment_name = "unitree_go2_pushcube_4l"

    actor = {
        "class_name": "MLPModel",
        "hidden_dims": [512, 256, 128],
        "activation": "elu",
        "distribution_cfg": {
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "log",
        },
    }
