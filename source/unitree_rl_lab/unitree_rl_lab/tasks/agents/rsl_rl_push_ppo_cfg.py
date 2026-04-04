# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import BasePPORunnerCfg

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

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

    critic = {
        "class_name": "MLPModel",
        "hidden_dims": [512, 256, 128],
        "activation": "elu",
    }

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=6, # 4
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )