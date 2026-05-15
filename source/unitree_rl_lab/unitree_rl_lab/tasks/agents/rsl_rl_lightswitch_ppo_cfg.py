from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg

from .rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class LightSwitchPPORunnerCfg(BasePPORunnerCfg):
    num_steps_per_env = 32
    max_iterations = 10000
    save_interval = 1000
    experiment_name = "unitree_go2_lightswitch_4l"

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
        num_mini_batches=6,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
