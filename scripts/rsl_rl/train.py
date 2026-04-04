# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""


import gymnasium as gym
import os
import pathlib
import sys

sys.path.insert(0, f"{pathlib.Path(__file__).parent.parent}")
from list_envs import import_packages  # noqa: F401

sys.path.pop(0)

tasks = []
for task_spec in gym.registry.values():
    if "Unitree" in task_spec.id and "Isaac" not in task_spec.id:
        tasks.append(task_spec.id)

import argparse

import argcomplete

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
VIDEO_WIDTH = 1280 # 1920
VIDEO_HEIGHT = 720 # 1080
# DEFAULT_VIDEO_NUM_ENVS = 32
# MAX_VIDEO_NUM_ENVS = 256
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--video_interval",
    type=int,
    default=200,
    help="Interval between video recordings (in learning iterations).",
)
parser.add_argument(
    "--camera_mode",
    type=str,
    default="fixed",
    choices=["fixed", "follow"],
    help="Camera mode used for rendering/video capture.",
)
parser.add_argument(
    "--camera_eye",
    type=float,
    nargs=3,
    # default=[-30.0, 64.0, 4.5],
    # default=[-60.0, 0.0, 7.0], # good for push task
    # default=[-20.0, 0.0, 5.0], # good for 32 envs
    default=[-85.0, 0.0, 10.0],
    help="Camera eye position for fixed/follow modes.",
)
parser.add_argument(
    "--camera_lookat",
    type=float,
    nargs=3,
    # default=[-28.0, 0.0, -20.0],
    default=[0.0, 0.0, -25.0],
    # default=[0.0, 0.0, 0.0], # good for 32 envs
    help="Camera look-at target for fixed/follow modes.",
)
parser.add_argument(
    "--camera_follow_prim",
    type=str,
    default="{ENV_REGEX_NS}/Robot/base",
    help="Prim path to follow when using follow camera mode.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, choices=tasks, help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument(
    "--low_level_policy_path",
    type=str,
    default=None,
    help="Path to the pretrained low-level policy.pt used by push tasks.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
argcomplete.autocomplete(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.low_level_policy_path:
    os.environ["GO2_PUSH_LOW_LEVEL_POLICY_PATH"] = args_cli.low_level_policy_path
os.environ["GO2_PUSH_COLOR_SEED"] = str(args_cli.seed)

if hasattr(args_cli, "width"):
    args_cli.width = VIDEO_WIDTH
if hasattr(args_cli, "height"):
    args_cli.height = VIDEO_HEIGHT

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True
    # if args_cli.num_envs is None:
    #     args_cli.num_envs = DEFAULT_VIDEO_NUM_ENVS
    #     print(
    #         f"[INFO] Video capture enabled without --num_envs; defaulting to {DEFAULT_VIDEO_NUM_ENVS} envs "
    #         "to avoid renderer failures with the 4L training default."
    #     )
    # elif args_cli.num_envs > MAX_VIDEO_NUM_ENVS:
    #     raise ValueError(
    #         f"Video capture is only supported up to {MAX_VIDEO_NUM_ENVS} envs in this script; "
    #         f"got --num_envs {args_cli.num_envs}. Reduce --num_envs or disable --video."
    #     )

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# for distributed training, check minimum supported rsl-rl version
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import inspect
import shutil
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner  # TODO: Consider printing the experiment name in the terminal.

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if agent_cfg.logger == "wandb":
        os.environ.setdefault("WANDB_ENTITY", "fetchrobot")
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # configure viewer/camera settings for rendering
    if hasattr(env_cfg, "viewer"):
        env_cfg.viewer.eye = list(args_cli.camera_eye)
        env_cfg.viewer.lookat = list(args_cli.camera_lookat)
        if args_cli.camera_mode == "follow":
            follow_attr_candidates = [
                "follow_prim_path",
                "follow_asset_path",
                "follow_target",
                "follow_path",
            ]
            for attr_name in follow_attr_candidates:
                if hasattr(env_cfg.viewer, attr_name):
                    setattr(env_cfg.viewer, attr_name, args_cli.camera_follow_prim)
                    break

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # This way, the Ray Tune workflow can extract experiment name.
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    video_folder = None

    # wrap for video recording
    if args_cli.video:
        video_folder = os.path.join(log_dir, "videos", "train")
        num_steps_per_env = getattr(agent_cfg, "num_steps_per_env", 1)
        video_interval_steps = max(1, args_cli.video_interval * num_steps_per_env)
        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step % video_interval_steps == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    if args_cli.video and agent_cfg.logger == "wandb":
        import wandb

        wandb_run = getattr(wandb, "run", None)
        wandb_save = getattr(wandb, "save", None)
        if wandb_run is not None and callable(wandb_save) and video_folder is not None:
            wandb_save(os.path.join(video_folder, "*.mp4"), policy="live")
            print(f"[INFO] W&B live upload enabled for videos in: {video_folder}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    export_deploy_cfg(env.unwrapped, log_dir)
    # copy the environment configuration file to the log directory
    shutil.copy(
        inspect.getfile(env_cfg.__class__),
        os.path.join(log_dir, "params", os.path.basename(inspect.getfile(env_cfg.__class__))),
    )

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
