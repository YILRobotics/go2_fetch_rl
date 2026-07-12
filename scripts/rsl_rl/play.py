# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
import os
from importlib.metadata import version
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
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
    # default=[-25.0, 0.0, 5.0],
    default=[-8.0, 8.0, 4.5],
    help="Camera eye position for fixed/follow modes.",
)
parser.add_argument(
    "--camera_lookat",
    type=float,
    nargs=3,
    # default=[-10.0, 0.0, 1.0],
    default=[-1.5, 1.5, 0.0],
    help="Camera look-at target for fixed/follow modes.",
)
parser.add_argument(
    "--camera_follow_prim",
    type=str,
    default="{ENV_REGEX_NS}/Robot/base",
    help="Prim path to follow when using follow camera mode.",
)
parser.add_argument(
    "--vel_arrows",
    action="store_true",
    default=False,
    help="Show base-velocity command debug arrows during play.",
)
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--low_level_policy_path",
    type=str,
    default=None,
    help="Path to the pretrained low-level policy.pt used by push tasks.",
)
parser.add_argument(
    "--play_reset_mode",
    type=str,
    default="standard",
    choices=["standard", "success_keep_robot"],
    help="Reset behavior for play mode. 'success_keep_robot' keeps the robot on success and only respawns the cube.",
)

# Terrain
parser.add_argument(
    "--terrain_rows",
    type=int,
    default=3,
    help="Override terrain-generator row count used in play mode.",
)
parser.add_argument(
    "--terrain_cols",
    type=int,
    default=3,
    help="Override terrain-generator column count used in play mode.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.low_level_policy_path:
    os.environ["GO2_PUSH_LOW_LEVEL_POLICY_PATH"] = args_cli.low_level_policy_path
os.environ["GO2_PUSH_COLOR_SEED"] = str(args_cli.seed)
os.environ["GO2_PUSH_PLAY_RESET_MODE"] = args_cli.play_reset_mode
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import time
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def _flatten_observation(value) -> list[float]:
    """Flatten a scalar or nested observation value into CSV-compatible floats."""
    if isinstance(value, (list, tuple)):
        return [component for item in value for component in _flatten_observation(item)]
    return [float(value)]


def _all_observation_columns(unwrapped) -> tuple[list[str], list[float]]:
    """Return named components for every cached policy and critic observation term."""
    columns: list[str] = []
    values: list[float] = []
    for qualified_name, term_value in unwrapped.observation_manager.get_active_iterable_terms(env_idx=0):
        group_name, term_name = qualified_name.split("-", maxsplit=1)
        components = _flatten_observation(term_value)
        columns.extend(
            f"obs__{group_name}__{term_name}__{component_index}"
            for component_index in range(len(components))
        )
        values.extend(components)
    return columns, values


def _record_high_level_observations(env) -> None:
    """Append environment zero's high-level play observations to one CSV row."""
    unwrapped = env.unwrapped
    foot_force = getattr(unwrapped, "_last_foot_force_observation", None)
    cube_position = getattr(unwrapped, "_last_cube_position_xy_observation", None)
    cube_velocity = getattr(unwrapped, "_last_cube_velocity_xy_observation", None)
    if foot_force is None or cube_position is None or cube_velocity is None:
        return

    record_path = Path(unwrapped.foot_force_record_path)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not record_path.exists()
    step = getattr(unwrapped, "_play_record_sample", 0)
    record_dt = unwrapped._play_record_dt
    command = unwrapped.action_manager.get_term("pre_trained_policy_action").raw_actions
    observation_columns, observation_values = _all_observation_columns(unwrapped)
    row = [step, step * record_dt]
    # Record only environment 0; the other parallel environments are not written to this CSV.
    row.extend(foot_force[0].cpu().tolist())
    row.extend(command[0, :3].detach().cpu().tolist())
    row.extend(cube_position[0].cpu().tolist())
    row.extend(cube_velocity[0].cpu().tolist())
    row.extend(observation_values)

    with record_path.open("a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if write_header:
            writer.writerow(
                ["step", "time_s"]
                + [f"normal_force_foot_{index}" for index in range(4)]
                + ["cmd_lin_vel_x", "cmd_lin_vel_y", "cmd_ang_vel_z"]
                + ["cube_pos_obs_x", "cube_pos_obs_y"]
                + ["cube_vel_obs_x", "cube_vel_obs_y"]
                + observation_columns
            )
        writer.writerow(row)
    unwrapped._play_record_sample = step + 1


def _apply_play_terrain_overrides(env_cfg):
    """Apply optional terrain-generator overrides from CLI for play runs."""
    if args_cli.terrain_rows is None and args_cli.terrain_cols is None:
        return

    scene_cfg = getattr(env_cfg, "scene", None)
    terrain_cfg = getattr(scene_cfg, "terrain", None) if scene_cfg is not None else None
    terrain_generator_cfg = getattr(terrain_cfg, "terrain_generator", None) if terrain_cfg is not None else None
    if terrain_cfg is None or terrain_generator_cfg is None:
        print("[WARN] Terrain overrides ignored: task does not use a terrain generator.")
        return

    if args_cli.terrain_rows is not None:
        if args_cli.terrain_rows < 1:
            raise ValueError(f"--terrain_rows must be >= 1, got {args_cli.terrain_rows}")
        terrain_generator_cfg.num_rows = args_cli.terrain_rows

    if args_cli.terrain_cols is not None:
        if args_cli.terrain_cols < 1:
            raise ValueError(f"--terrain_cols must be >= 1, got {args_cli.terrain_cols}")
        terrain_generator_cfg.num_cols = args_cli.terrain_cols

    print(
        "[INFO] Applied terrain overrides: "
        f"rows={terrain_generator_cfg.num_rows}, "
        f"cols={terrain_generator_cfg.num_cols}"
    )


def _unique_video_path(path: Path) -> Path:
    """Return a unique file path by appending _N before the extension."""
    if not path.exists():
        return path

    base_name = path.stem
    suffix = path.suffix
    index = 1
    while True:
        candidate = path.with_name(f"{base_name}_{index}{suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def _rotate_existing_video_file(path: Path):
    """Rename an existing video file to *_N before writing a new one."""
    if not path.exists():
        return
    dst = _unique_video_path(path)
    path.rename(dst)
    print(f"[INFO] Existing video renamed to avoid overwrite: {dst}")


def _rename_recorded_video(video_folder: str, prefix: str, timestamp: str) -> Path | None:
    """Give the newest Gym video a task-specific timestamped name."""
    folder = Path(video_folder)
    if not folder.exists():
        print(f"[WARN] Video folder does not exist: {folder}")
        return None

    videos = list(folder.glob("rl-video-step-*.mp4"))
    if not videos:
        print(f"[WARN] No videos found in: {folder}")
        return None

    source = max(videos, key=lambda path: path.stat().st_mtime)
    destination = _unique_video_path(folder / f"{prefix}_video_{timestamp}.mp4")
    source.rename(destination)

    source_metadata = source.with_suffix(".meta.json")
    if source_metadata.exists():
        source_metadata.rename(destination.with_suffix(".meta.json"))
    return destination


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    _apply_play_terrain_overrides(env_cfg)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    if args_cli.vel_arrows:
        commands_cfg = getattr(env_cfg, "commands", None)
        base_velocity_cfg = getattr(commands_cfg, "base_velocity", None) if commands_cfg is not None else None
        if base_velocity_cfg is not None and hasattr(base_velocity_cfg, "debug_vis"):
            base_velocity_cfg.debug_vis = True
            print("[INFO] Enabled velocity command arrows (commands.base_velocity.debug_vis=True).")
        else:
            print("[WARN] --vel_arrows ignored: commands.base_velocity.debug_vis is unavailable for this task.")

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

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO*q experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    recording_timestamp = time.strftime("%Y%m%d_%H%M%S")
    task_name = (args_cli.task or "").lower()
    video_prefix = "push" if "push" in task_name else "vel" if "velocity" in task_name else "play"

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    video_folder = None
    if args_cli.video:
        video_folder = os.path.join(log_dir, "videos", "play")
        _rotate_existing_video_file(Path(video_folder) / "rl-video-step-0.mp4")
        _rotate_existing_video_file(Path(video_folder) / "rl-video-step-0.meta.json")
        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step == 1,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    env.unwrapped.print_foot_force = True
    try:
        env.unwrapped.scene["cube"]
        record_high_level_observations = True
    except KeyError:
        record_high_level_observations = False
    # Non-push tasks keep the existing observation-level foot-force recorder.
    env.unwrapped.record_foot_force = not record_high_level_observations
    env.unwrapped.foot_force_record_path = os.path.join(
        log_dir, "recordings", "play", f"recording_{recording_timestamp}.csv"
    )
    if record_high_level_observations:
        high_level_action = env.unwrapped.action_manager.get_term("pre_trained_policy_action")
        env.unwrapped._play_record_sample = 0
        env.unwrapped._play_record_dt = env.unwrapped.physics_dt * high_level_action.cfg.low_level_decimation
        env.unwrapped._play_record_callback = lambda: _record_high_level_observations(env)
    print(f"[INFO]: Recording play observations to: {env.unwrapped.foot_force_record_path}")

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner

        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # # extract the neural network module
    # # we do this in a try-except to maintain backwards compatibility.
    # try:
    #     # version 2.3 onwards
    #     policy_nn = runner.alg.policy
    # except AttributeError:
    #     # version 2.2 and below
    #     policy_nn = runner.alg.actor_critic

    # # extract the normalizer
    # if hasattr(policy_nn, "actor_obs_normalizer"):
    #     normalizer = policy_nn.actor_obs_normalizer
    # elif hasattr(policy_nn, "student_obs_normalizer"):
    #     normalizer = policy_nn.student_obs_normalizer
    # else:
    #     normalizer = None

    # # export policy to onnx/jit
    # export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    # export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    # export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    runner.export_policy_to_jit(export_model_dir, filename="policy.pt")
    runner.export_policy_to_onnx(export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()
    if args_cli.video and video_folder is not None:
        newest_video = _rename_recorded_video(video_folder, video_prefix, recording_timestamp)
        if newest_video is not None:
            print(f"[INFO] Newest video file: {newest_video.resolve()}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
