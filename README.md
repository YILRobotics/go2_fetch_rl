# Go2 Fechrobot RL with IsaacSim/Lab

## Environment Setup 

### 1. Follow the original README.md instructions (below)

### 2. Activate the environment
```bash
source ~/miniconda3/bin/activate
conda activate isaac_lab_ferdinand
or
conda activate isaac_lab_alessandro
or
conda activate isaac_lab
```
### ⚠️ DO NOT CONDA INIT
Do not run `conda init` or uncomment conda initialization in `.bashrc` or add it to your PATH, as it interferes with ROS2.

## Training

### Listing the available tasks:

```bash
./unitree_rl_lab.sh -l # This is a faster version than isaaclab
```

### Unitree-Go2-Velocity-4L Task

```bash
python scripts/rsl_rl/train.py --task Unitree-Go2-Velocity-4L --headless --logger wandb --video --video_interval 500 --video_length 300 --log_project_name f_vel_4l --run_name walk_1
```

**For live:** (don't use `--headless` and set low number of envs)

```bash
python scripts/rsl_rl/train.py --task Unitree-Go2-Velocity-4L --num_envs 32
```

### Unitree-Go2-PushCube-4L Task

```bash
python scripts/rsl_rl/train.py --task Unitree-Go2-PushCube-4L --low_level_policy_path /home/ferdinand/fetchrobot/ferdinand/go2_fetch_rl/logs/rsl_rl/unitree_go2_velocity_4l/2026-04-05_12-01-56_walk_2/exported/policy.pt --headless --logger wandb --video --video_interval 100 --video_length 300 --log_project_name f_pushcube_4l --run_name test_76
```

- #### -> policies are saved in `unitree_rl_lab/logs/rsl_rl`

### 1. BE CAREFUL WHEN CHANGING num_envs: adapt the terrain to cover all the area

### 2. BE CAREFUL WHEN CHANGING ROBOR/CUBE SPAWN RADIUS: adjust the env_spacing so that the envs dont collide and crash

### Env Frames and Why Robots Can Look Overlapped (Velocity vs Push)

- Each vectorized environment (`env_i`) has its own local frame and world origin.
- `env_spacing` controls the default grid distance between those env origins.
- In terrain-generator tasks, `scene.env_origins` can come from sampled terrain patch origins (not only from the default grid).
- Velocity tasks reset robot roots relative to `scene.env_origins`, so many envs can appear in similar world regions when many envs share limited terrain origins.
- Push task uses a custom reset around goal/cube and anchors to cloned env grid origins (`_default_env_origins` when available), and it uses larger spacing (`env_spacing=10.0`), so envs are visually more separated.
- Cross-env collisions are normally filtered by IsaacLab, so overlap in velocity usually looks like visual stacking of different env instances, not true physical interaction between those envs.


### Learning iteration -> is one full PPO cycle:

  1. Collect rollout data for all envs (num_steps_per_env steps each).
  2. Run policy/value updates on that collected batch (multiple epochs/mini-batches).
  3. Print one log block (Learning iteration X/Y).

  For:

  - num_envs = 4096
  - num_steps_per_env = 32

  One iteration collects:

  - 4096 * 32 = 131,072 transitions


### PushCube performance note (tensor caching)

For `Unitree-Go2-PushCube-4L`, frequently reused tensors are cached in `source/unitree_rl_lab/unitree_rl_lab/tasks/push_mdp.py` (for example env ids, goal vectors, and goal-radius observation tensors) to reduce per-step tensor allocations.

This optimization is meant to improve rollout throughput only. Task logic, rewards, and terminations are unchanged, so training behavior should remain the same while collection speed may increase.

How it works:
- Frequently used tensors are created once and reused (`env_ids`, `goal_xy`, `goal_radius`) instead of being recreated every step.
- Scalar broadcasting is used where possible (instead of building full-size tensors just to multiply by one value).
- `goal_position_xy` uses `expand` semantics instead of repeated allocation.

Why it helps:
- Fewer allocations in hot per-step paths reduces overhead in rollout collection.
- At large `num_envs` this can improve `Steps per second` without changing the PPO/task logic.



### Unitree-Go2-LightSwitch-4L Task

```bash
python scripts/rsl_rl/train.py --task Unitree-Go2-LightSwitch-4L --num_envs 16
```

```bash
python scripts/rsl_rl/train.py --task Unitree-Go2-LightSwitch-4L --headless --logger wandb --video --video_interval 75 --video_length 300 --log_project_name f_lightswitch_test --run_name test_1
```
  

## Play/Inference

**Use the newest Run:**

```bash
python scripts/rsl_rl/play.py --task Unitree-Go2-Velocity-4L --num_envs 32
```

**Load one specific checkpoint:**

```bash
python scripts/rsl_rl/play.py --task Unitree-Go2-Velocity-4L --num_envs 32 --checkpoint logs/rsl_rl/unitree_go2_velocity_4l/2026-04-04_21-30-01_walk_1/model_5000.pt
```

### Unitree-Go2-PushCube-4L Task

```bash
python scripts/rsl_rl/play.py \
  --task Unitree-Go2-PushCube-4L \
  --num_envs 16 \
  --checkpoint /home/ferdinand/fetchrobot/ferdinand/go2_fetch_rl/logs/rsl_rl/unitree_go2_pushcube_4l/2026-04-05_21-11-33_test_77/model_2399.pt \
  --low_level_policy_path /home/ferdinand/fetchrobot/ferdinand/go2_fetch_rl/logs/rsl_rl/unitree_go2_velocity_4l/2026-04-05_12-01-56_walk_2/exported/policy.pt \
  --play_reset_mode success_keep_robot
```
- When omitting --low_level_policy_path, the env tries to auto-pick the latest exported 4L velocity policy.

- The push task already wraps the 4-leg velocity policy inside the push action stack in source/unitree_rl_lab/unitree_rl_lab/tasks/push_env_cfg.py:401, so Unitree-Go2-PushCube-4L runs the high-level push policy and the low-level one.

- `play.py` also auto-exports the policy at:

  `logs/rsl_rl/<task_name>/<run_timestamp>_<run_name>/exported/policy.pt`

---

## Use Background Terminal with TMUX

```bash
tmux new -s train
# ctrl+b, then (while holding ctrl) press d to detach
tmux ls
tmux attach -t train
tmux kill-session -t train
```

<br>

---

# Fetchrobot Isaac Sim Extension

### There are 2 options:

### 1. Add Path to Extension filder in Isaac Sim to make it visible

Go to window -> extensions -> three bar menu -> settings -> add the path `/home/ferdinand/fetchrobot/ferdinand/go2_fetch_rl/isaacsim_extensions/exts` 
Then the extenison will show up in the user category

### 2. Launch Isaac Sim with the custom extension

```bash
./isaac-sim.sh --ext-folder /home/ferdinand/fetchrobot/ferdinand/go2_fetch_rl/isaacsim_extensions/exts --enable ferdinand.fetchrobot
```

Then in IsaacSim: Window → Examples → Robotics Examples → General section → Extension. Click load.

If changing something in the `setup_scene()` function: File → New From Stage Template → Empty, then LOAD. (Do this if you change `setup_scene`). Otherwise, just press LOAD.

---

## Fetchrobot Foam Material Parameters

The foam cube in the extension is configured in:
`isaacsim_extensions/exts/ferdinand.fetchrobot/python/ferdinand/fetchrobot/fetchrobot.py`

### Physics material parameters

- `youngs_modulus`: stiffness. Higher = harder/less squishy. Lower = softer/more jelly.
- `poissons_ratio`: sideways bulging when compressed. Closer to `0.5` behaves more incompressible/rubbery.
- `damping_scale`: internal damping. Higher = less wobble/oscillation.
- `elasticity_damping`: extra damping in elastic response. Higher = less bounce.
- `dynamic_friction`: sliding friction while in contact. Higher = less sliding.
- `density`: mass per volume. Higher = heavier, lower = lighter.

### Deformable body / solver parameters

- `simulation_hexahedral_resolution`: deformable mesh resolution. Higher = better shape quality but slower.
- `collision_simplification`: simplifies collision representation for stability/performance.
- `self_collision`: whether the deformable collides with itself.
- `solver_position_iteration_count`: solver effort per step. Higher = more stable/stiffer contact but more compute.

### Ground contact parameters (hover control)

- `rest_offset`: resting separation distance in contact. If too high, object may appear to float.
- `contact_offset`: distance where contacts start being detected. If too high, visible hovering can happen.

### Visual material parameters (matte vs shiny)

- `roughness`: higher = more matte.
- `metallic`: keep near `0.0` for foam.
- `specularColor`: lower values reduce highlights/shininess.
- `diffuseColor`: base color only (visual, not physics).

### Quick tuning cheatsheet

- Too deformable/jelly: increase `youngs_modulus` and/or `solver_position_iteration_count`.
- Too bouncy: increase `damping_scale` and `elasticity_damping`.
- Too heavy: decrease `density`.
- Hovering above ground: decrease `contact_offset`, keep `rest_offset` near `0.0`.
- Too shiny: increase `roughness`, decrease `specularColor`.

---

### Kill a process

```bash
kill -9 PID # PID = number of process
```

<br>

<br>

<br>

<br>

<br>

## Original Repo Readme:

<br>

<br>


# Unitree RL Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)
[![Discord](https://img.shields.io/badge/-Discord-5865F2?style=flat&logo=Discord&logoColor=white)](https://discord.gg/ZwcVwxv5rq)


## Overview

This project provides a set of reinforcement learning environments for Unitree robots, built on top of [IsaacLab](https://github.com/isaac-sim/IsaacLab).

Currently supports Unitree **Go2**, **H1** and **G1-29dof** robots.

<div align="center">

| <div align="center"> Isaac Lab </div> | <div align="center">  Mujoco </div> |  <div align="center"> Physical </div> |
|--- | --- | --- |
| [<img src="https://oss-global-cdn.unitree.com/static/d879adac250648c587d3681e90658b49_480x397.gif" width="240px">](g1_sim.gif) | [<img src="https://oss-global-cdn.unitree.com/static/3c88e045ab124c3ab9c761a99cb5e71f_480x397.gif" width="240px">](g1_mujoco.gif) | [<img src="https://oss-global-cdn.unitree.com/static/6c17c6cf52ec4e26bbfab1fbf591adb2_480x270.gif" width="240px">](g1_real.gif) |

</div>

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
- Install the Unitree RL IsaacLab standalone environments.

  - Clone or copy this repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

    ```bash
    git clone https://github.com/unitreerobotics/unitree_rl_lab.git
    ```
  - Use a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    conda activate env_isaaclab
    ./unitree_rl_lab.sh -i
    # restart your shell to activate the environment changes.
    ```
- Download unitree robot description files

  *Method 1: Using USD Files*
  - Download unitree usd files from [unitree_model](https://huggingface.co/datasets/unitreerobotics/unitree_model/tree/main), keeping folder structure
    ```bash
    git clone https://huggingface.co/datasets/unitreerobotics/unitree_model
    ```
  - Config `UNITREE_MODEL_DIR` in `source/unitree_rl_lab/unitree_rl_lab/assets/unitree.py`.

    ```bash
    UNITREE_MODEL_DIR = "</home/user/projects/unitree_usd>"
    ```

  *Method 2: Using URDF Files [Recommended]* Only for Isaacsim >= 5.0
  -  Download unitree robot urdf files from [unitree_ros](https://github.com/unitreerobotics/unitree_ros)
      ```
      git clone https://github.com/unitreerobotics/unitree_ros.git
      ```
  - Config `UNITREE_ROS_DIR` in `source/unitree_rl_lab/unitree_rl_lab/assets/unitree.py`.
    ```bash
    UNITREE_ROS_DIR = "</home/user/projects/unitree_ros/unitree_ros>"
    ```
  - [Optional]: change *robot_cfg.spawn* if you want to use urdf files



- Verify that the environments are correctly installed by:

  - Listing the available tasks:

    ```bash
    ./unitree_rl_lab.sh -l # This is a faster version than isaaclab
    ```
  - Running a task:

    ```bash
    ./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity # support for autocomplete task-name
    # same as
    python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Velocity
    ```
  - Inference with a trained agent:

    ```bash
    ./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Velocity # support for autocomplete task-name
    # same as
    python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Velocity
    ```

## Deploy

After the model training is completed, we need to perform sim2sim on the trained strategy in Mujoco to test the performance of the model.
Then deploy sim2real.

### Setup

```bash
# Install dependencies
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
# Install unitree_sdk2
git clone git@github.com:unitreerobotics/unitree_sdk2.git
cd unitree_sdk2
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=OFF # Install on the /usr/local directory
sudo make install
# Compile the robot_controller
cd unitree_rl_lab/deploy/robots/g1_29dof # or other robots
mkdir build && cd build
cmake .. && make
```

### Sim2Sim

Installing the [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco?tab=readme-ov-file#installation).

- Set the `robot` at `/simulate/config.yaml` to g1
- Set `domain_id` to 0
- Set `enable_elastic_hand` to 1
- Set `use_joystck` to 1.

```bash
# start simulation
cd unitree_mujoco/simulate/build
./unitree_mujoco
# ./unitree_mujoco -i 0 -n eth0 -r g1 -s scene_29dof.xml # alternative
```

```bash
cd unitree_rl_lab/deploy/robots/g1_29dof/build
./g1_ctrl
# 1. press [L2 + Up] to set the robot to stand up
# 2. Click the mujoco window, and then press 8 to make the robot feet touch the ground.
# 3. Press [R1 + X] to run the policy.
# 4. Click the mujoco window, and then press 9 to disable the elastic band.
```

### Sim2Real

You can use this program to control the robot directly, but make sure the on-borad control program has been closed.

```bash
./g1_ctrl --network eth0 # eth0 is the network interface name.
```

## Acknowledgements

This repository is built upon the support and contributions of the following open-source projects. Special thanks to:

- [IsaacLab](https://github.com/isaac-sim/IsaacLab): The foundation for training and running codes.
- [mujoco](https://github.com/google-deepmind/mujoco.git): Providing powerful simulation functionalities.
- [robot_lab](https://github.com/fan-ziqi/robot_lab): Referenced for project structure and parts of the implementation.
- [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking): Versatile humanoid control framework for motion tracking.
