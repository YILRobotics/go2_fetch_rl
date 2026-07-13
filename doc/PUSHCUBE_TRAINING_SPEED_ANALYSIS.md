# PushCube Training Speed Analysis

Date: 2026-04-03  
Task: `Unitree-Go2-PushCube-4L`  
Observed run: `test_65` (and stale `test_57`)

## 1) Baseline from your current run

Command pattern:

```bash
python scripts/rsl_rl/train.py \
  --task Unitree-Go2-PushCube-4L \
  --low_level_policy_path /home/ferdinand/fetchrobot/ferdinand/go2_fetch_rl/logs/rsl_rl/unitree_go2_velocity/2026-03-25_23-05-55_e30_allterain/exported/policy.pt \
  --headless --logger wandb --video --video_interval 100 --video_length 300 \
  --log_project_name f_pushcube_4l --run_name test_65
```

Recent training stats (from `tmux` pane):

- `Steps per second`: ~`20.3k` to `21.5k`
- `Collection time`: ~`6.33s` to `6.50s`
- `Learning time`: ~`0.09s`
- Per-iteration steps = `4096 envs * 32 steps_per_env = 131072`

Key observation:

- Runtime is collection-bound. Roughly `>98%` of iteration time is simulation/rollout, not PPO update.

## 2) Runtime profiling snapshot (`nvtop` + `htop`)

`nvtop` sample (3s snapshot):

- GPU: `RTX 4090`
- GPU util: ~`76-80%`
- VRAM used: ~`16.0 / 24.0 GiB`
- Power: ~`141-150W` (far from 450W limit)
- Active process (`test_65`, PID `269839`): ~`9.8 GiB` GPU memory
- Another process (`test_57`, PID `50875`): ~`6.1 GiB` GPU memory

`htop` sample:

- Host load average around `3.65 / 2.77 / 2.03` (not CPU-saturated for a 32-thread machine)
- Main training process CPU around `~140-250%` depending on sampling
- Main training process RES around `~13.8 GiB`

Process state check:

- PID `269839` (`test_65`) is running.
- PID `50875` (`test_57`) is in `T` (stopped) state but still holding ~`6.1 GiB` VRAM.

## 3) Bottlenecks and why

## A. Video capture in training is expensive

You run with `--video`, which forces:

- camera enable (`train.py`)
- `render_mode="rgb_array"` (`train.py`)
- `RecordVideo` wrapper with mp4 writing (`train.py`)
- W&B live video upload (`wandb.save(..., policy="live")`)

Also video is configured at `1920x1080` in `scripts/rsl_rl/train.py`.

Impact: direct overhead on collection speed and periodic stalls during capture/encoding/upload.

## B. High-level control rate implies heavy physics stepping

In `push_env_cfg.py`:

- `SIM_DT = 0.005`
- `HIGH_LEVEL_POLICY_HZ = 15.0`
- `decimation = round(1 / (SIM_DT * HZ)) = round(1 / 0.075) = 13`

So each RL step advances ~13 physics steps. This is expensive and explains long collection time.

## C. Unused stopped process wastes GPU memory

Stopped run `test_57` is occupying ~`6.1 GiB` VRAM.  
Even if not computing, this reduces headroom for scaling env count and can increase memory pressure.

## D. Extra per-step tensor allocations in push MDP code

In `source/unitree_rl_lab/unitree_rl_lab/tasks/push_mdp.py`, several functions repeatedly allocate tensors each step (`torch.arange`, `torch.full`, `torch.tensor(...).repeat(...)`), for example:

- `_cube_goal_distance`
- `robot_in_goal_area_penalty`
- `cube_goal_reached_robot_outsid_goal`
- `goal_position_xy`

Impact: additional GPU kernel launches/allocations and synchronization overhead at 4096 envs.

## E. Debug visualization flag enabled in command generator

In `push_env_cfg.py`, `CommandsCfg.base_velocity.debug_vis=True`.  
For pure throughput training, this should be disabled.

## 4) Prioritized improvements

| Priority | Change | Expected speed impact | Risk |
|---|---|---:|---|
| P0 | Stop stale process `test_57` (PID `50875`) | small to medium (frees 6.1 GiB VRAM) | none |
| P0 | Train without `--video` (and do separate eval video runs) | medium to high (often 10-35%) | none |
| P0 | Use `tensorboard` logger during throughput runs | small to medium | none |
| P1 | Disable `debug_vis` in `CommandsCfg` | small | none |
| P1 | Raise `HIGH_LEVEL_POLICY_HZ` from 15 to 20 (decimation 13 -> 10) | medium to high | may require retuning rewards |
| P1 | Tune `--num_envs` upward after freeing VRAM (`5120`, `6144`) | medium | may OOM if too high |
| P2 | Cache/reuse env-id and goal tensors in `push_mdp.py` | small to medium | low (code changes) |
| P2 | If video is required during training, lower capture resolution and frequency | medium | lower visual quality |

## 5) Concrete actions to run now

1. Free stale VRAM:

```bash
kill 50875
```

2. Throughput-focused training command:

```bash
python scripts/rsl_rl/train.py \
  --task Unitree-Go2-PushCube-4L \
  --low_level_policy_path /home/ferdinand/fetchrobot/ferdinand/go2_fetch_rl/logs/rsl_rl/unitree_go2_velocity/2026-03-25_23-05-55_e30_allterain/exported/policy.pt \
  --headless --logger tensorboard \
  --log_project_name f_pushcube_4l --run_name speed_no_video
```

3. If you want to keep W&B metrics but maximize speed, avoid live video upload in training runs.

## 6) Code-level edits I recommend next

1. `source/unitree_rl_lab/unitree_rl_lab/tasks/push_env_cfg.py`
- set `CommandsCfg.base_velocity.debug_vis = False`
- test `HIGH_LEVEL_POLICY_HZ = 20.0` first

2. `scripts/rsl_rl/train.py`
- add a switch to disable W&B live `wandb.save("*.mp4", policy="live")` during training
- if training video is unavoidable, reduce hardcoded `VIDEO_WIDTH/VIDEO_HEIGHT`

3. `source/unitree_rl_lab/unitree_rl_lab/tasks/push_mdp.py`
- cache reusable tensors on env object (`env_ids`, `goal_xy_tensor`, repeated full tensors)
- avoid repeated `torch.arange(env.num_envs, ...)` in per-step reward/termination functions

## 7) Benchmark protocol (to validate gains)

Run each variant for ~`120-200` iterations, then compare median `Steps per second`:

- Baseline: current command
- Variant A: no video
- Variant B: no video + `HIGH_LEVEL_POLICY_HZ=20`
- Variant C: Variant B + higher `num_envs`

Use the same seed per comparison pair to reduce noise.
