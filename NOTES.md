# Notes

## Environment Setup 

```bash
source ~/miniconda3/bin/activate
conda activate isaac_lab
```

## Training


```bash
python scripts/rsl_rl/train.py --task Unitree-Go2-Velocity --headless --logger wandb --log_project_name test_example_vel_1 --run_name test_1
```

for live, dont put the --headless and define the number of envs

```bash
python scripts/rsl_rl/train.py --task Unitree-Go2-Velocity --num_envs 32
```

policies are saved in /home/ferdinand/fetchrobot/unitree_rl_lab/logs/rsl_rl

## Inference

```bash
python scripts/rsl_rl/play.py --task Unitree-Go2-Velocity 
``` 

load one specific checkpoint

```bash
python scripts/rsl_rl/play.py --task Unitree-Go2-Velocity --checkpoint logs/rsl_rl/unitree_go2_velocity/2026-03-24_23-52-02/model_13200.pt
``` 


tmux new -s train
tmux ls
tmux attach -t train
ctrl b and then keep ctrl and press d



sudo apt install tmux -y



