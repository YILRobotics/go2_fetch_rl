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
python scripts/rsl_rl/play.py --task Unitree-Go2-Velocity --num_envs 32
``` 

load one specific run

```bash
python scripts/rsl_rl/play.py --task Unitree-Go2-Velocity --load_run 2025-03-25_12-30-00_test_1
```

load one specific checkpoint

```bash
python scripts/rsl_rl/play.py --task Unitree-Go2-Velocity --checkpoint logs/rsl_rl/unitree_go2_velocity/2026-03-24_23-52-02/model_13200.pt
``` 


tmux new -s train
tmux ls
tmux attach -t train
ctrl b and then keep ctrl and press d


## Fetchrobot Extension

### Launch isaac Sim with the custom extension

```bash
./isaac-sim.sh --ext-folder /home/ferdinand/fetchrobot/ferdinand/go2_fetch_rl/isaacsim_extensions/exts --enable ferdinand.fetchrobot
```
Then click in IsaacSim on: Window -> Examples -> Robotics Exacmples and go to the general section and there is the extension. Click load.

If changing somehting in the setup_scene() function: click File > New From Stage Template > Empty, then the LOAD button. (You need to perform this action if you change anything in the setup_scene). Otherwise, you only need to press the LOAD button.


---

## Fetchrobot Foam Material Parameters

The foam cube in the extension is configured in
`isaacsim_extensions/exts/ferdinand.fetchrobot/python/ferdinand/fetchrobot/fetchrobot.py`.

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

## Installed things:

```bash
sudo apt install tmux -y
sudo apt install nvtop -y 
sudo apt install htop -y
sudo apt  install tree -y
```