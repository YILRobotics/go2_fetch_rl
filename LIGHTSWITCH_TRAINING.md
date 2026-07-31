# Go2 Lightswitch Training

## Goal

`Unitree-Go2-LightSwitch-4L` trains a standalone policy that starts from the
default standing pose, crouches, raises its front-left foot while supported by
the other legs, touches the rocker with the front-left foot, lands, and becomes
stable again. It does not switch to or from the walking policy.

One episode lasts at most 5 seconds and follows this observable phase clock:

1. **Stand (0.50 s):** the action term holds the exact default pose.
2. **Crouch (up to 0.50 s):** keep all feet down and lower the base by 4.5–9.5 cm.
3. **Jump/reach (2.70 s):** lift the front-left foot, raise the base, and reach
   toward the switch. At least one rear foot stays planted while the other may
   step toward the wall; the front-right foot has no task requirement and may
   remain planted or move for balance.
4. **Land (at least 0.75 s stable):** return all feet to the floor and settle.
5. **Success:** terminate only if the rocker hold was completed and the landing
   was stable.

The robot initially spawns 34–35 cm in front of the switch with ±4 cm lateral
randomization. As the press-success EMA rises, those ranges widen to 34–38 cm
and ±16 cm. Every sampled distance is also clamped using the Go2 head collision
projection at the sampled yaw, leaving at least 1.5 cm clearance from the wall.
Yaw remains randomized by ±0.12 rad. Switch position spans ±12 cm left/right
and 0.50–1.00 m vertically: the low end is about two-thirds of the former
nominal height and the high end reaches the wall top. The wall is rigid and
extends to the floor. Legs may touch it, but
head or base contact terminates the episode.

## Observations and Press Detection

The deployable actor receives robot proprioception, the last action, switch
center position in the robot frame, the front-left-foot-to-switch vector in the
robot frame, phase one-hot, and normalized phase time. The foot vector is
computed from joint-state forward kinematics during deployment. It receives
**no switch contact, force, hold timer, or success signal**. The jump-to-land
transition therefore uses only the 2.70-second phase clock.

The asymmetric critic additionally receives base velocity, joint effort, the
simulated rocker state, filtered front-left-foot contact, and success state.
These privileged values are used only during training.

A dedicated filtered sensor detects front-left-foot/rocker contact above 2 N.
Only contact during a rear-supported, jump-ready reach counts. A quiet crouch is
still rewarded and diagnosed, but an imperfect crouch no longer disables reach
learning or invalidates an otherwise physical press. Measured contact steps
advance the hold timer. Dropouts up to 0.10 s pause it; longer dropouts reset it.
Version 1 accepts either rocker half. Only the front-left foot receives a
Cartesian switch target or can count as switch contact; the front-right foot is
unconstrained by the task target.

## Rewards

Isaac Lab multiplies configured reward weights by the environment step time.
Milestones are one-shot terms; progress and stability terms are dense.

| Stage | Reward terms and weights |
|---|---|
| Stand/crouch | settled stance `+50`, crouch pose `+2`, crouch progress `+100`, forward position `+4`, crouch complete `+100`, overshoot `-8` |
| Jump/reach | normalized base-rise progress `+550`, early base rise `+14`, low-vertical-speed raised-posture hold `+12`, stable raised torso front `+15`, left-foot clearance `+3`, left-foot-only waypoint `+30`, left-foot height `+6`, normalized switch approach `+400`, close press target `+20`, base-forward progress `+40`, half-start body-to-wall distance `+15` |
| Jump control | front impact `-3`, excess foot speed `-4`, heading error `-1`, roll `-2`, post-rise vertical speed `-2`, complete rear-support loss `-8`, hard rear-foot impact `-0.25`, rear calf/thigh ground contact `-12` |
| Press | hold progress `+25`, measured push force `+30`, contact stability `+5`, early first contact `+5`, completed press `+500` |
| Recovery | landing recovery `+4`, safe landing `+100`, complete task `+500`, landing impact `-2` |
| General | phase stability `+0.02`, initial pose/height `+2/+2`, initial contact/stability `+1/+1`, failure `-500`, vertical velocity `-0.2`, joint acceleration `-3e-7`, action rate `-0.12`, alive-time cost `-0.20`, joint limits `-8` |

A valid crouch is 4.5–9.5 cm deep, keeps all feet grounded, and holds for
0.04 s. Motion and fore-aft shift are logged and softly shaped but are not hard
readiness conditions. A timeout can still advance the phase without an extra
uncrouched-jump penalty and does not suppress subsequent reach shaping.

The left-foot-only waypoint combines a broad 25 cm exploration basin with a
precise 3 cm basin weighted toward final accuracy. Its pre-contact target moves
forward on the phase clock before jump readiness, so early policies learn to
reach instead of only hopping. Once jump-ready, the final waypoint, approach,
and close-reach rewards target 3 cm through the rocker rather than its center.
This removes the high-value hovering solution and creates contact force. Only
measured FL-rocker force receives the `+30` push reward. The right foot
contributes nothing to either proximity term.

Rear stepping remains general rather than target-driven. It receives only a
very small signed base-progress signal, with no desired step length or rear-foot
position. Normal touchdowns are unpenalized; the `-0.25` impact term activates
only above 0.8 m/s downward foot speed. At least one rear foot must remain
planted, and rear calf/thigh ground contact remains a safety penalty.

During the jump, a separate absolute-position reward targets a base-to-wall
distance equal to half the base distance measured at reset. Its one-sided 14 cm
Gaussian supplies a useful gradient from the initial pose, reaches full value at
the target, and does not reward continual motion through the wall. Base-forward
progress is available immediately during the rear-supported jump rather than
waiting for jump readiness.

The torso-front reward rises as the robot's forward body axis reaches about
29 degrees above horizontal. It is multiplied by the achieved base-rise
fraction and by low vertical/angular-speed scores, so pitching while remaining
low or rotating unstably does not collect the full reward.

## Curriculum and Diagnostics

The curriculum tracks an exponential moving average of completed press holds
with rate 0.20. Difficulty begins increasing at 20% EMA success and reaches full
difficulty at 50%, so jump, lift, hold-time, and reset randomization advance
quickly after the policy discovers reliable contact:

| Requirement | Start | Full |
|---|---:|---:|
| Base rise | 0.18 m | 0.28 m |
| Front-left-foot lift | 0.12 m | 0.28 m |
| Measured rocker-contact hold | 0.20 s | 1.00 s |
| Robot forward distance | 0.34–0.35 m | 0.34–0.38 m |
| Robot lateral offset | ±0.04 m | ±0.16 m |

Useful logged diagnostics are maximum crouch depth, base rise, front-foot lift,
front-left-foot forward extension, rear-foot step distance, minimum 3-D,
horizontal, and vertical foot errors, minimum body-to-wall distance,
first-contact fraction and time, maximum
press hold, contact dropouts, valid-crouch fraction, individual crouch-condition
pass rates, rear-support fraction, press success, and actuator-effort saturation.

## PPO and Commands

The actor and critic use `512 → 256 → 128` ELU MLPs. PPO collects 32 steps per
environment, uses 5 epochs and 6 minibatches, starts with action standard
deviation 0.4, and has zero entropy bonus. The default run is 4096 environments
for 2500 iterations.

The explicit foot-to-switch observation adds three actor inputs. Checkpoints
from the earlier observation layout are therefore not directly resumable; start
a fresh run for this version.

Interactive training:

```bash
python scripts/rsl_rl/train.py --task Unitree-Go2-LightSwitch-4L --num_envs 16
```

Headless W&B training:

```bash
python scripts/rsl_rl/train.py --task Unitree-Go2-LightSwitch-4L --headless \
  --logger wandb --video --video_interval 75 --video_length 300 \
  --log_project_name f_lightswitch_test --run_name lightswitch_1
```

Use `scripts/rsl_rl/play.py` with the same task and an explicit checkpoint for
evaluation. Judge new runs first by valid crouch and physical progress metrics,
then by press-hold duration and stable task success.
