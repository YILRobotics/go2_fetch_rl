import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.unitree import UNITREE_GO2_CFG as ROBOT_CFG
from unitree_rl_lab.tasks import mdp

from . import lightswitch_mdp

SIM_DT = 0.005
DECIMATION = 4

SWITCH_CENTER_XY = (0.45, 0.0)
SWITCH_X_RANGE = (0.42, 0.48)
SWITCH_Y_RANGE = (-0.04, 0.04)
SWITCH_HEIGHT_RANGE = (0.72, 0.78)
SWITCH_YAW_RAD = math.pi * 0.5
WALL_X_OFFSET = 0.036

# Start closer to the switch so approach is easier in early curriculum.
ROBOT_FORWARD_RANGE = (0.18, 0.30)
ROBOT_LATERAL_RANGE = (-0.10, 0.10)

def _phase_tracking_params() -> dict:
    """Fresh entity selectors for every manager term that advances the task phase."""
    return {
        "robot_cfg": SceneEntityCfg("robot"),
        "foot_cfg": SceneEntityCfg("robot", body_names="FL_foot.*"),
        "other_feet_cfg": SceneEntityCfg(
            "robot", body_names=["RL_foot.*", "RR_foot.*"]
        ),
        "foot_sensor_cfg": SceneEntityCfg("contact_forces", body_names="FL_foot.*"),
        "other_feet_sensor_cfg": SceneEntityCfg(
            "contact_forces", body_names=["RL_foot.*", "RR_foot.*"]
        ),
        "front_feet_sensor_cfg": SceneEntityCfg(
            "contact_forces", body_names=["FL_foot.*", "FR_foot.*"]
        ),
        "all_feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot.*"),
        "settle_time_s": 0.25,
        "lift_height": 0.30,
        "lift_hold_time_s": 0.06,
        "jump_base_rise": 0.08,
        "land_hold_time_s": 0.50,
    }


LIGHTSWITCH_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(2.0, 2.0),
    border_width=10.0,
    # 64x64 = 4096 terrain tiles so terrain coverage matches 4096 envs.
    num_rows=64,
    num_cols=64,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
    },
)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    replicate_physics = False

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=LIGHTSWITCH_TERRAIN_CFG,
        max_init_terrain_level=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.95,
            dynamic_friction=0.9,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 1.8),
            "dynamic_friction_range": (0.7, 1.5),
            "restitution_range": (0.0, 0.1),
            "make_consistent": True,
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    setup_switch_stage = EventTerm(
        func=lightswitch_mdp.setup_lightswitch_stage,
        mode="prestartup",
        params={
            "switch_center_xy": SWITCH_CENTER_XY,
            "switch_x_range": SWITCH_X_RANGE,
            "switch_y_range": SWITCH_Y_RANGE,
            "switch_default_height": 0.75,
            "switch_height_range": SWITCH_HEIGHT_RANGE,
            "switch_yaw": SWITCH_YAW_RAD,
        },
    )

    reset_robot_and_switch = EventTerm(
        func=lightswitch_mdp.reset_robot_and_lightswitch,
        mode="reset",
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "switch_height_range": SWITCH_HEIGHT_RANGE,
            "switch_yaw": SWITCH_YAW_RAD,
            "wall_x_offset": WALL_X_OFFSET,
            "robot_forward_range": ROBOT_FORWARD_RANGE,
            "robot_lateral_range": ROBOT_LATERAL_RANGE,
            "robot_yaw_range": (-0.12, 0.12),
            "robot_velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        rel_standing_envs=1.0,
        debug_vis=False,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    # Same action interface as Velocity-4L.
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # Same policy observations as Velocity-4L + switch center position.
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.2,
            clip=(-100, 100),
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            clip=(-100, 100),
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            clip=(-100, 100),
            params={"command_name": "base_velocity"},
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            clip=(-100, 100),
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            clip=(-100, 100),
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        switch_center_pos = ObsTerm(
            func=lightswitch_mdp.switch_center_position_robot_frame,
            clip=(-2.0, 2.0),
            params={"robot_cfg": SceneEntityCfg("robot")},
        )
        behavior_phase = ObsTerm(func=lightswitch_mdp.behavior_phase_one_hot, clip=(0.0, 1.0))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        # Same critic observations as Velocity-4L + switch center + privileged contact.
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            clip=(-100, 100),
            params={"command_name": "base_velocity"},
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100))
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100, 100))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        switch_center_pos = ObsTerm(
            func=lightswitch_mdp.switch_center_position_robot_frame,
            clip=(-2.0, 2.0),
            params={"robot_cfg": SceneEntityCfg("robot")},
        )
        behavior_phase = ObsTerm(func=lightswitch_mdp.behavior_phase_one_hot, clip=(0.0, 1.0))
        switch_state = ObsTerm(func=lightswitch_mdp.privileged_switch_state, clip=(-20.0, 20.0))
        switch_contact = ObsTerm(
            func=lightswitch_mdp.switch_contact_proxy_obs,
            clip=(0.0, 1.0),
            params={
                "foot_cfg": SceneEntityCfg("robot", body_names="FL_foot.*"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names="FL_foot.*"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    # Ordered one-shot milestones.  Isaac Lab multiplies these weights by step_dt.
    settled_stance = RewTerm(
        func=lightswitch_mdp.phase_milestone_reward,
        weight=5.0,
        params={**_phase_tracking_params(), "target_phase": lightswitch_mdp.PHASE_JUMP},
    )
    jump_base_progress = RewTerm(
        func=lightswitch_mdp.jump_base_progress_reward,
        weight=400.0,
        params=_phase_tracking_params(),
    )
    front_feet_air = RewTerm(
        func=lightswitch_mdp.front_feet_air_milestone_reward,
        weight=100.0,
        params=_phase_tracking_params(),
    )
    rear_support_loss = RewTerm(
        func=lightswitch_mdp.rear_support_loss_penalty,
        weight=-5.0,
        params=_phase_tracking_params(),
    )
    jump_complete = RewTerm(
        func=lightswitch_mdp.phase_milestone_reward,
        weight=250.0,
        params={**_phase_tracking_params(), "target_phase": lightswitch_mdp.PHASE_PRESS},
    )
    switch_approach = RewTerm(
        func=lightswitch_mdp.press_approach_progress_reward,
        weight=100.0,
        params=_phase_tracking_params(),
    )
    switch_pressed = RewTerm(
        func=lightswitch_mdp.phase_milestone_reward,
        weight=250.0,
        params={**_phase_tracking_params(), "target_phase": lightswitch_mdp.PHASE_LAND},
    )
    landing_recovery = RewTerm(
        func=lightswitch_mdp.landing_recovery_reward,
        weight=5.0,
        params=_phase_tracking_params(),
    )
    safe_landing = RewTerm(
        func=lightswitch_mdp.phase_milestone_reward,
        weight=500.0,
        params={**_phase_tracking_params(), "target_phase": lightswitch_mdp.PHASE_SUCCESS},
    )

    stability = RewTerm(
        func=lightswitch_mdp.phase_stability_reward,
        weight=0.02,
        params=_phase_tracking_params(),
    )
    landing_impact = RewTerm(
        func=lightswitch_mdp.landing_impact_penalty,
        weight=-2.0,
        params={**_phase_tracking_params(), "force_threshold": 250.0},
    )
    failure_termination_penalty = RewTerm(
        func=mdp.is_terminated_term,
        weight=-10.0,
        params={"term_keys": ["base_contact", "bad_orientation"]},
    )
    base_linear_velocity = RewTerm(
        func=lightswitch_mdp.phase_vertical_velocity_penalty,
        weight=-0.2,
        params=_phase_tracking_params(),
    )
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.01)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.003)
    time_penalty = RewTerm(func=mdp.is_alive, weight=-0.05)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-8.0)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.02)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    success = DoneTerm(
        func=lightswitch_mdp.lightswitch_goal_reached,
        params=_phase_tracking_params(),
    )

    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
            "threshold": 1.0,
        },
    )

    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})


@configclass
class CurriculumCfg:
    common_step_counter = CurrTerm(func=lightswitch_mdp.curriculum_common_step_counter, params={})
    behavior_difficulty = CurrTerm(
        func=lightswitch_mdp.curriculum_behavior_difficulty,
        params={
            "jump_rise_start": 0.03,
            "jump_rise_target": 0.08,
            "success_rate_start": 0.40,
            "success_rate_full": 0.70,
            "ema_rate": 0.10,
        },
    )
    phase_stand_fraction = CurrTerm(
        func=lightswitch_mdp.curriculum_phase_fraction, params={"phase": lightswitch_mdp.PHASE_STAND}
    )
    phase_jump_fraction = CurrTerm(
        func=lightswitch_mdp.curriculum_phase_fraction, params={"phase": lightswitch_mdp.PHASE_JUMP}
    )
    phase_press_fraction = CurrTerm(
        func=lightswitch_mdp.curriculum_phase_fraction, params={"phase": lightswitch_mdp.PHASE_PRESS}
    )
    phase_land_fraction = CurrTerm(
        func=lightswitch_mdp.curriculum_phase_fraction, params={"phase": lightswitch_mdp.PHASE_LAND}
    )


@configclass
class RobotLightSwitchEnvCfg(ManagerBasedRLEnvCfg):
    # Keep 4096 envs, reduce spacing to the smallest practical value.
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=1.8)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = DECIMATION
        self.episode_length_s = 20.0

        self.sim.dt = SIM_DT
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        self.scene.contact_forces.update_period = self.sim.dt
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotLightSwitchPlayEnvCfg(RobotLightSwitchEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.observations.policy.enable_corruption = False
