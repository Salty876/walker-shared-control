from __future__ import annotations

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass
from isaaclab.sensors import CameraCfg


from . import mdp
from .mdp.actions import AdmittanceDampingActionCfg
from .mdp import observations as walker_obs
from .mdp import rewards as walker_rewards
from .mdp import camera_perception

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)

##
# Pre-defined configs
##

from .walker_shared_control_robot_cfg import WALKER_CFG

##
# Scene definition
##
COURSE_SHIFT_X = 1.0


@configclass
class WalkerSharedControlSceneCfg(InteractiveSceneCfg):
    


    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot
    robot: ArticulationCfg = WALKER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")




    camera = CameraCfg(
        prim_path=(
            "{ENV_REGEX_NS}/Robot/Geometry/base_link/"
            "ZED_X/base_link/ZED_X/CameraLeft"
        ),
        update_period=0.02,
        height=240,
        width=320,
        data_types=[
            "distance_to_image_plane",
        ],
        spawn=None,
        update_latest_camera_pose=True,
    )

    zed_right = CameraCfg(
        prim_path=(
            "{ENV_REGEX_NS}/Robot/Geometry/base_link/"
            "ZED_X/base_link/ZED_X/CameraRight"
        ),
        update_period=0.02,
        height=240,
        width=320,
        data_types=[
            "rgb",
        ],
        spawn=None,
        update_latest_camera_pose=True,
    )
    

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # ---------- TERRAIN TEST COURSE ----------


    # 8 degree incline
    uphill = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/UphillRamp",
        spawn=sim_utils.CuboidCfg(
            size=(4.0, 3.0, 0.15),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(3.99 + COURSE_SHIFT_X, 0.0, 0.204),
            # -8 degrees around Y
            rot=(0.0, -0.069756, 0.0, 0.997564),
        ),
    )

    # Raised flat section
    platform = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Platform",
        spawn=sim_utils.CuboidCfg(
            size=(2.05, 3.0, 0.15),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(6.98 + COURSE_SHIFT_X, 0.0, 0.482),
        ),
    )

    # 8 degree decline
    downhill = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/DownhillRamp",
        spawn=sim_utils.CuboidCfg(
            size=(4.0, 3.0, 0.15),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(9.99 + COURSE_SHIFT_X, 0.0, 0.204),
            # +8 degrees around Y
            rot=(0.0, 0.069756, 0.0, 0.997564),
        ),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:

    admittance_damping = AdmittanceDampingActionCfg(
        asset_name="robot",
    )    

@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):

        slope_now = ObsTerm(
            func=walker_obs.slope_now,
            scale=1.0 / 0.13962634,
        )

        slope_ahead = ObsTerm(
            func=walker_obs.slope_ahead,
            scale=1.0 / 0.13962634,
        )

        velocity = ObsTerm(
            func=walker_obs.forward_velocity,
            scale=1.0 / 0.35,
        )

        human_force = ObsTerm(
            func=walker_obs.human_force,
            scale=1.0 / 20.0,
        )

        damping = ObsTerm(
            func=walker_obs.current_damping,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class CameraObservationCfg:

    @configclass
    class CameraPolicyCfg(ObsGroup):
    
        slope_now = ObsTerm(
            func=walker_obs.slope_now,
            scale=1.0 / 0.13962634,
        )

        slope_ahead = ObsTerm(
            # func=camera_perception.slope_ahead_camera_safe,
            func=walker_obs.zero_slope_ahead,
            scale=1.0 / 0.13962634,
        )

        velocity = ObsTerm(
            func=walker_obs.forward_velocity,
            scale=1.0 / 0.35,
        )

        human_force = ObsTerm(
            func=walker_obs.human_force,
            scale=1.0 / 20.0,
        )

        damping = ObsTerm(
            func=walker_obs.current_damping,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: CameraPolicyCfg = CameraPolicyCfg()


@configclass
class EventCfg:

    reset_scene = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
    )


@configclass
class RewardsCfg:

    velocity_tracking = RewTerm(
        func=walker_rewards.track_target_velocity,
        weight=3.0,
        params={
            "target_velocity": 0.35,
            "sigma": 0.05,
        },
    )

    voluntary_effort = RewTerm(
        func=walker_rewards.voluntary_effort,
        weight=-0.6,
        params={
            "force_scale": 20.0,
        },
    )

    overspeed = RewTerm(
        func = walker_rewards.overspeed,
        weight=-0.5,
        params={
            "target_velocity": 0.35,
            "sigma": 0.10
            }
    )
    

    damping_deviation = RewTerm(
        func=mdp.action_l2,
        weight=-0.05,
    )

    damping_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.02,
    )

@configclass
class TerminationsCfg:

    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True,
    )


##
# Environment configuration
##


@configclass
class WalkerSharedControlEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: WalkerSharedControlSceneCfg = WalkerSharedControlSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: CameraObservationCfg = CameraObservationCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 45
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation

