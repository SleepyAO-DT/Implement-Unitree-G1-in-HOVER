# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

import torch

from neural_wbc.core.modes import NeuralWBCModes
from neural_wbc.data import get_data_path

from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab_assets import G1_CFG

from .events import NeuralWBCPlayEventCfg, NeuralWBCTrainEventCfg
from .neural_wbc_env_cfg import NeuralWBCEnvCfg
from .terrain import HARD_ROUGH_TERRAINS_CFG, flat_terrain

DISTILL_MASK_MODES_ALL = {
    "exbody": {
        "upper_body": [".*torso_joint.*", ".*shoulder.*joint.*", ".*elbow.*joint.*"],
        "lower_body": ["root.*"],
    },
    "humanplus": {
        "upper_body": [".*torso_joint.*", ".*shoulder.*joint.*", ".*elbow.*joint.*"],
        "lower_body": [".*hip.*joint.*", ".*knee.*joint.*", ".*ankle.*joint.*", "root.*"],
    },
    "h2o": {
        "upper_body": [
            ".*shoulder.*link.*",
            ".*elbow.*link.*",
            ".*hand.*link.*",
        ],
        "lower_body": [".*ankle.*link.*"],
    },
    "omnih2o": {
        "upper_body": [".*hand.*link.*", ".*head.*link.*"],
    },
}


@configclass
class NeuralWBCEnvCfgG1(NeuralWBCEnvCfg):
    # General parameters:
    action_space = 23
    observation_space = 1073
    state_space = 1162

    # Distillation parameters:
    single_history_dim = 63
    observation_history_length = 25

    # Mask setup for an OH2O specialist policy as default:
    # OH2O mode is tracking the head and hand positions. This can be modified to train a different specialist
    # or use the full DISTILL_MASK_MODES_ALL to train a generalist policy.
    distill_mask_sparsity_randomization_enabled = False
    distill_mask_modes = DISTILL_MASK_MODES_ALL

    # Robot geometry / actuation parameters:
    actuators = {
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_hip_roll_joint",
                ".*_hip_yaw_joint",
                ".*_knee_joint",
                ".*_ankle_pitch_joint",
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint",
            ],
            effort_limit={
                ".*_hip_pitch_joint": 88.0,
                ".*_hip_roll_joint": 88.0,
                ".*_hip_yaw_joint": 88.0,
                ".*_knee_joint": 139.0,
                ".*_ankle_pitch_joint": 50.0,
                "waist_yaw_joint": 88.0,
                "waist_roll_joint": 50.0,
                "waist_pitch_joint": 50.0,
            },
            velocity_limit={
                ".*_hip_pitch_joint": 32.0,
                ".*_hip_roll_joint": 32.0,
                ".*_hip_yaw_joint": 32.0,
                ".*_knee_joint": 20.0,
                ".*_ankle_pitch_joint": 37.0,
                "waist_yaw_joint": 32.0,
                "waist_roll_joint": 37.0,
                "waist_pitch_joint": 37.0,
            },
            stiffness=0,
            damping=0,
        ),
        "feet": IdealPDActuatorCfg(
            joint_names_expr=[".*_ankle_roll_joint"],
            effort_limit=50.0,
            velocity_limit=37.0,
            stiffness=0,
            damping=0,
        ),
        "arms": IdealPDActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_joint"],
            effort_limit={
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
            },
            velocity_limit={
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
            },
            stiffness=0,
            damping=0,
        ),
    }

    robot: ArticulationCfg = G1_CFG.replace(prim_path="/World/envs/env_.*/Robot", actuators=actuators)

    body_names = [
        "pelvis",
        "left_hip_pitch_link",
        "left_hip_roll_link",
        "left_hip_yaw_link",
        "left_knee_link",
        'left_ankle_pitch_link', 
        'left_ankle_roll_link',
        "right_hip_pitch_link",
        "right_hip_roll_link",
        "right_hip_yaw_link",
        "right_knee_link",
        'right_ankle_pitch_link', 
        'right_ankle_roll_link',
        'waist_yaw_link',
        'waist_roll_link',
        "torso_link",
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
    ]

    # Joint names by the order in the MJCF model.
    joint_names = [
        'left_hip_pitch_joint', 
        'left_hip_roll_joint', 
        'left_hip_yaw_joint', 
        'left_knee_joint', 
        'left_ankle_pitch_joint', 
        'left_ankle_roll_joint',
        'right_hip_pitch_joint', 
        'right_hip_roll_joint', 
        'right_hip_yaw_joint', 
        'right_knee_joint', 
        'right_ankle_pitch_joint', 
        'right_ankle_roll_joint',
        'waist_yaw_joint', 
        'waist_roll_joint', 
        'waist_pitch_joint',
        'left_shoulder_pitch_joint', 
        'left_shoulder_roll_joint', 
        'left_shoulder_yaw_joint', 
        'left_elbow_joint', 
        'right_shoulder_pitch_joint', 
        'right_shoulder_roll_joint', 
        'right_shoulder_yaw_joint', 
        'right_elbow_joint',
    ]

    # Lower and upper body joint ids in the MJCF model.
    lower_body_joint_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # hips, knees, ankles
    upper_body_joint_ids = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]  # torso, shoulders, elbows

    base_name = "torso_link"
    root_id = body_names.index(base_name)

    feet_name = ".*_ankle_roll_link"

    extend_body_parent_names = ["left_elbow_link", "right_elbow_link", "torso_link"]
    extend_body_names = ["left_hand_link", "right_hand_link", "head_link"]
    extend_body_pos = torch.tensor([[0.25, 0, 0], [0.25, 0, 0], [0, 0, 0.42]])

    # These are the bodies that are tracked by the teacher. They may also contain the extended
    # bodies.
    tracked_body_names = [
        "pelvis",
        "left_hip_pitch_link",
        "left_hip_roll_link",
        "left_hip_yaw_link",
        "left_knee_link",
        "left_ankle_pitch_link",
        "left_ankle_roll_link",
        "right_hip_pitch_link",
        "right_hip_roll_link",
        "right_hip_yaw_link",
        "right_knee_link",
        "right_ankle_pitch_link",
        "right_ankle_roll_link",
        "waist_yaw_link",
        "waist_roll_link",
        "torso_link",
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
        "left_hand_link",
        "right_hand_link",
        "head_link",
    ]

    # control parameters
    stiffness = {
        'left_hip_pitch_joint': 100.0, 
        'left_hip_roll_joint': 100.0, 
        'left_hip_yaw_joint': 100.0, 
        'left_knee_joint': 200.0, 
        'left_ankle_pitch_joint': 20.0, 
        'left_ankle_roll_joint': 20.0,
        'right_hip_pitch_joint': 100.0, 
        'right_hip_roll_joint': 100.0, 
        'right_hip_yaw_joint': 100.0, 
        'right_knee_joint': 200.0, 
        'right_ankle_pitch_joint': 20.0, 
        'right_ankle_roll_joint': 20.0,
        'waist_yaw_joint': 400.0, 
        'waist_roll_joint': 400.0, 
        'waist_pitch_joint': 400.0,
        'left_shoulder_pitch_joint': 90.0, 
        'left_shoulder_roll_joint': 60.0, 
        'left_shoulder_yaw_joint': 20.0, 
        'left_elbow_joint': 60.0, 
        'right_shoulder_pitch_joint': 90.0, 
        'right_shoulder_roll_joint': 60.0, 
        'right_shoulder_yaw_joint': 20.0, 
        'right_elbow_joint': 60.0,
    }

    damping = {
        'left_hip_pitch_joint': 2.5, 
        'left_hip_roll_joint': 2.5, 
        'left_hip_yaw_joint': 2.5, 
        'left_knee_joint': 5.0, 
        'left_ankle_pitch_joint': 0.2, 
        'left_ankle_roll_joint': 0.1,
        'right_hip_pitch_joint': 2.5, 
        'right_hip_roll_joint': 2.5, 
        'right_hip_yaw_joint': 2.5, 
        'right_knee_joint': 5.0, 
        'right_ankle_pitch_joint': 0.2, 
        'right_ankle_roll_joint': 0.1,
        'waist_yaw_joint': 5.0, 
        'waist_roll_joint': 5.0, 
        'waist_pitch_joint': 5.0,
        'left_shoulder_pitch_joint': 2.0, 
        'left_shoulder_roll_joint': 1.0, 
        'left_shoulder_yaw_joint': 0.4, 
        'left_elbow_joint': 1.0, 
        'right_shoulder_pitch_joint': 2.0, 
        'right_shoulder_roll_joint': 1.0, 
        'right_shoulder_yaw_joint': 0.4, 
        'right_elbow_joint': 1.0,
    }

    mass_randomized_body_names = [
        'pelvis', 
        'left_hip_yaw_link', 
        'left_hip_roll_link', 
        'left_hip_pitch_link', 
        'right_hip_yaw_link', 
        'right_hip_roll_link', 
        'right_hip_pitch_link', 
        'torso_link',
    ]

    undesired_contact_body_names = [
        "pelvis",
        "left_hip_pitch_link",
        "left_hip_roll_link",
        "left_hip_yaw_link",
        "left_knee_link",
        "left_ankle_pitch_link",
        "right_hip_pitch_link",
        "right_hip_roll_link",
        "right_hip_yaw_link",
        "right_knee_link",
        "right_ankle_pitch_link",
        "waist_yaw_link",
        "waist_roll_link",
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
    ]

    # Add a height scanner to the torso to detect the height of the terrain mesh
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/pelvis",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        attach_yaw_only=True,
        # Apply a grid pattern that is smaller than the resolution to only return one height value.
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.05, 0.05]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    def __post_init__(self):
        super().__post_init__()

        self.reference_motion_manager.motion_path = get_data_path("motions/amass_all_g1.pkl")
        self.reference_motion_manager.skeleton_path = get_data_path("motion_lib/g1_29dof_anneal_23dof.xml")

        if self.terrain.terrain_generator == HARD_ROUGH_TERRAINS_CFG:
            self.events.update_curriculum.params["penalty_level_up_threshold"] = 125

        if self.mode == NeuralWBCModes.TRAIN:
            self.episode_length_s = 20.0
            self.max_ref_motion_dist = 0.5
            self.events = NeuralWBCTrainEventCfg()
            self.events.reset_robot_rigid_body_mass.params["asset_cfg"].body_names = self.mass_randomized_body_names
            self.events.reset_robot_base_com.params["asset_cfg"].body_names = "torso_link"
        elif self.mode == NeuralWBCModes.DISTILL:
            self.max_ref_motion_dist = 0.5
            self.events = NeuralWBCTrainEventCfg()
            self.events.reset_robot_rigid_body_mass.params["asset_cfg"].body_names = self.mass_randomized_body_names
            self.events.reset_robot_base_com.params["asset_cfg"].body_names = "torso_link"
            self.add_policy_obs_noise = False
            self.reset_mask = True
            # Do not reset mask when there is only one mode.
            num_regions = len(self.distill_mask_modes)
            if num_regions == 1:
                region_modes = list(self.distill_mask_modes.values())[0]
                if len(region_modes) == 1:
                    self.reset_mask = False
        elif self.mode == NeuralWBCModes.TEST:
            self.terrain = flat_terrain
            self.events = NeuralWBCPlayEventCfg()
            self.ctrl_delay_step_range = (2, 2)
            self.max_ref_motion_dist = 0.5
            self.add_policy_obs_noise = False
            self.resample_motions = False
            self.distill_mask_sparsity_randomization_enabled = False
            self.distill_mask_modes = {"omnih2o": DISTILL_MASK_MODES_ALL["omnih2o"]}
        elif self.mode == NeuralWBCModes.DISTILL_TEST:
            self.terrain = flat_terrain
            self.events = NeuralWBCPlayEventCfg()
            self.distill_teleop_selected_keypoints_names = []
            self.ctrl_delay_step_range = (2, 2)
            self.max_ref_motion_dist = 0.5
            self.default_rfi_lim = 0.0
            self.add_policy_obs_noise = False
            self.resample_motions = False
            self.distill_mask_sparsity_randomization_enabled = False
            self.distill_mask_modes = {"omnih2o": DISTILL_MASK_MODES_ALL["omnih2o"]}
        else:
            raise ValueError(f"Unsupported mode {self.mode}")
