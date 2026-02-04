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

from isaaclab.utils import configclass

# Reward configuration for G1 robot

@configclass
class NeuralWBCRewardCfg:
    # Reward and penalty scales
    scales = {
        "reward_track_joint_positions": 32.0,
        "reward_track_joint_velocities": 16.0,
        "reward_track_body_velocities": 8.0,
        "reward_track_body_angular_velocities": 8.0,
        "reward_track_body_position_extended": 30.0,
        "reward_track_body_position_vr_key_points": 50.0,
        "penalize_torques": -0.0001,
        "penalize_by_torque_limits": -2,
        "penalize_joint_accelerations": -0.000011,
        "penalize_joint_velocities": -0.004,
        "penalize_lower_body_action_changes": -3.0,
        "penalize_upper_body_action_changes": -0.625,
        "penalize_by_joint_pos_limits": -125.0,
        "penalize_by_joint_velocity_limits": -50.0,
        "penalize_early_termination": -250.0,
        "penalize_feet_contact_forces": -0.75,
        "penalize_stumble": -1000.0,
        "penalize_slippage": -37.5,
        "penalize_feet_orientation": -62.5,
        "penalize_feet_air_time": 1000.0,
        "penalize_both_feet_in_air": -200.0,
        "penalize_orientation": -200.0,
        "penalize_max_feet_height_before_contact": -2500.0,
    }

    # Sigmas for exponential terms
    # The larger the sigma, the more tolerance for error, more radical the robot can be
    body_pos_lower_body_sigma = 0.5
    body_pos_upper_body_sigma = 0.03
    body_pos_vr_key_points_sigma = 0.03
    body_rot_sigma = 0.1
    body_vel_sigma = 10
    body_ang_vel_sigma = 10
    joint_pos_sigma = 0.5
    joint_vel_sigma = 1.0

    # Weights for weighted sums
    body_pos_lower_body_weight = 0.5
    body_pos_upper_body_weight = 1.0

    # Limits
    torque_limits_scale = 0.95
    # The order here follows the order in cfg.joint_names
    torque_limits = [
        88.0,   # joint 0 - left_hip_pitch_joint
        88.0,   # joint 1 - left_hip_roll_joint
        88.0,   # joint 2 - left_hip_yaw_joint
        139.0,  # joint 3 - left_knee_joint
        50.0,   # joint 4 - left_ankle_pitch_joint
        50.0,   # joint 5 - left_ankle_roll_joint
        88.0,   # joint 6 - right_hip_pitch_joint
        88.0,   # joint 7 - right_hip_roll_joint
        88.0,   # joint 8 - right_hip_yaw_joint
        139.0,  # joint 9 - right_knee_joint
        50.0,   # joint 10 - right_ankle_pitch_joint
        50.0,   # joint 11 - right_ankle_roll_joint
        88.0,   # joint 12 - waist_yaw_joint
        50.0,   # joint 13 - waist_roll_joint
        50.0,   # joint 14 - waist_pitch_joint
        25.0,   # joint 15 - left_shoulder_pitch_joint
        25.0,   # joint 16 - left_shoulder_roll_joint
        25.0,   # joint 17 - left_shoulder_yaw_joint
        25.0,   # joint 18 - left_elbow_joint
        25.0,   # joint 19 - right_shoulder_pitch_joint
        25.0,   # joint 20 - right_shoulder_roll_joint
        25.0,   # joint 21 - right_shoulder_yaw_joint
        25.0,   # joint 22 - right_elbow_joint
    ]
    # Joint pos limits, in the form of (lower_limit, upper_limit)
    joint_pos_limits = [
        (-2.5307, 2.8798),     # joint 0
        (-0.5236, 2.9671),     # joint 1
        (-2.7576, 2.7576),     # joint 2
        (-0.087267, 2.8798),   # joint 3
        (-0.87267, 0.5236),    # joint 4
        (-0.2618, 0.2618),     # joint 5
        (-2.5307, 2.8798),     # joint 6
        (-2.9671, 0.5236),     # joint 7
        (-2.7576, 2.7576),     # joint 8
        (-0.087267, 2.8798),   # joint 9
        (-0.87267, 0.5236),    # joint 10
        (-0.2618, 0.2618),     # joint 11
        (-2.618, 2.618),       # joint 12
        (-0.52, 0.52),         # joint 13
        (-0.52, 0.52),         # joint 14
        (-3.0892, 2.6704),     # joint 15
        (-1.5882, 2.2515),     # joint 16
        (-2.618, 2.618),       # joint 17
        (-1.0472, 2.0944),     # joint 18
        (-3.0892, 2.6704),     # joint 19
        (-2.2515, 1.5882),     # joint 20
        (-2.618, 2.618),       # joint 21
        (-1.0472, 2.0944),     # joint 22
    ]

    joint_vel_limits_scale = 0.95
    joint_vel_limits = [
        32.0,  # joint 0
        32.0,  # joint 1
        32.0,  # joint 2
        20.0,  # joint 3
        37.0,  # joint 4
        37.0,  # joint 5
        32.0,  # joint 6
        32.0,  # joint 7
        32.0,  # joint 8
        20.0,  # joint 9
        37.0,  # joint 10
        37.0,  # joint 11
        32.0,  # joint 12
        37.0,  # joint 13
        37.0,  # joint 14
        37.0,  # joint 15
        37.0,  # joint 16
        37.0,  # joint 17
        37.0,  # joint 18
        37.0,  # joint 19
        37.0,  # joint 20
        37.0,  # joint 21
        37.0,  # joint 22
    ]
    max_contact_force = 200.0
    max_feet_height_limit_before_contact = 0.25
