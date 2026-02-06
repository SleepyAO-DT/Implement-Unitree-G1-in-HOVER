# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
===================================================================================
DETAILED EXPLANATION OF NEURAL WBC REWARD SYSTEM FOR G1 HUMANOID ROBOT
===================================================================================

本文件详细说明了G1人形机器人的奖励函数计算过程。

核心原理：
1. reward_sum = Σ (reward_function_result × scale_factor)
2. 每个reward函数计算一个张量，形状为 (num_envs,)，表示每个并行环境的奖励值
3. 通过配置中的scale字典加权求和得到最终奖励

数据流向：
articulation_data ──> 关节力矩、关节位置、关节速度、关节加速度
body_state ──> 身体各部位的位置、旋转、线速度、角速度
ref_motion_state ──> 参考动作的关节位置/速度、身体速度等
contact_sensor ──> 脚部接触信息、接触力

===================================================================================
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from neural_wbc.core import math_utils
from isaaclab.assets import ArticulationData
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from ._env import NeuralWBCEnv

import phc.utils.torch_utils as torch_utils
from neural_wbc.core.body_state import BodyState
from neural_wbc.core.reference_motion import ReferenceMotionState
from .reward_cfg import NeuralWBCRewardCfg


class NeuralWBCRewards:
    """
    奖励计算类
    
    负责计算所有奖励函数并返回加权求和结果。
    支持的奖励类型：
    - 追踪奖励（reward_*）：与参考动作匹配程度越高，奖励越大
    - 惩罚项（penalize_*）：机器人执行不良行为（大力矩、超限等）时的负奖励
    """

    def __init__(
        self,
        env: NeuralWBCEnv,
        reward_cfg: NeuralWBCRewardCfg,
        contact_sensor: ContactSensor,
        contact_sensor_feet_ids: list,
        body_state_feet_ids: list,
    ):
        # 获取环境基本信息
        self._num_envs = env.num_envs  # 并行运行的环境数量
        self._device = env.device  # 计算设备（CPU或GPU）
        self._cfg = reward_cfg  # 奖励配置对象
        
        # 将配置中的限制转换为张量并缩放
        # torque_limits: 每个关节的最大许可力矩值，单位N·m
        # torque_limits_scale = 0.85 表示使用额定值的85%
        self._torque_limits = torch.tensor(self._cfg.torque_limits, device=self._device) * self._cfg.torque_limits_scale
        
        # joint_pos_limits: 每个关节的位置限制对 [(min, max), ...]
        # 例如关节0的range是 (-2.5307, 2.8798) 弧度
        self._joint_pos_limits = torch.tensor(self._cfg.joint_pos_limits, device=self._device)
        
        # joint_vel_limits: 每个关节的速度限制，单位 rad/s
        # joint_vel_limits_scale = 0.85 表示使用额定值的85%
        self._joint_vel_limits = (
            torch.tensor(self._cfg.joint_vel_limits, device=self._device) * self._cfg.joint_vel_limits_scale
        )
        
        # 接触传感器配置
        self.contact_sensor = contact_sensor  # IsaacLab中的接触传感器对象
        self.contact_sensor_feet_ids = contact_sensor_feet_ids  # 脚部传感器索引 [左脚, 右脚]
        self._body_state_feet_ids = body_state_feet_ids  # 脚部身体索引
        
        # 仿真时间步
        self._dt = env.step_dt  # 单步仿真时间，单位秒
        
        # 关节变换
        self._joint_id_reorder = env._joint_ids  # 用于重新排列关节的索引
        
        # 重力向量（IsaacSim中Z轴向上）
        # gravity = [0, 0, -1] 表示重力沿Z轴负方向（指向地面）
        # repeat((num_envs, 1)) 将其扩展为 (num_envs, 3) 的张量
        self._gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self._device).repeat((self._num_envs, 1))
        
        # 脚部在空中时的最大高度记录
        # 用于计算脚在接触地面前能达到的最大高度
        self._feet_max_height_in_air = torch.zeros(
            self._num_envs, len(self.contact_sensor_feet_ids), dtype=torch.float, device=self._device
        )

    def compute_reward(
        self,
        articulation_data: ArticulationData,
        body_state: BodyState,
        ref_motion_state: ReferenceMotionState,
        previous_actions: torch.Tensor,
        actions: torch.Tensor,
        reset_buf: torch.Tensor,
        timeout_buf: torch.Tensor,
        penalty_scale: float,
    ) -> tuple[torch.Tensor, dict]:
        """
        计算总奖励（加权求和）
        
        这是主入口函数，被调用一次时计算所有奖励。
        
        输入参数说明：
        ─────────────────────────────────────────
        
        articulation_data (ArticulationData)：
            IsaacLab提供的关节数据对象，包含：
            - applied_torque: (num_envs, num_joints) 每个关节的应用力矩
            - joint_acc: (num_envs, num_joints) 每个关节的加速度
            - joint_vel: (num_envs, num_joints) 每个关节的速度
            - projected_gravity_b: (num_envs, 3) 在身体坐标系中投影的重力向量
        
        body_state (BodyState)：
            机器人身体状态，包含：
            - joint_pos: (num_envs, num_joints) 关节位置（弧度）
            - joint_vel: (num_envs, num_joints) 关节速度（弧度/秒）
            - body_lin_vel: (num_envs, num_body_parts, 3) 各身体部位的线速度
            - body_ang_vel: (num_envs, num_body_parts, 3) 各身体部位的角速度
            - body_pos: (num_envs, num_body_parts, 3) 各身体部位的位置
            - body_rot: (num_envs, num_body_parts, 4) 各身体部位的旋转（四元数wxyz）
            - body_pos_extend: (num_envs, total_features) 扩展的身体位置特征
        
        ref_motion_state (ReferenceMotionState)：
            参考动作状态（来自动作捕捉或参考动作库），包含：
            - joint_pos: (num_envs, num_joints) 参考关节位置
            - joint_vel: (num_envs, num_joints) 参考关节速度
            - body_lin_vel: (num_envs, num_body_parts, 3) 参考身体线速度
            - body_ang_vel: (num_envs, num_body_parts, 3) 参考身体角速度
            - body_pos_extend: (num_envs, total_features) 参考身体位置特征
            - body_rot: (num_envs, num_body_parts, 4) 参考身体旋转
        
        previous_actions (torch.Tensor): (num_envs, num_joints) 上一时步的动作
        actions (torch.Tensor): (num_envs, num_joints) 当前时步的动作
        
        reset_buf (torch.Tensor): (num_envs,) bool，True表示环境需要重置
        timeout_buf (torch.Tensor): (num_envs,) bool，True表示到达时间限制
        penalty_scale (float): 惩罚项的全局缩放因子
        
        返回值：
        ──────────────────────────────────────────
        reward_sum: (num_envs,) 每个环境的总奖励
        rewards: dict，键为奖励名称，值为该奖励的计算结果张量
        """
        
        # 初始化总奖励张量，形状(num_envs)，全0
        reward_sum = torch.zeros([self._num_envs], device=self._device)
        
        # 字典用于存储每个奖励项的详细计算结果
        rewards = {}
        
        # 遍历配置中定义的所有奖励项
        for reward_name, scale in self._cfg.scales.items():
            # reward_name: 奖励函数名称，如 "reward_track_joint_positions"
            # scale: 该奖励的权重系数，如 32.0（正数）或 -0.0001（负数）
            
            try:
                # 使用反射获取对应的奖励函数方法
                # 例如 reward_name="reward_track_joint_positions" 时，
                # 会调用 self.reward_track_joint_positions()
                reward_fn = getattr(self, reward_name)
            except AttributeError:
                raise AttributeError(f"No reward or penalty function is defined for {reward_name}")

            # 调用奖励函数，传入所有需要的数据
            # 各函数通过 **kwargs 接收不需要的参数（忽略处理）
            rewards[reward_name] = reward_fn(
                body_state=body_state,
                ref_motion_state=ref_motion_state,
                articulation_data=articulation_data,
                previous_actions=previous_actions,
                actions=actions,
                reset_buf=reset_buf,
                timeout_buf=timeout_buf,
            )
            
            # 如果是惩罚项（名称以"penalize"开头），乘以全局惩罚缩放因子
            # 这允许在训练中动态调整惩罚强度
            if reward_name.startswith("penalize"):
                rewards[reward_name] *= penalty_scale
            
            # 累加加权后的奖励到总奖励
            # reward_sum += rewards[reward_name] * scale
            reward_sum += rewards[reward_name] * scale

        return reward_sum, rewards

    # ═════════════════════════════════════════════════════════════════════════
    # 追踪类奖励 (Tracking Rewards)
    # 这些奖励鼓励机器人的状态与参考动作相匹配
    # ═════════════════════════════════════════════════════════════════════════

    def reward_track_joint_positions(
        self,
        body_state: BodyState,
        ref_motion_state: ReferenceMotionState,
        **kwargs,
    ) -> torch.Tensor:
        """
        关节位置追踪奖励
        
        计算公式：
        reward = exp(-mean_squared_error / sigma)
                = exp( -mean((ref_pos - actual_pos)²) / joint_pos_sigma )
        
        其中sigma = 0.5（配置中的body_pos_sigma）
        
        解释：
        - 当机器人关节位置与参考动作完全一致时，error=0，reward≈1
        - 当偏差增加时，exp指数衰减，奖励快速减少
        - 在所有19个关节上的平均误差越小，奖励越高
        
        张量操作细节：
        - body_state.joint_pos: (num_envs, 23) 当前关节位置
        - ref_motion_state.joint_pos: (num_envs, 23) 参考关节位置
        - torch.square(): 平方操作
        - torch.mean(., dim=1): 沿关节维度求平均，得到(num_envs,)
        - torch.exp(): 指数函数
        """
        # 获取当前关节位置，shape: (num_envs, 23)
        joint_pos = body_state.joint_pos
        
        # 获取参考关节位置，shape: (num_envs, 23)
        ref_joint_pos = ref_motion_state.joint_pos
        
        # 计算位置差，shape: (num_envs, 23)
        # diff = ref_pos - actual_pos
        
        # 平方后沿关节维度求平均，shape: (num_envs,)
        # mean_squared_error = mean((ref - actual)²)
        mean_joint_pos_diff_squared = torch.mean(torch.square(ref_joint_pos - joint_pos), dim=1)
        
        # 指数衰减奖励
        # 当误差为0时，reward=1；误差越大，reward越小
        return torch.exp(-mean_joint_pos_diff_squared / self._cfg.joint_pos_sigma)

    def reward_track_joint_velocities(
        self,
        body_state: BodyState,
        ref_motion_state: ReferenceMotionState,
        **kwargs,
    ) -> torch.Tensor:
        """
        关节速度追踪奖励
        
        计算公式：
        reward = exp(-mean((ref_vel - actual_vel)²) / sigma)
        where sigma = 1.0 (joint_vel_sigma)
        
        原理与关节位置奖励相同，但作用于速度维度。
        鼓励机器人以与参考动作相同的速度移动关节。
        """
        joint_vel = body_state.joint_vel
        ref_joint_vel = ref_motion_state.joint_vel
        mean_joint_vel_diff_squared = torch.mean(torch.square(ref_joint_vel - joint_vel), dim=1)
        return torch.exp(-mean_joint_vel_diff_squared / self._cfg.joint_vel_sigma)

    def reward_track_body_velocities(
        self,
        body_state: BodyState,
        ref_motion_state: ReferenceMotionState,
        **kwargs,
    ) -> torch.Tensor:
        """
        身体线速度追踪奖励
        
        计算公式：
        reward = exp(-mean_diff_vel_squared / body_vel_sigma)
        where body_vel_sigma = 10
        
        这里计算的是身体各部位的线速度追踪。
        
        张量操作：
        - body_lin_vel: (num_envs, num_body_parts, 3)
          num_body_parts 包括：骨盆(pelvis)、脊椎(spine)、胸部(chest)、
          颈部(neck)、头部(head)、左右肩膀、左右肘部、左右手、左右脚等
          
        - (diff_vel**2).mean(dim=-1): 在3维坐标上求平均
          输出shape: (num_envs, num_body_parts)
          
        - .mean(dim=-1): 再在身体部位维度求平均
          输出shape: (num_envs,)
        """
        # 获取实际身体线速度，shape: (num_envs, num_body_parts, 3)
        body_vel = body_state.body_lin_vel
        
        # 获取参考身体线速度，shape: (num_envs, num_body_parts, 3)
        ref_body_vel = ref_motion_state.body_lin_vel
        
        # 计算速度差，shape: (num_envs, num_body_parts, 3)
        diff_vel = ref_body_vel - body_vel
        
        # 计算均方差：
        # (diff_vel**2) -> (num_envs, num_body_parts, 3)
        # .mean(dim=-1) -> (num_envs, num_body_parts) 沿x,y,z方向求平均
        # .mean(dim=-1) -> (num_envs,) 沿身体部位求平均
        mean_diff_vel_squared = (diff_vel**2).mean(dim=-1).mean(dim=-1)
        
        return torch.exp(-mean_diff_vel_squared / self._cfg.body_vel_sigma)

    def reward_track_body_angular_velocities(
        self,
        body_state: BodyState,
        ref_motion_state: ReferenceMotionState,
        **kwargs,
    ) -> torch.Tensor:
        """
        身体角速度追踪奖励
        
        与线速度奖励类似，但作用于角速度。
        鼓励机器人身体的旋转速度与参考动作相匹配。
        
        where body_ang_vel_sigma = 10
        """
        body_ang_vel = body_state.body_ang_vel
        ref_body_ang_vel = ref_motion_state.body_ang_vel
        diff_ang_vel = ref_body_ang_vel - body_ang_vel
        mean_diff_ang_vel_squared = (diff_ang_vel**2).mean(dim=-1).mean(dim=-1)
        return torch.exp(-mean_diff_ang_vel_squared / self._cfg.body_ang_vel_sigma)

    def reward_track_body_position_extended(
        self,
        body_state: BodyState,
        ref_motion_state: ReferenceMotionState,
        **kwargs,
    ) -> torch.Tensor:
        """
        扩展身体位置追踪奖励
        
        这是一个更细粒度的位置追踪，将H1的身体分为上半身和下半身分别计算：
        - body_pos_extend: (num_envs, 22) 或更多维度的特征向量
          前11维对应下半身（两条腿）
          后11+维对应上半身（躯干、手臂、头部等）

        如果变为G1, 则下半身为13个维度。
        
        - body_pos_extend: (num_envs, 27) 或更多维度的特征向量
          前13维对应下半身（两条腿）
          后13+维对应上半身（躯干、手臂、头部等）
        
        计算公式：
        for lower_body (前13维):
            r_lower = exp(-mean_diff² / body_pos_lower_body_sigma)
        for upper_body (后续维):
            r_upper = exp(-mean_diff² / body_pos_upper_body_sigma)
        
        final_reward = r_lower * weight_lower + r_upper * weight_upper
        
        其中：
        - body_pos_lower_body_sigma = 0.5 (下半身容差更大，更灵活)
        - body_pos_upper_body_sigma = 0.03 (上半身要求精确，需要平衡)
        - weight_lower = 0.5
        - weight_upper = 1.0 (上半身权重更高，强调平衡)
        """
        body_pos_extend = body_state.body_pos_extend
        ref_body_pos_extend = ref_motion_state.body_pos_extend

        # 计算位置差
        diff_global_body_pos = ref_body_pos_extend - body_pos_extend
        
        # 分割为下半身和上半身
        # 前13维为下半身（两条腿的所有信息）
        diff_global_body_pos_lower = diff_global_body_pos[:, :13]
        # 后续维为上半身（躯干、手臂等）
        diff_global_body_pos_upper = diff_global_body_pos[:, 13:]
        
        # 计算下半身的均方差
        diff_body_pos_dist_lower = (diff_global_body_pos_lower**2).mean(dim=-1).mean(dim=-1)
        # 计算上半身的均方差
        diff_body_pos_dist_upper = (diff_global_body_pos_upper**2).mean(dim=-1).mean(dim=-1)
        
        # 计算指数衰减奖励
        r_body_pos_lower = torch.exp(-diff_body_pos_dist_lower / self._cfg.body_pos_lower_body_sigma)
        r_body_pos_upper = torch.exp(-diff_body_pos_dist_upper / self._cfg.body_pos_upper_body_sigma)

        # 加权求和：上半身权重更高，因为保持平衡更难
        return (
            r_body_pos_lower * self._cfg.body_pos_lower_body_weight
            + r_body_pos_upper * self._cfg.body_pos_upper_body_weight
        )

    def reward_track_body_position_vr_key_points(
        self,
        body_state: BodyState,
        ref_motion_state: ReferenceMotionState,
        **kwargs,
    ) -> torch.Tensor:
        """
        VR关键点位置追踪奖励
        
        这个奖励函数只追踪身体的3个关键点（可能是头、两手或其他重要位置）。
        用于VR动作捕捉场景，只关心特定位置的精确度。
        
        计算公式：
        reward = exp(-mean((ref_keypoints - actual_keypoints)²) / sigma)
        where body_pos_vr_key_points_sigma = 0.03
        
        body_pos_extend的最后3维代表VR关键点。
        """
        body_pos_extend = body_state.body_pos_extend
        ref_body_pos_extend = ref_motion_state.body_pos_extend

        diff_global_body_pos = ref_body_pos_extend - body_pos_extend
        
        # 只取最后3维（VR关键点）
        diff_global_body_pos_vr_key_points = diff_global_body_pos[:, -3:]
        
        # 计算均方差
        diff_body_pos_dist_vr_key_points = (diff_global_body_pos_vr_key_points**2).mean(dim=-1).mean(dim=-1)

        return torch.exp(-diff_body_pos_dist_vr_key_points / self._cfg.body_pos_vr_key_points_sigma)

    def reward_track_body_rotation(
        self,
        body_state: BodyState,
        ref_motion_state: ReferenceMotionState,
        **kwargs,
    ) -> torch.Tensor:
        """
        身体旋转追踪奖励
        
        这个函数计算的是身体的全局旋转与参考动作的匹配程度。
        用到四元数的复杂数学运算。
        
        计算步骤：
        1. 获取当前和参考的身体旋转（四元数格式）
           shape: (num_envs, num_body_parts, 4) 或类似
        
        2. 计算旋转差：
           Q_diff = Q_ref * conj(Q_actual)
           这个差值四元数代表从实际旋转到参考旋转所需的旋转
        
        3. 将四元数转换为轴-角表示：
           - 轴：旋转轴（三维向量）
           - 角：旋转角度（标量）
           实际使用的是旋转角度
        
        4. 计算奖励：
           reward = exp(-angle² / body_rot_sigma)
           其中 body_rot_sigma = 0.1
        
        四元数格式说明：
        IsaacLab使用 wxyz 格式（w为标量, xyz为向量）
        但torch_utils.quat_to_angle_axis 的输入期望 xyzw 格式
        所以需要 math_utils.convert_quat() 进行转换
        """
        # 获取实际和参考的身体旋转（四元数 xyzw或wxyz格式）
        body_rot = body_state.body_rot
        ref_body_rot = ref_motion_state.body_rot

        # 计算旋转差：从当前旋转到参考旋转
        # Q_diff = Q_ref * conj(Q_current)
        # 这给出了需要应用的旋转变换
        diff_global_body_rot = math_utils.quat_mul(ref_body_rot, math_utils.quat_conjugate(body_rot))
        
        # 转换四元数格式从 wxyz 到 xyzw
        diff_global_body_rot_xyzw = math_utils.convert_quat(diff_global_body_rot, to="xyzw")
        
        # 将四元数转换为轴-角表示，提取旋转角度
        # quat_to_angle_axis 返回 (angle, axis)
        # [0] 取第一个输出（旋转角度）
        diff_global_body_angle = torch_utils.quat_to_angle_axis(diff_global_body_rot_xyzw)[0]
        
        # 计算均方差：沿身体部位维度求平均
        diff_global_body_angle_dist = (diff_global_body_angle**2).mean(dim=-1)

        return torch.exp(-diff_global_body_angle_dist / self._cfg.body_rot_sigma)

    # ═════════════════════════════════════════════════════════════════════════
    # 惩罚类函数 (Penalty Functions)
    # 这些函数计算机器人不良行为的负奖励（惩罚）
    # ═════════════════════════════════════════════════════════════════════════

    def penalize_torques(
        self,
        articulation_data: ArticulationData,
        **kwargs,
    ) -> torch.Tensor:
        """
        力矩惩罚
        
        惩罚机器人使用过大的力矩，鼓励能量效率。
        
        计算公式：
        penalty = sum((applied_torque)²)
        
        - applied_torque: (num_envs, num_joints) 23个关节的实际应用力矩
        - 平方后沿关节维度求和，得到 (num_envs,)
        
        这鼓励机器人使用尽可能小的力矩来执行动作。
        
        配置中的scale = -0.0001，是一个很小的惩罚（能量效率不是主要目标）
        """
        # applied_torque shape: (num_envs, 23)
        # torch.square() -> (num_envs, 23)
        # torch.sum(., dim=1) -> (num_envs,) 对所有关节求和
        return torch.sum(torch.square(articulation_data.applied_torque), dim=1)

    def penalize_by_torque_limits(
        self,
        articulation_data: ArticulationData,
        **kwargs,
    ) -> torch.Tensor:
        """
        超力矩限制惩罚
        
        当机器人的力矩超过定义的限制时，惩罚该机器人。
        这是一个硬约束，防止物理上不可行的动作。
        
        计算公式：
        for each joint i:
            if |torque[i]| > torque_limit[i]:
                penalty += |torque[i]| - torque_limit[i]
            else:
                penalty += 0
        
        使用torch.clip(min=0.0)实现这个逻辑。
        
        张量操作细节：
        - articulation_data.applied_torque[:, self._joint_id_reorder]
          使用_joint_id_reorder进行关节重新排列
          确保与cfg中定义的torque_limits顺序一致
        
        - torch.abs(): 取绝对值（力矩有正负）
        - subtract torque_limits
        - .clip(min=0.0): 只保留超限的部分
        
        配置中的scale = -2，这是一个相对强的惩罚
        """
        # 获取应用的力矩，通过_joint_id_reorder进行重排列以匹配配置
        # 计算超过限制的部分：max(0, |torque| - limit)
        return torch.sum(
            (torch.abs(articulation_data.applied_torque[:, self._joint_id_reorder]) - self._torque_limits).clip(
                min=0.0
            ),
            dim=1,
        )

    def penalize_joint_accelerations(
        self,
        articulation_data: ArticulationData,
        **kwargs,
    ) -> torch.Tensor:
        """
        关节加速度惩罚
        
        限制关节加速度，鼓励平滑的动作。
        
        计算公式：
        penalty = sum((joint_acceleration)²)
        
        - joint_acc: (num_envs, 23) 23个关节的加速度 rad/s²
        - 平方并求和得到 (num_envs,)
        
        大的加速度会导致：
        1. 物理上的抖动
        2. 更高的力矩需求
        3. 不自然的动作
        
        配置中的scale = -0.000011，这是一个很小的惩罚
        """
        return torch.sum(torch.square(articulation_data.joint_acc), dim=1)

    def penalize_joint_velocities(
        self,
        articulation_data: ArticulationData,
        **kwargs,
    ) -> torch.Tensor:
        """
        关节速度惩罚
        
        限制关节速度，有几个目的：
        1. 避免物理上不合理的高速运动
        2. 鼓励平滑低速的关节移动
        3. 降低电机磨损
        
        计算公式：
        penalty = sum((joint_velocity)²)
        
        注意：这个惩罚作用于实际关节速度，而不是目标速度。
        
        配置中的scale = -0.004
        """
        return torch.sum(torch.square(articulation_data.joint_vel), dim=1)

    def penalize_lower_body_action_changes(
        self,
        previous_actions: torch.Tensor,
        actions: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        下半身动作变化惩罚
        
        限制下半身（腿部）相邻时步之间的动作变化幅度。
        鼓励平稳的动作序列。
        
        计算公式：
        penalty = sum((action[t] - action[t-1])²) for lower_body_joints
        
        G1机器人的下半身关节：
        - 关节 0-12 是腿部（对应注释中的13个下半身关节）
          0-6: 左腿 (左髋x,y,z, 左膝, 左踝x,y)
          7-12: 右腿 (右髋x,y,z, 右膝, 右踝x,y)
        
        这个惩罚防止关节突然跳跃，提高运动的连贯性。
        
        配置中的scale = -3.0，这是一个相对强的惩罚
        """
        # 限制范围：[:, :13] 表示前13个关节（下半身）
        # 计算与前一时步的差
        # 平方后求和
        return torch.sum(torch.square(previous_actions[:, :13] - actions[:, :13]), dim=1)

    def penalize_upper_body_action_changes(
        self,
        previous_actions: torch.Tensor,
        actions: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        上半身动作变化惩罚
        
        与下半身类似，但作用于上半身（躯干和手臂）。
        
        G1机器人的上半身关节：
        - 关节 13-23 是上半身
          13-15: 腰部 (腰yaw, 腰roll, 腰pitch)
          16-23: 双臂 (左臂4个关节, 右臂4个关节)
        
        配置中的scale = -0.625，比下半身低，因为上半身动作可以更频繁变化
        """
        # 限制范围：[:, 13:] 表示从关节13到末尾（上半身）
        return torch.sum(torch.square(previous_actions[:, 13:] - actions[:, 13:]), dim=1)

    def penalize_by_joint_pos_limits(
        self,
        body_state: BodyState,
        **kwargs,
    ) -> torch.Tensor:
        """
        关节位置限制惩罚
        
        当关节位置超出定义的范围时，进行惩罚。
        这是一个硬约束，防止关节反向弯折。
        
        计算公式：
        for each joint i:
            if joint_pos[i] < limit_min[i]:
                penalty += -(joint_pos[i] - limit_min[i])  # 负值，绝对值大
            if joint_pos[i] > limit_max[i]:
                penalty += joint_pos[i] - limit_max[i]     # 正值
        
        使用torch.clip实现这个逻辑：
        - out_of_bounds_lower = -(pos - min).clip(max=0.0)
          当 pos < min 时，(pos - min) 为负，
          -(负数).clip(max=0) 给出正的惩罚
        
        - out_of_bounds_upper = (pos - max).clip(min=0.0)
          当 pos > max 时，(pos - max) 为正，
          .clip(min=0.0) 保留正值
        
        例：
        joint_pos=-2.6, limit_min=-2.5, limit_max=2.9
        out_lower = -(-2.6 - (-2.5)).clip(max=0) = -(-0.1).clip(max=0) = 0
        out_upper = (-2.6 - 2.9).clip(min=0) = (-5.5).clip(min=0) = 0
        
        joint_pos=-3.0, limit_min=-2.5, limit_max=2.9
        out_lower = -(-3.0 - (-2.5)).clip(max=0) = -(-0.5).clip(max=0) = 0.5
        out_upper = (-3.0 - 2.9).clip(min=0) = (-5.9).clip(min=0) = 0
        
        配置中的scale = -125.0，这是一个非常强的惩罚
        """
        joint_pos = body_state.joint_pos
        
        # 计算下限超出：当joint_pos < limit_min时
        # (joint_pos - limit_min) 是负数
        # .clip(max=0.0) 保留负值或0
        # -(负数) 得到正的惩罚
        out_of_bounds = -(joint_pos - self._joint_pos_limits[:, 0]).clip(max=0.0)
        
        # 计算上限超出：当joint_pos > limit_max时
        # (joint_pos - limit_max) 是正数
        # .clip(min=0.0) 保留正值
        out_of_bounds += (joint_pos - self._joint_pos_limits[:, 1]).clip(min=0.0)
        
        # 对所有关节的超出部分求和
        return torch.sum(out_of_bounds, dim=1)

    def penalize_by_joint_velocity_limits(
        self,
        body_state: BodyState,
        **kwargs,
    ) -> torch.Tensor:
        """
        关节速度限制惩罚
        
        当关节速度超过定义的最大值时，进行惩罚。
        
        计算公式：
        for each joint i:
            if |vel[i]| > limit[i]:
                penalty += |vel[i]| - limit[i]
            else:
                penalty += 0
        
        与torque_limits类似，但作用于速度。
        
        注意：这里使用 .clip(min=0.0, max=1.0)
        max=1.0 表示单个关节的惩罚上限为1.0
        这是一个截断机制，防止某个关节的超速惩罚过度主导
        
        配置中的scale = -50.0，相对强的惩罚
        """
        # 取绝对值
        # 计算超过限制的部分
        # clip(min=0.0, max=1.0) 保留在[0, 1]范围内的超限值
        return torch.sum(
            (torch.abs(body_state.joint_vel) - self._joint_vel_limits).clip(min=0.0, max=1.0),
            dim=1,
        )

    def penalize_early_termination(self, reset_buf: torch.Tensor, timeout_buf: torch.Tensor, **kwargs):
        """
        提前终止惩罚
        
        当环境在达到时间限制前就终止时（例如摔倒），进行惩罚。
        
        计算公式：
        penalty = 1.0 if (reset_buf AND NOT timeout_buf) else 0.0
        
        含义：
        - reset_buf=True, timeout_buf=False: 环境提前终止（失败），penalty=1
        - reset_buf=True, timeout_buf=True: 正常超时（成功），penalty=0
        - reset_buf=False: 环境继续运行，penalty=0
        
        这鼓励机器人在整个剧集期间保持平衡和稳定。
        
        配置中的scale = -250.0，这是一个非常强的惩罚
        """
        # ~timeout_buf 是 NOT timeout_buf，逻辑非
        # reset_buf * ~timeout_buf 是 reset_buf AND NOT timeout_buf
        # .float() 转换为浮点数 0.0 或 1.0
        return (reset_buf * ~timeout_buf).float()

    def penalize_feet_contact_forces(self, **kwargs):
        """
        脚部接触力惩罚
        
        惩罚脚部与地面的接触力过大。
        大的接触力意味着：
        1. 不稳定的着陆
        2. 笨重的步态
        3. 效率低下
        
        计算公式：
        feet_contact_forces: (num_envs, 2, 3)
          2个脚，每个脚3维力(Fx, Fy, Fz)
        
        for each environment:
            for each foot:
                force_magnitude = norm(contact_force)  # sqrt(Fx² + Fy² + Fz²)
                if force_magnitude > max_contact_force:
                    penalty += force_magnitude - max_contact_force
                else:
                    penalty += 0
            sum_over_feet
        
        配置中：max_contact_force = 500.0 N
                scale = -0.75（相对温和的惩罚）
        """
        # 获取脚部接触力：(num_envs, 2, 3)
        feet_contact_forces = self._get_feet_contact_forces()
        
        # torch.norm(., dim=-1) 计算3维力向量的模长
        # 输出：(num_envs, 2)
        
        # 与max_contact_force比较，超过部分才惩罚
        # .clip(min=0.0) 保留正值（超限部分）
        
        # torch.sum(., dim=1) 对两个脚求和
        # 输出：(num_envs,)
        return torch.sum((torch.norm(feet_contact_forces, dim=-1) - self._cfg.max_contact_force).clip(min=0.0), dim=1)

    def penalize_stumble(self, **kwargs):
        """
        绊脚惩罚
        
        检测机器人是否在跌倒（绊脚）。
        绊脚被定义为：脚部的水平接触力（Fx, Fy）相对于竖直力（Fz）过大。
        
        物理直觉：
        - 稳定行走时：竖直力很大，水平力很小（向下压）
        - 绊脚时：竖直力突然变小，但水平力依然很大（转向或摔倒）
        
        计算公式：
        feet_contact_forces: (num_envs, 2, 3) [Fx, Fy, Fz]
        
        for each environment:
            for each foot:
                horizontal_force = norm([Fx, Fy])
                vertical_force = |Fz|
                if horizontal_force > 5 * vertical_force:
                    stumble = True
                    break
            if stumble:
                penalty = 1.0
            else:
                penalty = 0.0
        
        torch.norm(forces[:, :, :2], dim=2) 计算水平力
        torch.abs(forces[:, :, 2]) 得到竖直力绝对值
        
        torch.any(., dim=1) 检查是否存在任何一只脚绊倒
        .float() 转换为浮点0.0或1.0
        
        配置中：scale = -1000.0，非常强的惩罚！
        """
        feet_contact_forces = self._get_feet_contact_forces()
        
        # 计算：水平力的模长 > 5 * 竖直力的绝对值
        # 如果这个条件对任何一只脚为真，就认为绊倒了
        # torch.any(., dim=1) 检查每个环境是否有脚绊倒
        return torch.any(
            torch.norm(feet_contact_forces[:, :, :2], dim=2) > 5 * torch.abs(feet_contact_forces[:, :, 2]), dim=1
        ).float()

    def penalize_slippage(self, body_state: BodyState, **kwargs):
        """
        滑动惩罚
        
        当脚部与地面有接触但仍在滑动时，进行惩罚。
        理想的行走应该是脚部接触时不滑动（静摩擦）。
        
        计算公式：
        脚部线速度 > 0 AND 脚部有接触 → 滑动
        
        张量操作：
        - body_state.body_lin_vel[:, body_state_feet_ids]
          获取脚部的线速度：(num_envs, 2, 3)
        
        - torch.norm(., dim=-1): 计算速度模长
          输出：(num_envs, 2)
        
        - contact_forces = _get_feet_contact_forces(): (num_envs, 2, 3)
        - torch.norm(., dim=-1) > 1.0: 判断接触力是否存在（大于1N）
          输出：(num_envs, 2) bool
        
        - feet_vel * contact_bool: 只有在接触时才计算滑动
        
        - torch.sum(., dim=1): 对两个脚求和
          输出：(num_envs,)
        
        配置中：scale = -37.5，相对中等的惩罚
        """
        # 获取脚部速度
        feet_vel = body_state.body_lin_vel[:, self._body_state_feet_ids]
        
        # 计算脚部速度的模长：(num_envs, 2)
        feet_vel_norm = torch.norm(feet_vel, dim=-1)
        
        # 获取脚部接触力
        feet_contact_forces = self._get_feet_contact_forces()
        
        # 计算接触力的模长，大于1.0表示有接触
        contact_exists = torch.norm(feet_contact_forces, dim=-1) > 1.0
        
        # 滑动惩罚 = 脚部速度 * 接触判断（只在接触时计算）
        return torch.sum(feet_vel_norm * contact_exists, dim=1)

    def penalize_feet_orientation(self, body_state: BodyState, **kwargs):
        """
        脚部姿态惩罚
        
        脚部应该与地面平行（或足够接近）保持平衡。
        这个惩罚鼓励脚部在x-y平面上的重力投影接近0。
        
        计算公式：
        对于每只脚：
        1. 获取脚部的旋转四元数 Q_foot
        2. 在脚部坐标系中投影重力向量（局部坐标）
           local_gravity = quat_rotate_inverse(Q_foot, world_gravity)
           其中 world_gravity = [0, 0, -1]（指向地面）
        3. 计算x,y分量的平方和（z应该指向地面）
           penalty_foot = sqrt(local_gravity.x² + local_gravity.y²)
        
        物理含义：
        - 如果脚部与地面平行：local_gravity = [0, 0, -1]，penalty = 0
        - 如果脚部倾斜：local_gravity有非零的x,y分量，penalty > 0
        
        配置中：scale = -62.5，中等强度惩罚
        """
        # 获取左脚旋转（四元数）
        left_quat = body_state.body_rot[:, self._body_state_feet_ids[0]]
        
        # 在左脚坐标系中投影重力
        # quat_rotate_inverse: 将向量从世界坐标系变换到局部坐标系
        left_gravity = math_utils.quat_rotate_inverse(left_quat, self._gravity_vec)
        
        # 获取右脚旋转
        right_quat = body_state.body_rot[:, self._body_state_feet_ids[1]]
        
        # 在右脚坐标系中投影重力
        right_gravity = math_utils.quat_rotate_inverse(right_quat, self._gravity_vec)
        
        # 计算两只脚的惩罚：只考虑x,y分量（[:, :2]）的平方和
        # sqrt() 得到2D模长
        return (
            torch.sum(torch.square(left_gravity[:, :2]), dim=1) ** 0.5
            + torch.sum(torch.square(right_gravity[:, :2]), dim=1) ** 0.5
        )

    def penalize_feet_air_time(self, ref_motion_state: ReferenceMotionState, **kwargs):
        """
        脚部空中时间惩罚（实际上是奖励）
        
        这个函数名叫"penalize"但实际上可能是负奖励变成正奖励。
        鼓励脚部在空中停留适当的时间（跳跃），但在参考动作有动作时。
        
        计算公式：
        last_air_time[foot] = 自上次接触地面以来的时间
        
        reward = sum((last_air_time - 0.25) * first_contact) 
                 * (ref_velocity > 0.1)
        
        含义：
        - 当脚部第一次接触地面时（first_contact=True）：
          - 如果空中时间 > 0.25s（好的跳跃）：reward += (air_time - 0.25)
          - 如果空中时间 < 0.25s（差的跳跃）：reward -= ...（惩罚）
        
        - 只有在参考动作有足够的速度（>0.1 m/s）时才计算
          静止时不应该跳跃
        
        配置中：scale = 1000.0，正的大值，这是一个强奖励
        """
        # 获取参考动作中骨盆的水平速度
        ref_pelvis_vel_xy = ref_motion_state.body_lin_vel[:, 0, :2]
        
        # 获取脚部是否第一次接触地面
        first_contact = self._get_feet_first_contact()
        
        # 获取脚部最后一次空中的时间
        last_feet_air_time = self._get_last_air_time_for_feet()
        
        # 计算奖励：(air_time - 0.25秒) × 第一次接触
        # 鼓励脚部在空中停留超过0.25秒
        reward = torch.sum(
            (last_feet_air_time - 0.25) * first_contact, dim=1
        )
        
        # 只在参考动作速度足够大时计算（不是静止）
        reward *= torch.norm(ref_pelvis_vel_xy, dim=1) > 0.1
        
        return reward

    def penalize_both_feet_in_air(self, **kwargs):
        """
        双脚离地惩罚
        
        在正常行走中，总应该有至少一只脚接触地面。
        双脚离地只在跳跃时短暂发生，其他情况下应该被惩罚。
        
        计算公式：
        feet_in_air: (num_envs, 2) bool
            True 表示脚部离地（接触力z < 1.0）
        
        penalty = 1.0 if all(feet_in_air) else 0.0
        
        torch.all(., dim=1) 检查是否两只脚都离地
        .float() 转换为浮点0.0或1.0
        
        配置中：scale = -200.0，强惩罚
        """
        # 获取脚部是否离地
        feet_in_air = self._get_feet_in_the_air()
        
        # 检查两只脚是否都离地，返回 (num_envs,)
        return torch.all(feet_in_air, dim=1).float()

    def penalize_orientation(self, articulation_data: ArticulationData, **kwarg):
        """
        身体姿态惩罚
        
        鼓励机器人身体保持竖直，不要倾斜过度。
        
        计算公式：
        projected_gravity_b: 在身体坐标系中的重力向量投影
        
        如果身体竖直：projected_gravity = [0, 0, -1]
        倾斜时：x,y分量不为0
        
        penalty = sum(projected_gravity_b.x², projected_gravity_b.y²)
        
        这是一个平方和，不取平方根。
        
        配置中：scale = -200.0，强惩罚
        """
        # 获取在身体坐标系中投影的重力
        # shape: (num_envs, 3)
        projected_gravity = articulation_data.projected_gravity_b
        
        # 计算x,y分量的平方和（只保留水平分量）
        # torch.square().sum(dim=1) 得到 (num_envs,)
        return torch.sum(torch.square(projected_gravity[:, :2]), dim=1)

    def penalize_max_feet_height_before_contact(self, body_state: BodyState, **kwargs):
        """
        脚部最大高度惩罚
        
        鼓励脚部着陆时不要太高。太高的着陆意味着：
        1. 不必要的大幅度摆动
        2. 能量浪费
        3. 不稳定的步态
        
        计算公式：
        在脚部离地期间，记录脚部达到的最大高度。
        当脚部再次接触地面时（first_contact=True）：
            if max_height > max_height_limit (0.25m):
                penalty = max_height - max_height_limit
            else:
                penalty = 0
        
        然后在脚部再次离地时重置记录。
        
        这是一个有状态的奖励函数，需要在环境重置时也重置内部状态。
        
        配置中：scale = -2500.0，非常强的惩罚！
        """
        # 获取是否第一次接触（从离地到接触）
        first_contact = self._get_feet_first_contact()
        
        # 获取脚部高度（Z坐标）
        # body_pos: (num_envs, num_body_parts, 3)
        # body_pos[:, feet_ids, 2] 得到脚的Z坐标 (num_envs, 2)
        feet_height = body_state.body_pos[:, self._body_state_feet_ids, 2]
        
        # 更新脚部最大高度（脚的历史最大高度）
        # torch.max(old_max, current_height)
        self._feet_max_height_in_air = torch.max(self._feet_max_height_in_air, feet_height)
        
        # 计算超过限制的部分，只在第一次接触时计算
        # torch.clamp_min(x, 0) = max(x, 0) 确保非负
        feet_max_height = torch.sum(
            (torch.clamp_min(self._cfg.max_feet_height_limit_before_contact - self._feet_max_height_in_air, 0))
            * first_contact,
            dim=1,
        )
        
        # 获取脚部是否在空中
        feet_in_air = self._get_feet_in_the_air()
        
        # 当脚部离地时，重置该脚的最大高度记录
        # 只有在离地时才清除记录，为下一个周期做准备
        self._feet_max_height_in_air *= feet_in_air
        
        return feet_max_height

    # ═════════════════════════════════════════════════════════════════════════
    # 辅助函数 (Helper Methods)
    # 这些是内部工具函数，从传感器和状态中提取信息
    # ═════════════════════════════════════════════════════════════════════════

    def _get_feet_contact_forces(self):
        """
        获取脚部接触力
        
        从接触传感器中提取脚部的接触力。
        
        返回值：
        torch.Tensor: shape (num_envs, num_feet, 3)
            每个环境中两只脚的接触力 [Fx, Fy, Fz]，单位N
        
        组件说明：
        - net_forces_w: IsaacLab中的属性，存储世界坐标系中的合力
        - [:, contact_sensor_feet_ids, :] 提取脚部数据
          select axes: 所有环境, 脚部索引, 3维力向量
        """
        return self.contact_sensor.data.net_forces_w[:, self.contact_sensor_feet_ids, :]

    def _get_last_air_time_for_feet(self):
        """
        获取脚部最后一次空中时间
        
        从接触传感器的数据中获取脚部自上次接触以来的时间。
        
        返回值：
        torch.Tensor: shape (num_envs, num_feet)
            每只脚自上次失去接触以来的时间，单位秒
        
        这用于计算脚部的摆动周期和跳跃高度。
        """
        return self.contact_sensor.data.last_air_time[:, self.contact_sensor_feet_ids]

    def _get_feet_in_the_air(self):
        """
        检查脚部是否在空中
        
        通过检查接触力的z分量来判断。
        
        返回值：
        torch.Tensor: shape (num_envs, num_feet) bool
            True 表示脚部离地，False 表示有接触
        
        逻辑：
        feet_contact_forces[:, :, 2] 是z分量（竖直方向）
        <= 1.0 表示接触力很小，认为脚部在空中
        
        阈值1.0是根据经验设置的，大于1N的竖直力表示足够的接触。
        """
        return self._get_feet_contact_forces()[:, :, 2] <= 1.0

    def _get_feet_first_contact(self):
        """
        检查脚部是否刚接触地面
        
        比较当前和上一时步的接触状况，找出从无接触到有接触的转换。
        
        返回值：
        torch.Tensor: shape (num_envs, num_feet) bool
            True 表示在本时步从空中接触了地面
        
        实现方式：
        contact_sensor.compute_first_contact(dt) 计算接触转换
        返回的是该时步是否有新接触事件
        
        这在计算步长周期和脚部着陆质量时很重要。
        """
        return self.contact_sensor.compute_first_contact(self._dt)[:, self.contact_sensor_feet_ids]
