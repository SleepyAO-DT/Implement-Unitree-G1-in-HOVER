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

"""
===================================================================================
参考动作系统详解（Reference Motion System）
===================================================================================

本文件负责从mocap数据库加载参考动作，并将其与真实机器人关节进行匹配比较。

核心流程：
1. ReferenceMotionManager 从动作库（.bvh/.h5文件）加载动作数据
2. 根据当前时间戳从动作库中查询参考状态
3. 将参考状态的body和joint顺序重新排列，以匹配真实机器人的定义
4. 返回ReferenceMotionState对象，包含关节位置、速度等，用于reward计算

数据来源：
- motion_lib: MotionLibG1 对象，从.h5/.bvh文件加载mocap数据
- 包含多个独立的动作序列（clips）
- 每个动作有多个frame，每帧包含完整的骨骼信息

数据对应关系：
Reference Motion:                    | Real Robot (IsaacLab):
├─ body_pos (rg_pos)              | ← body_state.body_pos
├─ body_rot (rb_rot)              | ← body_state.body_rot  
├─ body_lin_vel (body_vel)        | ← body_state.body_lin_vel
├─ body_ang_vel (body_ang_vel)    | ← body_state.body_ang_vel
├─ joint_pos (dof_pos)            | ← body_state.joint_pos
└─ joint_vel (dof_vel)            | ← body_state.joint_vel

关键问题：顺序对应
─────────────────────────
reference motion的身体部位/关节顺序 可能与 IsaacLab模拟器中的顺序不同
因此需要通过 body_ids 和 joint_ids 进行映射和重新排列，
以确保在reward计算中进行的比较是对齐的。

===================================================================================
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import List

# TODO: Replace MotionLibH1 with the agnostic version when it's ready.
# from phc.utils.motion_lib_h1 import MotionLibH1
from phc.utils.motion_lib_g1 import MotionLibG1
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree


class ReferenceMotionState:
    """
    参考动作状态存储类
    
    封装从动作库中提取的参考动作的各个维度数据。
    这个类的主要作用是将动作库中的原始数据重新组织，
    使其与真实机器人的body和joint顺序相匹配。
    """

    def __init__(
        self,
        ref_motion: dict,
        body_ids: List[int] | None = None,
        joint_ids: List[int] | None = None,
    ):
        """
        初始化参考动作状态
        
        核心作用：将动作库中的多维数据按照指定的body_ids和joint_ids进行重新排列，
        以确保与真实机器人的body/joint对齐。
        
        参数说明：
        ───────────────────────────────────────────
        
        ref_motion (dict): 
            从MotionLibG1.get_motion_state()返回的动作数据字典
            包含以下关键字段：
            
            • 'root_pos': (num_envs, 3) 根节点（骨盆）的世界坐标位置
            • 'root_rot': (num_envs, 4) 根节点的旋转四元数 (wxyz)
            • 'root_vel': (num_envs, 3) 根节点的线速度
            • 'root_ang_vel': (num_envs, 3) 根节点的角速度
            
            • 'rg_pos': (num_envs, num_bodies_original, 3) 
              所有身体部位的世界坐标位置
              原始顺序来自mocap骨骼树（如BVH文件的顺序）
              例如G1机器人可能有20多个部位（骨盆、头、
              左肩、左肘、左手、右肩、右肘、右手、左髋、左膝、左脚、右髋、...）
            
            • 'rb_rot': (num_envs, num_bodies_original, 4)
              所有身体部位的旋转四元数
            
            • 'body_vel': (num_envs, num_bodies_original, 3)
              所有身体部位的线速度
            
            • 'body_ang_vel': (num_envs, num_bodies_original, 3)
              所有身体部位的角速度
            
            • 'rg_pos_t': (num_envs, num_bodies_extended, 3)
              扩展的身体位置特征
              "t"表示"tracked"或"transformed"
              比'rg_pos'包含更多的中间点位置（用于VR追踪或精细控制）
            
            • 'rg_rot_t': (num_envs, num_bodies_extended, 4)
              扩展的旋转定义
            
            • 'body_vel_t': (num_envs, num_bodies_extended, 3)
              扩展的线速度
            
            • 'body_ang_vel_t': (num_envs, num_bodies_extended, 3)
              扩展的角速度
            
            • 'dof_pos': (num_envs, num_joints_original, 1)
              所有关节的位置（弧度）
              原始顺序来自mocap骨骼树
              G1有23个关节（下肢12 + 上肢11）
            
            • 'dof_vel': (num_envs, num_joints_original, 1)
              所有关节的速度（弧度/秒）
        
        body_ids (List[int] | None):
            指定要使用的body索引的列表
            例如：[0, 1, 2, 3, 4, 5, ...]
            
            用途：
            - 当mocap中的body顺序与IsaacLab中的顺序不同时，用于重新映射
            - 例如mocap可能有25个body，而IsaacLab只需要其中的12个
            - 通过body_ids可以选择这12个，并按IsaacLab的顺序排列
            
            重新排列过程：
            位置从 rg_pos[:, 原始顺序, :] 
            重排为 rg_pos[:, body_ids, :]
            
            示例：
            mocap中的body顺序: [pelvis, spine, chest, neck, head, L_shoulder, ...]
            IsaacLab期望: [pelvis, chest, neck, head, L_shoulder, ...]
            则 body_ids = [0, 2, 3, 4, 5, ...] (跳过spine)
        
        joint_ids (List[int] | None):
            指定要使用的joint索引的列表
            
            用途：与body_ids类似，用于joint重新排列
            例如：mocap有25个关节，IsaacLab需要23个
            通过joint_ids选择并排列
            
            注意：关节顺序必须与reward函数中使用的顺序完全相同！
            否则reward中的比较会出错。
            例如：
            - reward计算中：ref_joint_pos[0] 对应"left_hip_pitch"
            - 如果ref_motion返回的joint_pos[0] 是"right_hip_pitch"
            - 则计算出的error会完全错误
            
        返回值说明：
        ───────────────────────────────────────────
        初始化后，该对象包含以下属性（均已按body_ids/joint_ids重新排列）：
        - root_pos, root_rot, root_lin_vel, root_ang_vel
        - body_pos, body_rot, body_lin_vel, body_ang_vel
        - body_pos_extend, body_rot_extend, body_lin_vel_extend, body_ang_vel_extend
        - joint_pos, joint_vel
        """

        # ────────────────────────────────────────────────────────────────────
        # 第1步：确定body_ids，如果未指定则使用全量
        # ────────────────────────────────────────────────────────────────────
        
        if body_ids is None:
            # 从ref_motion中读取原始body数量
            # rg_pos shape: (num_envs, num_bodies_original, 3)
            # .shape[1] 获取body维度的大小
            num_bodies = ref_motion["rg_pos"].shape[1]
            
            # 生成完整的索引列表 [0, 1, 2, ..., num_bodies-1]
            body_ids = list(range(0, num_bodies))

        # 保存body_ids用于对齐
        self._body_ids = body_ids

        # ────────────────────────────────────────────────────────────────────
        # 第2步：确定joint_ids，如果未指定则使用全量
        # ────────────────────────────────────────────────────────────────────
        
        if joint_ids is None:
            # 从ref_motion中读取原始joint数量
            # dof_pos shape: (num_envs, num_joints_original)
            num_joints = ref_motion["dof_pos"].shape[1]
            
            # 生成完整的索引列表
            joint_ids = list(range(0, num_joints))

        # 保存joint_ids用于对齐
        self._joint_ids = joint_ids

        # ────────────────────────────────────────────────────────────────────
        # 第3步：提取并存储根节点(Root)数据
        # ────────────────────────────────────────────────────────────────────
        
        # Root是特殊的，代表整个角色的基坐标系（通常是骨盆）
        # Root数据不按body_ids重新排列，直接使用原始值
        
        # 根节点位置：(num_envs, 3)
        # 例如 [1.5, 0.0, 1.0] 表示骨盆在世界坐标系中的位置
        self.root_pos = ref_motion["root_pos"]
        
        # 根节点旋转：(num_envs, 4) wxyz格式四元数
        self.root_rot = ref_motion["root_rot"]
        
        # 根节点线速度：(num_envs, 3) m/s
        # 例如 [0.5, 0.0, 0.0] 表示沿X轴以0.5m/s的速度运动
        self.root_lin_vel = ref_motion["root_vel"]
        
        # 根节点角速度：(num_envs, 3) rad/s
        self.root_ang_vel = ref_motion["root_ang_vel"]

        # ────────────────────────────────────────────────────────────────────
        # 第4步：提取并重新排列身体部位(Links/Bodies)数据
        # ────────────────────────────────────────────────────────────────────
        
        # 这是最关键的一步！这里需要确保body的顺序与IsaacLab中的相同
        
        # 身体位置：从 (num_envs, num_bodies_original, 3) 
        # 重排为 (num_envs, num_bodies_selected, 3)
        # 
        # 例如：
        # 原始ref_motion["rg_pos"][:, :, :] 有25个body
        # 通过[:, body_ids, :] 选择指定的body并按给定顺序排列
        # 最终得到与IsaacLab匹配的body顺序
        self.body_pos = ref_motion["rg_pos"][:, body_ids, :]
        
        # 身体旋转：(num_envs, num_bodies_selected, 4) wxyz四元数
        # 每个body的旋转都重新排列
        self.body_rot = ref_motion["rb_rot"][:, body_ids, :]
        
        # 身体线速度：(num_envs, num_bodies_selected, 3)
        # 每个身体部位的线速度（X, Y, Z方向）
        self.body_lin_vel = ref_motion["body_vel"][:, body_ids, :]
        
        # 身体角速度：(num_envs, num_bodies_selected, 3)
        # 每个身体部位的旋转速度（绕X, Y, Z轴）
        self.body_ang_vel = ref_motion["body_ang_vel"][:, body_ids, :]

        # ────────────────────────────────────────────────────────────────────
        # 第5步：提取扩展的身体位置数据
        # ────────────────────────────────────────────────────────────────────
        
        # "Extended" 指的是额外记录的身体部位或中间点
        # 用于更精细的动作控制或VR应用
        # 例如可能包括手指位置、头顶位置等
        
        # 注意：这些不通过body_ids进行重新排列！
        # 这可能是因为它们的顺序已经是固定的，或者
        # 会在后续的get_state_from_motion_lib_cache中处理
        
        self.body_pos_extend = ref_motion["rg_pos_t"]
        self.body_rot_extend = ref_motion["rg_rot_t"]
        self.body_lin_vel_extend = ref_motion["body_vel_t"]
        self.body_ang_vel_extend = ref_motion["body_ang_vel_t"]

        # ────────────────────────────────────────────────────────────────────
        # 第6步：提取并重新排列关节数据
        # ────────────────────────────────────────────────────────────────────
        
        # 这是最关键的部分！关节顺序必须与reward硬件对齐！
        
        # 关节位置：从 (num_envs, num_joints_original)
        # 重排为 (num_envs, num_joints_selected)
        #
        # 关键约束：
        # reward中的reward_track_joint_positions这样做比较：
        #   diff = ref_joint_pos[0] - body_state.joint_pos[0]  # 都应该是left_hip_pitch
        #
        # 如果ref_motion的joint_pos[0]是right_hip_pitch，
        # 而body_state.joint_pos[0]是left_hip_pitch，
        # 那么error会导致奖励完全错误！
        #
        # 因此joint_ids的排列顺序必须与IsaacLab中的关节顺序一致
        
        self.joint_pos = ref_motion["dof_pos"][:, joint_ids]
        
        # 关节速度：(num_envs, num_joints_selected)
        self.joint_vel = ref_motion["dof_vel"][:, joint_ids]


@dataclass
class ReferenceMotionManagerCfg:
    """
    参考动作管理器的配置类
    
    包含加载动作库所需的文件路径信息
    """
    motion_path: str  # 动作数据库文件路径，例如 "data/motions.h5"
    skeleton_path: str  # 骨骼定义文件路径，例如 "data/g1.mjcf"

    def __init__(self):
        pass


class ReferenceMotionManager:
    """
    参考动作管理器
    
    负责：
    1. 从动作库文件加载mocap数据
    2. 按照当前模拟时间查询参考动作状态
    3. 确保查询到的body和joint顺序与IsaacLab相匹配
    4. 管理多个并行环境的动作播放进度
    """

    def __init__(
        self,
        cfg: ReferenceMotionManagerCfg,
        device: torch.device,
        num_envs: int,
        random_sample: bool,
        extend_head: bool,
        dt: float,
    ):
        """
        初始化参考动作管理器
        
        参数说明：
        ───────────────────────────────────────────
        
        cfg (ReferenceMotionManagerCfg):
            配置对象，包含：
            - motion_path: 动作库文件路径（.h5或.bvh格式）
              例如 "datasets/g1_motions.h5"
              这个文件包含多个mocap录制的人类动作片段
            
            - skeleton_path: 骨骼定义文件路径（MJCF格式）
              例如 "assets/g1.mjcf"
              定义了G1机器人的骨骼结构（身体部位、关节关系等）
        
        device (torch.device):
            计算设备，GPU或CPU
            动作数据将加载到这个设备上
        
        num_envs (int):
            并行环境数量
            每个环境会加载一个动作片段并独立播放
            例如128表示同时运行128个并行的动作播放实例
        
        random_sample (bool):
            是否随机采样动作
            - True: 从动作库中随机选择不同的动作片段分配给每个环境
            - False: 按顺序从动作库中选择
        
        extend_head (bool):
            是否扩展头部表示
            对于H1机器人这可能很重要（SMPL模型的头部处理）
            对于G1可能会被忽略
        
        dt (float):
            策略时间步长（秒）
            这是RL算法的控制频率，例如0.01秒表示100Hz
            
            重要：时间步对应关系
            ──────────────────────
            策略步数 t ──× dt────→ 真实时间 t*dt (秒)
            
            当我们要查询第t步的参考动作时：
            实际查询的动作库时间 = episode_start_time + t * dt
            
            例如：dt=0.01, t=50
            则查询 episode_start_time + 50*0.01 = episode_start_time + 0.5秒
        """
        
        # ────────────────────────────────────────────────────────────────────
        # 存储基本参数
        # ────────────────────────────────────────────────────────────────────
        
        self._device = device
        self._num_envs = num_envs  # 并行环境数
        self._dt = dt  # 策略时间步
        
        # ────────────────────────────────────────────────────────────────────
        # 第1步：加载动作库（MotionLibG1）
        # ────────────────────────────────────────────────────────────────────
        
        # MotionLibG1是PHC库中的一个核心类，负责：
        # - 加载mocap数据文件（.h5或.bvh）
        # - 解析并缓存动作数据
        # - 提供高效的时间查询接口
        #
        # 参数说明：
        # - motion_file: 动作库文件路径
        # - mjcf_file: 骨骼定义文件（用于理解关节关系）
        # - device: GPU/CPU
        # - masterfoot_config: None（对于G1可能不需要）
        # - fix_height: False（不固定高度，允许运动）
        # - multi_thread: False（单线程查询）
        # - extend_head: 是否扩展头部表示
        
        self._motion_lib = MotionLibG1(
            motion_file=cfg.motion_path,
            mjcf_file=cfg.skeleton_path,
            device=self._device,
            masterfoot_conifg=None,
            fix_height=False,
            multi_thread=False,
            extend_head=extend_head,
        )
        

        # ────────────────────────────────────────────────────────────────────
        # 第2步：构建骨骼树
        # ────────────────────────────────────────────────────────────────────
        
        # 骨骼树定义了身体部位之间的关系（哪个是哪个的父节点）
        # 从MJCF文件中解析
        # 为每个环境创建一份副本（独立的骨骼树实例）
        
        self._skeleton_trees = [SkeletonTree.from_mjcf(cfg.skeleton_path)] * self._num_envs
        
        # 可以检查骨骼树的结构
        if len(self._skeleton_trees) > 0:
            skeleton_tree = self._skeleton_trees[0]
            # skeleton_tree 包含以下信息：
            # - num_nodes: 骨骼节点总数
            # - parent: 每个节点的父节点索引
            # - names: 每个节点的名称

        # ────────────────────────────────────────────────────────────────────
        # 第3步：初始化运动ID和时间跟踪
        # ────────────────────────────────────────────────────────────────────
        
        # motion_ids：标识每个环境当前播放的是哪个动作
        # 形状 (num_envs,)，值为 0 到 num_unique_motions-1
        self._motion_ids = torch.arange(self._num_envs).to(self._device)
        
        # motion_start_times：每个环境的动作开始时间
        # 用于时间戳计算：
        #   动作库查询时间 = episode_time + motion_start_time
        # 
        # 例如：
        # - motion_start_time = 0: 从动作的0秒开始播放
        # - motion_start_time = 5: 从动作的5秒处开始播放（用于多样化）
        self._motion_start_times = torch.zeros(
            self._num_envs, dtype=torch.float32, device=self._device, requires_grad=False
        )

        # ────────────────────────────────────────────────────────────────────
        # 第4步：加载动作数据
        # ────────────────────────────────────────────────────────────────────
        
        # 从动作库文件中加载具体的动作clips
        # 这个操作会：
        # 1. 读取.h5/.bvh文件中的mocap数据
        # 2. 为每个环境分配一个动作
        # 3. 缓存数据以便快速查询
        
        self.load_motions(random_sample=random_sample, start_idx=0)

    # ════════════════════════════════════════════════════════════════════════
    # 属性访问接口
    # ════════════════════════════════════════════════════════════════════════

    @property
    def motion_lib(self):
        """
        返回MotionLibG1实例
        
        可以直接访问动作库的各种属性和方法
        """
        return self._motion_lib

    @property
    def motion_start_times(self):
        """
        获取动作开始时间张量
        
        返回 (num_envs,) 张量
        """
        return self._motion_start_times

    @property
    def motion_len(self):
        """
        获取加载的动作总长度
        
        返回 (num_envs,) 张量
        每个值表示该环境的动作持续多少秒
        
        用于检查：
        - 是否达到了动作的末尾
        - 需要循环播放还是重新加载新动作
        """
        return self._motion_len

    @property
    def num_unique_motions(self):
        """
        获取动作库中的独立动作数量
        
        返回整数，例如1000表示库中有1000个不同的动作片段
        """
        return self._motion_lib._num_unique_motions

    @property
    def body_extended_names(self) -> list[str]:
        """
        获取扩展的身体部位名称列表
        
        返回形如 ['pelvis', 'spine', 'chest', 'neck', 'head', 
                  'left_shoulder', 'left_elbow', 'left_hand',
                  'right_shoulder', 'right_elbow', 'right_hand',
                  'left_hip', 'left_knee', 'left_foot',
                  'right_hip', 'right_knee', 'right_foot', ...]
        
        这些名称对应于mocap数据中的body顺序
        在进行body重新排列时会用到
        """
        return self._motion_lib.mesh_parsers.model_names

    def _log_mapping(self, body_names: list[str] | None, base_index: dict[str, int] | None,
                     joint_names: list[str] | None):
        lines: list[str] = []
        if body_names is not None and base_index is not None:
            lines.append("[RefMotion] body mapping (sim_index: name <= ref_index)")
            for sim_idx, name in enumerate(body_names):
                ref_idx = base_index.get(name, None)
                if ref_idx is None:
                    lines.append(f"  {sim_idx:02d}: {name} <= MISSING")
                else:
                    lines.append(f"  {sim_idx:02d}: {name} <= ref[{ref_idx}]")

        if joint_names is not None:
            ref_joint_names = getattr(self._skeleton_trees[0], "node_names", None)
            if ref_joint_names is not None:
                ref_index = {name: idx for idx, name in enumerate(ref_joint_names)}
                lines.append("[RefMotion] joint mapping (sim_index: name <= ref_index)")
                for sim_idx, name in enumerate(joint_names):
                    ref_idx = ref_index.get(name, None)
                    if ref_idx is None:
                        lines.append(f"  {sim_idx:02d}: {name} <= MISSING")
                    else:
                        lines.append(f"  {sim_idx:02d}: {name} <= ref[{ref_idx}]")

        if lines:
            print("\n".join(lines))

    def get_motion_num_steps(self):
        """
        获取已加载动作的步数
        
        返回 (num_envs,) 张量
        每个值 = 动作持续时间 / 时间步长
        
        例如：动作持续5秒，dt=0.01，则num_steps=500
        """
        return self._motion_lib.get_motion_num_steps()

    # ════════════════════════════════════════════════════════════════════════
    # 动作加载和重置
    # ════════════════════════════════════════════════════════════════════════

    def load_motions(self, random_sample: bool, start_idx: int):
        """
        从动作库中加载动作
        
        为num_envs个环境各分配一个动作片段
        
        参数说明：
        ───────────────────────────────────────────
        
        random_sample (bool):
            - True: 随机选择动作片段
            - False: 按顺序从start_idx开始选择
        
        start_idx (int):
            起始索引，当random_sample=False时有效
            例如start_idx=10表示从第10个动作开始
        
        内部过程：
        ──────────────
        1. 调用 _motion_lib.load_motions()
           传入：
           - skeleton_trees: 每个环境的骨骼树定义
           - gender_betas: SMPL模型参数（全零）
           - limb_weights: 身体部位权重（全零）
           - random_sample: 是否随机
           - start_idx: 起始索引
        
        2. 调用 _motion_lib.get_motion_length()
           获取每个环境中加载的动作的持续时间
           返回 (num_envs,) 张量，单位秒
           
        3. 调用 reset_motion_start_times()
           初始化motion_start_times为0或随机值
        """
        # 调用动作库的load_motions方法加载动作
        self._motion_lib.load_motions(
            skeleton_trees=self._skeleton_trees,
            gender_betas=[torch.zeros(17)] * self._num_envs,  # SMPL参数，对G1可能不用
            limb_weights=[np.zeros(10)] * self._num_envs,      # 身体部位权重
            random_sample=random_sample,
            start_idx=start_idx,
        )
        
        # 获取已加载动作的持续时间
        self._motion_len = self._motion_lib.get_motion_length(self._motion_ids)
        # 重置动作开始时间
        self.reset_motion_start_times(env_ids=self._motion_ids, sample=False)

    def reset_motion_start_times(self, env_ids: torch.Tensor, sample: bool):
        """
        重置指定环境的动作开始时间
        
        参数说明：
        ───────────────────────────────────────────
        
        env_ids (torch.Tensor):
            要重置的环境索引，形状 (num_envs,) 或 (subset,)
            例如：torch.tensor([0, 1, 2, 3, ...])
        
        sample (bool):
            - True: 从动作时间范围内随机采样
              这样不同环境会在动作的不同位置开始
              增加训练的多样性
              
            - False: 全部设为0
              所有环境都从动作的开头（0秒）开始播放
              
        示例：
        ──────
        reset_motion_start_times(env_ids=torch.tensor([5, 10, 15]), sample=True)
        
        会为环境5、10、15分别随机选择一个开始时间，
        例如分别是 2.3秒、3.5秒、1.8秒
        这样这三个环境会在不同的动作位置开始
        """
        if sample:
            # 从该动作的有效时间范围内随机采样
            # _motion_lib.sample_time() 返回 (num_sampled,) 的随机时间值
            self._motion_start_times[env_ids] = self._motion_lib.sample_time(
                self._motion_ids[env_ids]
            )
        else:
            # 设置为0，从动作开头开始
            self._motion_start_times[env_ids] = 0

    # ════════════════════════════════════════════════════════════════════════
    # 动作查询
    # ════════════════════════════════════════════════════════════════════════

    def episodes_exceed_motion_length(
        self, episode_times: torch.Tensor, env_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        检查是否到达了动作的末尾
        
        在每个环境的每个时步调用，判断是否需要重新加载新动作
        
        参数说明：
        ───────────────────────────────────────────
        
        episode_times (torch.Tensor):
            当前的回合时间，形状 (num_envs,)
            例如 [50, 51, 52, ...] 表示第50、51、52步
            
            实际的动作库查询时间 = episode_times * dt + motion_start_times
            例如：episode_times=50, dt=0.01, motion_start_times=0
            查询时间 = 50 * 0.01 + 0 = 0.5秒
        
        env_ids (torch.Tensor | None):
            指定要检查的环境，None表示全部
        
        返回值：
        ──────────────────────────────────────────
        torch.Tensor: (num_envs,) bool张量
        True表示该环境的时间已超过动作长度，应该重置
        False表示该环境仍在动作时间范围内
        
        公式：
        ──────
        result = (episode_times + motion_start_times) > motion_len
        
        例如：
        env 0: episode_times=500, motion_start_times=0, motion_len=400秒
               500 + 0 > 400 → True (应该重置)
        env 1: episode_times=350, motion_start_times=0, motion_len=400秒
               350 + 0 > 400 → False (继续)
        """
        if env_ids is None:
            # 检查全部环境
            return (episode_times + self._motion_start_times) > self._motion_len
        else:
            # 检查指定环境
            return (episode_times[env_ids] + self._motion_start_times[env_ids]) > self._motion_len[env_ids]

    # ════════════════════════════════════════════════════════════════════════
    # 核心查询函数：从动作库提取参考状态
    # ════════════════════════════════════════════════════════════════════════

    def get_state_from_motion_lib_cache(
        self,
        episode_length_buf: torch.Tensor,
        terrain_heights: torch.Tensor | None = None,
        offset: torch.Tensor | None = None,
        quaternion_is_xyzw=True,
        body_names: list[str] | None = None,
        body_names_extend: list[str] | None = None,
        joint_names: list[str] | None = None,
        log_mapping: bool = True,
    ) -> ReferenceMotionState:
        """
        从动作库缓存中查询参考动作状态
        
        这是最核心的函数！被called in training loop中的每一步。
        
        参数说明：
        ═══════════════════════════════════════════════════════════════════════
        
        episode_length_buf (torch.Tensor):
            当前回合已运行的时步数，形状 (num_envs,)
            例如 [10, 20, 30, ...] 表示第10、20、30步
            
            时间转换：
            动作库查询时间 = episode_length_buf * dt + motion_start_times
            例如：
            - episode_length_buf[0] = 50
            - dt = 0.01
            - motion_start_times[0] = 0.5
            - 查询时间 = 50 * 0.01 + 0.5 = 1.0秒
            
            这意味着查询第一个环境在动作的1.0秒处的状态
        
        terrain_heights (torch.Tensor | None):
            地形高度信息，形状 (num_envs, 1)
            用于调整参考动作的高度，使其与地形匹配
            
            例如：
            - ref_motion的root在z=1.0
            - terrain_heights[env] = 0.5
            - 则最终的root_pos[env, 2] = 1.0 + 0.5 = 1.5
            
            这允许同一个动作片段在不同高度的地形上重复
        
        offset (torch.Tensor | None):
            位置和旋转偏移，形状 (num_envs, 7)
            [x, y, z, qx, qy, qz, qw]
            
            用于平移/旋转参考动作到指定位置
            例如多个智能体启动时的位置不同
        
        quaternion_is_xyzw (bool):
            输出四元数的格式
            - True: xyzw格式（向量在前）
            - False: wxyz格式（标量在前）
            
            IsaacLab使用wxyz，通常设为False
            PHC使用xyzw，通常设为True
            需要与下游处理保持一致
        
        body_names (list[str] | None):
            指定要使用的body名称列表，顺序很重要！
            
            例如：
            body_names = ['pelvis', 'spine', 'chest', 'neck', 'head', 
                         'left_shoulder', 'left_elbow', 'left_hand',
                         'right_shoulder', 'right_elbow', 'right_hand',
                         'left_hip', 'left_knee', 'left_foot',
                         'right_hip', 'right_knee', 'right_foot']
            
            这list的顺序必须与IsaacLab中body_state.body_pos的顺序相同！
            否则reward计算会出错。
            
            如果为None，使用mocap数据中的全部body，不进行重新排列
        
        body_names_extend (list[str] | None):
            扩展body名称列表
            同样需要与IsaacLab的顺序相匹配
        
        返回值：
        ═══════════════════════════════════════════════════════════════════════
        
        ReferenceMotionState:
            包含重新排列后的参考动作数据：
            - root_pos, root_rot, root_lin_vel, root_ang_vel
            - body_pos, body_rot, body_lin_vel, body_ang_vel
              (已按body_names重新排列)
            - joint_pos, joint_vel
              (已按关节顺序重新排列)
            - 各项的形状与IsaacLab中的body_state相同
        
        关键步骤：
        ═══════════════════════════════════════════════════════════════════════
        """
        
        # ────────────────────────────────────────────────────────────────────
        # 步骤1：计算动作库查询时间
        # ────────────────────────────────────────────────────────────────────
        
        # 时间计算公式：
        #   motion_time = episode_step * dt + motion_start_time
        # 
        # 例如：
        # episode_length_buf = [10, 20, 30] (当前步数)
        # dt = 0.01 (时间步长)
        # motion_start_times = [0.0, 1.0, 2.0] (各环境的开始时间)
        # 
        # 则 motion_times = [10*0.01+0.0, 20*0.01+1.0, 30*0.01+2.0]
        #                 = [0.1, 1.2, 2.3]
        
        motion_times = episode_length_buf * self._dt + self._motion_start_times

        # ────────────────────────────────────────────────────────────────────
        # 步骤2：从动作库中查询参考状态
        # ────────────────────────────────────────────────────────────────────
        
        # 调用MotionLibG1的get_motion_state方法
        # 传入参数：
        # - motion_ids: 各环境的动作ID
        # - motion_times: 各环境的查询时间
        # - offset: 可选的位置/旋转偏移
        #
        # 返回值是一个字典，包含：
        # 'rg_pos', 'rb_rot', 'body_vel', 'body_ang_vel',
        # 'root_pos', 'root_rot', 'root_vel', 'root_ang_vel',
        # 'dof_pos', 'dof_vel',
        # 'rg_pos_t', 'rg_rot_t', 'body_vel_t', 'body_ang_vel_t'
        
        motion_res = self._motion_lib.get_motion_state(
            self._motion_ids,  # (num_envs,) 各环境的动作ID
            motion_times,      # (num_envs,) 各环境的查询时间
            offset=offset      # (num_envs, 7) 可选的偏移
        )
        

        # ────────────────────────────────────────────────────────────────────
        # 步骤3：处理地形高度调整
        # ────────────────────────────────────────────────────────────────────
        
        # 当智能体在不同高度的地形上运动时，需要调整参考动作的高度
        # 这样使用同一个动作片段就能适配不同方向的地形
        
        if terrain_heights is not None:
            # 复制地形高度，避免修改原始张量
            delta_height = terrain_heights.clone()
            
            # 如果有偏移，需要考虑偏移中的z分量
            if offset is not None:
                # offset[:, 2] 是z方向的偏移
                # unsqueeze(1) 改变形状以匹配delta_height
                delta_height -= offset[:, 2].unsqueeze(1)
            
            # 调整root位置的z坐标
            # root_pos[:, 2] 是z坐标（高度）
            motion_res["root_pos"][:, 2] += delta_height.flatten()
            
            # 调整所有body位置的z坐标
            if "rg_pos" in motion_res:
                # rg_pos shape: (num_envs, num_bodies, 3)
                # delta_height shape after broadcast: (num_envs, 1)
                motion_res["rg_pos"][:, :, 2] += delta_height
            
            # 调整扩展body位置的z坐标
            if "rg_pos_t" in motion_res:
                motion_res["rg_pos_t"][:, :, 2] += delta_height

        # ────────────────────────────────────────────────────────────────────
        # 步骤4：重新排列body顺序以匹配IsaacLab
        # ────────────────────────────────────────────────────────────────────
        
        # 这是关键步骤！确保body顺序与reward计算相匹配
        
        if body_names is not None:
            # 获取mocap中所有的body名称
            full_names = self.body_extended_names
            # 处理可能的_remove_idx（某些body被移除的情况）
            remove_idx = getattr(self._motion_lib.mesh_parsers, "_remove_idx", 0)
            if remove_idx > 0 and len(full_names) >= remove_idx:
                # 如果有body被移除，从末尾截断
                base_names = full_names[:-remove_idx]
            else:
                base_names = full_names
            
            # 构建名称→索引的映射字典
            # 这样可以快速查找每个body在mocap数据中的对应索引
            base_index = {name: idx for idx, name in enumerate(base_names)}
            # 检查所有请求的body名称是否都在mocap中存在
            missing = [name for name in body_names if name not in base_index]
            if missing:
                raise ValueError(f"Missing body names in reference motion: {missing}")
            
            # 根据body_names的顺序，生成对应的索引列表
            # 例如：
            # body_names = ['pelvis', 'chest', 'neck']
            # base_index = {'pelvis': 0, 'spine': 1, 'chest': 2, 'neck': 3, ...}
            # 则 body_ids = [0, 2, 3]
            body_ids = [base_index[name] for name in body_names]
            # 使用计算出的body_ids重新排列所有body相关的数据
            for key in ("rg_pos", "rb_rot", "body_vel", "body_ang_vel"):
                if key in motion_res:
                    # 沿第二维（body维）按body_ids重新排列
                    motion_res[key] = motion_res[key][:, body_ids, ...]

            if log_mapping:
                self._log_mapping(body_names=body_names, base_index=base_index, joint_names=joint_names)

        if body_names is None and log_mapping and joint_names is not None:
            self._log_mapping(body_names=None, base_index=None, joint_names=joint_names)

        # ────────────────────────────────────────────────────────────────────
        # 步骤5：处理扩展body名称
        # ────────────────────────────────────────────────────────────────────
        
        if body_names_extend is not None and "rg_pos_t" in motion_res:
            full_names = self.body_extended_names
            full_index = {name: idx for idx, name in enumerate(full_names)}
            missing = [name for name in body_names_extend if name not in full_index]
            if missing:
                raise ValueError(f"Missing extended body names in reference motion: {missing}")
            
            body_ids_extend = [full_index[name] for name in body_names_extend]
            # 重新排列扩展body数据
            for key in ("rg_pos_t", "rg_rot_t", "body_vel_t", "body_ang_vel_t"):
                if key in motion_res:
                    motion_res[key] = motion_res[key][:, body_ids_extend, ...]

        # ────────────────────────────────────────────────────────────────────
        # 步骤6：保持root信息与body顺序一致
        # ────────────────────────────────────────────────────────────────────
        
        # Root通常代表身体的第一个部位（骨盆）
        # 需要确保重排后，root信息与第一个body一致
        
        if "rg_pos" in motion_res:
            # 将root_pos设置为第一个body（重排后的）的位置
            motion_res["root_pos"] = motion_res["rg_pos"][..., 0, :].clone()
        
        if "rb_rot" in motion_res:
            motion_res["root_rot"] = motion_res["rb_rot"][..., 0, :].clone()
        
        if "body_vel" in motion_res:
            motion_res["root_vel"] = motion_res["body_vel"][..., 0, :].clone()
        
        if "body_ang_vel" in motion_res:
            motion_res["root_ang_vel"] = motion_res["body_ang_vel"][..., 0, :].clone()

        # ────────────────────────────────────────────────────────────────────
        # 步骤7：处理四元数格式转换
        # ────────────────────────────────────────────────────────────────────
        
        # PHC库使用xyzw格式（向量在前）
        # IsaacLab使用wxyz格式（标量在前）
        # 需要根据quaternion_is_xyzw参数进行转换
        
        if quaternion_is_xyzw:
            # 需要从wxyz转换为xyzw
            # 转换规律：[w, x, y, z] → [x, y, z, w]
            # 在PyTorch中用 [..., [3, 0, 1, 2]] 进行重排列
            for key, value in motion_res.items():
                # 检查是否是四元数（最后一维为4）
                if value.shape[-1] == 4:
                    # 重新排列：w, x, y, z → x, y, z, w
                    # 对应索引：0, 1, 2, 3 → 1, 2, 3, 0
                    # 用 [..., [3, 0, 1, 2]] 表示为 [..., [w_idx, x_idx, y_idx, z_idx]]
                    # 不对，应该是 [..., [x_idx, y_idx, z_idx, w_idx]]
                    # wxyz的索引是 [0(w), 1(x), 2(y), 3(z)]
                    # xyzw的索引应该是 [1(x), 2(y), 3(z), 0(w)]
                    motion_res[key] = value[..., [1, 2, 3, 0]]

        # ────────────────────────────────────────────────────────────────────
        # 步骤8：创建ReferenceMotionState对象
        # ────────────────────────────────────────────────────────────────────
        
        # 使用重排和调整后的motion_res数据创建ReferenceMotionState
        # ReferenceMotionState会进一步按body_ids/joint_ids重新排列
        ref_motion_state = ReferenceMotionState(motion_res)
        
        return ref_motion_state
