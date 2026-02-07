# Implement-Unitree-G1-in-HOVER
This repo records what changes have I done in the original HOVER repo.
Each file can be find through the path in this repo.

## G1 -> HOVER Workflow And Checklist

> Goal: deploy G1 into HOVER and debug failures in teacher training.

### 1) Prepare G1 Source Files

Folder layout (example):

```
g1/
	g1_29dof_anneal_23dof_fitmotionONLY.xml   # removed 6 wrist dof
	g1_29dof_anneal_23dof.xml                 # full joints
	g1_29dof_anneal_23dof.usd
	g1_29dof_anneal_23dof.urdf
	g1_29dof_anneal_23dof.yaml
	meshes/
```

Checklist:
- [ ] Verify 29-dof XML removes the 6 wrist dof.
- [ ] Verify full XML matches official G1 joint order.
- [ ] Verify USD uses the full G1 joints.
- [ ] Verify URDF and meshes are consistent.

### 2) Prepare AMASS And SMPL

Checklist:
- [ ] AMASS data downloaded and accessible.
- [ ] SMPL assets available for fitting.

### 3) Fit Shape And Motions

Process:
1. Use `g1_29dof_anneal_23dof_fitmotionONLY.xml` to fit shape.
2. Fit all AMASS motions (no filtering, full ~14k).
3. Export combined motion file: `amass_all_g1.pkl`.

Checklist:
- [ ] Shape fit completed without errors.
- [ ] All AMASS motions fit and exported.
- [ ] `amass_all_g1.pkl` exists and loads.

### 4) HOVER Adaptation Steps

#### 4.1 Train Entry Point

Change: if CLI flag `--robot g1`, use G1 config.

Checklist:
- [ ] `train_teacher_policy` routes `--robot g1` to `NeuralWBCEnvCfgG1`.

#### 4.2 G1 Env Config

File: `neural_wbc/isaac_lab_wrapper/neural_wbc/isaac_lab_wrapper/neural_wbc_env_cfg_g1.py`

Changes:
- Spaces:
	- `action_space = 23`
	- `observation_space = 1073`
	- `state_space = 1162`
- Updated fields:
	- `actuators`
	- `body_names`
	- `joint_names`
	- `lower_body_joint_ids` / `upper_body_joint_ids`
	- `feet_name`
	- `extend_body_pos`
	- `tracked_body_names`
	- `stiffness`
	- `damping`
	- `undesired_contact_body_names`
	- `ref_motion_path`
	- `ref_motion_skeleton_path`

Checklist:
- [ ] All names match G1 joint/body order.
- [ ] Sizes match G1 dof counts.
- [ ] Ref motion paths point to G1 files.

#### 4.3 G1 Asset Config

File: `IsaacLab/source/isaaclab_assets/robots/unitree.py` (G1_CFG)

Changes:
- Use full `g1_29dof_anneal_23dof.usd`.
- Adjust `init_state` per ASAP yaml.
- Actuators replaced in 4.2.

Checklist:
- [ ] USD path points to full G1 asset.
- [ ] `init_state` matches ASAP yaml.

#### 4.4 Reference Motion Switch

File: `neural_wbc/core/neural_wbc/core/reference_motion.py`

Change:
- Use `MotionLibG1` instead of H1.

Checklist:
- [ ] G1 motion lib is selected for G1 runs.

#### 4.5 Motion Lib For G1

File: `third_party/human2humanoid/phc/phc/utils/motion_lib_g1.py`

Changes:
- Read `mjcf_file` from `g1_29dof_anneal_23dof_fitmotionONLY.xml`.
- Fix AMASS data loading: nested motion-name key before frames.

Checklist:
- [ ] MJCF file points to 29-dof fitmotion XML.
- [ ] AMASS loader reads nested keys correctly.

#### 4.6 Human PID Batch

File: `third_party/human2humanoid/phc/phc/utils/torch_g1_humanoid_batch.py`

Changes:
- Use `g1_29dof_anneal_23dof_fitmotionONLY.xml`.
- Update `extend_hand` parent indices to 19, 26 (elbows).
- Set `extend_hand` length to 0.25m (from g1.yaml).
- Update `extend_head` similarly (g1.yaml).

Checklist:
- [ ] Parent indices match elbow links in XML order.
- [ ] Extend lengths match g1.yaml.

#### 4.7 Reward Config

File: `neural_wbc/isaac_lab_wrapper/neural_wbc/isaac_lab_wrapper/rewards/reward_cfg.py`

Changes:
- Resize `torque_limits`, `joint_pos_limits`, `joint_vel_limits` to G1 size.
- `max_contact_force = 200` (H1 was 500).

Checklist:
- [ ] Limits vector length matches G1 dof.
- [ ] Contact force limit set to 200.

#### 4.8 Reward Computation Split

File: `neural_wbc/isaac_lab_wrapper/neural_wbc/isaac_lab_wrapper/rewards/rewards_g1.py`

Change:
- Upper/lower split index set to 13 (pelvis included in body list).

Checklist:
- [ ] Split index matches body list ordering.

#### 4.9 Body ID Extension Logic

File: `neural_wbc/isaac_lab_wrapper/neural_wbc/isaac_lab_wrapper/neural_wbc_env.py`

Change:
- Modify `self._body_ids_extend` to add hand/head ids.
- Exact ID logic still unclear.

Checklist:
- [ ] Extended body ids match actual body indices in G1 asset.
- [ ] `extend_body_parent_ids` maps to correct parents.

#### 4.10 Debug Prints

Files:
- `neural_wbc/isaac_lab_wrapper/neural_wbc/isaac_lab_wrapper/neural_wbc_env.py`
- `neural_wbc/isaac_lab_wrapper/neural_wbc/isaac_lab_wrapper/rewards/rewards_g1.py`
- `neural_wbc/core/neural_wbc/core/reference_motion.py`

Change:
- Added debug output for joint info.

Checklist:
- [ ] Debug output shows correct joint/body order.
- [ ] Debug output does not flood training logs.

### 5) Current Status

- Teacher training for G1 still fails (root cause unknown).

### 6) Triage Checklist (Next Debug Pass)

- [ ] Confirm G1 joint order consistency across XML, USD, URDF.
- [ ] Verify `body_names` and `joint_names` align with asset order.
- [ ] Validate `self._body_ids_extend` mapping with actual body ids.
- [ ] Check reference motion skeleton matches G1 joint count.
- [ ] Ensure reward limits sizes match G1 dof.
- [ ] Run a single env step with debug prints and inspect diffs.

