#!/bin/bash
#SBATCH -J egoverse_RBY1         # Job name
#SBATCH --partition=overcap
#SBATCH --nodes=1   # NOTE: try to use GPUs on the same node to reduce communication overhead
#SBATCH --gres=gpu:a40:1               # Request how many H100 GPUs per node
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-gpu=32G               # RAM per GPU Memory per core
#SBATCH -t 24:00:00                     # Duration of the job
#SBATCH --requeue
#SBATCH -o ./logs/slurm/%A/task_%a/out.log
#SBATCH -e ./logs/slurm/%A/task_%a/err.log
#SBATCH --exclude='puma,deebot,qt-1'

# xaea-12,dendrite,synapse,uniblab,

cd /coc/flash7/zhenyang/EgoVerse

source emimic/bin/activate

export TMPDIR=/tmp

# After cd and source activate, before python
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

# python egomimic/trainHydra.py model=hpt_bc_flow_rby1 name=RBY1_test_resume_0301_2 description=resume_last_ckpt

# python egomimic/trainHydra.py model=hpt_bc_flow_eva name=EgoWM_test

# python egomimic/trainHydra.py model=hpt_bc_flow_rby1 name=RBY1_test_0305_lowdim description=debug


################## MOBILE: PUSH CART experiments ######
DATASET=RBY1_human_data_0423_push_cart
CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 python egomimic/trainHydra.py \
        --config-name=train_rby1_sew_human_mobie_img_act32_proprio_dropout \
        name=${DATASET} \
        data.train_datasets.dataset1.datasets.rl2_lab.folder_path=/coc/flash7/zhenyang/EgoVerse/datasets/${DATASET} \
        data.valid_datasets.dataset1.datasets.eth_lab.folder_path=/coc/flash7/zhenyang/EgoVerse/datasets/${DATASET}

################## SEW experiments ######
# DATASET=RBY1_human_data_0416_pickupbox
# DATASET=RBY1_human_data_0417_pickupbox
# DATASET=RBY1_0418_rotatebox_human_data
# CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 python egomimic/trainHydra.py \
#         --config-name=train_rby1_sew_human_no_mobie_img_act32 \
#         name=${DATASET} \
#         data.train_datasets.dataset1.datasets.rl2_lab.folder_path=/coc/flash7/zhenyang/EgoVerse/datasets/${DATASET} \
#         data.valid_datasets.dataset1.datasets.eth_lab.folder_path=/coc/flash7/zhenyang/EgoVerse/datasets/${DATASET}
# DATASET=RBY1_0418_rotatebox_human_data
# CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 python egomimic/trainHydra.py \
#         --config-name=train_rby1_sew_human_no_mobie_img_act32_proprio_dropout \
#         name=${DATASET}_dropout_test \
#         data.train_datasets.dataset1.datasets.rl2_lab.folder_path=/coc/flash7/zhenyang/EgoVerse/datasets/${DATASET} \
#         data.valid_datasets.dataset1.datasets.eth_lab.folder_path=/coc/flash7/zhenyang/EgoVerse/datasets/${DATASET}


################## EgoEngine experiments ######
# RBY1 0411Egoengine (Hydra: egomimic/hydra_configs/train_rby1_egoengine_eef_hand_img_act32.yaml).
# Data: LeRobot root with obs.images + eef/hand proprio and actions.right_arm_eef_hand (18-D concat).
# Build concat set: python egomimic/scripts/egoengine_process/egoengine_lerobot_combine_action.py <src> --output-path datasets/<name>_right_arm_eef_hand
# Uncomment exactly one training block in this file (Egoengine vs 0331 below).
# DATASET=RBY1_egoengine_mustard_cropped_right_arm_eef_hand
# DATASET=RBY1_egoengine_mustard_inpainted_right_arm_hand
# CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 python egomimic/trainHydra.py \
#         --config-name=train_rby1_egoengine_eef_hand_img_act32 \
#         name=${DATASET} \
#         data.train_datasets.dataset1.datasets.rl2_lab.folder_path=/coc/flash7/zhenyang/EgoVerse/datasets/${DATASET} \
#         data.valid_datasets.dataset1.datasets.eth_lab.folder_path=/coc/flash7/zhenyang/EgoVerse/datasets/${DATASET}

################## WORKING VERSION ######
# RBY1 0331 experiments (Hydra: egomimic/hydra_configs/train_rby1_0331_*.yaml).
# Composes train + model + data=rby1_0331_filtered_torso_actions + data_schematic.norm_mode=minmax.
# Uncomment exactly one run at a time (and comment out the Egoengine block above to avoid two jobs).
# DATASET=RBY1_human_data_0401_v8_egoengine

# (1) Image + robot0_joint_pos + hand_left_qpos + hand_right_qpos.
# # Action: right_hand + right_arm + torso (25-dim), 20-step chunk.
# CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 python egomimic/trainHydra.py \
#         --config-name=train_rby1_0331_01_right_hand_arm_torso_img_robot0_bihand \
#         data.train_datasets.dataset1.datasets.rl2_lab.folder_path=/coc/flash7/zhenyang/EgoVerse/datasets/${DATASET} \
#         data.valid_datasets.dataset1.datasets.eth_lab.folder_path=/coc/flash7/zhenyang/EgoVerse/datasets/${DATASET}
# #
# (2) Image + robot0_joint_pos + hand_left_qpos + hand_right_qpos.
# Action: left/right hand + left/right arm + torso (44-dim), 20-step chunk.
# CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 python egomimic/trainHydra.py \
#         --config-name=train_rby1_0331_02_hand_arm_torso_img_robot0_bihand \
#         data.train_datasets.dataset1.datasets.rl2_lab.folder_path=/coc/flash7/zhenyang/EgoVerse/datasets/${DATASET} \
#         data.valid_datasets.dataset1.datasets.eth_lab.folder_path=/coc/flash7/zhenyang/EgoVerse/datasets/${DATASET}
# 
##################
# RBY1 0321 experiments (Hydra: egomimic/hydra_configs/train_rby1_0320_*.yaml).
# Composes train + model + data=rby1_0320_filtered + data_schematic.norm_mode=minmax.
# Uncomment exactly one run at a time.
#
# (1) Image + right hand qpos + right arm proprio (model=hpt_bc_flow_rby1_egoengine)
# CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 python egomimic/trainHydra.py \
#         --config-name=train_rby1_0320_01_right_hand_img_hand_arm
# #
# (2) Low-dim only: hand + right arm, no image (model=rby1_0320_right_hand_lowdim_hand_arm)
# CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 python egomimic/trainHydra.py \
#         --config-name=train_rby1_0320_02_right_hand_lowdim_hand_arm
# #
# (3) Image-only policy, no proprio stems (model=rby1_0320_right_hand_img_only)
# CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 python egomimic/trainHydra.py \
#         --config-name=train_rby1_0320_03_right_hand_img_only

# (4) Bimanual arm+hand (38-dim), image + robot0 + hand_right, data 0320_filtered
# CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 python egomimic/trainHydra.py \
#         --config-name=train_rby1_0320_04_biarm_hand_act20_filtered

##################

##################
# Total 3 exp: high quality with right arm hand and biarm hand. All data with full body
# CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 python egomimic/trainHydra.py \
#         model=hpt_bc_flow_rby1_egoengine \
#         data=test_RBY1_0309 \
#         data.zero_obs=false \
#         data_schematic.norm_mode=minmax \
#         data.train_datasets.dataset1.datasets.rl2_lab.folder_path=/coc/flash7/zhenyang/EgoVerse/datasets/0320_filtered_high_quality \
#         data.valid_datasets.dataset1.datasets.eth_lab.folder_path=/coc/flash7/zhenyang/EgoVerse/datasets/0320_filtered_high_quality \
#         name=RBY1_0320_rightarm_hand_act20

# 0320 biarm hand
# obs: aria_image, robot0_joint_pos, hand_right_qpos
# action: actions_joint_arm_hand or right_arm_hand
# CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 python egomimic/trainHydra.py \
#         model=hpt_bc_flow_rby1_biarm_hand \
#         data=test_RBY1_0309 \
#         data.zero_obs=false \
#         data_schematic.norm_mode=minmax \
#         data.train_datasets.dataset1.datasets.rl2_lab.folder_path=/coc/flash7/zhenyang/EgoVerse/datasets/0320_filtered \
#         data.valid_datasets.dataset1.datasets.eth_lab.folder_path=/coc/flash7/zhenyang/EgoVerse/datasets/0320_filtered \
#         name=RBY1_0320_biarm_hand_act20_all



#######################################################################
# full body mobile!
# CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 python egomimic/trainHydra.py \
#         model=hpt_bc_flow_rby1_mobile \
#         data=test_RBY1_mobile \
#         data.zero_obs=true \
#         data_schematic.norm_mode=minmax \
#         name=RBY1_0316_mobile_obs2_act20

########################################################################
# img obs + right arm and hand 
# CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 python egomimic/trainHydra.py \
#         model=hpt_bc_flow_rby1_egoengine_act20_armhand \
#         data=test_RBY1_0309 \
#         data.zero_obs=false \
#         data_schematic.norm_mode=minmax \
#         name=RBY1_0309_egoengine_image_right_arm_hand_act20_obs2_batch32_minmax
# TODO: using 20/40 action horzion. Using minmax norm. Enable zero_obs.
# CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 python egomimic/trainHydra.py \
#         model=hpt_bc_flow_rby1_egoengine_act40 \
#         data=test_RBY1_0309_40act \
#         data_schematic.norm_mode=minmax \
#         name=RBY1_0309_egoengine_image_right_arm_black_image_act40_obs4_batch32_minmax

# CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 python egomimic/trainHydra.py \
#         model=hpt_bc_flow_rby1_egoengine_act20 \
#         data=test_RBY1_0309 \
#         data_schematic.norm_mode=minmax \
#         name=RBY1_0309_egoengine_image_right_arm_black_image_act20_obs8_batch32_minmax

# control group test for zero_obs. Keep batch2, act10, quantile norm, as right_arm_image exp
# CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 python egomimic/trainHydra.py \
#         model=hpt_bc_flow_rby1_egoengine_act10 \
#         data=test_RBY1_zero_obs \
#         name=RBY1_0309_egoengine_image_right_arm_zero_obs_test
########################################################################

# 20260311 Egoengine right arm hand
# CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 python egomimic/trainHydra.py \
#         model=hpt_bc_flow_rby1_lowdim_egoengine \
#         name=RBY1_0309_egoengine_lowdim_right_arm \
#         data=test_RBY1_0309 \
#         data.train_datasets.dataset1.datasets.rl2_lab.folder_path=/coc/flash7/zhenyang/EgoVerse/datasets/0309/smoothed \
#         data.valid_datasets.dataset1.datasets.eth_lab.folder_path=/coc/flash7/zhenyang/EgoVerse/datasets/0309/smoothed \
#         model.robomimic_model.ac_keys.rby1=actions_joint_right_arm \
#         model.robomimic_model.head_specs.rby1.infer_ac_dims.rby1=7 \
#         model.robomimic_model.head_specs.rby1.model.act_dim=7
# TODO: turn off data.zero_obs=true
# CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 python egomimic/trainHydra.py \
#         model=hpt_bc_flow_rby1_egoengine \
#         data=test_RBY1_0309 \
#         name=RBY1_0309_egoengine_image_right_arm_black_image_act_20 \
#         data.zero_obs=true \
#         data.train_datasets.dataset1.datasets.rl2_lab.folder_path=/coc/flash7/zhenyang/EgoVerse/datasets/0309/smoothed \
#         data.valid_datasets.dataset1.datasets.eth_lab.folder_path=/coc/flash7/zhenyang/EgoVerse/datasets/0309/smoothed \
#         model.robomimic_model.ac_keys.rby1=actions_joint_right_arm \
#         model.robomimic_model.head_specs.rby1.infer_ac_dims.rby1=7 \
#         model.robomimic_model.head_specs.rby1.model.act_dim=7 \
#         model.robomimic_model.head_specs.rby1.action_horizon=20 \
#         model.robomimic_model.head_specs.rby1.model.act_seq=20 \
#         model.robomimic_model.trunk.action_horizon=20

# python egomimic/trainHydra.py \
#         model=hpt_bc_flow_rby1_egoengine \
#         data=test_RBY1_0309 \
#         name=RBY1_test_0309_egoengine_image_arm_hand \
#         description=RBY1_test_0309_egoengine_image_arm_hand \
#         model.robomimic_model.ac_keys.rby1=actions_joint_right_arm_hand \
#         model.robomimic_model.head_specs.rby1.infer_ac_dims.rby1=19 \
#         model.robomimic_model.head_specs.rby1.model.act_dim=19
# python egomimic/trainHydra.py \
#         model=hpt_bc_flow_rby1_lowdim_egoengine \
#         data=test_RBY1_0309 \
#         name=RBY1_test_0309_lowdim_egoengine_image_arm_hand \
#         description=RBY1_test_0309_lowdim_egoengine_image_arm_hand \
#         model.robomimic_model.ac_keys.rby1=actions_joint_right_arm_hand \
#         model.robomimic_model.head_specs.rby1.infer_ac_dims.rby1=19 \
#         model.robomimic_model.head_specs.rby1.model.act_dim=19 \

########################################################################
# 20260309 RBY1_test_2
# image (use data=test_RBY1_0309 for 0309 format: actions.joint_arm_hand columns)
# python egomimic/trainHydra.py \
#         model=hpt_bc_flow_rby1 \
#         data=test_RBY1_0309 \
#         name=RBY1_test_0309_image_arm_hand \
#         description=RBY1_test_0309_image_arm \
#         model.robomimic_model.ac_keys.rby1=actions_joint_arm_hand \
#         model.robomimic_model.head_specs.rby1.infer_ac_dims.rby1=38 \
#         model.robomimic_model.head_specs.rby1.model.act_dim=38
# python egomimic/trainHydra.py \
#         model=hpt_bc_flow_rby1 \
#         data=test_RBY1_0309 \
#         name=RBY1_test_0309_image_arm \
#         description=RBY1_test_0309_image_arm \
#         model.robomimic_model.ac_keys.rby1=actions_joint_arm \
#         model.robomimic_model.head_specs.rby1.infer_ac_dims.rby1=14 \
#         model.robomimic_model.head_specs.rby1.model.act_dim=14

# lowdim (use data=test_RBY1_0309 for 0309 format)
# python egomimic/trainHydra.py \
#         model=hpt_bc_flow_rby1_lowdim \
#         data=test_RBY1_0309 \
#         name=RBY1_test_0309_lowdim_arm_hand \
#         description=RBY1_test_0309_lowdim_arm_hand \
#         model.robomimic_model.ac_keys.rby1=actions_joint_arm_hand \
#         model.robomimic_model.head_specs.rby1.infer_ac_dims.rby1=38 \
#         model.robomimic_model.head_specs.rby1.model.act_dim=38
# python egomimic/trainHydra.py \
#         model=hpt_bc_flow_rby1_lowdim    \
#         name=RBY1_test_0309_lowdim_arm \
#         data=test_RBY1_0309 \
#         description=RBY1_test_0309_lowdim_arm \
#         model.robomimic_model.ac_keys.rby1=actions_joint_arm \
#         model.robomimic_model.head_specs.rby1.infer_ac_dims.rby1=14 \
#         model.robomimic_model.head_specs.rby1.model.act_dim=14
########################################################################

# use the test_2 dataset
# Test1: image + proprio → bi arm action
# Test2:  proprio → bi arm action
# actions_biarm: 14-dim (vs actions 49-dim). Override schematic, delta_timestamps, and act_dim.
# python egomimic/trainHydra.py \
#         model=hpt_bc_flow_rby1 \
#         name=RBY1_test_biarm_action_image_0305 \s
#         description=RBY1_test_biarm_action_image_0305 \
#         data_schematic.schematic_dict.rby1.actions.lerobot_key=actions_biarm \
#         model.robomimic_model.head_specs.rby1.infer_ac_dims.rby1=14 \
#         model.robomimic_model.head_specs.rby1.model.act_dim=14

