# RBY1 pipeline on Egoverse

By Zhenyang Chen Feb, 2026.

## Data Processing

1. Change the `raw_path` to the combined HDF5 file.

`python egomimic/rldb/scripts/robomimic_hd5.py --name RBY1_test_2 --dataset-repo-id RBY1_test_0223_2 --config-path ./egomimic/rldb/configs/RBY1_HDF5_config.json --output-dir ./datasets --fps 10 --ignore_episode_keys --robot-type rby1 --raw-path /coc/flash7/zhenyang/teleop_1771887047.hdf5`
or if we have several HDF5 under the same folder:
`python egomimic/rldb/scripts/robomimic_hd5.py --name RBY1_0303_smoothed --raw-path /coc/flash7/czhang883/smoothed --dataset-repo-id RBY1_0303_smoothed --config-path ./egomimic/rldb/configs/RBY1_HDF5_config_0309.json --output-dir ./datasets/0309/smoothed --fps 10 --ignore_episode_keys --robot-type rby1`

1. (optional) Egoengine Data Processing

Postprocess data and get the right hand and right arm data only.
Options: currently mapping right hand. And with `--black-image` it will create a black obs.  
`python egomimic/scripts/egoengine_process/egoengine_lerobot_extract_arm_hand.py datasets/0309/RBY1_0309`

Visualize and sanity check the data
`python egomimic/scripts/egoengine_process/visualize_lerobot_dataset.py /coc/flash7/zhenyang/EgoVerse/datasets/RBY1_human_data_0401_v8_egoengine/LeRobot -k actions.joint_arm --dims 0:14 -e 0`

## Training setup

1. Set cache path: `export TMPDIR=/tmp`
2. Run `python egomimic/trainHydra.py model=hpt_bc_flow_rby1`

## Config Setup Verbose

1. `egomimic/rldb/configs/RBY1_HDF5_config.json`
2. delta_timestampes to stack action chunk: note that this is relevant to control frequency of the dataset. `egomimic/hydra_configs/data/test_RBY1.yaml`
  For `proprio`, need to have a `state`_ as appendix to match the robomimic format. You will see `_robomimic_to_hpt_data`
3. Main changes: `test_RBY1.yaml` and `hpt_bc_flow_rby1.yaml`

TODO:

1. Provide offline eval (camera transform and the IK setup)
2. Config multi GPU training

## Utils - Dataset Visualization

1. [don't use this] lerobot-dataset-visualizer (only for HTTP + video format. Not useful for us):
  - Setup: [https://github.com/huggingface/lerobot-dataset-visualizer](https://github.com/huggingface/lerobot-dataset-visualizer)
    - On sky2: conda activate lerobot_viz && export BUN_INSTALL="$HOME/.bun" && export PATH="$BUN_INSTALL/bin:$PATH" && bun dev
2. `uv run external/lerobot/lerobot/scripts/visualize_dataset.py     --repo-id RBY1_test_0227     --root /coc/flash7/zhenyang/EgoVerse/datasets/RBY1_test_0227/     --local-files-only 1   --ws-port 9087  --episode-index 0 --mode distant --save 1 --output-dir ./logs/dataset_vis/0228`
3. Visualization with rerun:
  - (optional) If streaming from remote and visualize on local: run `ssh -L 9087:localhost:9087 sky2` where 9087 is the ws_port, sky2 is the server streaming the data.
    - Then launch rerun: `rerun ws://localhost:9087`

## Offline eval

This is supposed to run on your local computer for now.

1. Serve the policy

`python egomimic/scripts/serve_policy.py --checkpoint /home/droid_robot/zhenyang/EgoVerse/checkpoints/RBY1_0309_image_arm_hand.ckpt`
2. Plot and verify
To plot and verify the data, you can download a LeRobot dataset, the script will load the dataset and generate predictions to compare with the GT data.
`python egomimic/scripts/test_serve_policy_client.py --episode-idx 0 --max-steps 30  --dataset-folder ~/zhenyang/dataset/RBY0309 --trajectory`
3. TODO: test with rollout_sim in SEW_teleop

## Policy Training Note:

1. Smooth out action
2. Check act frequency and delta timestamped in the LeRobot dataset.
3. Default to batch 32, testing action seq=20/40

