# RBY1 pipeline on Egoverse

By Zhenyang Chen Feb, 2026.

## Data Processing
Change the `raw_path` to the combined HDF5 file.
`
python egomimic/rldb/scripts/robomimic_hd5.py --name RBY1_test --dataset-repo-id RBY1_test_0223 --config-path ./egomimic/rldb/configs/RBY1_HDF5_config.json --output-dir ./dataset --fps 10 --ignore_episode_keys --robot-type rby1 --raw-path /coc/flash7/zhenyang/teleop_1771887047.hdf5 
`
## Training setup
1. Set cache path: `export TMPDIR=/tmp`
2. Run `python egomimic/trainHydra.py model=hpt_bc_flow_rby1`

## Config Setup Verbose
1. `egomimic/rldb/configs/RBY1_HDF5_config.json`
2. delta_timestampes to stack action chunk: note that this is relevant to control frequency of the dataset. `egomimic/hydra_configs/data/test_RBY1.yaml`
 - For `proprio`, need to have a `state_` as appendix to match the robomimic format. You will see `_robomimic_to_hpt_data`
3. Main changes: `test_RBY1.yaml` and `hpt_bc_flow_rby1.yaml`

TODO:
1. Wandb log picture vis? should viz original pic -> check original config
2. Config multi GPU training