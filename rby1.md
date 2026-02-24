Hello World.

By Zhenyang Chen Feb, 2026.

## Data Processing
/coc/flash7/zhenyang/teleop_1771887047.hdf5

`
python egomimic/rldb/scripts/robomimic_hd5.py --name RBY1_test --raw-path /coc/flash7/zhenyang/teleop_1771887047.hdf5 --dataset-repo-id RBY1_test_0223 --config-path ./egomimic/rldb/configs/RBY1_HDF5_config.json --output-dir ../data --fps 10 --ignore_episode_keys --robot_type rby1
`