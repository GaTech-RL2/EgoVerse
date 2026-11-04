"""Yam station: GELLO demo collection and rollout of EgoVerse checkpoints
(dashboard, IK, cameras, episode/video capture).

    python -m egomimic.robot.yam_station.collect_gello --config egomimic/hydra_configs/robot/yam_rl2_gello_collect.yaml [--check-config|--list-ports|--check-devices]
    python -m egomimic.robot.yam_station.calibrate_gello --config <same profile> --output <calibration.yaml>
    YAM_EGOVERSE_CHECKPOINT=/abs/model.ckpt python -m egomimic.robot.yam_station.rollout --config egomimic/hydra_configs/robot/yam_rl2_egoverse_rollout.yaml [--check-model]

The Yam SDK is not a project dependency (i2rt pins its own numpy). Install it
into the station environment:

    uv pip install pyrealsense2 aiohttp dynamixel-sdk==4.0.5 \
        "i2rt @ git+https://github.com/rohan-bansal/i2rt.git@2b851d7f4511b6bda442db85e98ec1122835221b"
    uv pip install pytracik==0.0.3  # optional TRAC-IK endpoint solver
"""
