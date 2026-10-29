"""Yam rollout station (dashboard, IK, cameras, episode/video capture) for
EgoVerse checkpoints; see ``hydra_configs/robot/yam_rl2_egoverse_rollout.yaml``.

The Yam SDK is not a project dependency (i2rt pins its own numpy). Install it
into the station environment:

    uv pip install pyrealsense2 aiohttp pure-python-adb dynamixel-sdk==4.0.5 \
        "i2rt @ git+https://github.com/rohan-bansal/i2rt.git@2b851d7f4511b6bda442db85e98ec1122835221b"
    uv pip install pytracik==0.0.3  # optional TRAC-IK endpoint solver
"""
