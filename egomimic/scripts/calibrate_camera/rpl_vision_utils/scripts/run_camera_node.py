import argparse
import json
import os
import struct
import time

import cv2
import init_path
import numpy as np
import redis
from easydict import EasyDict
from rpl_vision_utils.networking.camera_redis_interface import CameraRedisPubInterface


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-type", type=str, choices=["k4a", "rs"])
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--camera-serial", type=str, default="")

    # 207222071445
    # 207222072020

    parser.add_argument("--eval", action="store_true")

    parser.add_argument("--no-color", action="store_true")

    parser.add_argument("--no-depth", action="store_true")

    parser.add_argument("--visualization", action="store_true")

    args = parser.parse_args()

    camera_id = args.camera_id
    camera_serial = args.camera_serial

    host = "172.16.0.2"
    port = 6379
    camera2redis_pub_interface = CameraRedisPubInterface(
        redis_host=host, redis_port=port, camera_id=camera_id
    )
    # Check redis if the camera id is occupied or not.
    camera_interface = None

    node_config = EasyDict(use_color=True, use_depth=True)
    if args.no_color:
        node_config.use_color = False

    if args.no_depth:
        node_config.use_depth = False

    if args.camera_type == "k4a":
        from rpl_vision_utils.k4a.k4a_interface import K4aInterface

        camera_interface = K4aInterface(device_id=camera_id)
        assert camera_interface is not None
    elif args.camera_type == "rs":
        from rpl_vision_utils.rs.rs_interface import RSInterface

        import pyrealsense2 as rs

        color_cfg = EasyDict(
            enabled=node_config.use_color,
            img_w=640,
            img_h=480,
            img_format=rs.format.bgr8,
            fps=30,
        )
        # color_cfg = EasyDict(
        #     enabled=node_config.use_color, img_w=1920, img_h=1080, img_format=rs.format.bgr8, fps=30
        # )

        depth_cfg = EasyDict(
            enabled=node_config.use_depth,
            img_w=640,
            img_h=480,
            img_format=rs.format.z16,
            fps=30,
        )

        pc_cfg = EasyDict(enabled=False)
        camera_interface = RSInterface(
            device_id=camera_id,
            color_cfg=color_cfg,
            depth_cfg=depth_cfg,
            pc_cfg=pc_cfg,
            serial_number=camera_serial,
        )

    else:
        raise NotImplementedError
    camera_interface.start()
    # NOTE: needed to add this sleep when we added depth filtering
    print("Sleeping to give time for camera to start up...")
    time.sleep(2)
    print("Starting")
    t = time.time()
    save_dir = f"/tmp/{camera2redis_pub_interface.camera_name}_{t}"
    file_ext = "jpg"
    camera_num = 0
    os.makedirs(save_dir)

    COUNT_THRESH = 5
    counter = COUNT_THRESH

    img_counter = 0
    freq = 50.0
    while True:
        start_time = time.time_ns()

        capture = camera_interface.get_last_obs()

        if capture is None:
            continue

        # t = time.time_ns()
        if capture is None:
            continue

        save_img = camera2redis_pub_interface.get_save_img_info()

        if save_img:
            counter = 0
        else:
            counter += 1
        img_info = {}
        imgs = {}
        img_info["time"] = t
        img_info["camera_type"] = args.camera_type
        img_info["intrinsics"] = {}
        if node_config.use_color:
            img_color = capture["color"]
            color_img_name = f"{save_dir}/color_{img_counter:09d}.{file_ext}"
            img_info["color"] = color_img_name
            img_info["intrinsics"]["color"] = camera_interface.get_color_intrinsics(
                mode="dict"
            )
            intrinsics_matrix = camera_interface.get_color_intrinsics(mode="matrix")
            imgs["color"] = img_color

        if node_config.use_depth:
            img_depth = capture["depth"]
            depth_img_name = f"{save_dir}/depth_{img_counter:09d}.{file_ext}"
            img_info["depth"] = depth_img_name
            img_info["intrinsics"]["depth"] = camera_interface.get_depth_intrinsics(
                mode="dict"
            )
            imgs["depth"] = img_depth
            if "unaligned_depth" in capture:
                imgs["unaligned_depth"] = capture["unaligned_depth"]

        camera2redis_pub_interface.set_img_info(img_info)
        camera2redis_pub_interface.set_img_buffer(imgs=imgs)

        if not args.eval:
            if counter < COUNT_THRESH:
                # Save img to tmp file
                if node_config.use_color:
                    cv2.imwrite(color_img_name, imgs["color"])
                if node_config.use_depth:
                    cv2.imwrite(depth_img_name, imgs["depth"])
                img_counter += 1

        if args.visualization:
            cv2.imshow("", imgs["color"])
            cv2.waitKey(10)
        end_time = time.time_ns()

        time_interval = (end_time - start_time) / (10**9)
        if time_interval < 1.0 / freq:
            time.sleep(1.0 / freq - time_interval)

        if camera2redis_pub_interface.finished:
            break


if __name__ == "__main__":
    main()
