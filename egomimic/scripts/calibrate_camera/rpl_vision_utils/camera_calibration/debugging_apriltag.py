import cv2
import time
import numpy as np

import init_path
# from rpl_vision_utils.k4a.k4a_interface import K4aInterface
from rpl_vision_utils.rs.rs_interface import RSInterface
from rpl_vision_utils.utils.apriltag_detector import AprilTagDetector
import gprs.utils.transform_utils as T
from easydict import EasyDict

import pyrealsense2 as rs

camera_id = 0
serial_number = '207222071445'

# camera_id = 1
# serial_number = '207222072020'

color_cfg = EasyDict(enabled=True,
                     img_w=640,
                     img_h=480,
                     img_format=rs.format.bgr8,
                     fps=30)

depth_cfg = EasyDict(enabled=True,
                     img_w=640,
                     img_h=480,
                     img_format=rs.format.z16,
                     fps=30)

pc_cfg = EasyDict(enabled=False)
k4a_interface = RSInterface(device_id=camera_id,
                            color_cfg=color_cfg,
                            depth_cfg=depth_cfg,
                            pc_cfg=pc_cfg,
                            serial_number=serial_number)

k4a_interface.start()

while True:
    capture = k4a_interface.get_last_obs()
    if capture is None:
        continue

    image = capture["color"].astype(np.uint8)
    intrinsics = k4a_interface.get_color_intrinsics(mode='dict')
    print(intrinsics)

    # Get AprilTag Detection

    april_detector = AprilTagDetector()
    april_detector.detect(image,
                          intrinsics=intrinsics,
                          tag_size=0.039
                          # tag_size=0.06
                          # tag_size = 0.06
                          )

    image = april_detector.vis_tag(image)

    cv2.imshow("", image)
    cv2.waitKey(1)

    if len(april_detector.results) == 1:
        print("Translation: ", april_detector.results[0].pose_t)
        print("Rotation: ", april_detector.results[0].pose_R)
        # print("Rotation: ", T.mat2quat(april_detector.results[0].pose_R.transpose()))
        print("=========================================")
        time.sleep(0.05)

# for detection in april_detector.results:
#     print(detection)
#     print("=================================")
