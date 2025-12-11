import time

import cv2
import numpy as np

from rpl_vision_utils.k4a.k4a_interface import K4aInterface
from rpl_vision_utils.utils.apriltag_detector import AprilTagDetector

k4a_interface = K4aInterface()

k4a_interface.start()

while True:
    capture = k4a_interface.get_last_obs()
    if capture is not None:
        break
    time.sleep(0.05)

image = capture["color"].astype(np.uint8)
intrinsics = k4a_interface.get_depth_intrinsics()

# Get AprilTag Detection

april_detector = AprilTagDetector()
april_detector.detect(image, intrinsics=intrinsics, tag_size=0.04255)

image = april_detector.vis_tag(image)

cv2.imshow("", image)
cv2.waitKey(0)

for detection in april_detector.results:
    print(detection)
    print("=================================")
