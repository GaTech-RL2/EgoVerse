"""Calibrate Intrinsics"""

import argparse
import json
import os
import pprint

import cv2
import gprs.utils.transform_utils as T
import init_path
import matplotlib.pyplot as plt
import numpy as np
from gprs import config_root
from gprs.camera_redis_interface import CameraRedisSubInterface
from gprs.franka_interface import FrankaInterface
from gprs.franka_interface.visualizer import PybulletVisualizer
from gprs.utils import load_yaml_config
from gprs.utils.input_utils import input2action
from gprs.utils.io_devices import SpaceMouse

# from rpl_vision_utils.k4a.k4a_interface import K4aInterface
from rpl_vision_utils.utils.apriltag_detector import AprilTagDetector
from rpl_vision_utils.utils.transform_manager import RPLTransformManager
from urdf_models.urdf_models import URDFModel

# folder_path = os.path.join(os.path.dirname(__file__))


def main():
    pp = pprint.PrettyPrinter(indent=4)
    # cr_interface = CameraRedisSubInterface(camera_id=2,
    #                                        use_color=True,
    #                                        use_depth=True)
    # cr_interface.start()

    # imgs = cr_interface.get_img()
    # img_info = cr_interface.get_img_info()

    images = []
    for idx in range(13):
        img = cv2.imread(f"aruco_calibration_imgs/{idx}_color.png")
        if img is None:
            print("img is none")
            break
        images.append(img)
    # import pdb; pdb.set_trace()
    workdir = "./workdir/"
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard_create(
        7,
        5,
        squareLength=0.2863,
        markerLength=0.2308,
        # 1,
        # .8,
        dictionary=aruco_dict,
    )
    imboard = board.draw((2000, 2000))

    def read_chessboards(images):
        """
        Charuco base pose estimation.
        """
        print("POSE ESTIMATION STARTS:")
        allCorners = []
        allIds = []
        decimator = 0
        # SUB PIXEL CORNER DETECTION CRITERION
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

        for im in images:
            print("=> Processing image {0}".format(im))
            # frame = cv2.imread(im)
            frame = im
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)

            if len(corners) > 0:
                # SUB PIXEL DETECTION
                for corner in corners:
                    cv2.cornerSubPix(
                        gray, corner, winSize=(3, 3), zeroZone=(-1, -1), criteria=criteria
                    )
                res2 = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
                if (
                    res2[1] is not None
                    and res2[2] is not None
                    and len(res2[1]) > 3
                    and decimator % 1 == 0
                ):
                    allCorners.append(res2[1])
                    allIds.append(res2[2])

            decimator += 1

        imsize = gray.shape
        return allCorners, allIds, imsize

    def calibrate_camera(allCorners, allIds, imsize):
        """
        Calibrates the camera using the dected corners.
        """
        print("CAMERA CALIBRATION")

        cameraMatrixInit = np.array(
            [[1000.0, 0.0, imsize[0] / 2.0], [0.0, 1000.0, imsize[1] / 2.0], [0.0, 0.0, 1.0]]
        )

        distCoeffsInit = np.zeros((5, 1))
        flags = (
            cv2.CALIB_USE_INTRINSIC_GUESS
            # + cv2.CALIB_RATIONAL_MODEL
            + cv2.CALIB_FIX_ASPECT_RATIO
        )
        # flags = (cv2.CALIB_RATIONAL_MODEL)
        (
            ret,
            camera_matrix,
            distortion_coefficients0,
            rotation_vectors,
            translation_vectors,
            stdDeviationsIntrinsics,
            stdDeviationsExtrinsics,
            perViewErrors,
        ) = cv2.aruco.calibrateCameraCharucoExtended(
            charucoCorners=allCorners,
            charucoIds=allIds,
            board=board,
            imageSize=imsize,
            cameraMatrix=cameraMatrixInit,
            distCoeffs=distCoeffsInit,
            flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9),
        )

        return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors

    allCorners, allIds, imsize = read_chessboards(images)

    ret, mtx, dist, rvecs, tvecs = calibrate_camera(allCorners, allIds, imsize)

    # for i in range(len(images)):
    #     # i=0# select image id
    #     plt.figure()
    #     frame = images[i]
    #     img_undist = cv2.undistort(frame,mtx,dist,None)
    #     plt.subplot(1,2,1)
    #     plt.imshow(frame)
    #     plt.title("Raw image")
    #     plt.axis("off")
    #     plt.subplot(1,2,2)
    #     plt.imshow(img_undist)
    #     plt.title("Corrected image")
    #     plt.axis("off")
    #     plt.show()
    # import pdb; pdb.set_trace()
    print(dist)
    print(mtx)


if __name__ == "__main__":
    main()
