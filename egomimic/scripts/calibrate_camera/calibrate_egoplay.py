import os

folder_path = os.path.join(os.path.dirname(__file__))

import numpy as np
import cv2
import argparse
import json
import h5py
from tqdm import tqdm

from egomimic.utils.egomimicUtils import (
    # WIDE_LENS_ROBOT_LEFT_K,
    # WIDE_LENS_ROBOT_LEFT_D,
    ARIA_INTRINSICS,
)

from scipy.spatial.transform import Rotation as Rot
import matplotlib.pyplot as plt
from rpl_vision_utils.utils.apriltag_detector import AprilTagDetector


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--h5py-path",
        type=str,
    )

    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--store-matrix", action="store_true")

    return parser.parse_args()


def store_matrix(path, R, t):
    file = h5py.File(path, "r+")

    for demo_name in file.keys():
        demo = file[demo_name]
        calib_matrix_group = demo.create_group("calibration_matrix")
        calib_matrix_group.create_dataset("rotation", data=R)
        calib_matrix_group.create_dataset("translation", data=t)

    print("Appended calibration matrix: ")
    print(R.round(3))
    print(t.round(3))
    print("==============================")


def main():
    args = parse_args()

    calib = h5py.File(args.h5py_path, "r+")

    april_detector = AprilTagDetector(quad_decimate=1.0)

    # TODO get intrinsics
    # with open(os.path.join(args.config_folder, f"camera_{args.camera_id}_{args.camera_type}.json"), "r") as f:
    #     intrinsics = json.load(f)
    # TODO: THESE ARE JUST TEMP VALUES
    intrinsics = ARIA_INTRINSICS
    intrinsics = {
        "color": {
            "fx": intrinsics[0, 0],
            "fy": intrinsics[1, 1],
            "cx": intrinsics[0, 2],
            "cy": intrinsics[1, 2],
        }
    }

    print(intrinsics)

    R_base2gripper_list = []
    t_base2gripper_list = []
    R_target2cam_list = []
    t_target2cam_list = []
    calib = calib["data"]
    count = 0
    for key in calib.keys():
        demo = calib[key]
        T, H, W, _ = demo["obs/front_img_1"].shape
        for t in tqdm(range(T)):
            img = demo["obs/front_img_1"][t]
            # img = cv2.undistort(
            #     img, WIDE_LENS_ROBOT_LEFT_K[:, :3], WIDE_LENS_ROBOT_LEFT_D
            # )

            detect_result = april_detector.detect(
                img,
                intrinsics=intrinsics["color"],
                # tag_size=0.0958)
                tag_size=0.17541875,
            )

            if len(detect_result) != 1:
                print(f"wrong detection, skipping img {t}")
                if args.debug:
                    plt.imsave(f"calibration_imgs/{t}_fail.png", img)

                continue

            bounding_box_corners = detect_result[0].corners
            # draw bounding box on img and save
            if args.debug:
                img = april_detector.vis_tag(img)
                plt.imsave(f"calibration_imgs/{t}_detection.png", img)

            count += 1
            pose = demo["obs/ee_pose_robot_frame"][t]
            assert pose.shape == (7,)
            pos = pose[0:3]
            rot = Rot.from_quat(pose[3:])

            R_base2gripper_list.append(rot.as_matrix().T)
            t_base2gripper_list.append(
                -rot.as_matrix().T @ np.array(pos)[:, np.newaxis]
            )

            R_target2cam_list.append(detect_result[0].pose_R)
            pose_t = detect_result[0].pose_t

            # if args.debug:
            #     print("Detected: ", pose_t, T.quat2axisangle(T.mat2quat(detect_result[0].pose_R)))

            t_target2cam_list.append(pose_t)

    print(f"==========Using {count} images================")

    for method in [
        cv2.CALIB_HAND_EYE_TSAI,
        cv2.CALIB_HAND_EYE_PARK,
        # cv2.CALIB_HAND_EYE_DANIILIDIS,
        # cv2.CALIB_HAND_EYE_ANDREFF,
        # cv2.CALIB_HAND_EYE_HORAUD
    ]:
        R, t = cv2.calibrateHandEye(
            R_base2gripper_list,
            t_base2gripper_list,
            R_target2cam_list,
            t_target2cam_list,
            method=method,
        )
        # print("Rotation matrix: ", R.round(3))
        # print("Axis Angle: ", T.quat2axisangle(T.mat2quat(R)))
        # print("Quaternion: ", T.mat2quat(R))
        # print("Translation: ", t.T.round(3))
        fullT = np.concatenate((R, t), axis=1)
        fullT = np.concatenate((fullT, np.array([[0, 0, 0, 1]])), axis=0)
        print("T: ", repr(fullT))

    print("==============================")

    if args.store_matrix:
        store_matrix(args.h5py_path, R, t.T)


if __name__ == "__main__":
    main()
