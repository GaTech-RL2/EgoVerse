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

    parser.add_argument("--every-k", type=int, default=5, help="sample every k frames")

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

    def reproject_tag_corners(det, intrinsics_color, tag_size, distCoeffs=None):
        """
        Reproject detected AprilTag corners using tag->cam pose and intrinsics.
        Returns projected pixel coords (4,2) and per-corner residuals to detector corners.
        """
        fx, fy, cx, cy = (
            intrinsics_color["fx"],
            intrinsics_color["fy"],
            intrinsics_color["cx"],
            intrinsics_color["cy"],
        )
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        if distCoeffs is None:
            distCoeffs = np.zeros(5)

        s = float(tag_size) / 2.0
        objp = np.array(
            [[-s, -s, 0.0], [s, -s, 0.0], [s, s, 0.0], [-s, s, 0.0]],
            dtype=np.float64,
        )

        rvec, _ = cv2.Rodrigues(det.pose_R.astype(np.float64))  # tag->cam
        tvec = det.pose_t.reshape(3, 1).astype(np.float64)
        imgpts, _ = cv2.projectPoints(objp, rvec, tvec, K, distCoeffs)
        proj = imgpts.reshape(-1, 2)

        det_corners = np.array(det.corners, dtype=np.float64)
        residual = np.linalg.norm(proj - det_corners, axis=1)
        return proj, residual

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
    missed_count = 0
    for key in calib.keys():
        demo = calib[key]
        T, H, W, _ = demo["obs/front_img_1"].shape
        for t in tqdm(range(0, T, args.every_k)):
            img = demo["obs/front_img_1"][t]
            # img = cv2.undistort(
            #     img, WIDE_LENS_ROBOT_LEFT_K[:, :3], WIDE_LENS_ROBOT_LEFT_D
            # )

            detect_result = april_detector.detect(
                img,
                intrinsics=intrinsics["color"],
                # tag_size=0.0958)
                tag_size=0.155,
            )

            if len(detect_result) != 1:
                if args.debug:
                    os.makedirs("calibration_imgs", exist_ok=True)
                    plt.imsave(f"calibration_imgs/{t}_fail.png", img)
                missed_count += 1
                continue

            bounding_box_corners = detect_result[0].corners
            # draw bounding box on img and save
            # if args.debug:
            #     os.makedirs("calibration_imgs", exist_ok=True)
            #     img = april_detector.vis_tag(img)
            #     plt.imsave(f"calibration_imgs/{t}_detection.png", img)

            # Optional: Reprojection check to validate intrinsics / tag size
            proj, resid = reproject_tag_corners(
                detect_result[0], intrinsics["color"], tag_size=0.155
            )
            if args.debug:
                print(
                    f"[t={t}] reprojection error px per-corner: {np.round(resid,2)}, mean={resid.mean():.2f}"
                )
                # Save overlay image with detected (green) vs projected (red) corners
                os.makedirs("calibration_imgs", exist_ok=True)
                vis = img.copy()
                # detected corners (green)
                for (u, v) in detect_result[0].corners:
                    cv2.circle(vis, (int(u), int(v)), 4, (0, 255, 0), -1)
                # projected corners (red)
                for (u, v) in proj:
                    cv2.circle(vis, (int(u), int(v)), 4, (0, 0, 255), -1)
                cv2.putText(
                    vis,
                    f"mean err: {resid.mean():.2f}px",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2,
                )
                cv2.imwrite(f"calibration_imgs/{t}_reproj.png", vis)

            count += 1
            pose = demo["obs/ee_pose_robot_frame"][t]
            assert pose.shape == (7,)
            pos = pose[0:3]
            # Orientation stored as [yaw, pitch, roll] (radians); ignore gripper at pose[6]
            angles_ypr = pose[3:6]
            # Use ZYX (yaw, pitch, roll) convention
            rot = Rot.from_euler("ZYX", angles_ypr, degrees=False)

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
    print(f"==========Missed {missed_count} images================")

    for method in [
        # cv2.CALIB_HAND_EYE_TSAI,
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
