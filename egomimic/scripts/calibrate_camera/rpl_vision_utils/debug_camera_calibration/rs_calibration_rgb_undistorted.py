import argparse
import json
import os
import pprint
import time

import cv2
import gprs.utils.transform_utils as T
import init_path
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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config-folder",
        type=str,
        default=os.path.expanduser("~/.rpl_vision_utils/calibration"),
    )

    parser.add_argument("--config-filename", type=str, default="joints_info.json")

    parser.add_argument("--camera-id", type=int, default=0)

    parser.add_argument("--camera-type", type=str, default="k4a")

    parser.add_argument("--use-saved-images", action="store_true")

    parser.add_argument("--debug", action="store_true")

    parser.add_argument("-p", "--post-fix", type=str, default="")

    return parser.parse_args()


def main():

    args = parse_args()
    # load a list of joints to move
    joints_json_file_name = f"{args.config_folder}/{args.config_filename}"

    joint_info = json.load(open(joints_json_file_name, "r"))
    joint_list = joint_info["joints"]

    # Iniitalize camera interface

    use_saved_images = args.use_saved_images

    identity_matrix_3x3 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    identity_matrix_4x4 = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )

    new_joint_list = []

    calibration_img_folder = "calibration_imgs"

    """Take images"""
    if not use_saved_images:
        os.makedirs(calibration_img_folder, exist_ok=True)
        camera_id = args.camera_id
        cr_interface = CameraRedisSubInterface(camera_id=camera_id, use_depth=True)
        cr_interface.start()

        # Load robot controller configs
        controller_cfg = load_yaml_config(config_root + "/osc-controller.yml")
        robot_interface = FrankaInterface(config_root + "/alice.yml", use_visualizer=False)
        controller_type = "JOINT_POSITION"

        intrinsics = cr_interface.get_img_info()["intrinsics"]
        print(intrinsics)

        for (idx, robot_joints) in enumerate(joint_list):
            action = robot_joints + [-1]

            while True:
                if len(robot_interface._state_buffer) > 0:
                    # print(np.round(np.array(robot_interface._state_buffer[-1].qq) - np.array(reset_joint_positions), 5))
                    if (
                        np.max(
                            np.abs(
                                np.array(robot_interface._state_buffer[-1].q)
                                - np.array(robot_joints)
                            )
                        )
                        < 5e-3
                    ):
                        break
                robot_interface.control(
                    control_type=controller_type, action=action, controller_cfg=controller_cfg
                )

            # save image

            time.sleep(0.8)
            while not np.linalg.norm(robot_interface._state_buffer[-1].q) > 0:
                time.sleep(0.01)
            new_joint_list.append(robot_interface._state_buffer[-1].q)
            imgs = cr_interface.get_img()
            img_info = cr_interface.get_img_info()

            color_img = imgs["color"]
            depth_img = imgs["depth"]
            cv2.imshow("", color_img)
            cv2.imwrite(f"{calibration_img_folder}/{idx}_color.png", color_img)
            cv2.imwrite(f"{calibration_img_folder}/{idx}_depth.png", depth_img)
            cv2.waitKey(1)

            time.sleep(0.3)

        with open(
            os.path.join(args.config_folder, f"camera_{args.camera_id}_{args.camera_type}.json"),
            "w",
        ) as f:
            json.dump(intrinsics, f)
        # joint_list = new_joint_list
        robot_interface.close()

    rpl_transform_manager = RPLTransformManager()
    april_detector = AprilTagDetector()

    with open(
        os.path.join(args.config_folder, f"camera_{args.camera_id}_{args.camera_type}.json"), "r"
    ) as f:
        intrinsics = json.load(f)

    print(intrinsics)
    intrinsics["color"]["fx"] = 635.64732655
    intrinsics["color"]["fy"] = 635.64732655
    intrinsics["color"]["cx"] = 321.29969964
    intrinsics["color"]["cy"] = 241.43618241
    distortion_coeff = np.array(
        [
            [4.78724231e-02],
            [3.49537234e01],
            [-4.84510371e-04],
            [-6.76381113e-03],
            [-5.18494520e00],
            [2.62149250e-02],
            [3.35731787e01],
            [-2.17898416e00],
            [0.00000000e00],
            [0.00000000e00],
            [0.00000000e00],
            [0.00000000e00],
            [0.00000000e00],
            [0.00000000e00],
        ]
    )

    R_gripper2base_list = []
    t_gripper2base_list = []

    R_target2cam_list = []
    t_target2cam_list = []

    marker_pose_computation = URDFModel()

    count = 0

    imgs = []

    for (idx, robot_joints) in enumerate(joint_list):
        img = cv2.imread(f"{calibration_img_folder}/{idx}_color.png")

        # Yifeng: temporrary modification
        img = cv2.undistort(
            img,
            np.array(
                [
                    [635.64732655, 0.0, 321.29969964],
                    [0.0, 635.64732655, 241.43618241],
                    [0.0, 0.0, 1.0],
                ]
            ),
            distortion_coeff,
            None,
        )

        detect_results = april_detector.detect(
            img,
            intrinsics=intrinsics["color"],
            # tag_size=0.1718
            tag_size=0.080,
        )
        if len(detect_results) != 1:
            continue

        imgs.append(img)
        count += 1
        pose = marker_pose_computation.get_gripper_pose(robot_joints)[:2]
        pos = pose[0]
        rot = T.quat2mat(pose[1])

        rpl_transform_manager.add_transform(f"ee_{idx}", "base", rot, np.array(pos))

        R_gripper2base_list.append(rot)
        t_gripper2base_list.append(np.array(pos)[:, np.newaxis])

        R_target2cam_list.append(detect_results[0].pose_R)
        pose_t = detect_results[0].pose_t

        rpl_transform_manager.add_transform(
            f"target_{idx}", f"cam_view_{idx}", detect_results[0].pose_R, pose_t
        )
        if args.debug:
            print("Detected: ", pose_t, T.quat2axisangle(T.mat2quat(detect_results[0].pose_R)))

        t_target2cam_list.append(pose_t)

        if args.debug:
            img = april_detector.vis_tag(img)
            cv2.imwrite(f"calibration_imgs/{idx}_detection.png", img)
            cv2.imshow("", img)
            cv2.waitKey(0)

    print(f"==========Using {count} images================")
    pp = pprint.PrettyPrinter(indent=2)
    for method in [cv2.CALIB_HAND_EYE_HORAUD, cv2.CALIB_HAND_EYE_TSAI]:
        R, t = cv2.calibrateHandEye(
            R_gripper2base_list,
            t_gripper2base_list,
            R_target2cam_list,
            t_target2cam_list,
            method=method,
        )
        print("Rotation matrix: ", R)
        print("Axis Angle: ", T.quat2axisangle(T.mat2quat(R)))
        print("Quaternion: ", T.mat2quat(R))
        print("Translation: ", t.transpose())

        with open(
            os.path.join(
                args.config_folder,
                f"camera_{args.camera_id}_{args.camera_type}_{args.post_fix}extrinsics.json",
            ),
            "w",
        ) as f:
            extrinsics = {"translation": t.tolist(), "rotation": R.tolist()}
            json.dump(extrinsics, f)

    for idx in range(len(joint_list)):
        rpl_transform_manager.add_transform(f"cam_view_{idx}", f"ee_{idx}", R, t)

    print("==============================")
    for idx in range(len(joint_list)):
        pp.pprint(f"View {idx}:")
        target2base = rpl_transform_manager.get_transform(f"target_{idx}", "base")
        pp.pprint(np.round(target2base, 3))

    # if args.debug:
    #     def multiply(R1, t1, R2, t2):
    #         return R1 @ R2, R1 @ t2 + t1

    #     for (R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, robot_joints) in zip(R_gripper2base_list,
    #                                                                             t_gripper2base_list,
    #                                                                             R_target2cam_list,
    #                                                                             t_target2cam_list,
    #                                                                             joint_list):

    #         R_target2base, t_target2base = multiply(R, t, R_target2cam, t_target2cam)

    #         # The eye-to-base calibration is optimized based on the
    #         # fact that the transformation between the marker and the
    #         # gripper is fixed in this setting.
    #         R_target2gripper, t_target2gripper = multiply(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam)

    #         pose = marker_pose_computation.get_marker_pose(robot_joints)[:2]
    #         pos = pose[0]
    #         rot = T.quat2mat(pose[1])


if __name__ == "__main__":
    main()
