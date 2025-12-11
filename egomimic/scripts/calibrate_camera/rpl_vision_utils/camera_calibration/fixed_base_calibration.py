import os

folder_path = os.path.join(os.path.dirname(__file__))

import numpy as np
import cv2
import argparse
import json

import init_path

from gprs.franka_interface import FrankaInterface
from gprs.camera_redis_interface import CameraRedisSubInterface
from gprs.utils.io_devices import SpaceMouse
from gprs.utils.input_utils import input2action
from gprs.franka_interface.visualizer import PybulletVisualizer
from gprs.utils import load_yaml_config
import gprs.utils.transform_utils as T
from gprs import config_root

# from rpl_vision_utils.k4a.k4a_interface import K4aInterface
from rpl_vision_utils.utils.apriltag_detector import AprilTagDetector

from urdf_models.urdf_models import URDFModel


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config-folder',
        type=str,
        default=os.path.expanduser("~/.rpl_vision_utils/calibration"),
    )

    parser.add_argument(
        '--config-filename',
        type=str,
        default="calib.json"
    )

    parser.add_argument(
        '--camera-id',
        type=int,
        default=0
    )

    parser.add_argument(
        '--camera-type',
        type=str,
        default="k4a"
    )

    parser.add_argument(
        '--use-saved-images',
        action='store_true'
    )

    parser.add_argument(
        '--debug',
        action='store_true'
    )

    return parser.parse_args()


def main():
    args = parse_args()
    # load a list of joints to move
    joints_json_file_name = f"{args.config_folder}/{args.config_filename}"

    joint_info = json.load(open(joints_json_file_name, "r"))
    joint_list = joint_info["joints"]

    # Iniitalize camera interface

    use_saved_images = args.use_saved_images

    identity_matrix_3x3 = np.array([[1., 0., 0.],
                                    [0., 1., 0.],
                                    [0., 0., 1.]])

    identity_matrix_4x4 = np.array([[1., 0., 0., 0.],
                                    [0., 1., 0., 0.],
                                    [0., 0., 1., 0.],
                                    [0., 0., 0., 1.]])

    new_joint_list = []
    new_ee_pos_list = []
    new_ee_rot_list = []

    calibration_img_folder = "calib"
    if not use_saved_images:
        os.makedirs(calibration_img_folder, exist_ok=True)
        camera_id = args.camera_id
        cr_interface = CameraRedisSubInterface(camera_id=camera_id)
        cr_interface.start()

        # Load robot controller configs
        controller_cfg = load_yaml_config(config_root + "/osc-controller.yml")
        robot_interface = FrankaInterface(config_root + "/alice.yml", use_visualizer=False)
        controller_type = "JOINT_POSITION"

        print(cr_interface.get_img_info())
        intrinsics = cr_interface.get_img_info()["intrinsics"]
        # distortion = cr_interface.get_img_info()["distortion"]
        print(intrinsics)
        for (idx, robot_joints) in enumerate(joint_list):
            action = robot_joints + [-1]

            while True:
                if len(robot_interface._state_buffer) > 0:
                    # print(np.round(np.array(robot_interface._state_buffer[-1].qq) - np.array(reset_joint_positions), 5))
                    if np.max(np.abs(np.array(robot_interface._state_buffer[-1].q) - np.array(robot_joints))) < 5e-3:
                        break
                robot_interface.control(control_type=controller_type,
                                        action=action,
                                        controller_cfg=controller_cfg)

            # save image

            import time;
            time.sleep(0.8)
            while not np.linalg.norm(robot_interface._state_buffer[-1].q) > 0:
                time.sleep(0.01)
            new_joint_list.append(robot_interface._state_buffer[-1].q)
            print("+++++++++++++++++++++ robot ee pose", robot_interface._state_buffer[-1].O_T_EE)
            ee_pose = np.array(robot_interface._state_buffer[-1].O_T_EE)
            new_ee_pos_list.append(ee_pose.reshape(4, 4).T[:3, 3])
            new_ee_rot_list.append(ee_pose.reshape(4, 4).T[:3, :3])
            imgs = cr_interface.get_img()
            img_info = cr_interface.get_img_info()

            color_img = imgs["color"]
            cv2.imshow("", color_img)
            cv2.imwrite(f"{calibration_img_folder}/{idx}.png", color_img)
            cv2.waitKey(1)
            import time;
            time.sleep(0.3)

        with open(os.path.join(args.config_folder, f"camera_{args.camera_id}_{args.camera_type}.json"), "w") as f:
            json.dump(intrinsics, f)

        # with open(os.path.join(args.config_folder, f"camera_{args.camera_id}_{args.camera_type}_distortion.json"), "w") as f:
        #     json.dump(distortion, f)

        # joint_list = new_joint_list
        robot_interface.close()

    april_detector = AprilTagDetector()

    with open(os.path.join(args.config_folder, f"camera_{args.camera_id}_{args.camera_type}.json"), "r") as f:
        intrinsics = json.load(f)

    print(intrinsics)

    R_base2gripper_list = []
    t_base2gripper_list = []

    R_target2cam_list = []
    t_target2cam_list = []

    marker_pose_computation = URDFModel()

    count = 0
    for (idx, robot_joints) in enumerate(joint_list):

        print(f"{calibration_img_folder}/{idx}.png")
        img = cv2.imread(f"{calibration_img_folder}/{idx}.png")

        detect_result = april_detector.detect(img,
                                              intrinsics=intrinsics["color"],
                                              tag_size=0.039)

        if len(detect_result) != 1:
            print(f"wrong detection, skipping img {idx}")
            continue

        count += 1
        pose = marker_pose_computation.get_gripper_pose(robot_joints)[:2]
        # print("--------------------- print the pose of gripper pose", pose, T, np.array(pose[1]))
        pos = pose[0]
        # from IPython import embed; embed()
        rot = T.quat2mat(np.array(pose[1]))

        # print("HERE", rot.transpose(), -rot.transpose() @ np.array(pos)[:, np.newaxis])
        # pos = new_ee_pos_list[idx]
        print("-----------gripper pose from urdf", rot)
        # print("-----------gripper pose from ee", new_ee_rot_list[idx])

        # rot = new_ee_rot_list[idx]

        R_base2gripper_list.append(rot.transpose())
        t_base2gripper_list.append(-rot.transpose() @ np.array(pos)[:, np.newaxis])

        R_target2cam_list.append(detect_result[0].pose_R)
        pose_t = detect_result[0].pose_t

        if args.debug:
            print("Detected: ", pose_t, T.quat2axisangle(T.mat2quat(detect_result[0].pose_R)))

        t_target2cam_list.append(pose_t)

        if args.debug:
            img = april_detector.vis_tag(img)
            cv2.imwrite(f"calibration_imgs/{idx}_detection.png", img)
            cv2.imshow("", img)
            cv2.waitKey(0)

    print(f"==========Using {count} images================")

    for method in [
        cv2.CALIB_HAND_EYE_TSAI
    ]:
        R, t = cv2.calibrateHandEye(
            R_base2gripper_list, t_base2gripper_list,
            R_target2cam_list, t_target2cam_list,
            method=method,
        )
        print("Rotation matrix: ", R)
        print("Axis Angle: ", T.quat2axisangle(T.mat2quat(R)))
        print("Quaternion: ", T.mat2quat(R))
        print("Translation: ", t.transpose())

        with open(os.path.join(args.config_folder, f"camera_{args.camera_id}_{args.camera_type}_extrinsics.json"),
                  "w") as f:
            extrinsics = {"translation": t.tolist(),
                          "rotation": R.tolist()}
            json.dump(extrinsics, f)
    print("==============================")

    if args.debug:
        def multiply(R1, t1, R2, t2):
            return R1 @ R2, R1 @ t2 + t1

        for (R_base2gripper, t_base2gripper, R_target2cam, t_target2cam, robot_joints) in zip(R_base2gripper_list,
                                                                                              t_base2gripper_list,
                                                                                              R_target2cam_list,
                                                                                              t_target2cam_list,
                                                                                              joint_list):
            R_target2base, t_target2base = multiply(R, t, R_target2cam, t_target2cam)

            # The eye-to-base calibration is optimized based on the
            # fact that the transformation between the marker and the
            # gripper is fixed in this setting.
            R_target2gripper, t_target2gripper = multiply(R_base2gripper, t_base2gripper, R_target2cam, t_target2cam)

            pose = marker_pose_computation.get_marker_pose(robot_joints)[:2]
            pos = pose[0]
            rot = T.quat2mat(pose[1])


if __name__ == "__main__":
    main()
