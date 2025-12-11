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
        # default="./",
    )

    parser.add_argument("--config-filename", type=str, default="joints_info.json")

    parser.add_argument("--camera-id", type=int, default=0)

    parser.add_argument("--camera-type", type=str, default="k4a")

    parser.add_argument("--use-saved-images", action="store_true")

    parser.add_argument("--debug", action="store_true")

    parser.add_argument("-p", "--post-fix", type=str, default="")

    parser.add_argument("--calibration-img-folder", type=str, default="calibration_imgs")

    return parser.parse_args()


def main():
    pp = pprint.PrettyPrinter(indent=4)

    args = parse_args()
    # load a list of joints to move
    joints_json_file_name = f"{args.config_folder}/{args.config_filename}"

    joint_info = json.load(open(joints_json_file_name, "r"))
    joint_list = joint_info["joints"]

    with open(
        os.path.join(args.config_folder, f"camera_{args.camera_id}_{args.camera_type}.json"), "r"
    ) as f:
        intrinsics = json.load(f)

    # TODO: Load extrinsics of camera2gripper
    camera_extrinsics_json_file_name = "debug_camera_calibration/camera_2_rs_1105_extrinsics.json"
    # camera_extrinsics_json_file_name = os.path.expanduser("~/.rpl_vision_utils/calibration/camera_2_rs_extrinsics.json")
    extrinsics = json.load(open(camera_extrinsics_json_file_name, "r"))
    R_cam2ee, p_cam2ee = (
        np.array(extrinsics["rotation"]),
        np.array(extrinsics["translation"]).reshape((3)),
    )

    use_saved_images = args.use_saved_images

    identity_matrix_3x3 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    identity_matrix_4x4 = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )

    new_joint_list = []

    rpl_transform_manager = RPLTransformManager()

    # intrinsics = cr_interface.get_img_info()["intrinsics"]
    # print(intrinsics)

    if not use_saved_images:

        os.makedirs(args.calibration_img_folder, exist_ok=True)
        camera_id = args.camera_id
        cr_interface = CameraRedisSubInterface(camera_id=camera_id, use_color=True, use_depth=True)
        cr_interface.start()

        os.makedirs(args.calibration_img_folder, exist_ok=True)
        # camera_id = args.camera_id
        # cr_interface = CameraRedisSubInterface(camera_id=camera_id)
        # cr_interface.start()

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
            cv2.imwrite(f"{args.calibration_img_folder}/{idx}_color.png", color_img)
            cv2.imwrite(f"{args.calibration_img_folder}/{idx}_depth.png", depth_img)
            cv2.waitKey(1)

            time.sleep(0.3)

    ###########################################
    # Start visualizing images
    ###########################################
    april_detector = AprilTagDetector()

    with open(
        os.path.join(args.config_folder, f"camera_{args.camera_id}_{args.camera_type}.json"), "r"
    ) as f:
        intrinsics = json.load(f)

    print(intrinsics)

    R_gripper2base_list = []
    t_gripper2base_list = []

    R_target2cam_list = []
    t_target2cam_list = []

    urdf_model = URDFModel()

    count = 0

    imgs = []
    tag_observations = {}

    # intrinsics['color']['fx'] = 635.64732655
    # intrinsics['color']['fy'] = 635.64732655
    # intrinsics['color']['cx'] = 321.29969964
    # intrinsics['color']['cy'] = 241.43618241
    # distortion_coeff = np.array([[ 4.78724231e-02],
    #                        [ 3.49537234e+01],
    #                        [-4.84510371e-04],
    #                        [-6.76381113e-03],
    #                        [-5.18494520e+00],
    #                        [ 2.62149250e-02],
    #                        [ 3.35731787e+01],
    #                        [-2.17898416e+00],
    #                        [ 0.00000000e+00],
    #                        [ 0.00000000e+00],
    #                        [ 0.00000000e+00],
    #                        [ 0.00000000e+00],
    #                        [ 0.00000000e+00],
    #                        [ 0.00000000e+00]])

    for (idx, robot_joints) in enumerate(new_joint_list):
        rpl_transform_manager.add_transform(f"cam_{idx}", f"ee_{idx}", R_cam2ee, p_cam2ee)
        print(idx)
        img = cv2.imread(f"{args.calibration_img_folder}/{idx}_color.png")
        # img = cv2.undistort(img, np.array([[635.64732655, 0.,  321.29969964], [0., 635.64732655, 241.43618241], [0., 0., 1.]]), distortion_coeff, None)
        imgs.append(img)

        count += 1
        pose = urdf_model.get_gripper_pose(robot_joints)[:2]
        p_ee2base = np.array(pose[0]).reshape((3))
        R_ee2base = T.quat2mat(pose[1])
        rpl_transform_manager.add_transform(f"ee_{idx}", "base", R_ee2base, p_ee2base)

        detect_results = april_detector.detect(
            img, intrinsics=intrinsics["color"], tag_size=0.080
        )  # 0.043)
        tag_poses = {}
        for single_result in detect_results:
            tag_id = single_result.tag_id
            R_target2cam = single_result.pose_R
            p_target2cam = single_result.pose_t.reshape((3))

            # compute T_target2base
            tag_poses[single_result.tag_id] = {"R": R_target2cam, "p": p_target2cam}
            rpl_transform_manager.add_transform(
                f"cam_{idx}_tag_{tag_id}", f"cam_{idx}", R_target2cam, p_target2cam
            )

        # img = april_detector.vis_tag(img)
        # cv2.imshow("", img)
        # cv2.waitKey(0)

        tag_observations[idx] = tag_poses

    for tag_id in range(11):
        pp.pprint(f"================{tag_id}====================")
        for idx in range(len(joint_list)):
            if tag_id in tag_observations[idx]:
                target2base = rpl_transform_manager.get_transform(
                    f"cam_{idx}_tag_{tag_id}", "base"
                )
                pp.pprint(np.round(target2base, 3))


if __name__ == "__main__":
    main()
