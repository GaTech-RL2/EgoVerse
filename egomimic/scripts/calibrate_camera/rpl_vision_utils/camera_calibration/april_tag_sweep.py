"""
Script to take a json containing a recorded sequence of joint positions 
(e.g. from camera_calibration_gprs.py) and to move the robot through each 
joint position. For each location, we try to detect apriltags from the 
current wrist camera frame, and use that to estimate the pose of the 
apriltag with respect to the base frame.
"""

import time
import os
import cv2
import imageio
import json
import numpy as np
from PIL import Image, ImageDraw

from gprs.utils import load_yaml_config
from gprs.franka_interface import FrankaInterface
from gprs.camera_redis_interface import CameraRedisSubInterface
from gprs import config_root
from rpl_vision_utils.utils.apriltag_detector import AprilTagDetector

# wrist camera id
WRIST_CAMERA_ID = 0

# tag ids to detect
TAG_IDS_TO_DETECT = [6, 7]

# json with sequence of joint positions to follow
JSON_FILE = os.path.join(os.path.expanduser("~/.rpl_vision_utils/calibration"), "no_tag_arm_joints_ajay.json")


def get_wrist_camera_extrinsic_matrix():
    """
    Fill out this function to put the extrinsic matrix of your camera.
    This should correspond to the camera pose in the robot base frame. 

    Returns:
        R (np.array): 4x4 camera extrinsic matrix
    """
    R = np.eye(4)
    R[:3, :3] = np.array([
        [0.00180954, -0.82874178, -0.55962826],
        [0.99990155,  0.00928646, -0.01051896],
        [0.01391446, -0.55955413, 0.82867699],
    ])
    R[:3, 3] = np.array([0.03004587, -0.00614714, -0.04158355])
    return R


def pose_in_A_to_pose_in_B(pose_in_A, pose_A_in_B):
    """
    Converts homogenous matrices corresponding to a point C in frame A
    to homogenous matrices corresponding to the same point C in frame B.

    Args:
        pose_in_A (np.array): batch of homogenous matrices corresponding to the pose of C in frame A
        pose_A_in_B (np.array): batch of homogenous matrices corresponding to the pose of A in frame B

    Returns:
        pose_in_B (np.array): batch of homogenous matrices corresponding to the pose of C in frame B
    """
    print("mat shape", pose_A_in_B.shape, pose_in_A.shape)
    return np.matmul(pose_A_in_B, pose_in_A)


def get_robot_eef_pose(robot_interface):
    """
    Returns eef pose in base frame as 4x4 matrix.
    """
    last_robot_state = robot_interface._state_buffer[-1]
    ee_pose = np.array(last_robot_state.O_T_EE).reshape((4, 4)).T
    return ee_pose


def move_robot_to_joint_position(robot_interface, joint_pos):
    """
    Should move robot to a target joint configuration.
    """
    action = list(joint_pos) + [-1.]
    controller_cfg = load_yaml_config(os.path.join(config_root, "osc-controller.yml"))
    while True:
        if len(robot_interface._state_buffer) > 0:
            print("target: {}".format(joint_pos))
            current = robot_interface._state_buffer[-1].q
            print("current: {}".format(current))
            print("-----------------------")

            if np.max(np.abs(np.array(current) - np.array(joint_pos))) < 1e-3:
                break

        robot_interface.control(
            control_type="JOINT_POSITION",
            action=action,
            controller_cfg=controller_cfg,
        )


def detect_april_tag(april_detector, camera_interface, camera_intrinsics, display_str):
    """
    Helper function to get april tag detections and show them. Returns list of dictionaries
    containing the tag poses with respect to the camera, and the tag ID (to distinguish
    between different tags if multiple are present).

    NOTE: assumes that all tags are the same family and size.
    """

    # read image
    print("READ")
    img = camera_interface.get_img()["color"]

    # show all detections
    detect_result = april_detector.detect(img,
        intrinsics=camera_intrinsics["color"],
        tag_size=0.039,
    )
    img = april_detector.vis_tag(img)
    cv2.imshow(display_str, img)
    cv2.waitKey(1)

    # return list of dicts with pose + tag id
    ret = []
    for res in detect_result:
        tag_family = res.tag_family.decode('utf-8')
        # print('tag_family', tag_family)
        assert tag_family == "tag36h11"

        tag_id = res.tag_id
        R = res.pose_R
        t = res.pose_t
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = t.reshape(-1)
        ret.append(dict(pose=pose, tag_id=tag_id))
    return ret


if __name__ == "__main__":
    # get extrinsic for wrist camera
    wrist_camera_pose_in_eef = get_wrist_camera_extrinsic_matrix()

    # april tag detector used to detect tag poses with respect to each camera
    april_detector = AprilTagDetector()

    # set up robot interface to read eef pose (needed for wrist camera pose conversion)
    # and to send joint position commands
    robot_interface = FrankaInterface(config_root + "/alice.yml", use_visualizer=False)

    # set up camera interfaces to read images
    wrist_camera_interface = CameraRedisSubInterface(camera_id=WRIST_CAMERA_ID)
    wrist_camera_interface.start()
    wrist_camera_intrinsics = wrist_camera_interface.get_img_info()["intrinsics"]

    # read json for sequence of robot joints to follow
    with open(JSON_FILE, "r") as f:
        json_dic = json.load(f)
    joint_position_seq = json_dic["joints"]

    # keep track of all tag pose estimates for each tag ID (with respect to robot base frame)
    tag_pose_estimates = { x : [] for x in TAG_IDS_TO_DETECT }

    for i, joint_pos in enumerate(joint_position_seq):
        print("moving to joint position {}: {}".format(i, joint_pos))

        # move robot to next joint position in sequence
        move_robot_to_joint_position(robot_interface=robot_interface, joint_pos=joint_pos)

        # read wrist image, detect april tag, and show result
        tag_poses_in_wrist_camera_frame = detect_april_tag(
            april_detector=april_detector,
            camera_interface=wrist_camera_interface,
            camera_intrinsics=wrist_camera_intrinsics,
            display_str="wrist",
        )

        # if we have some detections, read them
        for tag_pose_dict in tag_poses_in_wrist_camera_frame:

            tag_id = tag_pose_dict["tag_id"]
            if tag_id in TAG_IDS_TO_DETECT:
                # wrist camera pose is with respect to eef frame, so convert to base frame
                # using current end effector pose
                eef_pose_in_base = get_robot_eef_pose(robot_interface)
                wrist_camera_pose_in_base = pose_in_A_to_pose_in_B(
                    pose_in_A=wrist_camera_pose_in_eef,
                    pose_A_in_B=eef_pose_in_base,
                )

                # then convert april tag pose with respect to camera frame to be with
                # respect to base frame
                tag_pose_in_base_frame = pose_in_A_to_pose_in_B(
                    pose_in_A=tag_pose_dict["pose"],
                    pose_A_in_B=wrist_camera_pose_in_base,
                )

                # add to results
                tag_pose_estimates[tag_id].append(tag_pose_in_base_frame)

            else:
                print("*" * 50)
                print("WARNING: detected tag id {} not in targets {} to detect".format(tag_id, TAG_IDS_TO_DETECT))
                print("*" * 50)


    print("\nDone with joint position sequence!\n")

    # print results
    estimates = dict(raw={}, avg={}, num={})
    for x in TAG_IDS_TO_DETECT:
        estimates["raw"][x] = dict(
            pose=[y.reshape(-1).tolist() for y in tag_pose_estimates[x]],
            pos=[y[:3, 3].reshape(-1).tolist() for y in tag_pose_estimates[x]],
        )
        estimates["avg"][x] = dict(
            pose=np.array(tag_pose_estimates[x]).mean(axis=0).tolist(),
            pos=np.array(tag_pose_estimates[x])[:, :3, 3].mean(axis=0).tolist(),
        )
        estimates["num"][x] = len(tag_pose_estimates[x])
    print("RAW")
    print(json.dumps(estimates["raw"], indent=4))
    print("AVG")
    print(json.dumps(estimates["avg"], indent=4))
    print("NUM")
    print(json.dumps(estimates["num"], indent=4))
