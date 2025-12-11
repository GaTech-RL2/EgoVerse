"""
GPRS version of testing wrist camera calibration.

This script tries to detect an april tag using the wrist and front camera,
and uses their extrinsics to compute the april tag pose with respect to the robot base
frame. We measure the agreement.
"""

import time
import cv2
import imageio
import numpy as np
from PIL import Image, ImageDraw

from gprs.franka_interface import FrankaInterface
from gprs.camera_redis_interface import CameraRedisSubInterface
from gprs import config_root
from rpl_vision_utils.utils.apriltag_detector import AprilTagDetector

# wrist camera id
WRIST_CAMERA_ID = 0

# front camera id
FRONT_CAMERA_ID = 1


def get_camera_extrinsic_matrix():
    return dict(
        front=get_front_camera_extrinsic_matrix(),
        wrist=get_wrist_camera_extrinsic_matrix(),
    )


# def get_camera_intrinsic_matrix():
#     return dict(
#         front=get_front_camera_intrinsic_matrix(),
#         wrist=get_wrist_camera_intrinsic_matrix(),
#     )


# # FRONT CAMERA
# def get_front_camera_intrinsic_matrix():
#     """
#     Fill out this function to put the intrinsic matrix of your camera.
#     Returns:
#         K (np.array): 3x3 camera matrix
#     """
#     K = np.array([
#         [608.21350098, 0., 327.76547241],
#         [0., 608.25396729, 242.55935669],
#         [0., 0., 1.],
#     ])
#     return K


def get_front_camera_extrinsic_matrix():
    """
    Fill out this function to put the extrinsic matrix of your camera.
    This should correspond to the camera pose in the robot base frame. 
    Returns:
        R (np.array): 4x4 camera extrinsic matrix
    """
    R = np.eye(4)
    R[:3, :3] = np.array([
        [0.03039011, 0.6968179, -0.71660398],
        [ 0.99723176, -0.06981243, -0.02559373],
        [-0.06786204, -0.71384245, -0.69701055],
    ])
    R[:3, 3] = np.array([1.110382, 0.07760241, 0.70286399])
    return R


# # WRIST CAMERA
# def get_wrist_camera_intrinsic_matrix():
#     """
#     Fill out this function to put the intrinsic matrix of your camera.

#     Returns:
#         K (np.array): 3x3 camera matrix
#     """
#     K = np.array([
#         [607.26049805, 0., 327.08224487],
#         [0., 607.22131348, 243.40344238],
#         [0., 0., 1.],
#     ])
#     return K


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


def pose_inv(pose):
    """
    Computes the inverse of a homogeneous matrix corresponding to the pose of some
    frame B in frame A. The inverse is the pose of frame A in frame B.
    Args:
        pose (np.array): 4x4 matrix for the pose to inverse
    Returns:
        np.array: 4x4 matrix for the inverse pose
    """

    # Note, the inverse of a pose matrix is the following
    # [R t; 0 1]^-1 = [R.T -R.T*t; 0 1]

    # Intuitively, this makes sense.
    # The original pose matrix translates by t, then rotates by R.
    # We just invert the rotation by applying R-1 = R.T, and also translate back.
    # Since we apply translation first before rotation, we need to translate by
    # -t in the original frame, which is -R-1*t in the new frame, and then rotate back by
    # R-1 to align the axis again.

    pose_inv = np.zeros((4, 4))
    pose_inv[:3, :3] = pose[:3, :3].T
    pose_inv[:3, 3] = -pose_inv[:3, :3].dot(pose[:3, 3])
    pose_inv[3, 3] = 1.0
    return pose_inv


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
    return np.matmul(pose_A_in_B, pose_in_A)


def pose_in_camera_to_pose_in_base(pose_in_camera, camera_pose_in_base):
    """
    Takes 4x4 pose in camera frame and transforms to 4x4 pose in robot base frame.
    """
    return pose_in_A_to_pose_in_B(pose_in_A=pose_in_camera, pose_A_in_B=camera_pose_in_base)


def get_robot_eef_pose(robot_interface):
    """
    Returns eef pose in base frame as 4x4 matrix
    """
    last_robot_state = robot_interface._state_buffer[-1]
    ee_pose = np.array(last_robot_state.O_T_EE).reshape((4, 4)).T
    return ee_pose
    return ee_pose[:3, 3]


def detect_april_tag(april_detector, camera_interface, camera_intrinsics, display_str):
    """
    Helper function to get april tag detection and show it.
    """

    # read image
    img = camera_interface.get_img()["color"]

    # show detection
    detect_result = april_detector.detect(img,
        intrinsics=camera_intrinsics["color"],
        tag_size=0.039,
    )
    img = april_detector.vis_tag(img)
    cv2.imshow(display_str, img)
    cv2.waitKey(1)

    # return detection pose
    if len(detect_result) != 1:
        return None
    R = detect_result[0].pose_R
    t = detect_result[0].pose_t
    # from IPython import embed; embed()
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = t.reshape(-1)
    return pose


def compare_pose(pose_1, pose_2):
    """
    Helper function to compute error in pose.
    """
    pos_1 = pose_1[:3, 3]
    pos_2 = pose_2[:3, 3]
    pos_err = np.sum((pos_1 - pos_2) ** 2)
    print("pos 1: {}".format(pos_1))
    print("pos 2: {}".format(pos_2))
    print("square L2 error: {}".format(pos_err))


if __name__ == "__main__":
    # get extrinsics for both cameras
    camera_extrinsics = get_camera_extrinsic_matrix()

    # april tag detector used to detect tag poses with respect to each camera
    april_detector = AprilTagDetector()

    # set up robot interface to read eef pose (needed for wrist camera pose conversion)
    robot_interface = FrankaInterface(config_root + "/alice.yml", use_visualizer=False)

    # set up camera interfaces to read images
    front_camera_interface = CameraRedisSubInterface(camera_id=FRONT_CAMERA_ID)
    front_camera_interface.start()
    front_camera_intrinsics = front_camera_interface.get_img_info()["intrinsics"]
    wrist_camera_interface = CameraRedisSubInterface(camera_id=WRIST_CAMERA_ID)
    wrist_camera_interface.start()
    wrist_camera_intrinsics = wrist_camera_interface.get_img_info()["intrinsics"]


    while True:

        # read front image, detect april tag, and show result
        tag_front_pose = detect_april_tag(
            april_detector=april_detector,
            camera_interface=front_camera_interface,
            camera_intrinsics=front_camera_intrinsics,
            display_str="front",
        )

        # read wrist image, detect april tag, and show result
        tag_wrist_pose = detect_april_tag(
            april_detector=april_detector,
            camera_interface=wrist_camera_interface,
            camera_intrinsics=wrist_camera_intrinsics,
            display_str="wrist",
        )

        # if both detections are non-empty, compare the poses
        if (tag_front_pose is not None) and (tag_wrist_pose is not None):
            # # maybe invert april tag pose
            # tag_front_pose = pose_inv(tag_front_pose)
            # tag_wrist_pose = pose_inv(tag_wrist_pose)

            print("tag pos in front cam")
            print(tag_front_pose[:3, 3])

            # convert pose detections to base frame
            tag_front_pose = pose_in_camera_to_pose_in_base(
                pose_in_camera=tag_front_pose,
                camera_pose_in_base=camera_extrinsics["front"],
            )
            # wrist camera pose is with respect to eef frame, so convert to base frame
            eef_pose_in_base = get_robot_eef_pose(robot_interface)
            print("eef pos")
            print(eef_pose_in_base[:3, 3])

            wrist_camera_pose_in_eef = camera_extrinsics["wrist"]
            wrist_camera_pose_in_base = pose_in_A_to_pose_in_B(
                pose_in_A=wrist_camera_pose_in_eef,
                pose_A_in_B=eef_pose_in_base,
            )
            tag_wrist_pose = pose_in_camera_to_pose_in_base(
                pose_in_camera=tag_wrist_pose,
                camera_pose_in_base=wrist_camera_pose_in_base,
            )
            print("compare pose!")
            compare_pose(tag_front_pose, tag_wrist_pose)
