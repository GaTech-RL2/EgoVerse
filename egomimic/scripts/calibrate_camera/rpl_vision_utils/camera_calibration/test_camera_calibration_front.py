"""
GPRS version of testing front camera calibration

Script to test camera calibration by reading robot end effector pose and projecting it onto camera image.
Many functions are based on https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/utils/camera_utils.py
"""

import time
import cv2
import imageio
import numpy as np
import os
from PIL import Image, ImageDraw

from deoxys.franka_interface import FrankaInterface
from rpl_vision_utils.networking.camera_redis_interface import CameraRedisSubInterface
from deoxys import config_root

# camera id
from hardware_config import *


# FRONT CAMERA
def get_camera_intrinsic_matrix():
    """
    Fill out this function to put the intrinsic matrix of your camera.
    Returns:
        K (np.array): 3x3 camera matrix
    """
    K = np.array(
        [
            [606.9329833984375, 0.0, 642.0451049804688],
            [0.0, 606.7354736328125, 364.87799072265625],
            [0.0, 0.0, 1.0],
        ]
    )
    return K


def get_camera_extrinsic_matrix():
    """
    Fill out this function to put the extrinsic matrix of your camera.
    This should correspond to the camera pose in the robot base frame.
    Returns:
        R (np.array): 4x4 camera extrinsic matrix
    """
    R = np.eye(4)
    R[:3, :3] = np.array(
        [
            [0.23820846, -0.23325683, 0.94278735],
            [-0.97031034, -0.01529059, 0.24137945],
            [-0.04188763, -0.97229494, -0.22997386],
        ]
    )
    R[:3, 3] = np.array([-0.04609977, -0.42817777, 0.24079107])

    return R


# # WRIST CAMERA
# def get_camera_intrinsic_matrix():
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


# def get_camera_extrinsic_matrix():
#     """
#     Fill out this function to put the extrinsic matrix of your camera.
#     This should correspond to the camera pose in the robot base frame.

#     Returns:
#         R (np.array): 4x4 camera extrinsic matrix
#     """
#     R = np.eye(4)


#     R[:3, :3] = np.array([
#         [0.03039011, 0.6968179, -0.71660398],
#         [ 0.99723176, -0.06981243, -0.02559373],
#         [-0.06786204, -0.71384245, -0.69701055],
#     ])
#     R[:3, 3] = np.array([1.110382, 0.07760241, 0.70286399])
#     return R


def get_robot_eef_position(robot_interface):
    """
    Fill out this function to get the robot end effector position in the robot base frame.
    Returns:
        e (np.array): end effector position of shape (3,)
    """
    # e = np.zeros(3)
    # return e
    last_robot_state = robot_interface._state_buffer[-1]
    ee_pose = np.array(last_robot_state.O_T_EE).reshape((4, 4)).T
    return ee_pose[:3, 3]


def get_camera_image(camera_interface):
    """
    Fill out this function to get an RGB image from the camera.
    Returns:
        I (np.array): array of shape (H, W, 3) and type np.uint8
    """
    # I = np.zeros((480, 640, 3), dtype=np.uint8)
    # return I

    # get image
    imgs = camera_interface.get_img()
    im = imgs["color"]

    # resize image
    im = Image.fromarray(im).resize((IM_SIZE[1], IM_SIZE[0]), Image.BILINEAR)
    return np.array(im).astype(np.uint8)


def get_camera_transform_matrix():
    """
    Camera transform matrix to project from world coordinates to pixel coordinates.
    Returns:
        K (np.array): 4x4 camera matrix to project from world coordinates to pixel coordinates
    """
    R = get_camera_extrinsic_matrix()
    K = get_camera_intrinsic_matrix()
    K_exp = np.eye(4)
    K_exp[:3, :3] = K

    # Takes a point in world, transforms to camera frame, and then projects onto image plane.
    return K_exp @ pose_inv(R)


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


def project_points_from_base_to_camera(
    points, base_to_camera_transform, camera_height, camera_width
):
    """
    Helper function to project a batch of points in the base frame
    into camera pixels using the base to camera transformation.
    Args:
        points (np.array): 3D points in base frame to project onto camera pixel locations. Should
            be shape [..., 3].
        base_to_camera_transform (np.array): 4x4 Tensor to go from robot coordinates to pixel
            coordinates.
        camera_height (int): height of the camera image
        camera_width (int): width of the camera image
    Return:
        pixels (np.array): projected pixel indices of shape [..., 2]
    """
    assert points.shape[-1] == 3  # last dimension must be 3D
    assert len(base_to_camera_transform.shape) == 2
    assert (
        base_to_camera_transform.shape[0] == 4
        and base_to_camera_transform.shape[1] == 4
    )

    # convert points to homogenous coordinates -> (px, py, pz, 1)
    ones_pad = np.ones(points.shape[:-1] + (1,))
    points = np.concatenate((points, ones_pad), axis=-1)  # shape [..., 4]

    # batch matrix multiplication of 4 x 4 matrix and 4 x 1 vectors to do robot frame to pixels transform
    mat_reshape = [1] * len(points.shape[:-1]) + [4, 4]
    cam_trans = base_to_camera_transform.reshape(mat_reshape)  # shape [..., 4, 4]
    pixels = np.matmul(cam_trans, points[..., None])[..., 0]  # shape [..., 4]

    # re-scaling from homogenous coordinates to recover pixel values
    # (x, y, z) -> (x / z, y / z)
    pixels = pixels / pixels[..., 2:3]
    pixels = pixels[..., :2].round().astype(int)  # shape [..., 2]

    # swap first and second coordinates to get pixel indices that correspond to (height, width)
    # and also clip pixels that are out of range of the camera image
    pixels = np.concatenate(
        (
            pixels[..., 1:2].clip(0, camera_height - 1),
            pixels[..., 0:1].clip(0, camera_width - 1),
        ),
        axis=-1,
    )

    return pixels


if __name__ == "__main__":
    # get camera matrix (base to camera transform, or equivalently, camera pose in base frame)
    base_to_camera = get_camera_transform_matrix()
    camera_to_base = np.linalg.inv(base_to_camera)

    # connect to robot and camera
    robot_interface = FrankaInterface(
        config_root + "/charmander.yml", use_visualizer=False
    )
    camera_interface = CameraRedisSubInterface(
        redis_host=REDIS_HOST, redis_port=REDIS_PORT, camera_id=CAMERA_ID
    )
    camera_interface.start()

    # record video
    vid_path = "saved_figs"
    isExist = os.path.exists(vid_path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(vid_path)

    writer = imageio.get_writer(os.path.join(vid_path, "vid.mp4"), fps=20)

    num_pics = 400
    for i in range(num_pics):
        # input("press enter to continue")

        # get eef pos
        eef_pos_in_base = get_robot_eef_position(robot_interface)

        # transform end effector position into camera pixel
        eef_pixel = project_points_from_base_to_camera(
            points=eef_pos_in_base,
            base_to_camera_transform=base_to_camera,
            camera_height=IM_SIZE[0],
            camera_width=IM_SIZE[1],
        )

        # get image and plot red circle at the pixel location
        im = get_camera_image(camera_interface)
        # print("im: {}".format(im))
        image = Image.fromarray(im)
        draw = ImageDraw.Draw(image)
        r = 10  # size of circle
        x, y = eef_pixel[0], eef_pixel[1]
        print("iter {} h_coord: {} w_coord: {}".format(i, eef_pixel[0], eef_pixel[1]))
        y, x = x, y
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 0, 0))
        im = np.array(image).astype(np.uint8)

        # # plot image
        # cv2.imshow('test', im[:, :, ::-1])
        # # cv2.waitKey(0)
        # cv2.imwrite('saved_figs/{0}.png'.format(i), im[:, :, ::-1])
        writer.append_data(im)

        time.sleep(0.05)

    writer.close()
