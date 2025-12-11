import argparse
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

from urdf_models.urdf_models import URDFModel

EPS = np.finfo(float).eps * 4.0


def quat2mat(quaternion):
    """
    Converts given quaternion to matrix.
    Args:
        quaternion (np.array): (x,y,z,w) vec4 float angles
    Returns:
        np.array: 3x3 rotation matrix
    """
    # awkward semantics for use with numba
    inds = np.array([3, 0, 1, 2])
    q = np.asarray(quaternion).copy().astype(np.float32)[inds]

    n = np.dot(q, q)
    if n < EPS:
        return np.identity(3)
    q *= math.sqrt(2.0 / n)
    q2 = np.outer(q, q)
    return np.array(
        [
            [1.0 - q2[2, 2] - q2[3, 3], q2[1, 2] - q2[3, 0], q2[1, 3] + q2[2, 0]],
            [q2[1, 2] + q2[3, 0], 1.0 - q2[1, 1] - q2[3, 3], q2[2, 3] - q2[1, 0]],
            [q2[1, 3] - q2[2, 0], q2[2, 3] + q2[1, 0], 1.0 - q2[1, 1] - q2[2, 2]],
        ]
    )


def multiple(T1, T2):
    return HomogeneousMatrix(R=T1.R @ T2.R, pos=T1.R @ T2.pos + T1.pos)


class HomogeneousMatrix:
    def __init__(self, R: np.ndarray, pos: np.ndarray):
        self.R = R
        self.pos = np.array(pos)

    def __mul__(self, other):
        return multiple(self, other)

    def __rmul__(self, other):
        return multiple(self, other)


def inverse(T):
    # print(T.R.shape, np.array(T.pos).shape)
    return HomogeneousMatrix(R=T.R.transpose(), pos=-T.R.transpose() @ np.array(T.pos))


def compute_transform(root, dst, camera_extrinsics_json_file_name, joints_json_file_name):
    joints_json_file_name = os.path.join(root, joints_json_file_name)
    joint_info = json.load(open(joints_json_file_name, "r"))

    joints_list = joint_info["joints"]

    urdf_model = URDFModel()

    # camera_extrinsics_json_file_name = os.path.join(root, '../', "camera_2_rs_extrinsics.json")
    camera_extrinsics = json.load(open(camera_extrinsics_json_file_name, "r"))

    pos_camera2gripper = camera_extrinsics["translation"]
    rot_camera2gripper = camera_extrinsics["rotation"]

    identity_matrix_3x3 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    T_camera_color_optical2gripper = HomogeneousMatrix(
        R=rot_camera2gripper, pos=pos_camera2gripper
    )
    T_camera_color_optical2camera_color = HomogeneousMatrix(
        R=quat2mat(np.array([1.0, 0.0, 0.0, 0.0])), pos=[[0.0], [0.0], [0.0]]
    )
    T_camera_color2camera_link = HomogeneousMatrix(
        R=identity_matrix_3x3, pos=[[0.0], [0.0], [0.0]]
    )

    T_camera_depth_optical2camera_depth = HomogeneousMatrix(
        R=quat2mat(np.array([1.0, 0.0, 0.0, 0.0])), pos=[[0.0], [0.0], [0.0]]
    )
    T_camera_depth2camera_link = HomogeneousMatrix(
        R=identity_matrix_3x3, pos=[[0.0], [0.0], [0.0]]
    )

    # joints_lisappend([1.1425278949410889, 0.32150903321160856, -0.35035042289544266, -2.5733937344200335, 0.1988833357269323, 2.8731215853529757, 1.3560868128561296])

    camera2base_info = {"camera2base": []}
    for joints in joints_list:
        pose = urdf_model.get_gripper_pose(joints)[:2]
        pos = np.array(pose[0])[:, np.newaxis]
        rot = quat2mat(pose[1])

        T_gripper2base = HomogeneousMatrix(R=rot, pos=pos)

        T_camera_color_optical2base = T_gripper2base * T_camera_color_optical2gripper

        T_camera_depth_optical2camera_color_optical = (
            inverse(T_camera_color_optical2camera_color)
            * inverse(T_camera_color2camera_link)
            * T_camera_depth2camera_link
            * T_camera_depth_optical2camera_depth
        )

        T_camera_depth_optical2base = (
            T_camera_color_optical2base * T_camera_depth_optical2camera_color_optical
        )
        # print(T_gripper2base.pos)
        # print("-------------------------------------------------")
        # print(T_camera_color_optical2base.pos)
        # print("-------------------------------------------------")
        # print(T_camera_depth_optical2camera_color_optical.pos)
        # print("-------------------------------------------------")
        # print(T_camera_depth_optical2base.pos)

        # This is what you want:
        print(T_camera_depth_optical2base.R, T_camera_depth_optical2base.pos)

        matrix = np.zeros((4, 4))
        matrix[:3, :3] = T_camera_depth_optical2base.R
        matrix[:3, 3:] = T_camera_depth_optical2base.pos
        matrix[3, 3] = 1.0
        camera2base_info["camera2base"].append(matrix.tolist())
        print("=================================================")

    with open(dst, "w") as f:
        json.dump(camera2base_info, f)


def plot_3d_point_cloud(
    x,
    y,
    z,
    show=True,
    show_axis=True,
    in_u_sphere=False,
    marker=".",
    s=8,
    alpha=0.2,
    figsize=(5, 5),
    elev=10,
    azim=240,
    axis=None,
    title=None,
    lim=None,
    *args,
    **kwargs,
):

    if axis is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    else:
        ax = axis
        fig = axis

    if title is not None:
        plt.title(title)

    sc = ax.scatter(x, y, z, marker=marker, s=s, alpha=alpha, *args, **kwargs)
    ax.view_init(elev=elev, azim=azim)

    if lim:
        ax.set_xlim3d(*lim[0])
        ax.set_ylim3d(*lim[1])
        ax.set_zlim3d(*lim[2])
    elif in_u_sphere:
        ax.set_xlim3d(-1.0, 1.0)
        ax.set_ylim3d(-1.0, 1.0)
        ax.set_zlim3d(-1.0, 1.0)
    else:
        lim = (min(np.min(x), np.min(y), np.min(z)), max(np.max(x), np.max(y), np.max(z)))
        ax.set_xlim(1.3 * lim[0], 0.8 * lim[1])
        ax.set_ylim(1.3 * lim[0], 0.8 * lim[1])
        ax.set_zlim(1.3 * lim[0], 0.8 * lim[1])

        plt.tight_layout()

    if not show_axis:
        plt.axis("off")

    # if show:
    #     plt.show()

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Path to input directory")
    parser.add_argument(
        "-e",
        "--extrinsic",
        type=str,
        help="Path to extrinsic file",
        default="./real_results/camera_2_rs_1105_extrinsics.json",
    )
    parser.add_argument("--img-dir", type=str, help="name of image dir", default="imgs_1")
    parser.add_argument(
        "--joints", type=str, help="name of image dir", default="multiview_joints.json"
    )
    parser.add_argument(
        "--config-folder",
        type=str,
        default=os.path.expanduser("~/.rpl_vision_utils/calibration"),
    )

    args = parser.parse_args()
    # compute and save camera2base
    compute_transform(
        args.input, os.path.join(args.input, "camera2base.json"), args.extrinsic, args.joints
    )

    root = args.input

    with open(os.path.join(root, "camera2base.json")) as f:
        transform_dict = json.load(f)

    src_img_dir = os.path.join(root, args.img_dir)

    transforms = transform_dict["camera2base"]

    with open(os.path.join(args.config_folder, "camera_2_rs.json"), "r") as f:
        intrinsics_config = json.load(f)

    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=640,
        height=480,
        fx=intrinsics_config["color"]["fx"],
        fy=intrinsics_config["color"]["fy"],
        cx=intrinsics_config["color"]["cx"],
        cy=intrinsics_config["color"]["cy"],
    )
    # read data
    src_data = []
    for idx in range(len(transforms)):
        color_img = np.array(Image.open(os.path.join(src_img_dir, f"{idx}_color.png")))

        img = np.array(Image.open(os.path.join(src_img_dir, f"{idx}_depth.png")))
        img = img.astype(np.float32) * 0.001
        src_data.append((color_img, img, np.array(transforms[idx])))

    # back projection

    src_pc_list = []
    for (i, (color_img, img, trans)) in enumerate(src_data):
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_img),
            # o3d.geometry.Image(np.empty_like(img)),
            o3d.geometry.Image(img),
            depth_scale=1.0,
            depth_trunc=1.1,
            convert_rgb_to_intensity=False,
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
        pcd.transform(trans)
        src_pc_list.append(pcd)
    o3d.visualization.draw_geometries(src_pc_list)

    #    front_vec = np.array([0.5, 1.0, 0.5])
    #    o3d.visualization.draw_geometries(src_pc_list)
    # for view_idx in range(len(src_data)):
    #    src_pc_list = []
    #    print(f"Currently view: {view_idx}")
    #    for (i, (color_img, img, trans)) in enumerate(src_data):
    #        if view_idx < i:
    #            continue
    #        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    #            o3d.geometry.Image(color_img),
    #            # o3d.geometry.Image(np.empty_like(img)),
    #            o3d.geometry.Image(img),
    #            depth_scale=1.0,
    #            depth_trunc=1.1,
    #            convert_rgb_to_intensity=False,
    #        )
    #        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
    #        pcd.transform(trans)
    #        src_pc_list.append(pcd)

    #    front_vec = np.array([0.5, 1.0, 0.5])
    #    o3d.visualization.draw_geometries(src_pc_list)

    exit()
    if 1:
        # Die
        pc = np.asarray(pcd.points)
        pc_idx = np.random.randint(0, pc.shape[0], size=(8192,))
        pc = pc[pc_idx]
        pc_tmp = np.pad(pc, ((0, 0), (0, 1)), constant_values=1)
        pc_trans = trans.dot(pc_tmp.T)[:3].T
        pc_trans = pc_trans[pc_trans[:, 2] > 0.05]
        src_pc_list.append(pc_trans)

    src_fused_pc = np.concatenate(src_pc_list, axis=0)
    # normalize point cloud
    center = (np.min(src_fused_pc, 0) + np.max(src_fused_pc, 0)) / 2
    scale = (np.max(src_fused_pc, 0) - np.min(src_fused_pc, 0)).max()
    scale *= 1.1

    src_normed_pc_list = []
    for new_pc in src_pc_list:
        new_pc_normed = (new_pc - center) / scale
        src_normed_pc_list.append(new_pc_normed)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    for pc in src_normed_pc_list:
        plot_3d_point_cloud(*pc.T, axis=ax, azim=0, elev=0)
    plt.show()
