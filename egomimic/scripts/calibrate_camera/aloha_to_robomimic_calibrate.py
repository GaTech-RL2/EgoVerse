# Example
# python3 aloha_to_robomimic_calibrate.py --dataset /home/robot/robot_ws/egomimic/robot/calibration_demos/left --arm left --extrinsics left --out /home/robot/robot_ws/egomimic/robot/calibration_demos/left_proc/test2.hdf5 --data-type robot

import h5py
import numpy as np
import argparse
import os
from tqdm import tqdm
from egomimic.utils.egomimicUtils import nds, ee_pose_to_cam_frame, EXTRINSICS, AlohaFK
import pytorch_kinematics as pk
import torch

# from modern_robotics import FKinSpace
# from robomimic.scripts.split_train_val import split_train_val_from_hdf5
import json

"""
aloha_hdf5 has the following format
dict with keys:  <KeysViewHDF5 ['action', 'observations']>
action: (500, 14)
observations: dict with keys:  <KeysViewHDF5 ['effort', 'images', 'qpos', 'qvel']>
        effort: (500, 14)
        images: dict with keys:  <KeysViewHDF5 ['cam_high', 'cam_right_wrist']>
                cam_high: (500, 480, 640, 3)
                cam_right_wrist: (500, 480, 640, 3)
        qpos: (500, 14)
        qvel: (500, 14)
"""

# def convert_qpos_to_eef(qpos):
#     M = np.array([[1.0, 0.0, 0.0, 0.536494],
#                   [0.0, 1.0, 0.0, 0.0],
#                   [0.0, 0.0, 1.0, 0.42705],
#                   [0.0, 0.0, 0.0, 1.0]])

#     Slist = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
#                       [0.0, 1.0, 0.0, -0.12705, 0.0, 0.0],
#                       [0.0, 1.0, 0.0, -0.42705, 0.0, 0.05955],
#                       [1.0, 0.0, 0.0, 0.0, 0.42705, 0.0],
#                       [0.0, 1.0, 0.0, -0.42705, 0.0, 0.35955],
#                       [1.0, 0.0, 0.0, 0.0, 0.42705, 0.0]]).T

#     T_obs = FKinSpace(M, Slist, qpos)
#     return T_obs


def get_future_points(arr, POINT_GAP=15, FUTURE_POINTS_COUNT=10):
    """
    arr: (T, ACTION_DIM)
    POINT_GAP: how many timesteps to skip
    FUTURE_POINTS_COUNT: how many future points to collect
    given an array arr, prepack the future points into each timestep.  return an array of size (T, FUTURE_POINTS_COUNT, ACTION_DIM).  If there are not enough future points, pad with the last point.
    do it purely vectorized
    """
    T, ACTION_DIM = arr.shape
    result = np.zeros((T, FUTURE_POINTS_COUNT, ACTION_DIM))

    for t in range(T):
        future_indices = np.arange(t, t + POINT_GAP * (FUTURE_POINTS_COUNT), POINT_GAP)
        future_indices = np.clip(future_indices, 0, T - 1)
        result[t] = arr[future_indices]

    return result

def is_valid_path(path):
    if os.path.isdir(path):
        return False
    if not path.endswith(".hdf5"):
        return False

    # Only treat as raw Aloha if it has the expected top-level keys
    try:
        with h5py.File(path, "r") as f:
            keys = set(f.keys())
            return "observations" in keys and "action" in keys
    except OSError:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to rawAloha folder",
    )
    parser.add_argument("--arm", type=str, help="which arm to convert data for")
    parser.add_argument("--extrinsics", type=str, help="which arm to convert data for")
    parser.add_argument(
        "--out",
        type=str,
        help="path to output dataset: /coc/flash7/datasets/oboov2/<ds_name>.hdf5",
    )
    parser.add_argument(
        "--data-type",
        type=str,
        required=True,
        choices=["hand", "robot"],  # Restrict to only 'hand' or 'robot'
        help="Choose which data-type - hand or robot",
    )
    parser.add_argument("--prestack", action="store_true")

    args = parser.parse_args()

    # chain = pk.build_serial_chain_from_urdf(
    #     open("/home/rl2-bonjour/EgoPlay/EgoPlay/egomimic/resources/model.urdf").read(),
    #     "vx300s/ee_gripper_link",
    # )

    # before converting everything, check it all at least opens
    for file in tqdm(os.listdir(args.dataset)):
        # print("Trying to open " + file)
        to_open = os.path.join(args.dataset, file)
        print(to_open)
        if is_valid_path(to_open):
            with h5py.File(to_open, "r") as f:
                pass

    with h5py.File(args.out, "w", rdcc_nbytes=1024**2 * 2) as dataset:
        data_group = dataset.create_group("data")
        data_group.attrs["env_args"] = json.dumps({})  # if no normalize obs

        for i, aloha_demo in enumerate(tqdm(os.listdir(args.dataset))):
            if not is_valid_path(os.path.join(args.dataset, aloha_demo)):
                continue

            aloha_demo_path = os.path.join(args.dataset, aloha_demo)

            with h5py.File(aloha_demo_path, "r") as aloha_hdf5:
                demo_number = aloha_demo.split("_")[1].split(".")[0]
                demo_i_group = data_group.create_group(f"demo_{demo_number}")
                demo_i_group.attrs["num_samples"] = aloha_hdf5["action"].shape[0]
                demo_i_obs_group = demo_i_group.create_group("obs")

                # Extract the data from the aloha hdf5 file
                if args.arm == "right":
                    joint_start = 7
                    joint_end = 14
                elif args.arm == "left":
                    joint_start = 0
                    joint_end = 7

                # obs
                demo_i_obs_group.create_dataset(
                    "front_img_1",
                    data=aloha_hdf5["observations"]["images"]["front_img_1"],
                    dtype="uint8",
                    chunks=(1, 480, 640, 3),
                )
                # demo_i_obs_group.create_dataset(
                #     "right_wrist_img",
                #     data=aloha_hdf5["observations"]["images"]["cam_right_wrist"],
                #     dtype="uint8",
                #     chunks=(1, 480, 640, 3),
                # )
                demo_i_obs_group.create_dataset(
                    "joint_positions",
                    data=aloha_hdf5["observations"]["joints"][:, joint_start:joint_end],
                )
                # fk_pos, fk_rot = pk.matrix_to_pos_rot(
                #     chain.forward_kinematics(
                #         torch.from_numpy(
                #             aloha_hdf5["observations"]["qpos"][
                #                 :, joint_start : joint_end - 1
                #             ]
                #         ),
                #         end_only=True,
                #     ).get_matrix()
                # )
                # fk_positions = torch.cat([fk_pos, fk_rot], dim=1)
                fk_positions = aloha_hdf5["observations"]["eepose"][:, joint_start:joint_end]

                demo_i_obs_group.create_dataset(
                    "ee_pose_robot_frame", data=fk_positions
                )

                # fk_positions = ee_pose_to_cam_frame(
                #     fk_positions, EXTRINSICS[args.extrinsics]
                # )[:, :3]

                # demo_i_obs_group.create_dataset("ee_pose", data=fk_positions)

                # breakpoint()
                if args.data_type == "hand":
                    POINT_GAP = 4
                    FUTURE_POINTS_COUNT = 10
                elif args.data_type == "robot":
                    POINT_GAP = 15
                    FUTURE_POINTS_COUNT = 10

                # actions_joints
                joint_actions = aloha_hdf5["action"][:, joint_start:joint_end]
                if args.prestack:
                    joint_actions = get_future_points(
                        joint_actions,
                        POINT_GAP=POINT_GAP,
                        FUTURE_POINTS_COUNT=FUTURE_POINTS_COUNT,
                    )
                demo_i_group.create_dataset("actions_joints", data=joint_actions)

                # actions_xyz
                # fk_positions = chain.forward_kinematics(
                #     torch.from_numpy(aloha_hdf5["action"][:, joint_start:joint_end-1]), end_only=True
                # ).get_matrix()[:, :3, 3]
                # fk_positions = ee_pose_to_cam_frame(
                #     fk_positions, EXTRINSICS[args.extrinsics]
                # )[:, :3]
                ee_pose_actions = aloha_hdf5["actions"]["eepose"]
                if args.prestack:
                    fk_positions = get_future_points(
                        fk_positions,
                        POINT_GAP=POINT_GAP,
                        FUTURE_POINTS_COUNT=FUTURE_POINTS_COUNT,
                    )
                demo_i_group.create_dataset("actions_xyz", data=ee_pose_actions)

                # print(chain.forward_kinematics(torch.from_numpy(aloha_hdf5["observations"]["qpos"][10, 7:13])[None, :], end_only=True).get_matrix()[:, :3, 3])
                # print(convert_qpos_to_eef(aloha_hdf5["observations"]["qpos"][10, 7:13]))

                # create a dataset for the end effector positions

    # with h5py.File(args.out, "r", rdcc_nbytes=1024**2*2) as dataset:
    #     breakpoint()
    #     print("hi")

    # split_train_val_from_hdf5(hdf5_path=args.out, val_ratio=0.2, filter_key=None)

    print("Successful Conversion!")
