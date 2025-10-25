import time

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
import arx5_interface as arx5
import click
import numpy as np


@click.command()
@click.argument("model")  # ARX arm model: X5 or L5
@click.argument("interface")  # can bus name (can0 etc.)
def main(model: str, interface: str):
    controller = arx5.Arx5CartesianController(model, interface)
    np.set_printoptions(precision=4, suppress=True)
    controller.set_log_level(arx5.LogLevel.DEBUG)
    home_pose = controller.get_home_pose()
    controller.reset_to_home()
    cartesian_waypoints = home_pose + np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.2, 0.0, 0.0, 0.0],
            [0.2, 0.0, 0.2, 0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    waypoint_interval_s = 1.0

    interpolate_interval_s = 0.1
    interpolated_waypoints = []
    for i in range(len(cartesian_waypoints) - 1):
        start = cartesian_waypoints[i]
        end = cartesian_waypoints[i + 1]
        num_interpolations = int(waypoint_interval_s / interpolate_interval_s)
        for j in range(num_interpolations):
            interpolated_waypoint = start + (end - start) * j / num_interpolations
            interpolated_waypoints.append(interpolated_waypoint)

    pose_error = np.zeros(6)
    pose_error_cnt = 0
    # Set waypoints one-by-one (only position control)
    for waypoint in interpolated_waypoints:
        eef_state = controller.get_eef_state()
        eef_cmd = arx5.EEFState(waypoint, 0.0)
        eef_cmd.timestamp = eef_state.timestamp + interpolate_interval_s
        controller.set_eef_cmd(eef_cmd)

        current_time = time.time()
        while time.time() < current_time + interpolate_interval_s:
            # You can do whatever you want here while the robot is moving
            eef_state = controller.get_eef_state()
            eef_cmd = controller.get_eef_cmd()
            # joint_vel = controller.get_joint_cmd().vel()
            # print(f"Vel cmd: {joint_vel}")
            pose_error += eef_cmd.pose_6d() - eef_state.pose_6d()
            pose_error_cnt += 1
            time.sleep(0.05)
    single_update_pose_err = pose_error / pose_error_cnt
    controller.reset_to_home()

    # Directly set a eef trajectory (with velocity automatically updated)
    eef_traj = []
    current_timestamp = controller.get_eef_state().timestamp
    for k, waypoint in enumerate(interpolated_waypoints):
        eef_cmd = arx5.EEFState(waypoint, 0.0)
        eef_cmd.timestamp = current_timestamp + interpolate_interval_s * (k + 1)
        eef_traj.append(eef_cmd)
    controller.set_eef_traj(eef_traj)
    pose_error = np.zeros(6)
    pose_error_cnt = 0
    current_time = time.time()
    while time.time() < current_time + interpolate_interval_s * len(
        interpolated_waypoints
    ):
        # You can do whatever you want here while the robot is moving
        eef_state = controller.get_eef_state()
        eef_cmd = controller.get_eef_cmd()
        # joint_vel = controller.get_joint_cmd().vel().copy()
        # print(f"Vel cmd: {joint_vel}")
        pose_error += eef_cmd.pose_6d() - eef_state.pose_6d()
        pose_error_cnt += 1
        time.sleep(0.05)
    traj_update_pose_err = pose_error / pose_error_cnt
    controller.reset_to_home()
    print(f"Single point update: average pose error {single_update_pose_err}")
    print(f"EEF trajectory update: average pose error {traj_update_pose_err}")
    # Since the input trajectory is already smooth enough, the tracking error difference is not significant


if __name__ == "__main__":
    main()
