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
    controller = arx5.Arx5JointController(model, interface)
    controller.set_log_level(arx5.LogLevel.DEBUG)
    controller.reset_to_home()
    joint_waypoints = np.array(
        [[1.0, 2.0, 2.0, 1.5, 1.5, -1.57], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    )
    waypoint_interval_s = 5.0

    # Single waypoint scheduling
    for waypoint in joint_waypoints:
        joint_state = controller.get_joint_state()
        joint_cmd = arx5.JointState(waypoint, np.zeros(6), np.zeros(6), 0.0)
        joint_cmd.timestamp = joint_state.timestamp + waypoint_interval_s
        controller.set_joint_cmd(joint_cmd)
        time.sleep(waypoint_interval_s)

    # Sending joint trajectory
    joint_traj = []
    init_timestamp = controller.get_joint_state().timestamp
    for waypoint in joint_waypoints:
        joint_traj.append(arx5.JointState(waypoint, np.zeros(6), np.zeros(6), 0.0))
        joint_traj[-1].timestamp = init_timestamp + waypoint_interval_s * len(
            joint_traj
        )
    controller.set_joint_traj(joint_traj)
    time.sleep(waypoint_interval_s * len(joint_traj))

    time.sleep(1.0)
    controller.reset_to_home()


if __name__ == "__main__":
    main()
