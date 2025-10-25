import time

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
import arx5_interface as arx5
import click
import numpy as np

"""
This script is to test the solver when the gravity vector is not the default value (pointing down)
To run this script successfully, please hang the robot arm upside down, so the gravity vector becomes [0, 0, 9.81].
"""


@click.command()
@click.argument("model")  # ARX arm model: X5 or L5
@click.argument("interface")  # can bus name (can0 etc.)
def main(model: str, interface: str):

    robot_config = arx5.RobotConfigFactory.get_instance().get_config(model)
    robot_config.gravity_vector = np.array([0, 0, 9.81])
    controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
        "joint_controller", robot_config.joint_dof
    )

    arx5_joint_controller = arx5.Arx5JointController(
        robot_config, controller_config, interface
    )

    gain = arx5.Gain(robot_config.joint_dof)
    gain.kd()[:] = 0.1

    arx5_joint_controller.set_gain(gain)

    time.sleep(1000)


if __name__ == "__main__":
    main()
