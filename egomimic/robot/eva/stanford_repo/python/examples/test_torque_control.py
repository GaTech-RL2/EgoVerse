import time

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
import arx5_interface as arx5
import click
import numpy as np


def easeInOutQuad(t):
    t *= 2
    if t < 1:
        return t * t / 2
    else:
        t -= 1
        return -(t * (t - 2) - 1) / 2


@click.command()
@click.argument("model")  # ARX arm model: X5 or L5
@click.argument("interface")  # can bus name (can0 etc.)
@click.option("--urdf_path", "-u", default="../models/arx5.urdf", help="URDF file path")
def main(model: str, interface: str, urdf_path: str):

    # To initialize robot with different configurations,
    # you can create RobotConfig and ControllerConfig by yourself and modify based on it
    robot_config = arx5.RobotConfigFactory.get_instance().get_config(model)
    controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
        "joint_controller", robot_config.joint_dof
    )
    controller_config.gravity_compensation = False
    
    # Modify the default configuration here
    # controller_config.controller_dt = 0.01 # etc.

    USE_MULTITHREADING = True
    if USE_MULTITHREADING:
        # Will create another thread that communicates with the arm, so each send_recv_once() will take no time
        # for the main thread to execute. Otherwise (without background send/recv), send_recv_once() will block the
        # main thread until the arm responds (usually 2ms).
        controller_config.background_send_recv = True
    else:
        controller_config.background_send_recv = False

    arx5_joint_controller = arx5.Arx5JointController(
        robot_config, controller_config, interface
    )

    # Or you can directly use the model and interface name
    # arx5_joint_controller = arx5.Arx5JointController(model, interface)

    np.set_printoptions(precision=3, suppress=True)
    arx5_joint_controller.set_log_level(arx5.LogLevel.INFO)
    robot_config = arx5_joint_controller.get_robot_config()


    arx5_joint_controller.reset_to_home()
    gain = arx5_joint_controller.get_gain()
    gain.kd()[:] *= 0.0
    gain.kp()[:] *= 0.0
    arx5_joint_controller.set_gain(gain) 
    
    target_joint_poses = np.array([1.0, 2.0, 2.0, 1.5, 1.5, -1.57])

    step_num = 1500
    try:
        while True:
            cmd = arx5.JointState(robot_config.joint_dof)
            # i = 0
            # cmd.torque()[:] = np.array([1.0, 1.0, 1.0, 0.2, 0.2, 0.0])
            cmd.torque()[0] = 5.0
            arx5_joint_controller.set_joint_cmd(cmd)
            if not USE_MULTITHREADING:
                arx5_joint_controller.send_recv_once()
            else:
                time.sleep(controller_config.controller_dt)
            joint_state = arx5_joint_controller.get_joint_state()
            arm_dof_pos = joint_state.pos().copy()
            arm_dof_vel = joint_state.vel().copy()
            arm_dof_torque = joint_state.torque().copy()

            joint_cmd = arx5_joint_controller.get_joint_cmd()


            print(f"{arm_dof_torque[0]:.05f}, {joint_cmd.torque()[0]:.05f}")

    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Resetting arms to home position...")
        arx5_joint_controller.reset_to_home()
        print("Arms reset to home position. Exiting.")

main()
