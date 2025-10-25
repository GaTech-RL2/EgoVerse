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
@click.option("--urdf_path", "-u", default="../models/arx5.urdf", help="URDF file path")
def main(model: str, interface: str, urdf_path: str):

    arx5_joint_controller = arx5.Arx5JointController(model, interface)
    dof = arx5_joint_controller.get_robot_config().joint_dof
    gain = arx5.Gain(dof)
    gain.kp()[:] = 0.0
    gain.kd()[:] = 0.0
    gain.gripper_kp = 0.0
    gain.gripper_kd = 0.0
    
    arx5_joint_controller.reset_to_home()
    arx5_joint_controller.set_gain(gain)
    
    gripper_acc = 0.0
    # last_gripper_vel = 0.0
    gripper_vels = [0.0] * 5
    dt = 0.01
    gripper_width_threshold = 0.01
    acc_threshold = 0.01
    torque_to_acc = 0.06

    
    while True:
        joint_state = arx5_joint_controller.get_joint_state()
        gripper_vels.append(joint_state.gripper_vel)
        gripper_vels.pop(0)
        gripper_acc = 1/2 *(gripper_vels[-1] - gripper_vels[0]) / (dt * (len(gripper_vels) - 1)) +\
             1/2 * (gripper_vels[-2] - gripper_vels[2]) / (dt * (len(gripper_vels) - 3))
        if gripper_acc > acc_threshold:
            gripper_acc -= acc_threshold
        elif gripper_acc < -acc_threshold:
            gripper_acc += acc_threshold
        else:
            gripper_acc = 0.0

        if joint_state.gripper_pos < gripper_width_threshold:
            gripper_acc = 0.0

        print(f"gripper pos: {joint_state.gripper_pos:+.3f}, vel: {joint_state.gripper_vel:+.3f}, acc: {gripper_acc:+.3f}, torque: {joint_state.gripper_torque:+.3f}")
        joint_cmd = arx5.JointState(dof)
        
        joint_cmd.gripper_torque = torque_to_acc * gripper_acc
        arx5_joint_controller.set_joint_cmd(joint_cmd)
        time.sleep(0.01)
    
    
    
    
    
if __name__ == "__main__":
    main()
