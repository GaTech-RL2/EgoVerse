import time
import numpy as np
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
import arx5_interface as arx5


def easeInOutQuad(t):
    t *= 2
    if t < 1:
        return t * t / 2
    else:
        t -= 1
        return -(t * (t - 2) - 1) / 2


def main():
    np.set_printoptions(precision=3, suppress=True)
    x7_left = arx5.Arx5JointController("X7Left", "can0")
    x7_left_config = x7_left.get_robot_config()
    # x7_left = arx5.Arx5JointController("X7Left", "can1")
    # x7_left.set_log_level(arx5.LogLevel.DEBUG)
    # x7_left.reset_to_home()
    gain = x7_left.get_gain()
    gain.kp()[:] = 0
    x7_left.set_gain(gain)
    x7_left_solver = arx5.Arx5Solver(
        "../models/arx7_left.urdf",
        7,
        x7_left_config.joint_pos_min,
        x7_left_config.joint_pos_max,
    )

    while True:
        joint_state = x7_left.get_joint_state()
        ee_pose = x7_left_solver.forward_kinematics(joint_state.pos())
        print(ee_pose, joint_state.pos())
        time.sleep(0.1)

    # arx5_0 = arx5.Arx5JointController("X7Left", "can0")
    # arx5_1 = arx5.Arx5JointController("X5", "can1")
    # robot_config = arx5_0.get_robot_config()
    # controller_config = arx5_0.get_controller_config()

    # arx5_0.enable_background_send_recv()
    # arx5_0.reset_to_home()
    # arx5_0.enable_gravity_compensation("../models/arx5.urdf")
    # arx5_1.enable_background_send_recv()
    # arx5_1.reset_to_home()
    # arx5_1.enable_gravity_compensation("../models/arx5.urdf")

    # target_joint_poses = np.array([1.0, 2.0, 2.0, 1.5, 1.5, -1.57])
    # step_num = 1500  # 3s
    # USE_TIMER = True
    # if not USE_TIMER:
    #     arx5_0.disable_background_send_recv()

    # for i in range(step_num):
    #     cmd = arx5.JointState(robot_config.joint_dof)
    #     # i = 0
    #     cmd.pos()[0:4] = easeInOutQuad(float(i) / step_num) * target_joint_poses[0:4]
    #     cmd.gripper_pos = easeInOutQuad((i / (step_num - 1))) * 0.08
    #     arx5_0.set_joint_cmd(cmd)
    #     arx5_1.set_joint_cmd(cmd)
    #     if not USE_TIMER:
    #         arx5_0.send_recv_once()
    #         arx5_1.send_recv_once()
    #     JointState = arx5_0.get_state()
    #     JointState = arx5_1.get_state()
    #     arm_dof_pos = JointState.pos().copy()
    #     arm_dof_vel = JointState.vel().copy()
    #     # print(arm_dof_pos, arm_dof_vel)
    #     # print(f"gripper: {JointState.gripper_pos:.05f}")
    #     time.sleep(controller_config.controller_dt)

    # for i in range(step_num):
    #     cmd = arx5.JointState(robot_config.joint_dof)
    #     cmd.pos()[0:4] = (
    #         easeInOutQuad((1 - float(i) / step_num)) * target_joint_poses[0:4]
    #     )
    #     cmd.gripper_pos = easeInOutQuad((1 - i / (step_num - 1))) * 0.08
    #     arx5_0.set_joint_cmd(cmd)
    #     arx5_1.set_joint_cmd(cmd)
    #     if not USE_TIMER:
    #         arx5_0.send_recv_once()
    #         arx5_1.send_recv_once()
    #     time.sleep(controller_config.controller_dt)
    #     JointState = arx5_0.get_state()
    #     JointState = arx5_1.get_state()
    #     # print(f"gripper: {JointState.gripper_pos:.05f}")


main()
