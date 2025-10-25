import time
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
import arx5_interface as arx5
import numpy as np


joint_dof = 6
robot_config = arx5.RobotConfigFactory.get_instance().get_config("X5")
solver = arx5.Arx5Solver(
    "../models/arx5.urdf",
    # "../models/arx7_left.urdf",
    joint_dof,
    robot_config.joint_pos_min,
    robot_config.joint_pos_max,
    # "base_link",
    # "eef_link",
    # np.array([0, 0, -9.807], dtype=np.float64),
)

# solver = arx5.Arx5Solver("../models/arx5_umi.urdf", joint_dof)

# joint_dof = 7
# solver = arx5.Arx5Solver(
#     "../models/arx7_left.urdf",
#     joint_dof,
#     "base_link",
#     "link7",
#     np.array([0, 0, -9.807]),
# )

print("Done initialization")


# joint_pos = np.array([-0.000, 2.487, -0.449, 2.937, -0.000, -0.000])

joint_pos = np.zeros(joint_dof)
# joint_pos = np.zeros(joint_dof)
# joint_pos[2] = np.pi / 2
# joint_pos[3] = -np.pi / 2
joint_vel = np.zeros(joint_dof)
joint_acc = np.zeros(joint_dof)
fk_pose = solver.forward_kinematics(joint_pos)

print(f"{fk_pose=}")

home_pose = solver.forward_kinematics(joint_pos)
init_pose = home_pose + np.array([0.05, 0, 0, 0, 0, 0])
ik_status, init_joint_pos = solver.inverse_kinematics(init_pose, np.zeros(6))

print(f"{init_joint_pos=}")

ik_status, ik_joint_pos = solver.inverse_kinematics(
    home_pose,
    np.array(
        [
            -3.69121061e-06,
            8.25553071e-01,
            1.94861662e-02,
            0.5,  # 0.76
            1.47200640e-06,
            -2.36060781e-06,
        ]
    ),
)

ik_status, multi_trial_ik_joint_pos = solver.multi_trial_ik(
    home_pose,
    np.array(
        [
            -3.69121061e-06,
            8.25553071e-01,
            1.94861662e-02,
            0.5,  # 0.76
            1.47200640e-06,
            -2.36060781e-06,
        ]
    ),
    5,
)


# ik_status, ik_joint_pos = solver.inverse_kinematics(home_pose, init_joint_pos)


print(f"{ik_joint_pos=}, {multi_trial_ik_joint_pos=}")

start_time = time.monotonic()
id_torque = solver.inverse_dynamics(joint_pos, joint_vel, joint_acc)
print(f"{id_torque=}")
