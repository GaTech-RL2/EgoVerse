#include "app/cartesian_controller.h"
#include "app/common.h"
#include <chrono>
#include <csignal>

using namespace arx;

Arx5CartesianController *arx5_cartesian_controller = new Arx5CartesianController("L5", "can0");

void signal_handler(int signal)
{
    std::cout << "SIGINT received" << std::endl;
    delete arx5_cartesian_controller;
    exit(signal);
}

int main()
{
    EEFState cmd;
    int loop_cnt = 0;
    RobotConfig robot_config = arx5_cartesian_controller->get_robot_config();
    int dof = robot_config.joint_dof;
    Gain gain{dof};
    Arx5Solver solver("../models/arx5.urdf", dof, robot_config.joint_pos_min, robot_config.joint_pos_max);
    arx5_cartesian_controller->reset_to_home();
    arx5_cartesian_controller->set_log_level(spdlog::level::debug);
    gain.kd = (arx5_cartesian_controller->get_controller_config()).default_kd / 10;
    std::signal(SIGINT, signal_handler);
    arx5_cartesian_controller->set_gain(gain);
    double current_time = arx5_cartesian_controller->get_timestamp();
    cmd.pose_6d = arx5_cartesian_controller->get_home_pose();
    cmd.timestamp = current_time + 0.1;
    arx5_cartesian_controller->set_eef_cmd(cmd);
    while (true)
    {
        EEFState eef_state = arx5_cartesian_controller->get_eef_state();
        JointState joint_state = arx5_cartesian_controller->get_joint_state();
        loop_cnt++;
        // printf("raw x: %.2f, y: %.2f, z: %.2f, r: %.2f, p: %.2f, y: %.2f, gripper: %.2f\n", eef_state.pose_6d[0],
        // eef_state.pose_6d[1], eef_state.pose_6d[2], eef_state.pose_6d[3], eef_state.pose_6d[4], eef_state.pose_6d[5],
        // eef_state.gripper_pos); printf("raw joint pos: %.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n",
        //        joint_state.pos[0], joint_state.pos[1], joint_state.pos[2],
        //        joint_state.pos[3], joint_state.pos[4], joint_state.pos[5]);
        std::tuple<int, VecDoF> result = solver.inverse_kinematics(eef_state.pose_6d, joint_state.pos);
        if (std::get<0>(result))
        {
            VecDoF ik_joint_pos = std::get<1>(result);
            // printf("ik  joint pos: %.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n",
            //        ik_joint_pos[0], ik_joint_pos[1], ik_joint_pos[2], ik_joint_pos[3],
            //        ik_joint_pos[4], ik_joint_pos[5]);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return 0;
}
