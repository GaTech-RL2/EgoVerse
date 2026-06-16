#include "app/cartesian_controller.h"
#include "app/common.h"
#include "app/config.h"
#include "utils.h"
#include <stdexcept>
#include <sys/syscall.h>
#include <sys/types.h>

using namespace arx;

Arx5CartesianController::Arx5CartesianController(RobotConfig robot_config, ControllerConfig controller_config,
                                                 std::string interface_name)
    : Arx5ControllerBase(robot_config, controller_config, interface_name)
{
    if (!controller_config.background_send_recv)
        throw std::runtime_error(
            "controller_config.background_send_recv should be set to true when running cartesian controller.");
}

Arx5CartesianController::Arx5CartesianController(std::string model, std::string interface_name)
    : Arx5CartesianController::Arx5CartesianController(
          RobotConfigFactory::get_instance().get_config(model),
          ControllerConfigFactory::get_instance().get_config(
              "cartesian_controller", RobotConfigFactory::get_instance().get_config(model).joint_dof),
          interface_name)
{
}

void Arx5CartesianController::set_eef_cmd(EEFState new_cmd)
{
    JointState current_joint_state = get_joint_state();

    // The following line only works under c++17
    // auto [success, target_joint_pos] = solver_->inverse_kinematics(new_cmd.pose_6d, current_joint_state.pos);

    std::tuple<int, VecDoF> ik_results;
    ik_results = multi_trial_ik(new_cmd.pose_6d, joint_state_.pos);
    int ik_status = std::get<0>(ik_results);

    if (new_cmd.timestamp == 0)
        new_cmd.timestamp = get_timestamp() + controller_config_.default_preview_time;

    JointState target_joint_state{robot_config_.joint_dof};
    target_joint_state.pos = std::get<1>(ik_results);
    target_joint_state.gripper_pos = new_cmd.gripper_pos;
    target_joint_state.gripper_vel = new_cmd.gripper_vel;
    target_joint_state.gripper_torque = new_cmd.gripper_torque;
    target_joint_state.timestamp = new_cmd.timestamp;

    double current_time = get_timestamp();
    // TODO: include velocity
    std::lock_guard<std::mutex> guard(cmd_mutex_);
    interpolator_.override_waypoint(get_timestamp(), target_joint_state);

    if (ik_status != 0)
    {
        logger_->warn("Inverse kinematics failed: {} ({})", solver_->get_ik_status_name(ik_status), ik_status);
    }
}

void Arx5CartesianController::set_eef_traj(std::vector<EEFState> new_traj)
{
    double start_time = get_timestamp();
    std::vector<JointState> joint_traj;
    double avg_window_s = 0.05;
    joint_traj.push_back(interpolator_.interpolate(start_time - 2 * avg_window_s));
    joint_traj.push_back(interpolator_.interpolate(start_time - avg_window_s));
    joint_traj.push_back(interpolator_.interpolate(start_time));

    double prev_timestamp = 0;
    for (auto eef_state : new_traj)
    {
        if (eef_state.timestamp <= start_time)
            continue;
        if (eef_state.timestamp == 0)
            throw std::invalid_argument("EEFState timestamp must be set for all waypoints");
        if (eef_state.timestamp <= prev_timestamp)
            throw std::invalid_argument("EEFState timestamps must be in ascending order");
        JointState current_joint_state = get_joint_state();
        std::tuple<int, VecDoF> ik_results;
        ik_results = multi_trial_ik(eef_state.pose_6d, current_joint_state.pos);
        int ik_status = std::get<0>(ik_results);

        JointState target_joint_state{robot_config_.joint_dof};
        target_joint_state.pos = std::get<1>(ik_results);
        target_joint_state.gripper_pos = eef_state.gripper_pos;
        target_joint_state.gripper_vel = eef_state.gripper_vel;
        target_joint_state.gripper_torque = eef_state.gripper_torque;
        target_joint_state.timestamp = eef_state.timestamp;

        joint_traj.push_back(target_joint_state);
        prev_timestamp = eef_state.timestamp;

        if (ik_status != 0)
        {
            logger_->warn("Inverse kinematics failed: {} ({})", solver_->get_ik_status_name(ik_status), ik_status);
        }
    }

    double ik_end_time = get_timestamp();

    // Include velocity: first and last point based on current state, others based on neighboring points
    calc_joint_vel(joint_traj, avg_window_s);

    double current_time = get_timestamp();
    std::lock_guard<std::mutex> guard(cmd_mutex_);

    interpolator_.override_traj(current_time, joint_traj);

    double end_override_traj_time = get_timestamp();
    // logger_->debug("IK time: {:.3f}ms, calc vel time: {:.3f}ms, override_traj time: {:.3f}ms",
    //                (ik_end_time - start_time) * 1000, (current_time - ik_end_time) * 1000,
    //                (end_override_traj_time - ik_end_time) * 1000);
}

EEFState Arx5CartesianController::get_eef_cmd()
{
    JointState joint_cmd = get_joint_cmd();
    EEFState eef_cmd;
    eef_cmd.pose_6d = solver_->forward_kinematics(joint_cmd.pos);
    eef_cmd.gripper_pos = joint_cmd.gripper_pos;
    eef_cmd.gripper_vel = joint_cmd.gripper_vel;
    eef_cmd.gripper_torque = joint_cmd.gripper_torque;
    eef_cmd.timestamp = joint_cmd.timestamp;
    return eef_cmd;
}

std::tuple<int, Eigen::VectorXd> Arx5CartesianController::multi_trial_ik(Eigen::Matrix<double, 6, 1> target_pose_6d,
                                                                         Eigen::VectorXd current_joint_pos,
                                                                         int additional_trial_num)
{
    if (additional_trial_num < 0)
        throw std::invalid_argument("Number of additional trials must be non-negative");
    // Solve IK with at least 2 init joint positions: current joint position and home joint position
    if (current_joint_pos.size() != robot_config_.joint_dof || target_pose_6d.size() != 6)
        throw std::invalid_argument(
            "Inverse kinematics input expected size 6, " + std::to_string(robot_config_.joint_dof) + " but got " +
            std::to_string(target_pose_6d.size()) + ", " + std::to_string(current_joint_pos.size()));
    Eigen::MatrixXd init_joint_positions = Eigen::MatrixXd::Zero(additional_trial_num + 2, robot_config_.joint_dof);
    init_joint_positions.row(0) = current_joint_pos;
    init_joint_positions.row(1) = Eigen::VectorXd::Zero(robot_config_.joint_dof);
    for (int i = 0; i < additional_trial_num; i++)
    {
        init_joint_positions.row(i + 2) = Eigen::VectorXd::Random(robot_config_.joint_dof);
        // Map the random values into the joint limits
        for (int j = 0; j < robot_config_.joint_dof; j++)
        {
            init_joint_positions(i + 2, j) =
                robot_config_.joint_pos_min[j] + (init_joint_positions(i + 2, j) + 1) / 2 *
                                                     (robot_config_.joint_pos_max[j] - robot_config_.joint_pos_min[j]);
        }
    }
    Eigen::MatrixXd target_joint_positions = Eigen::MatrixXd::Zero(additional_trial_num + 2, robot_config_.joint_dof);
    std::vector<int> all_ik_status(additional_trial_num + 2, 0);
    std::vector<double> distances(additional_trial_num + 2, 100000); // L2 distances, initialize to infinity

    for (int i = 0; i < additional_trial_num + 2; i++)
    {
        std::tuple<int, Eigen::VectorXd> result;
        result = solver_->inverse_kinematics(target_pose_6d, init_joint_positions.row(i));
        int ik_status = std::get<0>(result);
        Eigen::VectorXd target_joint_pos = std::get<1>(result);
        bool in_joint_limit = ((robot_config_.joint_pos_max - target_joint_pos).array() > 0).all() &&
                              ((robot_config_.joint_pos_min - target_joint_pos).array() < 0).all();
        if (ik_status == 0 && !in_joint_limit)
        {
            ik_status = -9; // E_EXCEED_JOINT_LIMIT
        }
        all_ik_status[i] = ik_status;
        // Check whether the target joint position is within the joint limits
        distances[i] = (target_joint_pos - current_joint_pos).norm();
        target_joint_positions.row(i) = target_joint_pos;
    }
    bool final_success = std::any_of(all_ik_status.begin(), all_ik_status.end(), [](bool success) { return success; });
    int min_idx = std::distance(distances.begin(), std::min_element(distances.begin(), distances.end()));
    int min_ik_status = all_ik_status[min_idx];
    Eigen::VectorXd min_target_joint_pos = target_joint_positions.row(min_idx);
    // clip the target joint position to the joint limits
    for (int i = 0; i < robot_config_.joint_dof; i++)
    {
        min_target_joint_pos[i] =
            std::max(robot_config_.joint_pos_min[i], std::min(robot_config_.joint_pos_max[i], min_target_joint_pos[i]));
    }
    return std::make_tuple(min_ik_status, min_target_joint_pos);
}
