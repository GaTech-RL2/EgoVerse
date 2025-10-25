#ifndef CONTROLLER_BASE_H
#define CONTROLLER_BASE_H
#include "app/common.h"
#include "app/config.h"
#include "app/solver.h"
#include "hardware/arx_can.h"
#include "utils.h"
#include <chrono>
#include <memory>
#include <mutex>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <stdlib.h>
#include <thread>
#include <unistd.h>

namespace arx
{
class Arx5ControllerBase // parent class for the other two controllers
{
  public:
    Arx5ControllerBase(RobotConfig robot_config, ControllerConfig controller_config, std::string interface_name);
    ~Arx5ControllerBase();
    JointState get_joint_cmd();
    JointState get_joint_state();
    EEFState get_eef_state();
    Pose6d get_home_pose();
    void set_gain(Gain new_gain);
    Gain get_gain();

    double get_timestamp();
    RobotConfig get_robot_config();
    ControllerConfig get_controller_config();
    void set_log_level(spdlog::level::level_enum level);

    void reset_to_home();
    void set_to_damping();

  protected:
    RobotConfig robot_config_;
    ControllerConfig controller_config_;

    int over_current_cnt_ = 0;
    JointState output_joint_cmd_{robot_config_.joint_dof};

    JointState joint_state_{robot_config_.joint_dof};
    Gain gain_{robot_config_.joint_dof};
    // bool prev_gripper_updated_ = false; // Declaring here leads to segfault

    ArxCan can_handle_;
    std::shared_ptr<spdlog::logger> logger_;
    std::thread background_send_recv_thread_;

    bool prev_gripper_updated_ = false; // To suppress the warning message
    bool background_send_recv_running_ = false;
    bool destroy_background_threads_ = false;

    std::mutex cmd_mutex_;
    std::mutex state_mutex_;

    long int start_time_us_;
    std::shared_ptr<Arx5Solver> solver_;
    JointStateInterpolator interpolator_{robot_config_.joint_dof, controller_config_.interpolation_method};
    void init_robot_();
    void update_joint_state_();
    void update_output_cmd_();
    void send_recv_();
    void recv_();
    void check_joint_state_sanity_();
    void over_current_protection_();
    void background_send_recv_();
    void enter_emergency_state_();
};
} // namespace arx

#endif