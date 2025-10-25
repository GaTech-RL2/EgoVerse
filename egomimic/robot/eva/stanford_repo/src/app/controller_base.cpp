#include "app/controller_base.h"
#include "app/common.h"
#include "utils.h"
#include <array>
#include <stdexcept>
#include <sys/syscall.h>
#include <sys/types.h>
using namespace arx;

Arx5ControllerBase::Arx5ControllerBase(RobotConfig robot_config, ControllerConfig controller_config,
                                       std::string interface_name)
    : can_handle_(interface_name),
      logger_(spdlog::stdout_color_mt(robot_config.robot_model + std::string("_") + interface_name)),
      robot_config_(robot_config), controller_config_(controller_config)
{
    start_time_us_ = get_time_us();
    logger_->set_pattern("[%H:%M:%S %n %^%l%$] %v");
    solver_ = std::make_shared<Arx5Solver>(
        robot_config_.urdf_path, robot_config_.joint_dof, robot_config_.joint_pos_min, robot_config_.joint_pos_max,
        robot_config_.base_link_name, robot_config_.eef_link_name, robot_config_.gravity_vector);
    if (robot_config_.robot_model == "X5" && !controller_config_.shutdown_to_passive)
    {
        logger_->warn("When shutting down X5 robot arms, the motors have to be set to passive. "
                      "controller_config_.shutdown_to_passive is set to `true`");
        controller_config_.shutdown_to_passive = true;
    }
    init_robot_();
    background_send_recv_thread_ = std::thread(&Arx5ControllerBase::background_send_recv_, this);
    background_send_recv_running_ = controller_config_.background_send_recv;
    logger_->info("Background send_recv task is running at ID: {}", syscall(SYS_gettid));
}

Arx5ControllerBase::~Arx5ControllerBase()
{
    if (controller_config_.shutdown_to_passive)
    {
        logger_->info("Set to damping before exit");
        Gain damping_gain{robot_config_.joint_dof};
        damping_gain.kd = controller_config_.default_kd;

        // Increase damping if needed
        // damping_gain.kd[0] *= 3;
        // damping_gain.kd[1] *= 3;
        // damping_gain.kd[2] *= 3;
        // damping_gain.kd[3] *= 1.5;

        set_gain(damping_gain);
        {
            std::lock_guard<std::mutex> guard(cmd_mutex_);
            output_joint_cmd_.vel = VecDoF::Zero(robot_config_.joint_dof);
            output_joint_cmd_.torque = VecDoF::Zero(robot_config_.joint_dof);
            interpolator_.init_fixed(output_joint_cmd_);
        }
        background_send_recv_running_ = true;
        controller_config_.gravity_compensation = false;
        sleep_ms(2000);
    }
    else
    {
        logger_->info("Disconnect motors without setting to damping");
    }

    destroy_background_threads_ = true;
    background_send_recv_thread_.join();
    logger_->info("background send_recv task joined");
    spdlog::drop(logger_->name());
    logger_.reset();
    solver_.reset();
}

JointState Arx5ControllerBase::get_joint_cmd()
{
    std::lock_guard<std::mutex> guard(cmd_mutex_);
    return output_joint_cmd_;
}

JointState Arx5ControllerBase::get_joint_state()
{
    std::lock_guard<std::mutex> guard(state_mutex_);
    return joint_state_;
}

EEFState Arx5ControllerBase::get_eef_state()
{
    EEFState eef_state;
    JointState joint_state = get_joint_state();
    Pose6d tool_pose = solver_->forward_kinematics(joint_state.pos);
    eef_state.pose_6d = tool_pose;
    eef_state.timestamp = joint_state.timestamp;
    eef_state.gripper_pos = joint_state.gripper_pos;
    eef_state.gripper_vel = joint_state.gripper_vel;
    eef_state.gripper_torque = joint_state.gripper_torque;
    return eef_state;
}

void Arx5ControllerBase::set_gain(Gain new_gain)
{
    // Make sure the robot doesn't jump when setting kp to non-zero
    if (gain_.kp.isZero() && !new_gain.kp.isZero())
    {
        JointState joint_state = get_joint_state();
        JointState joint_cmd = get_joint_cmd();
        double max_pos_error = (joint_state.pos - joint_cmd.pos).cwiseAbs().maxCoeff();
        double pos_error_threshold = 0.2;
        double kp_threshold = 1;
        if (max_pos_error > pos_error_threshold && new_gain.kp.maxCoeff() > kp_threshold)
        {
            logger_->error("Cannot set kp too large when the joint pos cmd is far from current pos.");
            logger_->error(
                "Target max kp: {}, kp threshold: {}. Current pos: {}, cmd pos: {}, position error threshold: {}",
                new_gain.kp.maxCoeff(), kp_threshold, vec2str(joint_state.pos), vec2str(joint_cmd.pos),
                pos_error_threshold);
            background_send_recv_running_ = false;
            throw std::runtime_error("Cannot set kp to non-zero when the joint pos cmd is far from current pos.");
        }
    }
    {
        std::lock_guard<std::mutex> guard(cmd_mutex_);
        gain_ = new_gain;
    }
}

Gain Arx5ControllerBase::get_gain()
{
    std::lock_guard<std::mutex> guard(cmd_mutex_);
    return gain_;
}

double Arx5ControllerBase::get_timestamp()
{
    return double(get_time_us() - start_time_us_) / 1e6;
}
RobotConfig Arx5ControllerBase::get_robot_config()
{
    return robot_config_;
}
ControllerConfig Arx5ControllerBase::get_controller_config()
{
    return controller_config_;
}
void Arx5ControllerBase::set_log_level(spdlog::level::level_enum level)
{
    logger_->set_level(level);
}

void Arx5ControllerBase::reset_to_home()
{
    JointState init_state = get_joint_state();
    Gain init_gain = get_gain();
    double init_gripper_kp = gain_.gripper_kp;
    double init_gripper_kd = gain_.gripper_kd;
    Gain target_gain{robot_config_.joint_dof};
    if (init_gain.kp.isZero())
    {
        logger_->info("Current kp is zero. Setting to default kp kd");
        target_gain = Gain(controller_config_.default_kp, controller_config_.default_kd,
                           controller_config_.default_gripper_kp, controller_config_.default_gripper_kd);
    }
    else
    {
        target_gain = init_gain;
    }

    JointState target_state{robot_config_.joint_dof};

    // calculate the maximum joint position error
    double max_pos_error = (init_state.pos - VecDoF::Zero(robot_config_.joint_dof)).cwiseAbs().maxCoeff();
    max_pos_error = std::max(max_pos_error, init_state.gripper_pos * 2 / robot_config_.gripper_width);
    // interpolate from current kp kd to default kp kd in max(max_pos_error, 0.5)s
    // and keep the target for max(max_pos_error, 0.5)s
    double wait_time = std::max(max_pos_error, 0.5);
    int step_num = int(wait_time / controller_config_.controller_dt);
    logger_->info("Start reset to home in {:.3f}s, max_pos_error: {:.3f}", std::max(max_pos_error, double(0.5)) + 0.5,
                  max_pos_error);

    bool prev_running = background_send_recv_running_;
    background_send_recv_running_ = true;
    target_state.timestamp = get_timestamp() + wait_time;
    target_state.pos[2] = 0.03; // avoiding clash

    {
        std::lock_guard<std::mutex> guard(cmd_mutex_);
        JointState start_state{robot_config_.joint_dof};
        start_state.pos = init_state.pos;
        start_state.gripper_pos = init_state.gripper_pos;
        start_state.timestamp = get_timestamp();
        interpolator_.init(start_state, target_state);
    }
    Gain new_gain{robot_config_.joint_dof};
    for (int i = 0; i <= step_num; i++)
    {
        double alpha = double(i) / step_num;
        new_gain = init_gain * (1 - alpha) + target_gain * alpha;
        set_gain(new_gain);
        sleep_us(int(controller_config_.controller_dt * 1e6));
    }

    target_state.pos[2] = 0.0;
    target_state.timestamp = get_timestamp() + 0.5;
    {
        std::lock_guard<std::mutex> guard(cmd_mutex_);
        interpolator_.override_waypoint(get_timestamp(), target_state);
    }
    logger_->info("Finish reset to home");
    background_send_recv_running_ = prev_running;
}

void Arx5ControllerBase::set_to_damping()
{
    Gain damping_gain{robot_config_.joint_dof};
    damping_gain.kd = controller_config_.default_kd;
    set_gain(damping_gain);
    sleep_ms(10);
    JointState joint_state = get_joint_state();
    {
        std::lock_guard<std::mutex> guard(cmd_mutex_);
        joint_state.vel = VecDoF::Zero(robot_config_.joint_dof);
        joint_state.torque = VecDoF::Zero(robot_config_.joint_dof);
        interpolator_.init_fixed(joint_state);
    }
}

// ---------------------- Private functions ----------------------

void Arx5ControllerBase::init_robot_()
{
    // Background send receive is disabled during initialization
    int init_rounds = 10; // Make sure the states of each motor is fully initialized

    for (int j = 0; j < init_rounds; j++)
    {
        recv_();
        check_joint_state_sanity_();
        over_current_protection_();
        // logger_->info("pos: {}", vec2str(joint_state_.pos));
    }

    Gain gain{robot_config_.joint_dof};
    gain.kd = controller_config_.default_kd;

    JointState init_joint_state = get_joint_state();
    init_joint_state.vel = VecDoF::Zero(robot_config_.joint_dof);
    init_joint_state.torque = VecDoF::Zero(robot_config_.joint_dof);
    set_gain(gain); // set to damping by default

    // std::lock_guard<std::mutex> guard(state_mutex_);
    // Check whether any motor has non-zero position
    if (joint_state_.pos == VecDoF::Zero(robot_config_.joint_dof))
    {
        logger_->error("None of the motors are initialized. Please check the connection or power of the arm.");
        throw std::runtime_error(
            "None of the motors are initialized. Please check the connection or power of the arm.");
    }
    {
        std::lock_guard<std::mutex> guard(cmd_mutex_);
        output_joint_cmd_ = init_joint_state;
        interpolator_.init_fixed(init_joint_state);
    }

    for (int j = 0; j < init_rounds; j++)
    {
        send_recv_();
        check_joint_state_sanity_();
        over_current_protection_();
    }
}

void Arx5ControllerBase::check_joint_state_sanity_()
{
    std::lock_guard<std::mutex> guard(state_mutex_);

    for (int i = 0; i < robot_config_.joint_dof; ++i)
    {
        if (std::abs(joint_state_.pos[i]) > robot_config_.joint_pos_max[i] + 3.14 ||
            std::abs(joint_state_.pos[i]) < robot_config_.joint_pos_min[i] - 3.14)
        {
            logger_->error("Joint {} pos data error: {:.3f}. Please restart the program.", i, joint_state_.pos[i]);
            enter_emergency_state_();
        }

        if (interpolator_.is_initialized())
        {
            std::lock_guard<std::mutex> guard(cmd_mutex_);
            JointState interpolator_cmd = interpolator_.interpolate(get_timestamp());
            if (std::abs(interpolator_cmd.pos[i]) > robot_config_.joint_pos_max[i] + 3.14 ||
                std::abs(interpolator_cmd.pos[i]) < robot_config_.joint_pos_min[i] - 3.14)
            {
                logger_->error("Joint {} interpolated command data error: {:.3f}. Please restart the program.", i,
                               interpolator_cmd.pos[i]);
                enter_emergency_state_();
            }
        }
        if (std::abs(joint_state_.torque[i]) > 100 * robot_config_.joint_torque_max[i])
        {
            logger_->error("Joint {} torque data error: {:.3f}. Please restart the program.", i,
                           joint_state_.torque[i]);
            enter_emergency_state_();
        }
    }
    // Gripper should be around 0~robot_config_.gripper_width
    double gripper_width_tolerance = 0.01; // m
    if (joint_state_.gripper_pos < -0.023 ||
        joint_state_.gripper_pos > robot_config_.gripper_width + gripper_width_tolerance)
    {
        logger_->error("Gripper position error: got {:.3f} but should be in 0~{:.3f} (m). Please close the gripper "
                       "before turning the arm on or recalibrate gripper home and width.",
                       joint_state_.gripper_pos, robot_config_.gripper_width);

        // enter_emergency_state_();
    }
}

void Arx5ControllerBase::over_current_protection_()
{
    bool over_current = false;
    for (int i = 0; i < robot_config_.joint_dof; ++i)
    {
        if (std::abs(joint_state_.torque[i]) > robot_config_.joint_torque_max[i])
        {
            over_current = true;
            logger_->error("Over current detected once on joint {}, current: {:.3f}", i, joint_state_.torque[i]);
            break;
        }
    }
    if (std::abs(joint_state_.gripper_torque) > robot_config_.gripper_torque_max)
    {
        over_current = true;
        logger_->error("Over current detected once on gripper, current: {:.3f}", joint_state_.gripper_torque);
    }
    if (over_current)
    {
        over_current_cnt_++;
        if (over_current_cnt_ > controller_config_.over_current_cnt_max)
        {
            logger_->error("Over current detected, robot is set to damping. Please restart the "
                           "program.");
            enter_emergency_state_();
        }
    }
    else
    {
        over_current_cnt_ = 0;
    }
}

void Arx5ControllerBase::enter_emergency_state_()
{
    Gain damping_gain{robot_config_.joint_dof};
    damping_gain.kd = controller_config_.default_kd;
    damping_gain.kd[1] *= 3;
    damping_gain.kd[2] *= 3;
    damping_gain.kd[3] *= 1.5;
    logger_->error("Emergency state entered. Please restart the program.");
    while (true)
    {
        std::lock_guard<std::mutex> guard(cmd_mutex_);
        set_gain(damping_gain);
        output_joint_cmd_.vel = VecDoF::Zero(robot_config_.joint_dof);
        output_joint_cmd_.torque = VecDoF::Zero(robot_config_.joint_dof);

        interpolator_.init_fixed(output_joint_cmd_);
        send_recv_();
        sleep_ms(5);
    }
}

void Arx5ControllerBase::update_joint_state_()
{
    // TODO: in the motor documentation, there shouldn't be these torque constants. Torque will go directly into the
    // motors
    const double torque_constant_EC_A4310 = 1.4; // Nm/A
    const double torque_constant_DM_J4310 = 0.424;
    const double torque_constant_DM_J4340 = 1.0;
    std::array<OD_Motor_Msg, 10> motor_msg = can_handle_.get_motor_msg();
    std::lock_guard<std::mutex> guard(state_mutex_);

    for (int i = 0; i < robot_config_.joint_dof; i++)
    {
        joint_state_.pos[i] = motor_msg[robot_config_.motor_id[i]].angle_actual_rad;
        joint_state_.vel[i] = motor_msg[robot_config_.motor_id[i]].speed_actual_rad;

        // Torque: matching the values (there must be something wrong)
        if (robot_config_.motor_type[i] == MotorType::EC_A4310)
        {
            joint_state_.torque[i] = motor_msg[robot_config_.motor_id[i]].current_actual_float *
                                     torque_constant_EC_A4310 * torque_constant_EC_A4310;
            // Why are there two torque_constant_EC_A4310?
        }
        else if (robot_config_.motor_type[i] == MotorType::DM_J4310)
        {
            joint_state_.torque[i] =
                motor_msg[robot_config_.motor_id[i]].current_actual_float * torque_constant_DM_J4310;
        }
        else if (robot_config_.motor_type[i] == MotorType::DM_J4340)
        {
            joint_state_.torque[i] =
                motor_msg[robot_config_.motor_id[i]].current_actual_float * torque_constant_DM_J4340;
        }
    }

    joint_state_.gripper_pos = motor_msg[robot_config_.gripper_motor_id].angle_actual_rad /
                               robot_config_.gripper_open_readout * robot_config_.gripper_width;

    joint_state_.gripper_vel = motor_msg[robot_config_.gripper_motor_id].speed_actual_rad /
                               robot_config_.gripper_open_readout * robot_config_.gripper_width;

    joint_state_.gripper_torque =
        motor_msg[robot_config_.gripper_motor_id].current_actual_float * torque_constant_DM_J4310;
    joint_state_.timestamp = get_timestamp();
}

void Arx5ControllerBase::update_output_cmd_()
{
    JointState prev_output_cmd = output_joint_cmd_;

    // TODO: deal with non-zero velocity and torque for joint control
    double timestamp = get_timestamp();
    {
        std::lock_guard<std::mutex> guard(cmd_mutex_);
        output_joint_cmd_ = interpolator_.interpolate(timestamp);
    }

    std::lock_guard<std::mutex> guard(state_mutex_);
    if (controller_config_.gravity_compensation)
    {
        output_joint_cmd_.torque += solver_->inverse_dynamics(joint_state_.pos, VecDoF::Zero(robot_config_.joint_dof),
                                                              VecDoF::Zero(robot_config_.joint_dof));
    }

    // Joint pos clipping
    for (int i = 0; i < robot_config_.joint_dof; ++i)
    {
        if (output_joint_cmd_.pos[i] < robot_config_.joint_pos_min[i])
        {
            logger_->debug("Joint {} pos {:.3f} pos cmd clipped from {:.3f} to min {:.3f}", i, joint_state_.pos[i],
                           output_joint_cmd_.pos[i], robot_config_.joint_pos_min[i]);
            output_joint_cmd_.pos[i] = robot_config_.joint_pos_min[i];
        }
        else if (output_joint_cmd_.pos[i] > robot_config_.joint_pos_max[i])
        {
            logger_->debug("Joint {} pos {:.3f} pos cmd clipped from {:.3f} to max {:.3f}", i, joint_state_.pos[i],
                           output_joint_cmd_.pos[i], robot_config_.joint_pos_max[i]);
            output_joint_cmd_.pos[i] = robot_config_.joint_pos_max[i];
        }
    }
    // Joint velocity clipping
    double dt = controller_config_.controller_dt;
    for (int i = 0; i < robot_config_.joint_dof; ++i)
    {
        if (gain_.kp[i] > 0)
        {

            double delta_pos = output_joint_cmd_.pos[i] - prev_output_cmd.pos[i];
            double max_vel = robot_config_.joint_vel_max[i];
            if (std::abs(delta_pos) > max_vel * dt)
            {
                double new_pos = prev_output_cmd.pos[i] + max_vel * dt * delta_pos / std::abs(delta_pos);
                if (new_pos > robot_config_.joint_pos_max[i])
                    new_pos = robot_config_.joint_pos_max[i];
                if (new_pos < robot_config_.joint_pos_min[i])
                    new_pos = robot_config_.joint_pos_min[i];
                logger_->debug("Joint velocity reaches limit: Joint {} pos {:.3f} pos cmd clipped: {:.3f} to {:.3f}", i,
                               joint_state_.pos[i], output_joint_cmd_.pos[i], new_pos);
                output_joint_cmd_.pos[i] = new_pos;
            }
        }
        else
        {
            output_joint_cmd_.pos[i] = joint_state_.pos[i];
        }

        // Gripper pos clipping
        if (gain_.gripper_kp > 0)
        {
            double gripper_delta_pos = output_joint_cmd_.gripper_pos - prev_output_cmd.gripper_pos;
            if (std::abs(gripper_delta_pos) / dt > robot_config_.gripper_vel_max)
            {
                double new_gripper_pos = prev_output_cmd.gripper_pos + robot_config_.gripper_vel_max * dt *
                                                                           gripper_delta_pos /
                                                                           std::abs(gripper_delta_pos);
                if (std::abs(output_joint_cmd_.gripper_pos - new_gripper_pos) >= 0.001)
                    logger_->debug("Gripper pos cmd clipped: {:.3f} to {:.3f}", output_joint_cmd_.gripper_pos,
                                   new_gripper_pos);
                output_joint_cmd_.gripper_pos = new_gripper_pos;
            }
        }
        else
        {
            output_joint_cmd_.gripper_pos = joint_state_.gripper_pos;
        }
    }

    // Gripper pos clipping
    if (output_joint_cmd_.gripper_pos < -0.02)
    {
        if (output_joint_cmd_.gripper_pos < -0.02)
            logger_->debug("Gripper pos cmd clipped from {:.3f} to min: {:.3f}", output_joint_cmd_.gripper_pos, 0.0);
        output_joint_cmd_.gripper_pos = -0.02;
    }
    else if (output_joint_cmd_.gripper_pos > robot_config_.gripper_width)
    {
        if (output_joint_cmd_.gripper_pos > robot_config_.gripper_width + 0.005)
            logger_->debug("Gripper pos cmd clipped from {:.3f} to max: {:.3f}", output_joint_cmd_.gripper_pos,
                           robot_config_.gripper_width);
        output_joint_cmd_.gripper_pos = robot_config_.gripper_width;
    }
    if (std::abs(joint_state_.gripper_torque) > robot_config_.gripper_torque_max / 2)
    {
        double sign = joint_state_.gripper_torque > 0 ? 1 : -1; // -1 for closing blocked, 1 for opening blocked
        double delta_pos =
            output_joint_cmd_.gripper_pos - prev_output_cmd.gripper_pos; // negative for closing, positive for opening
        if (delta_pos * sign > 0)
        {
            if (prev_gripper_updated_)
                logger_->warn("Gripper torque is too large, gripper pos cmd is not updated");
            output_joint_cmd_.gripper_pos = prev_output_cmd.gripper_pos;
            prev_gripper_updated_ = false;
        }
        else
            prev_gripper_updated_ = true;
    }

    // Torque clipping
    for (int i = 0; i < robot_config_.joint_dof; ++i)
    {
        if (output_joint_cmd_.torque[i] > robot_config_.joint_torque_max[i])
        {
            logger_->debug("Joint {} torque cmd clipped from {:.3f} to max {:.3f}", i, output_joint_cmd_.torque[i],
                           robot_config_.joint_torque_max[i]);
            output_joint_cmd_.torque[i] = robot_config_.joint_torque_max[i];
        }
        else if (output_joint_cmd_.torque[i] < -robot_config_.joint_torque_max[i])
        {
            logger_->debug("Joint {} torque cmd clipped from {:.3f} to min {:.3f}", i, output_joint_cmd_.torque[i],
                           -robot_config_.joint_torque_max[i]);
            output_joint_cmd_.torque[i] = -robot_config_.joint_torque_max[i];
        }
    }
    // Gripper torque clipping
    if (output_joint_cmd_.gripper_torque > robot_config_.gripper_torque_max)
    {
        logger_->debug("Gripper torque cmd clipped from {:.3f} to max {:.3f}", output_joint_cmd_.gripper_torque,
                       robot_config_.gripper_torque_max);
        output_joint_cmd_.gripper_torque = robot_config_.gripper_torque_max;
    }
    else if (output_joint_cmd_.gripper_torque < -robot_config_.gripper_torque_max)
    {
        logger_->debug("Gripper torque cmd clipped from {:.3f} to min {:.3f}", output_joint_cmd_.gripper_torque,
                       -robot_config_.gripper_torque_max);
        output_joint_cmd_.gripper_torque = -robot_config_.gripper_torque_max;
    }
}

void Arx5ControllerBase::send_recv_()
{
    // TODO: in the motor documentation, there shouldn't be these torque constants. Torque will go directly into the
    // motors
    const double torque_constant_EC_A4310 = 1.4; // Nm/A
    const double torque_constant_DM_J4310 = 0.424;
    const double torque_constant_DM_J4340 = 1.0;
    int start_time_us = get_time_us();

    update_output_cmd_();
    int update_cmd_time_us = get_time_us();
    int communicate_sleep_us = 150;

    for (int i = 0; i < robot_config_.joint_dof; i++)
    {
        int start_send_motor_time_us = get_time_us();
        {
            std::lock_guard<std::mutex> guard(cmd_mutex_);
            if (robot_config_.motor_type[i] == MotorType::EC_A4310)
            {
                can_handle_.send_EC_motor_cmd(robot_config_.motor_id[i], gain_.kp[i], gain_.kd[i],
                                              output_joint_cmd_.pos[i], output_joint_cmd_.vel[i],
                                              output_joint_cmd_.torque[i] / torque_constant_EC_A4310);
            }
            else if (robot_config_.motor_type[i] == MotorType::DM_J4310)
            {

                can_handle_.send_DM_motor_cmd(robot_config_.motor_id[i], gain_.kp[i], gain_.kd[i],
                                              output_joint_cmd_.pos[i], output_joint_cmd_.vel[i],
                                              output_joint_cmd_.torque[i] / torque_constant_DM_J4310);
            }
            else if (robot_config_.motor_type[i] == MotorType::DM_J4340)
            {
                can_handle_.send_DM_motor_cmd(robot_config_.motor_id[i], gain_.kp[i], gain_.kd[i],
                                              output_joint_cmd_.pos[i], output_joint_cmd_.vel[i],
                                              output_joint_cmd_.torque[i] / torque_constant_DM_J4340);
            }
            else
            {
                logger_->error("Motor type not supported.");
                return;
            }
        }
        int finish_send_motor_time_us = get_time_us();
        sleep_us(communicate_sleep_us - (finish_send_motor_time_us - start_send_motor_time_us));
    }

    // Send gripper command (gripper is using DM motor)
    if (robot_config_.gripper_motor_type == MotorType::DM_J4310)
    {
        int start_send_motor_time_us = get_time_us();

        double gripper_motor_pos =
            output_joint_cmd_.gripper_pos / robot_config_.gripper_width * robot_config_.gripper_open_readout;
        double gripper_motor_vel =
            output_joint_cmd_.gripper_vel / robot_config_.gripper_width * robot_config_.gripper_open_readout;
        double gripper_motor_torque = output_joint_cmd_.gripper_torque / torque_constant_DM_J4310;
        can_handle_.send_DM_motor_cmd(robot_config_.gripper_motor_id, gain_.gripper_kp, gain_.gripper_kd,
                                      gripper_motor_pos, gripper_motor_vel, gripper_motor_torque);
        int finish_send_motor_time_us = get_time_us();
        sleep_us(communicate_sleep_us - (finish_send_motor_time_us - start_send_motor_time_us));
    }

    // logger_->trace("update_cmd: {} us, send_motor_0: {} us, send_motor_1: {} us, send_motor_2: {} us, send_motor_3:
    // {} us, send_motor_4: {} us, send_motor_5: {} us, send_motor_6: {} us, get_motor_msg: {} us",
    //                update_cmd_time_us - start_time_us, send_motor_0_time_us - start_send_motor_0_time_us,
    //                send_motor_1_time_us - start_send_motor_1_time_us, send_motor_2_time_us -
    //                start_send_motor_2_time_us, send_motor_3_time_us - start_send_motor_3_time_us,
    //                send_motor_4_time_us - start_send_motor_4_time_us, send_motor_5_time_us -
    //                start_send_motor_5_time_us, send_motor_6_time_us - start_send_motor_6_time_us,
    //                get_motor_msg_time_us - start_get_motor_msg_time_us);

    update_joint_state_();
}

void Arx5ControllerBase::recv_()
{
    int communicate_sleep_us = 300;
    for (int i = 0; i < robot_config_.joint_dof; i++)
    {
        int start_send_motor_time_us = get_time_us();
        if (robot_config_.motor_type[i] == MotorType::EC_A4310)
        {
            // can_handle_.query_EC_motor_pos(robot_config_.motor_id[i]);
            // sleep_us(100);
            // can_handle_.query_EC_motor_vel(robot_config_.motor_id[i]);
            // sleep_us(100);
            // can_handle_.query_EC_motor_current(robot_config_.motor_id[i]);
        }
        else if (robot_config_.motor_type[i] == MotorType::DM_J4310 ||
                 robot_config_.motor_type[i] == MotorType::DM_J4340 ||
                 robot_config_.motor_type[i] == MotorType::DM_J8009)
        {
            can_handle_.enable_DM_motor(robot_config_.motor_id[i]);
        }
        else
        {
            logger_->error("Motor type not supported.");
            assert(false);
        }
        int finish_send_motor_time_us = get_time_us();
        sleep_us(communicate_sleep_us - (finish_send_motor_time_us - start_send_motor_time_us));
    }
    if (robot_config_.gripper_motor_type == MotorType::DM_J4310)
    {
        int start_send_motor_time_us = get_time_us();
        can_handle_.enable_DM_motor(robot_config_.gripper_motor_id);
        int finish_send_motor_time_us = get_time_us();
        sleep_us(communicate_sleep_us - (finish_send_motor_time_us - start_send_motor_time_us));
    }
    sleep_ms(1); // Wait until all the messages are updated
    update_joint_state_();
}

void Arx5ControllerBase::background_send_recv_()
{
    while (!destroy_background_threads_)
    {
        int start_time_us = get_time_us();
        if (background_send_recv_running_)
        {
            over_current_protection_();
            check_joint_state_sanity_();
            send_recv_();
        }
        int elapsed_time_us = get_time_us() - start_time_us;
        int sleep_time_us = int(controller_config_.controller_dt * 1e6) - elapsed_time_us;
        if (sleep_time_us > 0)
        {
            std::this_thread::sleep_for(std::chrono::microseconds(sleep_time_us));
        }
        else if (sleep_time_us < -500)
        {
            logger_->debug("Background send_recv task is running too slow, time: {} us", elapsed_time_us);
        }
    }
}

Pose6d Arx5ControllerBase::get_home_pose()
{
    return solver_->forward_kinematics(VecDoF::Zero(robot_config_.joint_dof));
}
