#include "app/cartesian_controller.h"
#include "app/common.h"
#include "app/config.h"
#include "app/controller_base.h"
#include "app/joint_controller.h"
#include "hardware/arx_can.h"
#include "spdlog/spdlog.h"
#include "utils.h"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
using namespace arx;
using Pose6d = Eigen::Matrix<double, 6, 1>;
using VecDoF = Eigen::VectorXd;
PYBIND11_MODULE(arx5_interface, m)
{
    py::enum_<spdlog::level::level_enum>(m, "LogLevel")
        .value("TRACE", spdlog::level::level_enum::trace)
        .value("DEBUG", spdlog::level::level_enum::debug)
        .value("INFO", spdlog::level::level_enum::info)
        .value("WARNING", spdlog::level::level_enum::warn)
        .value("ERROR", spdlog::level::level_enum::err)
        .value("CRITICAL", spdlog::level::level_enum::critical)
        .value("OFF", spdlog::level::level_enum::off)
        .export_values();
    py::class_<JointState>(m, "JointState")
        .def(py::init<int>())
        .def(py::init<VecDoF, VecDoF, VecDoF, double>())
        .def(py::init<VecDoF, VecDoF, VecDoF, double, double, double>())
        .def_readwrite("timestamp", &JointState::timestamp)
        .def_readwrite("gripper_pos", &JointState::gripper_pos)
        .def_readwrite("gripper_vel", &JointState::gripper_vel)
        .def_readwrite("gripper_torque", &JointState::gripper_torque)
        .def("__add__", [](const JointState &self, const JointState &other)
             { return self + other; })
        .def("__mul__", [](const JointState &self, const float &scalar)
             { return self * scalar; })
        .def("pos", &JointState::get_pos_ref, py::return_value_policy::reference)
        .def("vel", &JointState::get_vel_ref, py::return_value_policy::reference)
        .def("torque", &JointState::get_torque_ref, py::return_value_policy::reference);
    py::class_<EEFState>(m, "EEFState")
        .def(py::init<>())
        .def(py::init<Pose6d, double>())
        .def(py::init<Pose6d, double, double, double>())
        .def_readwrite("timestamp", &EEFState::timestamp)
        .def_readwrite("gripper_pos", &EEFState::gripper_pos)
        .def_readwrite("gripper_vel", &EEFState::gripper_vel)
        .def_readwrite("gripper_torque", &EEFState::gripper_torque)
        .def("__add__", [](const EEFState &self, const EEFState &other)
             { return self + other; })
        .def("__mul__", [](const EEFState &self, const float &scalar)
             { return self * scalar; })
        .def("pose_6d", &EEFState::get_pose_6d_ref, py::return_value_policy::reference);
    py::class_<Gain>(m, "Gain")
        .def(py::init<int>())
        .def(py::init<VecDoF, VecDoF, double, double>())
        .def_readwrite("gripper_kp", &Gain::gripper_kp)
        .def_readwrite("gripper_kd", &Gain::gripper_kd)
        .def("__add__", [](const Gain &self, const Gain &other)
             { return self + other; })
        .def("__mul__", [](const Gain &self, const float &scalar)
             { return self * scalar; })
        .def("kp", &Gain::get_kp_ref, py::return_value_policy::reference)
        .def("kd", &Gain::get_kd_ref, py::return_value_policy::reference);
    py::class_<Arx5JointController>(m, "Arx5JointController")
        .def(py::init<const std::string &, const std::string &>())
        .def(py::init<RobotConfig, ControllerConfig, const std::string &>())
        .def("send_recv_once", &Arx5JointController::send_recv_once)
        .def("recv_once", &Arx5JointController::recv_once)
        .def("get_joint_state", &Arx5JointController::get_joint_state)
        .def("get_timestamp", &Arx5JointController::get_timestamp)
        .def("set_joint_cmd", &Arx5JointController::set_joint_cmd)
        .def("set_joint_traj", &Arx5JointController::set_joint_traj)
        .def("get_home_pose", &Arx5JointController::get_home_pose)
        .def("get_eef_state", &Arx5JointController::get_eef_state)
        .def("get_joint_cmd", &Arx5JointController::get_joint_cmd)
        .def("set_gain", &Arx5JointController::set_gain)
        .def("get_gain", &Arx5JointController::get_gain)
        .def("get_robot_config", &Arx5JointController::get_robot_config)
        .def("get_controller_config", &Arx5JointController::get_controller_config)
        .def("reset_to_home", &Arx5JointController::reset_to_home)
        .def("set_to_damping", &Arx5JointController::set_to_damping)
        .def("set_log_level", &Arx5JointController::set_log_level)
        .def("calibrate_joint", &Arx5JointController::calibrate_joint)
        .def("calibrate_gripper", &Arx5JointController::calibrate_gripper);
    // .def("start_gripper_calibration_close_phase",
    //   &Arx5JointController::start_gripper_calibration_close_phase,
    //   py::call_guard<py::gil_scoped_release>())
    //   .def("finish_gripper_calibration_open_phase",
    //   &Arx5JointController::finish_gripper_calibration_open_phase,
    //   py::call_guard<py::gil_scoped_release>());
    py::class_<Arx5CartesianController>(m, "Arx5CartesianController")
        .def(py::init<const std::string &, const std::string &>())
        .def(py::init<RobotConfig, ControllerConfig, const std::string &>())
        .def("set_eef_cmd", &Arx5CartesianController::set_eef_cmd)
        .def("set_eef_traj", &Arx5CartesianController::set_eef_traj)
        .def("get_joint_cmd", &Arx5CartesianController::get_joint_cmd)
        .def("get_eef_cmd", &Arx5CartesianController::get_eef_cmd)
        .def("get_eef_state", &Arx5CartesianController::get_eef_state)
        .def("get_joint_state", &Arx5CartesianController::get_joint_state)
        .def("get_timestamp", &Arx5CartesianController::get_timestamp)
        .def("get_home_pose", &Arx5CartesianController::get_home_pose)
        .def("set_gain", &Arx5CartesianController::set_gain)
        .def("get_gain", &Arx5CartesianController::get_gain)
        .def("set_log_level", &Arx5CartesianController::set_log_level)
        .def("get_robot_config", &Arx5CartesianController::get_robot_config)
        .def("get_controller_config", &Arx5CartesianController::get_controller_config)
        .def("reset_to_home", &Arx5CartesianController::reset_to_home)
        .def("multi_trial_ik", &Arx5CartesianController::multi_trial_ik)
        .def("set_to_damping", &Arx5CartesianController::set_to_damping);
    py::class_<Arx5Solver>(m, "Arx5Solver")
        .def(py::init<const std::string &, int, Eigen::VectorXd, Eigen::VectorXd>())
        .def(py::init<const std::string &, int, Eigen::VectorXd, Eigen::VectorXd, const std::string &,
                      const std::string &, Eigen::Vector3d>())
        .def("inverse_dynamics", &Arx5Solver::inverse_dynamics)
        .def("forward_kinematics", &Arx5Solver::forward_kinematics)
        .def("inverse_kinematics", &Arx5Solver::inverse_kinematics)
        .def("get_ik_status_name", &Arx5Solver::get_ik_status_name)
        .def("multi_trial_ik", &Arx5Solver::multi_trial_ik);
    py::class_<RobotConfig>(m, "RobotConfig")
        .def_readwrite("robot_model", &RobotConfig::robot_model)
        .def_readwrite("joint_pos_min", &RobotConfig::joint_pos_min)
        .def_readwrite("joint_pos_max", &RobotConfig::joint_pos_max)
        .def_readwrite("joint_vel_max", &RobotConfig::joint_vel_max)
        .def_readwrite("joint_torque_max", &RobotConfig::joint_torque_max)
        .def_readwrite("ee_vel_max", &RobotConfig::ee_vel_max)
        .def_readwrite("gripper_vel_max", &RobotConfig::gripper_vel_max)
        .def_readwrite("gripper_torque_max", &RobotConfig::gripper_torque_max)
        .def_readwrite("gripper_width", &RobotConfig::gripper_width)
        .def_readwrite("gripper_open_readout", &RobotConfig::gripper_open_readout)
        .def_readwrite("joint_dof", &RobotConfig::joint_dof)
        .def_readwrite("motor_id", &RobotConfig::motor_id)
        .def_readwrite("motor_type", &RobotConfig::motor_type)
        .def_readwrite("gripper_motor_id", &RobotConfig::gripper_motor_id)
        .def_readwrite("gripper_motor_type", &RobotConfig::gripper_motor_type)
        .def_readwrite("gravity_vector", &RobotConfig::gravity_vector)
        .def_readwrite("base_link_name", &RobotConfig::base_link_name)
        .def_readwrite("eef_link_name", &RobotConfig::eef_link_name)
        .def_readwrite("urdf_path", &RobotConfig::urdf_path);

    py::class_<ControllerConfig>(m, "ControllerConfig")
        .def_readwrite("controller_type", &ControllerConfig::controller_type)
        .def_readwrite("default_kp", &ControllerConfig::default_kp)
        .def_readwrite("default_kd", &ControllerConfig::default_kd)
        .def_readwrite("default_gripper_kp", &ControllerConfig::default_gripper_kp)
        .def_readwrite("default_gripper_kd", &ControllerConfig::default_gripper_kd)
        .def_readwrite("over_current_cnt_max", &ControllerConfig::over_current_cnt_max)
        .def_readwrite("background_send_recv", &ControllerConfig::background_send_recv)
        .def_readwrite("gravity_compensation", &ControllerConfig::gravity_compensation)
        .def_readwrite("shutdown_to_passive", &ControllerConfig::shutdown_to_passive)
        .def_readwrite("interpolation_method", &ControllerConfig::interpolation_method)
        .def_readwrite("default_preview_time", &ControllerConfig::default_preview_time)
        .def_readwrite("controller_dt", &ControllerConfig::controller_dt);
    py::class_<RobotConfigFactory>(m, "RobotConfigFactory")
        .def_static("get_instance", &RobotConfigFactory::get_instance, py::return_value_policy::reference)
        .def("get_config", &RobotConfigFactory::get_config);
    py::class_<ControllerConfigFactory>(m, "ControllerConfigFactory")
        .def_static("get_instance", &ControllerConfigFactory::get_instance, py::return_value_policy::reference)
        .def("get_config", &ControllerConfigFactory::get_config);
    py::enum_<MotorType>(m, "MotorType")
        .value("EC_A4310", MotorType::EC_A4310)
        .value("DM_J4310", MotorType::DM_J4310)
        .value("DM_J4340", MotorType::DM_J4340)
        .value("NONE", MotorType::NONE);
}