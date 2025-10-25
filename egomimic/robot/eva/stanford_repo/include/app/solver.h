#ifndef SOLVER_H
#define SOLVER_H

#include <Eigen/Core>
#include <fstream>
#include <kdl/chain.hpp>
#include <kdl/chainfksolver.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainidsolver_recursive_newton_euler.hpp>
#include <kdl/chainiksolverpos_lma.hpp>
#include <kdl/kdl.hpp>
#include <kdl/segment.hpp>
#include <kdl/solveri.hpp>
#include <kdl/tree.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vector>

#include <unordered_map>

namespace arx
{

class Arx5Solver
{

  public:
    Arx5Solver(std::string urdf_path, int joint_dof, Eigen::VectorXd joint_pos_min, Eigen::VectorXd joint_pos_max);
    Arx5Solver(std::string urdf_path, int joint_dof, Eigen::VectorXd joint_pos_min, Eigen::VectorXd joint_pos_max,
               std::string base_link, std::string eef_link, Eigen::Vector3d gravity_vector);
    ~Arx5Solver() = default;

    Eigen::VectorXd inverse_dynamics(Eigen::VectorXd joint_pos, Eigen::VectorXd joint_vel, Eigen::VectorXd joint_acc);

    // Inverse kinematics will return now ik_status based on the definition in `<kdl/solveri.hpp>`
    // ik_status=0 means no error
    // Call `get_ik_status_name(ik_status)` to get the error message
    std::tuple<int, Eigen::VectorXd> multi_trial_ik(Eigen::Matrix<double, 6, 1> target_pose_6d,
                                                    Eigen::VectorXd current_joint_pos, int additional_trial_num = 5);
    // Inverse kinematics will return now ik_status based on the definition in `<kdl/solveri.hpp>`
    // ik_status=0 means no error
    // Call `get_ik_status_name(ik_status)` to get the error message
    std::tuple<int, Eigen::VectorXd> inverse_kinematics(Eigen::Matrix<double, 6, 1> target_pose_6d,
                                                        Eigen::VectorXd current_joint_pos);
    std::string get_ik_status_name(int ik_status);
    Eigen::Matrix<double, 6, 1> forward_kinematics(Eigen::VectorXd joint_pos);

    enum
    {
        E_EXCEED_JOITN_LIMIT = -9,
    };

  private:
    // parameters for ik solver
    const double EPS_ = 1E-4;
    const int MAXITER_ = 50;
    const double EPS_JOINTS_ = 1E-10;
    const int JOINT_DOF_;

    const Eigen::VectorXd JOINT_POS_MIN_;
    const Eigen::VectorXd JOINT_POS_MAX_;

    const std::unordered_map<int, std::string> IK_STATUS_MAP_ = {
        {KDL::SolverI::E_DEGRADED, "E_DEGRADED"},
        {KDL::SolverI::E_NOERROR, "E_NOERROR"},
        {KDL::SolverI::E_NO_CONVERGE, "E_NO_CONVERGE"},
        {KDL::SolverI::E_UNDEFINED, "E_UNDEFINED"},
        {KDL::SolverI::E_NOT_UP_TO_DATE, "E_NOT_UP_TO_DATE"},
        {KDL::SolverI::E_SIZE_MISMATCH, "E_SIZE_MISMATCH"},
        {KDL::SolverI::E_MAX_ITERATIONS_EXCEEDED, "E_MAX_ITERATIONS_EXCEEDED"},
        {KDL::SolverI::E_OUT_OF_RANGE, "E_OUT_OF_RANGE"},
        {KDL::SolverI::E_NOT_IMPLEMENTED, "E_NOT_IMPLEMENTED"},
        {KDL::SolverI::E_SVD_FAILED, "E_SVD_FAILED"},
        {E_EXCEED_JOITN_LIMIT, "E_EXCEED_JOITN_LIMIT"}};

    // These variables should be class members and will be used when solvers are called.
    KDL::Tree tree_;
    KDL::Chain chain_;
    KDL::Chain chain_without_fixed_joints_;

    std::shared_ptr<KDL::ChainIkSolverPos_LMA> ik_solver_;
    std::shared_ptr<KDL::ChainFkSolverPos_recursive> fk_solver_;
    std::shared_ptr<KDL::ChainIdSolver_RNE> id_solver_;
};
} // namespace arx

#endif