/********************************************************************************
Copyright (c) 2015, TRACLabs, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.

    3. Neither the name of the copyright holder nor the names of its contributors
       may be used to endorse or promote products derived from this software
       without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************************/

#include "nlopt_ik.hpp"
#include <limits>
#include <boost/date_time.hpp>
#include <boost/math/tools/precision.hpp>
#include "dual_quaternion.h"
#include <cmath>


namespace NLOPT_IK {

    dual_quaternion targetDQ;

    double minfunc(const std::vector<double> &x, std::vector<double> &grad, void *data) {
        // Auxilory function to minimize (Sum of Squared joint angle error
        // from the requested configuration).  Because we wanted a Class
        // without static members, but NLOpt library does not support
        // passing methods of Classes, we use these auxilary functions.

        NLOPT_IK *c = (NLOPT_IK *) data;

        return c->minJoints(x, grad);
    }

    double minfuncDQ(const std::vector<double> &x, std::vector<double> &grad, void *data) {
        // Auxilory function to minimize (Sum of Squared joint angle error
        // from the requested configuration).  Because we wanted a Class
        // without static members, but NLOpt library does not support
        // passing methods of Classes, we use these auxilary functions.
        NLOPT_IK *c = (NLOPT_IK *) data;

        std::vector<double> vals(x);

        double jump = boost::math::tools::epsilon<float>();
        double result[1];
        c->cartDQError(vals, result);

        if (!grad.empty()) {
            double v1[1];
            for (unsigned int i = 0; i < x.size(); i++) {
                double original = vals[i];

                vals[i] = original + jump;
                c->cartDQError(vals, v1);

                vals[i] = original;
                grad[i] = (v1[0] - result[0]) / (2 * jump);
            }
        }

        return result[0];
    }


    double minfuncSumSquared(const std::vector<double> &x, std::vector<double> &grad, void *data) {
        // Auxilory function to minimize (Sum of Squared joint angle error
        // from the requested configuration).  Because we wanted a Class
        // without static members, but NLOpt library does not support
        // passing methods of Classes, we use these auxilary functions.

        NLOPT_IK *c = (NLOPT_IK *) data;

        std::vector<double> vals(x);

        double jump = boost::math::tools::epsilon<float>();
        double result[1];
        c->cartSumSquaredError(vals, result);

        if (!grad.empty()) {
            double v1[1];
            for (unsigned int i = 0; i < x.size(); i++) {
                double original = vals[i];

                vals[i] = original + jump;
                c->cartSumSquaredError(vals, v1);

                vals[i] = original;
                grad[i] = (v1[0] - result[0]) / (2.0 * jump);
            }
        }

        return result[0];
    }


    double minfuncL2(const std::vector<double> &x, std::vector<double> &grad, void *data) {
        // Auxilory function to minimize (Sum of Squared joint angle error
        // from the requested configuration).  Because we wanted a Class
        // without static members, but NLOpt library does not support
        // passing methods of Classes, we use these auxilary functions.

        NLOPT_IK *c = (NLOPT_IK *) data;

        std::vector<double> vals(x);

        double jump = boost::math::tools::epsilon<float>();
        double result[1];
        c->cartL2NormError(vals, result);

        if (!grad.empty()) {
            double v1[1];
            for (unsigned int i = 0; i < x.size(); i++) {
                double original = vals[i];

                vals[i] = original + jump;
                c->cartL2NormError(vals, v1);

                vals[i] = original;
                grad[i] = (v1[0] - result[0]) / (2.0 * jump);
            }
        }

        return result[0];
    }


    void constrainfuncm(unsigned int m, double *result, unsigned int n, const double *x, double *grad, void *data) {
        //Equality constraint auxilary function for Euclidean distance .
        //This also uses a small walk to approximate the gradient of the
        //constraint function at the current joint angles.

        NLOPT_IK *c = (NLOPT_IK *) data;

        std::vector<double> vals(n);

        for (unsigned int i = 0; i < n; i++) {
            vals[i] = x[i];
        }

        double jump = boost::math::tools::epsilon<float>();

        c->cartSumSquaredError(vals, result);

        if (grad != NULL) {
            std::vector<double> v1(m);
            for (unsigned int i = 0; i < n; i++) {
                double o = vals[i];
                vals[i] = o + jump;
                c->cartSumSquaredError(vals, v1.data());
                vals[i] = o;
                for (unsigned int j = 0; j < m; j++) {
                    grad[j * n + i] = (v1[j] - result[j]) / (2 * jump);
                }
            }
        }
    }


    NLOPT_IK::NLOPT_IK(const KDL::Chain &_chain, const KDL::JntArray &_q_min, const KDL::JntArray &_q_max,
                       double _maxtime, double _eps, OptType _type) :
            chain(_chain), fksolver(chain), maxtime(_maxtime), eps(std::abs(_eps)), TYPE(_type) {
        assert(chain.getNrOfJoints() == _q_min.data.size());
        assert(chain.getNrOfJoints() == _q_max.data.size());

        //Constructor for an IK Class.  Takes in a Chain to operate on,
        //the min and max joint limits, an (optional) maximum number of
        //iterations, and an (optional) desired error.
        reset();

        if (chain.getNrOfJoints() < 2) {
            std::cout << "WARNING:" << "NLOpt_IK can only be run for chains of length 2 or more";
            return;
        }
        opt = nlopt::opt(nlopt::LD_SLSQP, _chain.getNrOfJoints());

        for (unsigned int i = 0; i < chain.getNrOfJoints(); i++) {
            lb.push_back(_q_min(i));
            ub.push_back(_q_max(i));
        }

        for (unsigned int i = 0; i < chain.segments.size(); i++) {
            std::string type = chain.segments[i].getJoint().getTypeName();
            if (type.find("Rot") != std::string::npos) {
                if (_q_max(types.size()) >= std::numeric_limits<float>::max() &&
                    _q_min(types.size()) <= std::numeric_limits<float>::lowest())
                    types.push_back(KDL::BasicJointType::Continuous);
                else
                    types.push_back(KDL::BasicJointType::RotJoint);
            } else if (type.find("Trans") != std::string::npos)
                types.push_back(KDL::BasicJointType::TransJoint);
        }

        assert(types.size() == lb.size());

        std::vector<double> tolerance(1, boost::math::tools::epsilon<float>());
        opt.set_xtol_abs(tolerance[0]);


        switch (TYPE) {
            case Joint:
                opt.set_min_objective(minfunc, this);
                opt.add_equality_mconstraint(constrainfuncm, this, tolerance);
                break;
            case DualQuat:
                opt.set_min_objective(minfuncDQ, this);
                break;
            case SumSq:
                opt.set_min_objective(minfuncSumSquared, this);
                break;
            case L2:
                opt.set_min_objective(minfuncL2, this);
                break;
        }
    }


    double NLOPT_IK::minJoints(const std::vector<double> &x, std::vector<double> &grad) {
        // Actual function to compute the error between the current joint
        // configuration and the desired.  The SSE is easy to provide a
        // closed form gradient for.

        bool gradient = !grad.empty();

        double err = 0;
        for (unsigned int i = 0; i < x.size(); i++) {
            err += pow(x[i] - des[i], 2);
            if (gradient)
                grad[i] = 2.0 * (x[i] - des[i]);
        }

        return err;

    }


    void NLOPT_IK::cartSumSquaredError(const std::vector<double> &x, double error[]) {
        // Actual function to compute Euclidean distance error.  This uses
        // the KDL Forward Kinematics solver to compute the Cartesian pose
        // of the current joint configuration and compares that to the
        // desired Cartesian pose for the IK solve.

        if (aborted || progress != -3) {
            opt.force_stop();
            return;
        }


        KDL::JntArray q(x.size());

        for (unsigned int i = 0; i < x.size(); i++)
            q(i) = x[i];

        int rc = fksolver.JntToCart(q, currentPose);

        if (rc < 0)
            std::cout << "KDL FKSolver is failing: " << q.data << std::endl;

        if (std::isnan(currentPose.p.x())) {
            std::cout << "NaNs from NLOpt!!" << std::endl;
            error[0] = std::numeric_limits<float>::max();
            progress = -1;
            return;
        }

        KDL::Twist delta_twist = KDL::diffRelative(targetPose, currentPose);

        for (int i = 0; i < 6; i++) {
            if (std::abs(delta_twist[i]) <= std::abs(bounds[i]))
                delta_twist[i] = 0.0;
        }

        error[0] = KDL::dot(delta_twist.vel, delta_twist.vel) + KDL::dot(delta_twist.rot, delta_twist.rot);

        if (KDL::Equal(delta_twist, KDL::Twist::Zero(), eps)) {
            progress = 1;
            best_x = x;
            return;
        }
    }


    void NLOPT_IK::cartL2NormError(const std::vector<double> &x, double error[]) {
        // Actual function to compute Euclidean distance error.  This uses
        // the KDL Forward Kinematics solver to compute the Cartesian pose
        // of the current joint configuration and compares that to the
        // desired Cartesian pose for the IK solve.

        if (aborted || progress != -3) {
            opt.force_stop();
            return;
        }

        KDL::JntArray q(x.size());

        for (unsigned int i = 0; i < x.size(); i++)
            q(i) = x[i];

        int rc = fksolver.JntToCart(q, currentPose);

        if (rc < 0)
            std::cout << "KDL FKSolver is failing: " << q.data << std::endl;


        if (std::isnan(currentPose.p.x())) {
            std::cout << "NaNs from NLOpt!!" << std::endl;
            error[0] = std::numeric_limits<float>::max();
            progress = -1;
            return;
        }

        KDL::Twist delta_twist = KDL::diffRelative(targetPose, currentPose);

        for (int i = 0; i < 6; i++) {
            if (std::abs(delta_twist[i]) <= std::abs(bounds[i]))
                delta_twist[i] = 0.0;
        }

        error[0] = std::sqrt(KDL::dot(delta_twist.vel, delta_twist.vel) + KDL::dot(delta_twist.rot, delta_twist.rot));

        if (KDL::Equal(delta_twist, KDL::Twist::Zero(), eps)) {
            progress = 1;
            best_x = x;
            return;
        }
    }


    void NLOPT_IK::cartDQError(const std::vector<double> &x, double error[]) {
        // Actual function to compute Euclidean distance error.  This uses
        // the KDL Forward Kinematics solver to compute the Cartesian pose
        // of the current joint configuration and compares that to the
        // desired Cartesian pose for the IK solve.

        if (aborted || progress != -3) {
            opt.force_stop();
            return;
        }

        KDL::JntArray q(x.size());

        for (unsigned int i = 0; i < x.size(); i++)
            q(i) = x[i];

        int rc = fksolver.JntToCart(q, currentPose);

        if (rc < 0)
            std::cout << "KDL FKSolver is failing: " << q.data << std::endl;


        if (std::isnan(currentPose.p.x())) {
            std::cout << "ERROR" << "NaNs from NLOpt!!";
            error[0] = std::numeric_limits<float>::max();
            progress = -1;
            return;
        }

        KDL::Twist delta_twist = KDL::diffRelative(targetPose, currentPose);

        for (int i = 0; i < 6; i++) {
            if (std::abs(delta_twist[i]) <= std::abs(bounds[i]))
                delta_twist[i] = 0.0;
        }

        math3d::matrix3x3<double> currentRotationMatrix(currentPose.M.data);
        math3d::quaternion<double> currentQuaternion = math3d::rot_matrix_to_quaternion<double>(currentRotationMatrix);
        math3d::point3d currentTranslation(currentPose.p.data);
        dual_quaternion currentDQ = dual_quaternion::rigid_transformation(currentQuaternion, currentTranslation);

        dual_quaternion errorDQ = (currentDQ * !targetDQ).normalize();
        errorDQ.log();
        error[0] = 4.0f * dot(errorDQ, errorDQ);


        if (KDL::Equal(delta_twist, KDL::Twist::Zero(), eps)) {
            progress = 1;
            best_x = x;
            return;
        }
    }


    int NLOPT_IK::CartToJnt(const KDL::JntArray &q_init, const KDL::Frame &p_in, KDL::JntArray &q_out,
                            const KDL::Twist _bounds, const KDL::JntArray &q_desired) {
        // User command to start an IK solve.  Takes in a seed
        // configuration, a Cartesian pose, and (optional) a desired
        // configuration.  If the desired is not provided, the seed is
        // used.  Outputs the joint configuration found that solves the
        // IK.

        // Returns -3 if a configuration could not be found within the eps
        // set up in the constructor.

        boost::posix_time::ptime start_time = boost::posix_time::microsec_clock::local_time();
        boost::posix_time::time_duration diff;

        bounds = _bounds;
        q_out = q_init;

        if (chain.getNrOfJoints() < 2) {
            std::cout << "ERROR" << "NLOpt_IK can only be run for chains of length 2 or more" << std::endl;
            return -3;
        }

        if (q_init.data.size() != types.size()) {
            std::cout << "ERROR" << "IK seeded with wrong number of joints.  Expected %d but got %d"
                      << (int) types.size() << (int) q_init.data.size() << std::endl;
            return -3;
        }

        opt.set_maxtime(maxtime);


        double minf; /* the minimum objective value, upon return */

        targetPose = p_in;

        if (TYPE == 1)   // DQ
        {
            math3d::matrix3x3<double> targetRotationMatrix(targetPose.M.data);
            math3d::quaternion<double> targetQuaternion = math3d::rot_matrix_to_quaternion<double>(
                    targetRotationMatrix);
            math3d::point3d targetTranslation(targetPose.p.data);
            targetDQ = dual_quaternion::rigid_transformation(targetQuaternion, targetTranslation);
        }
        // else if (TYPE == 1)
        // {
        //   z_target = targetPose*z_up;
        //   x_target = targetPose*x_out;
        //   y_target = targetPose*y_out;
        // }


        //    fksolver.JntToCart(q_init,currentPose);

        std::vector<double> x(chain.getNrOfJoints());

        for (unsigned int i = 0; i < x.size(); i++) {
            x[i] = q_init(i);

            if (types[i] == KDL::BasicJointType::Continuous)
                continue;

            if (types[i] == KDL::BasicJointType::TransJoint) {
                x[i] = std::min(x[i], ub[i]);
                x[i] = std::max(x[i], lb[i]);
            } else {

                // Below is to handle bad seeds outside of limits

                if (x[i] > ub[i]) {
                    //Find actual angle offset
                    double diffangle = fmod(x[i] - ub[i], 2 * M_PI);
                    // Add that to upper bound and go back a full rotation
                    x[i] = ub[i] + diffangle - 2 * M_PI;
                }

                if (x[i] < lb[i]) {
                    //Find actual angle offset
                    double diffangle = fmod(lb[i] - x[i], 2 * M_PI);
                    // Subtract that from lower bound and go forward a full rotation
                    x[i] = lb[i] - diffangle + 2 * M_PI;
                }

                if (x[i] > ub[i])
                    x[i] = (ub[i] + lb[i]) / 2.0;
            }
        }

        best_x = x;
        progress = -3;

        std::vector<double> artificial_lower_limits(lb.size());

        for (unsigned int i = 0; i < lb.size(); i++)
            if (types[i] == KDL::BasicJointType::Continuous)
                artificial_lower_limits[i] = best_x[i] - 2 * M_PI;
            else if (types[i] == KDL::BasicJointType::TransJoint)
                artificial_lower_limits[i] = lb[i];
            else
                artificial_lower_limits[i] = std::max(lb[i], best_x[i] - 2 * M_PI);

        opt.set_lower_bounds(artificial_lower_limits);

        std::vector<double> artificial_upper_limits(lb.size());

        for (unsigned int i = 0; i < ub.size(); i++)
            if (types[i] == KDL::BasicJointType::Continuous)
                artificial_upper_limits[i] = best_x[i] + 2 * M_PI;
            else if (types[i] == KDL::BasicJointType::TransJoint)
                artificial_upper_limits[i] = ub[i];
            else
                artificial_upper_limits[i] = std::min(ub[i], best_x[i] + 2 * M_PI);

        opt.set_upper_bounds(artificial_upper_limits);

        if (q_desired.data.size() == 0) {
            des = x;
        } else {
            des.resize(x.size());
            for (unsigned int i = 0; i < des.size(); i++)
                des[i] = q_desired(i);
        }

        try {
            opt.optimize(x, minf);
        }
        catch (...) {
        }

        if (progress == -1) // Got NaNs
            progress = -3;


        if (!aborted && progress < 0) {

            double time_left;
            diff = boost::posix_time::microsec_clock::local_time() - start_time;
            time_left = maxtime - diff.total_nanoseconds() / 1000000000.0;

            while (time_left > 0 && !aborted && progress < 0) {

                for (unsigned int i = 0; i < x.size(); i++)
                    x[i] = fRand(artificial_lower_limits[i], artificial_upper_limits[i]);

                opt.set_maxtime(time_left);

                try {
                    opt.optimize(x, minf);
                }
                catch (...) {}

                if (progress == -1) // Got NaNs
                    progress = -3;

                diff = boost::posix_time::microsec_clock::local_time() - start_time;
                time_left = maxtime - diff.total_nanoseconds() / 1000000000.0;
            }
        }


        for (unsigned int i = 0; i < x.size(); i++) {
            q_out(i) = best_x[i];
        }

        return progress;

    }


}
