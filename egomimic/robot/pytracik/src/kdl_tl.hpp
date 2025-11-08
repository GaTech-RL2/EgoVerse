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


#ifndef KDLCHAINIKSOLVERPOS_TL_HPP
#define KDLCHAINIKSOLVERPOS_TL_HPP

#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainiksolvervel_pinv.hpp>

#ifndef M_PI
#define M_PI 3.141592538
#endif //M_PI

namespace TRAC_IK {
    class TRAC_IK;
}

namespace KDL {

    enum BasicJointType {
        RotJoint, TransJoint, Continuous
    };

    class ChainIkSolverPos_TL {
        friend class TRAC_IK::TRAC_IK;

    public:
        ChainIkSolverPos_TL(const Chain &chain, const JntArray &q_min, const JntArray &q_max, double maxtime = 0.005,
                            double eps = 1e-3, bool random_restart = false, bool try_jl_wrap = false);

        ~ChainIkSolverPos_TL();

        int CartToJnt(const KDL::JntArray &q_init, const KDL::Frame &p_in, KDL::JntArray &q_out,
                      const KDL::Twist bounds = KDL::Twist::Zero());

        inline void setMaxtime(double t) {
            maxtime = t;
        }

    private:
        const Chain chain;
        JntArray q_min;
        JntArray q_max;

        KDL::Twist bounds;

        KDL::ChainIkSolverVel_pinv vik_solver;
        KDL::ChainFkSolverPos_recursive fksolver;
        JntArray delta_q;
        double maxtime;

        double eps;

        bool rr;
        bool wrap;

        std::vector<KDL::BasicJointType> types;

        inline void abort() {
            aborted = true;
        }

        inline void reset() {
            aborted = false;
        }

        bool aborted;

        Frame f;
        Twist delta_twist;

        inline static double fRand(double min, double max) {
            double f = (double) rand() / RAND_MAX;
            return min + f * (max - min);
        }


    };

/**
 * determines the rotation axis necessary to rotate from frame b1 to the
 * orientation of frame b2 and the vector necessary to translate the origin
 * of b1 to the origin of b2, and stores the result in a Twist
 * datastructure.  The result is w.r.t. frame b1.
 * \param F_a_b1 frame b1 expressed with respect to some frame a.
 * \param F_a_b2 frame b2 expressed with respect to some frame a.
 * \warning The result is not a real Twist!
 * \warning In contrast to standard KDL diff methods, the result of
 * diffRelative is w.r.t. frame b1 instead of frame a.
 */
    IMETHOD Twist diffRelative(const Frame &F_a_b1, const Frame &F_a_b2, double dt = 1) {
        return Twist(F_a_b1.M.Inverse() * diff(F_a_b1.p, F_a_b2.p, dt),
                     F_a_b1.M.Inverse() * diff(F_a_b1.M, F_a_b2.M, dt));
    }

}

#endif
