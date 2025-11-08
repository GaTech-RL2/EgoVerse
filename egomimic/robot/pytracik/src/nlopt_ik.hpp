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

#ifndef NLOPT_IK_HPP
#define NLOPT_IK_HPP

#include "kdl_tl.hpp"
#include <nlopt.hpp>


namespace NLOPT_IK
{

enum OptType { Joint, DualQuat, SumSq, L2 };


class NLOPT_IK
{
  friend class TRAC_IK::TRAC_IK;
public:
  NLOPT_IK(const KDL::Chain& chain, const KDL::JntArray& q_min, const KDL::JntArray& q_max, double maxtime = 0.005, double eps = 1e-3, OptType type = SumSq);

  ~NLOPT_IK() {};
  int CartToJnt(const KDL::JntArray& q_init, const KDL::Frame& p_in, KDL::JntArray& q_out, const KDL::Twist bounds = KDL::Twist::Zero(), const KDL::JntArray& q_desired = KDL::JntArray());

  double minJoints(const std::vector<double>& x, std::vector<double>& grad);
  //  void cartFourPointError(const std::vector<double>& x, double error[]);
  void cartSumSquaredError(const std::vector<double>& x, double error[]);
  void cartDQError(const std::vector<double>& x, double error[]);
  void cartL2NormError(const std::vector<double>& x, double error[]);

  inline void setMaxtime(double t)
  {
    maxtime = t;
  }

private:

  inline void abort()
  {
    aborted = true;
  }

  inline void reset()
  {
    aborted = false;
  }


  std::vector<double> lb;
  std::vector<double> ub;

  const KDL::Chain chain;
  std::vector<double> des;


  KDL::ChainFkSolverPos_recursive fksolver;

  double maxtime;
  double eps;
  int iter_counter;
  OptType TYPE;

  KDL::Frame targetPose;
  KDL::Frame z_up ;
  KDL::Frame x_out;
  KDL::Frame y_out;
  KDL::Frame z_target;
  KDL::Frame x_target;
  KDL::Frame y_target;

  std::vector<KDL::BasicJointType> types;

  nlopt::opt opt;

  KDL::Frame currentPose;

  std::vector<double> best_x;
  int progress;
  bool aborted;

  KDL::Twist bounds;

  inline static double fRand(double min, double max)
  {
    double f = (double)rand() / RAND_MAX;
    return min + f * (max - min);
  }


};

}

#endif
