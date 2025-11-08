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

#ifndef DUAL_QUATERNION_HPP
#define DUAL_QUATERNION_HPP

#include "math3d.h"

using math3d::point3d;
using math3d::matrix;
using math3d::quaternion;


template<typename T> inline int sign(T v)
{
  return (v < 0) ? -1 : 1;
}

void set_quaternion_matrix(matrix<double>&M, const quaternion<double>& q, int i = 0, int j = 0, double w = 1.0)
{
  //{{a, -b, -c, -d}, {b, a, -d, c}, {c, d, a, -b}, {d, -c, b, a}}
  M(i, j)  = w * q.w;
  M(i, j + 1)  = -w * q.i;
  M(i, j + 2)  = -w * q.j;
  M(i, j + 3)  = -w * q.k;
  M(i + 1, j) = w * q.i;
  M(i + 1, j + 1) = w * q.w;
  M(i + 1, j + 2) = -w * q.k;
  M(i + 1, j + 3) = w * q.j;
  M(i + 2, j) = w * q.j;
  M(i + 2, j + 1) = w * q.k;
  M(i + 2, j + 2) = w * q.w;
  M(i + 2, j + 3) = -w * q.i;
  M(i + 3, j) = w * q.k;
  M(i + 3, j + 1) = -w * q.j;
  M(i + 3, j + 2) = w * q.i;
  M(i + 3, j + 3) = w * q.w;
}

struct dual_quaternion
{
  quaternion<double> R, tR_2;

  dual_quaternion(double v = 1.0) : R(v), tR_2(0) {}

  static constexpr double dq_epsilon = 1e-8;

  static dual_quaternion rigid_transformation(const quaternion<double>& r, const point3d& t)
  {
    dual_quaternion result;
    result.R = r;
    result.tR_2 = (quaternion<double>::convert(t) * r) *= 0.5;
    return result;
  }
  static dual_quaternion convert(const double* p)
  {
    dual_quaternion result;
    result.R = quaternion<double>::convert(p);
    result.tR_2 = quaternion<double>::convert(p + 4);
    return result;
  }

  dual_quaternion& normalize()
  {
    double n = norm(R) * sign(R.w);
    R *= 1.0 / n;
    tR_2 *= 1.0 / n;
    double d = dot(R, tR_2);
    //tR_2 += (-d)*R;
    quaternion<double> r2 = R;
    r2 *= -d;
    tR_2 += r2;
    return *this;
  }

  point3d get_translation()
  {
    quaternion<double> t = tR_2 * ~R;
    point3d result;
    result.x = 2 * t.i;
    result.y = 2 * t.j;
    result.z = 2 * t.k;
    return result;
  }

  void to_vector(double* p)
  {
    R.to_vector(p);
    tR_2.to_vector(p + 4);
  }

  dual_quaternion& operator += (const dual_quaternion& a)
  {
    R += a.R;
    tR_2 += a.tR_2;
    return *this;
  }

  dual_quaternion& operator *= (double a)
  {
    R *= a;
    tR_2 *= a;
    return *this;
  }

  dual_quaternion& log()  //computes log map tangent at identity
  {
    //assumes qual_quaternion is unitary
    const double h0 = std::acos(R.w);
    if (h0 * h0 < dq_epsilon) //small angle approximation: sin(h0)=h0, cos(h0)=1
    {
      R.w = 0.0;
      R *= 0.5;
      tR_2.w = 0.0;
      tR_2 *= 0.5;
    }
    else
    {
      R.w = 0.0;
      const double ish0 = 1.0 / norm(R);
      //R *= ish0;
      math3d::normalize(R); //R=s0
      const double he = -tR_2.w * ish0;
      tR_2.w = 0.0;

      quaternion<double> Rp(R);
      Rp *= -dot(R, tR_2) / dot(R, R);
      tR_2 += Rp;
      tR_2 *= ish0; //tR_2=se

      tR_2 *= h0;
      Rp = R;
      Rp *= he;
      tR_2 += Rp;
      tR_2 *= 0.5;
      R *= h0 * 0.5;
    }

    return *this;
  }

  dual_quaternion& exp()  //computes exp map tangent at identity
  {
    //assumes qual_quaternion is on tangent space
    const double h0 = 2.0 * norm(R);

    if (h0 * h0 < dq_epsilon) //small angle approximation: sin(h0)=h0, cos(h0)=1
    {
      R *= 2.0;
      R.w = 1.0;
      tR_2 *= 2.0;
      //normalize();
    }
    else
    {
      const double he = 4.0 * math3d::dot(tR_2, R) / h0;
      const double sh0 = sin(h0), ch0 = cos(h0);
      quaternion<double> Rp(R);
      Rp *= -dot(R, tR_2) / dot(R, R);
      tR_2 += Rp;
      tR_2 *= 2.0 / h0; //tR_2=se


      tR_2 *= sh0;
      Rp = R;
      Rp *= he * ch0 * 2.0 / h0;
      tR_2 += Rp;
      tR_2.w = -he * sh0;

      R *= sh0 * 2.0 / h0;
      R.w = ch0;
    }
    normalize();
    return *this;
  }
};


dual_quaternion operator * (const dual_quaternion&a, const dual_quaternion& b)
{
  dual_quaternion result;
  result.R = a.R * b.R;
  result.tR_2 = a.R * b.tR_2 + a.tR_2 * b.R;
  return result;
}

dual_quaternion operator ~(const dual_quaternion& a)
{
  dual_quaternion result;
  result.R = ~a.R;
  result.tR_2 = ((~a.tR_2) *= -1);
  return result;
}

dual_quaternion operator !(const dual_quaternion& a)
{
  dual_quaternion result;
  result.R = ~a.R;
  result.tR_2 = ~a.tR_2;
  return result;
}

double dot(const dual_quaternion& a, const dual_quaternion& b)
{
  return dot(a.R, b.R) + dot(a.tR_2, b.tR_2);
}

void set_dual_quaternion_matrix(matrix<double>& M, const dual_quaternion& dq, int i = 0, int j = 0, double w = 1.0)
{
  set_quaternion_matrix(M, dq.R, i, j, w);
  M(i, j + 4) = M(i, j + 5) = M(i, j + 6) = M(i, j + 7) = 0;
  M(i + 1, j + 4) = M(i + 1, j + 5) = M(i + 1, j + 6) = M(i + 1, j + 7) = 0;
  M(i + 2, j + 4) = M(i + 2, j + 5) = M(i + 2, j + 6) = M(i + 2, j + 7) = 0;
  M(i + 3, j + 4) = M(i + 3, j + 5) = M(i + 3, j + 6) = M(i + 3, j + 7) = 0;
  set_quaternion_matrix(M, dq.tR_2, i + 4, j, w);
  set_quaternion_matrix(M, dq.R, i + 4, j + 4, w);
}

dual_quaternion log(dual_quaternion a)
{
  return a.log();
}
dual_quaternion exp(dual_quaternion a)
{
  return a.exp();
}


std::ostream& operator << (std::ostream& out, const dual_quaternion& dq)
{
  return out << "( " << dq.R.w << ", " << dq.R.i << ", " << dq.R.j << ", " << dq.R.k << ",  "
         << dq.tR_2.w << ", " << dq.tR_2.i << ", " << dq.tR_2.j << ", " << dq.tR_2.k << " )";
}

#endif
