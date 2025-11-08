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

#ifndef MATH3D_H
#define MATH3D_H

#include <string>
#include <iostream>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <cstdlib>

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

typedef unsigned char uint8_t;
typedef unsigned int uint32_t;

namespace math3d
{

static const double pi = M_PI;
static const double rad_on_deg = pi / 180.;
static const double deg_on_rad = 180. / pi;

template <typename T>
inline bool almost_zero(T a, double e)
{
  return (a == T(0)) || (a > 0 && a < e) || (a < 0 && a > -e);
}

struct color_rgb24
{
  uint8_t r, g, b;
  color_rgb24(uint8_t R, uint8_t G, uint8_t B) : r(R), g(G), b(B) {}
};

template <typename T>
struct vec3d
{
  T x, y, z;

  explicit vec3d() : x(0), y(0), z(0) {}
  vec3d(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {}

  template <typename S>
  vec3d(const vec3d<S>& s) : x(T(s.x)), y(T(s.y)), z(T(s.z)) {}

  template <typename S>
  vec3d(const S* s) : x(T(s[0])), y(T(s[1])), z(T(s[2])) {}

  vec3d<T> operator+(const vec3d<T>& p) const
  {
    return vec3d<T>(x + p.x, y + p.y, z + p.z);
  }

  vec3d<T> operator-(const vec3d<T>& p) const
  {
    return vec3d<T>(x - p.x, y - p.y, z - p.z);
  }

  vec3d<T> operator-() const
  {
    return vec3d<T>(-x, -y, -z);
  }

  template <typename S>
  vec3d<T>& operator+=(const vec3d<S>& p)
  {
    x += T(p.x);
    y += T(p.y);
    z += T(p.z);
    return *this;
  }

  // rules for partial ordering of function templates say that the new overload
  // is a better match, when it matches

  vec3d<T>& operator+=(const vec3d<T>& p)
  {
    x += p.x;
    y += p.y;
    z += p.z;
    return *this;
  }

  template <typename S>
  vec3d<T>& operator-=(const vec3d<S>& p)
  {
    x -= T(p.x);
    y -= T(p.y);
    z -= T(p.z);
    return *this;
  }

  vec3d<T>& operator-=(const vec3d<T>& p)
  {
    x -= p.x;
    y -= p.y;
    z -= p.z;
    return *this;
  }

  template <typename Scalar>
  vec3d<T>& operator/=(const Scalar& s)
  {
    const T i = T(1) / T(s);
    x *= i;
    y *= i;
    z *= i;
    return *this;
  }

  template <typename Scalar>
  vec3d<T>& operator*=(const Scalar& s)
  {
    x *= T(s);
    y *= T(s);
    z *= T(s);
    return *this;
  }

  bool operator==(const vec3d& o) const
  {
    return (x == o.x) && (y == o.y) && (z == o.z);
  }

  template <typename S>
  bool operator==(const vec3d<S>& o) const
  {
    return (x == T(o.x) && y == T(o.y) && z == T(o.z));
  }

  bool operator!=(const vec3d& o) const
  {
    return !(*this == o);
  }

  template <typename S>
  bool operator!=(const vec3d<S>& o) const
  {
    return !(*this == o);
  }

  template <typename Scalar>
  friend vec3d<T> operator*(const vec3d<T>& p, const Scalar& s)
  {
    return vec3d<T>(s * p.x, s * p.y, s * p.z);
  }

  template <typename Scalar>
  friend vec3d<T> operator*(const Scalar& s, const vec3d<T>& p)
  {
    return p * s;
  }

  template <typename Scalar>
  friend vec3d<T> operator/(const vec3d<T>& p, const Scalar& s)
  {
    return vec3d<T>(p.x / T(s), p.y / T(s), p.z / T(s));
  }

  friend std::ostream& operator<<(std::ostream& os, const vec3d<T>& p)
  {
    return (os << p.x << " " << p.y << " " << p.z);
  }

  friend std::istream& operator>>(std::istream& is, vec3d<T>& p)
  {
    return (is >> p.x >> p.y >> p.z);
  }

};

typedef vec3d<double> normal3d;
typedef vec3d<double> point3d;

class oriented_point3d : public point3d
{
public:
  normal3d n;

  explicit oriented_point3d() : point3d() {}
  oriented_point3d(const point3d& p) : point3d(p) {}
  oriented_point3d(double xx, double yy, double zz) : point3d(xx, yy, zz) {}
  oriented_point3d(const oriented_point3d& p) : point3d(p), n(p.n) {}
  oriented_point3d(const point3d& p, const normal3d& nn) : point3d(p), n(nn) {}
};


struct triangle
{
  oriented_point3d p0, p1, p2;
  int id0, id1, id2; // indices to vertices p0, p1, p2
  normal3d n;
  triangle() : id0(-1), id1(-1), id2(-1) {}
  triangle(int id0, int id1, int id2) : id0(id0), id1(id1), id2(id2) {}
  triangle(const oriented_point3d& p0_, const oriented_point3d& p1_, const oriented_point3d& p2_, const normal3d& n_)
    : p0(p0_), p1(p1_), p2(p2_), n(n_) {}
  triangle(const oriented_point3d& p0_, const oriented_point3d& p1_, const oriented_point3d& p2_, const normal3d& n_, int id0_, int id1_, int id2_)
    : p0(p0_), p1(p1_), p2(p2_), id0(id0_), id1(id1_), id2(id2_), n(n_) {}
  triangle(const point3d& p0_, const point3d& p1_, const point3d& p2_, const normal3d& n_)
    : p0(p0_), p1(p1_), p2(p2_), n(n_) {}
  friend std::ostream& operator<<(std::ostream& o, const triangle& t)
  {
    o << t.p0 << " " << t.p1 << " " << t.p2;
    return o;
  }
};


class invalid_vector : public std::logic_error
{
public:
  explicit invalid_vector() : std::logic_error("Exception invalid_vector caught.") {}
  invalid_vector(const std::string& msg) : std::logic_error("Exception invalid_vector caught: " + msg) {}
};


// ==============================================================
//                         Rotations
// ==============================================================

// NOTE: this is a std::vector derivation, thus a matrix<bool> will
// take 1 bit per element.

template<class T>
class matrix : private std::vector<T> // row-major order
{
private:
  typedef std::vector<T> super;
  int width_;
  int height_;

public:

  const int& width;
  const int& height;

  typedef typename super::iterator iterator;
  typedef typename super::const_iterator const_iterator;

  explicit matrix() : super(), width_(0), height_(0), width(width_), height(height_) {}

  matrix(int w, int h) : super(w * h), width_(w), height_(h), width(width_), height(height_) {}

  matrix(int w, int h, const T& v) : super(w * h, v), width_(w), height_(h), width(width_), height(height_) {}

  matrix(const matrix<T>& m) : super(), width(width_), height(height_)
  {
    resize(m.width_, m.height_);
    super::assign(m.begin(), m.end());
  }

  typename super::reference operator()(size_t r, size_t c)
  {
    return super::operator[](r * width_ + c);
  }
  typename super::const_reference operator()(size_t r, size_t c) const
  {
    return super::operator[](r * width_ + c);
  }

  typename super::reference at(const size_t r, const size_t c)
  {
    return super::at(r * width_ + c);
  }
  typename super::const_reference at(size_t r, size_t c) const
  {
    return super::at(r * width_ + c);
  }

  const T* to_ptr() const
  {
    return &(super::operator[](0)); // ok since std::vector is guaranteed to be contiguous
  }

  T* to_ptr()
  {
    return &(super::operator[](0));
  }

  void resize(int w, int h)
  {
    super::resize(w * h);
    width_ = w;
    height_ = h;
  }

  size_t size() const
  {
    return super::size();
  }

  iterator begin()
  {
    return super::begin();
  }
  const_iterator begin() const
  {
    return super::begin();
  }
  iterator end()
  {
    return super::end();
  }
  const_iterator end() const
  {
    return super::end();
  }

  bool operator==(const matrix<T>& m) const
  {
    if ((width_ != m.width_) || (height != m.height_))
      return false;

    const_iterator it1(begin()), it2(m.begin()), it1_end(end());

    for (; it1 != it1_end; ++it1, ++it2)
      if (*it1 != *it2) return false;

    return true;
  }

  bool operator!=(const matrix<T>& m) const
  {
    return !(*this == m);
  }

  matrix& operator=(const matrix<T>& m)
  {
    if (&m == this)
      return *this;

    if (width != m.width || height != m.height)
      throw invalid_vector("Cannot assign matrices with different sizes.");

    super::assign(m.begin(), m.end());
    return *this;
  }

  template <typename S>
  matrix<T>& operator*=(const S& s)
  {
    for (size_t i = 0; i < size(); ++i)
      super::operator[](i) *= T(s);
    return *this;
  }

  template <typename S>
  matrix<T>& operator/=(const S& s)
  {
    for (size_t i = 0; i < size(); ++i)
      super::operator[](i) /= T(s);
    return *this;
  }

  friend std::ostream& operator<<(std::ostream& s, const matrix<T>& m)
  {
    for (int y = 0; y < m.height_; ++y)
    {
      for (int x = 0; x < m.width_; ++x)
      {
        s << m[y * m.width_ + x] << " ";
      }
      s << std::endl;
    }
    return s;
  }
};


template<typename T>
struct matrix3x3
{
  T r00, r01, r02,
  r10, r11, r12,
  r20, r21, r22;

  int width, height;

  explicit matrix3x3()
    : r00(0), r01(0), r02(0)
    , r10(0), r11(0), r12(0)
    , r20(0), r21(0), r22(0)
    , width(3), height(3)
  {}

  template <typename S>
  explicit matrix3x3(const S* v)
    : r00(v[0]), r01(v[1]), r02(v[2])
    , r10(v[3]), r11(v[4]), r12(v[5])
    , r20(v[6]), r21(v[7]), r22(v[8])
    , width(3), height(3)
  {}

  void set_column(size_t c, const vec3d<T>& v)
  {
    T x = v.x;
    T y = v.y;
    T z = v.z;

    if (c == 0)
    {
      r00 = x;
      r10 = y;
      r20 = z;
    }
    else if (c == 1)
    {
      r01 = x;
      r11 = y;
      r21 = z;
    }
    else if (c == 2)
    {
      r02 = x;
      r12 = y;
      r22 = z;
    }
    else
      throw std::logic_error("Cannot set column for 3x3 matrix.");
  }

  T& operator()(size_t row, size_t col)
  {
    switch (row)
    {
    case 0:
      if (col == 0) return r00;
      if (col == 1) return r01;
      if (col == 2) return r02;
      else throw std::out_of_range("Cannot access element in 3x3 matrix");
    case 1:
      if (col == 0) return r10;
      if (col == 1) return r11;
      if (col == 2) return r12;
      else throw std::out_of_range("Cannot access element in 3x3 matrix");
    case 2:
      if (col == 0) return r20;
      if (col == 1) return r21;
      if (col == 2) return r22;
      else throw std::out_of_range("Cannot access element in 3x3 matrix");
    default:
      throw std::out_of_range("Cannot access element in 3x3 matrix");
    }
  }

  const T& operator()(size_t row, size_t col) const
  {
    switch (row)
    {
    case 0:
      if (col == 0) return r00;
      if (col == 1) return r01;
      if (col == 2) return r02;
      else throw std::out_of_range("Cannot access element in 3x3 matrix");
    case 1:
      if (col == 0) return r10;
      if (col == 1) return r11;
      if (col == 2) return r12;
      else throw std::out_of_range("Cannot access element in 3x3 matrix");
    case 2:
      if (col == 0) return r20;
      if (col == 1) return r21;
      if (col == 2) return r22;
      else throw std::out_of_range("Cannot access element in 3x3 matrix");
    default:
      throw std::out_of_range("Cannot access element in 3x3 matrix");
    }
  }

  friend std::ostream& operator<<(std::ostream& s, const matrix3x3<T>& m)
  {
    s << m.r00 << " " << m.r01 << " " << m.r02 << std::endl;
    s << m.r10 << " " << m.r11 << " " << m.r12 << std::endl;
    s << m.r20 << " " << m.r21 << " " << m.r22 << std::endl;
    return s;
  }
};


template <typename T>
inline matrix3x3<T> identity3x3()
{
  matrix3x3<T> m;
  set_identity(m);
  return m;
}


template <typename T>
inline void mult_matrix_inplace(const matrix3x3<T>& m1, const matrix3x3<T>& m2, matrix3x3<T>& r)
{
  const double r00 = m1.r00 * m2.r00 + m1.r01 * m2.r10 + m1.r02 * m2.r20;
  const double r01 = m1.r00 * m2.r01 + m1.r01 * m2.r11 + m1.r02 * m2.r21;
  const double r02 = m1.r00 * m2.r02 + m1.r01 * m2.r12 + m1.r02 * m2.r22;

  const double r10 = m1.r10 * m2.r00 + m1.r11 * m2.r10 + m1.r12 * m2.r20;
  const double r11 = m1.r10 * m2.r01 + m1.r11 * m2.r11 + m1.r12 * m2.r21;
  const double r12 = m1.r10 * m2.r02 + m1.r11 * m2.r12 + m1.r12 * m2.r22;

  const double r20 = m1.r20 * m2.r00 + m1.r21 * m2.r10 + m1.r22 * m2.r20;
  const double r21 = m1.r20 * m2.r01 + m1.r21 * m2.r11 + m1.r22 * m2.r21;
  const double r22 = m1.r20 * m2.r02 + m1.r21 * m2.r12 + m1.r22 * m2.r22;

  r.r00 = r00;
  r.r01 = r01;
  r.r02 = r02;
  r.r10 = r10;
  r.r11 = r11;
  r.r12 = r12;
  r.r20 = r20;
  r.r21 = r21;
  r.r22 = r22;
}

template <typename T>
inline void mult_matrix(const matrix3x3<T>& m1, const matrix3x3<T>& m2, matrix3x3<T>& r)
{
  if (&r == &m1 || &r == &m2) throw std::logic_error("math3d::mult_matrix() argument alias");
  return mult_matrix_inplace<T>(m1, m2, r);
}

// NO in-place!
template <typename Rot1, typename Rot2, typename Rot3>
void mult_matrix(const Rot1& m1, const Rot2& m2, Rot3& r)
{
  if ((char*)&r == (char*)&m1 || (char*)&r == (char*)&m2)
    throw std::logic_error("math3d::mult_matrix() argument alias");

  if (m1.width != m2.height || r.height != m1.height || r.width != m2.width)
    throw std::logic_error("Incompatible size matrices");

  double sum;

  for (int is = 0; is < m1.height; ++is)
  {
    for (int jd = 0; jd < m2.width; ++jd)
    {
      sum = 0.;
      for (int js = 0; js < m1.width; ++js)
      {
        sum += m1(is, js) * m2(js, jd);
      }
      r(is, jd) = sum;
    }
  }
}

// ==============================================================
//                         Quaternions
// ==============================================================

template <typename T>
struct quaternion
{
  T w, i, j, k;

  explicit quaternion(T v = 0) : w(v), i(0), j(0), k(0) {}
  //explicit quaternion(const T* p) : w(p[0]), i(p[1]), j(p[2]), k(p[3]) {}
  quaternion(T ww, T ii, T jj, T kk) : w(ww), i(ii), j(jj), k(kk) {}

  static quaternion<T> convert(const vec3d<T>& p)
  {
    return quaternion<T>(0, p.x, p.y, p.z);
  }

  static quaternion<T> convert(const T* p)
  {
    return quaternion<T>(p[0], p[1], p[2], p[3]);
  }

  quaternion<T>& operator+= (const quaternion<T>& a)
  {
    w += a.w;
    i += a.i;
    j += a.j;
    k += a.k;
    return *this;
  }

  quaternion<T>& operator*= (T a)
  {
    w *= a;
    i *= a;
    j *= a;
    k *= a;
    return *this;
  }

  void to_vector(T* p) const
  {
    p[0] = w;
    p[1] = i;
    p[2] = j;
    p[3] = k;
  }

  friend std::ostream& operator<<(std::ostream& os, const quaternion<T>& q)
  {
    return os << "[ " << q.w << " " << q.i << " " << q.j << " " << q.k << " ]";
  }

  friend std::istream& operator>>(std::istream& is, quaternion<T>& q)
  {
    std::string dump;
    return (is >> dump >> q.w >> q.i >> q.j >> q.k >> dump);
  }
};

template <typename T>
quaternion<T> operator+ (const quaternion<T>& a, const quaternion<T>& b)
{
  quaternion<T> result(a);
  result += b;
  return result;
}

template <typename T>
T dot(const quaternion<T>& a, const quaternion<T>& b)
{
  return a.w * b.w + a.i * b.i + a.j * b.j + a.k * b.k;
}

template <typename T>
T norm(const quaternion<T>& a)
{
  return std::sqrt(dot(a, a));
}

template <typename T>
quaternion<T> operator* (const quaternion<T>& a, const quaternion<T>& b)
{
  quaternion<T> result;

  result.w = a.w * b.w - a.i * b.i - a.j * b.j - a.k * b.k;
  result.i = a.i * b.w + a.w * b.i - a.k * b.j + a.j * b.k;
  result.j = a.j * b.w + a.k * b.i + a.w * b.j - a.i * b.k;
  result.k = a.k * b.w - a.j * b.i + a.i * b.j + a.w * b.k;

  return result;
}

template <typename T>
quaternion<T> operator~(const quaternion<T>& a)
{
  return quaternion<T>(a.w, -a.i, -a.j, -a.k);
}

template <typename T>
inline void conjugate(quaternion<T>& q)
{
  q.i = -q.i;
  q.j = -q.j;
  q.k = -q.k;
}

template <typename T>
inline void normalize(quaternion<T>& q)
{
  T mag = q.w * q.w + q.i * q.i + q.j * q.j + q.k * q.k;
  if (!almost_zero(mag - T(1), 1e-10))
  {
    mag = std::sqrt(mag);
    q.w /= mag;
    q.i /= mag;
    q.j /= mag;
    q.k /= mag;
  }
}

template <typename T>
inline void set_identity(quaternion<T>& q)
{
  q.w = T(1);
  q.i = q.j = q.k = T(0);
}

// returns a normalized unit quaternion
template<typename T>
quaternion<T> rot_matrix_to_quaternion(const matrix3x3<T>& m)
{
  const T m00 = m(0, 0);
  const T m11 = m(1, 1);
  const T m22 = m(2, 2);
  const T m01 = m(0, 1);
  const T m02 = m(0, 2);
  const T m10 = m(1, 0);
  const T m12 = m(1, 2);
  const T m20 = m(2, 0);
  const T m21 = m(2, 1);
  const T tr = m00 + m11 + m22;

  quaternion<T> ret;

  if (tr > 0)
  {
    T s = std::sqrt(tr + T(1)) * 2; // S=4*qw
    ret.w = 0.25 * s;
    ret.i = (m21 - m12) / s;
    ret.j = (m02 - m20) / s;
    ret.k = (m10 - m01) / s;
  }
  else if ((m00 > m11) & (m00 > m22))
  {
    const T s = std::sqrt(T(1) + m00 - m11 - m22) * 2; // S=4*qx
    ret.w = (m21 - m12) / s;
    ret.i = 0.25 * s;
    ret.j = (m01 + m10) / s;
    ret.k = (m02 + m20) / s;
  }
  else if (m11 > m22)
  {
    const T s = std::sqrt(T(1) + m11 - m00 - m22) * 2; // S=4*qy
    ret.w = (m02 - m20) / s;
    ret.i = (m01 + m10) / s;
    ret.j = 0.25 * s;
    ret.k = (m12 + m21) / s;
  }
  else
  {
    const T s = std::sqrt(T(1) + m22 - m00 - m11) * 2; // S=4*qz
    ret.w = (m10 - m01) / s;
    ret.i = (m02 + m20) / s;
    ret.j = (m12 + m21) / s;
    ret.k = 0.25 * s;
  }

  return ret;
}

// assumes a normalized unit quaternion
template <typename T>
matrix3x3<T> quaternion_to_rot_matrix(const quaternion<T>& q)
{
  matrix3x3<T> m;
  const T w = q.w, i = q.i, j = q.j, k = q.k;

  m(0, 0) = 1 - 2 * j * j - 2 * k * k;
  m(0, 1) = 2 * i * j - 2 * k * w;
  m(0, 2) = 2 * i * k + 2 * j * w;

  m(1, 0) = 2 * i * j + 2 * k * w;
  m(1, 1) = 1 - 2 * i * i - 2 * k * k;
  m(1, 2) = 2 * j * k - 2 * i * w;

  m(2, 0) = 2 * i * k - 2 * j * w;
  m(2, 1) = 2 * j * k + 2 * i * w;
  m(2, 2) = 1 - 2 * i * i - 2 * j * j;

  return m;
}

template <typename T>
inline void mult_quaternion(const quaternion<T>& a, const quaternion<T>& b, quaternion<T>& r)
{
  r.w = a.w * b.w - (a.i * b.i + a.j * b.j + a.k * b.k);
  r.i = a.w * b.i + b.w * a.i + a.j * b.k - a.k * b.j;
  r.j = a.w * b.j + b.w * a.j + a.k * b.i - a.i * b.k;
  r.k = a.w * b.k + b.w * a.k + a.i * b.j - a.j * b.i;
}

// ==============================================================
//
// ==============================================================

template <typename T>
void transpose(matrix<T>& m)
{
  matrix<T> old(m);
  const int w = m.width;
  const int h = m.height;
  m.resize(h, w);

  for (int row = 0; row < h; ++row)
  {
    for (int col = 0; col < w; ++col)
    {
      m(col, row) = old(row, col);
    }
  }
}

template <typename T>
inline void transpose(matrix3x3<T>& m)
{
  const T m01 = m.r01, m02 = m.r02, m12 = m.r12, m10 = m.r10, m21 = m.r21, m20 = m.r20;
  m.r01 = m10;
  m.r02 = m20;
  m.r10 = m01;
  m.r20 = m02;
  m.r12 = m21;
  m.r21 = m12;
}

template <typename T>
inline matrix3x3<T> get_transpose(const matrix3x3<T>& m)
{
  matrix3x3<T> ret;
  ret.r00 = m.r00;
  ret.r01 = m.r10;
  ret.r02 = m.r20;
  ret.r10 = m.r01;
  ret.r11 = m.r11;
  ret.r20 = m.r02;
  ret.r12 = m.r21;
  ret.r21 = m.r12;
  ret.r22 = m.r22;
  return ret;
}

// dest matrix must already be of the right size
template <typename T>
void transpose(const matrix<T>& src, matrix<T>& dest)
{
  if (src.width != dest.height || src.height != dest.width)
    throw math3d::invalid_vector("math3d::transpose(): Destination matrix must be of the right size.");

  const int w = src.width;
  const int h = src.height;

  for (int row = 0; row < h; ++row)
  {
    for (int col = 0; col < w; ++col)
    {
      dest(col, row) = src(row, col);
    }
  }
}

template <typename T>
inline void transpose(const matrix3x3<T>& src, matrix3x3<T>& dest)
{
  dest.r00 = src.r00;
  dest.r11 = src.r11;
  dest.r22 = src.r22;
  dest.r01 = src.r10;
  dest.r02 = src.r20;
  dest.r10 = src.r01;
  dest.r12 = src.r21;
  dest.r21 = src.r12;
  dest.r20 = src.r02;
}

template <typename T>
void set_identity(matrix<T>& m, T val = 1)
{
  if (m.width != m.height)
    throw invalid_vector("Cannot set identity on a rectangular matrix.");

  if (m.width == 0)
    return;

  const int n = m.width * m.height;
  const int w = m.width;

  int one = 0;
  for (int k = 0; k < n; ++k)
  {
    if (k == one)
    {
      m(k / w, k % w) = val;
      one += w + 1;
    }
    else
      m(k / w, k % w) = 0;
  }
}

template <typename T>
void set_identity(matrix3x3<T>& m, T val = 1)
{
  m.r00 = val;
  m.r01 = 0;
  m.r02 = 0;
  m.r10 = 0;
  m.r11 = val;
  m.r12 = 0;
  m.r20 = 0;
  m.r21 = 0;
  m.r22 = val;
}


// ==============================================================
//                         Rigid Motion
// ==============================================================

typedef std::pair<matrix3x3<double>, point3d> rigid_motion_t;

template <typename T>
void rotate(vec3d<T>& p, const matrix<T>& rot)
{
  if (rot.height != 3 || rot.width != 3)
    throw std::logic_error("Rotation matrix must be 3x3");

  T oldx = p.x, oldy = p.y, oldz = p.z;
  p.x = oldx * rot(0, 0) + oldy * rot(0, 1) + oldz * rot(0, 2);
  p.y = oldx * rot(1, 0) + oldy * rot(1, 1) + oldz * rot(1, 2);
  p.z = oldx * rot(2, 0) + oldy * rot(2, 1) + oldz * rot(2, 2);
}

template <typename T>
void rotate(vec3d<T>& p, const matrix3x3<T>& rot)
{
  T oldx = p.x, oldy = p.y, oldz = p.z;
  p.x = oldx * rot.r00 + oldy * rot.r01 + oldz * rot.r02;
  p.y = oldx * rot.r10 + oldy * rot.r11 + oldz * rot.r12;
  p.z = oldx * rot.r20 + oldy * rot.r21 + oldz * rot.r22;
}

template <typename T, typename S>
void rotate(vec3d<T>& p, const matrix<S>& rot)
{
  if (rot.height != 3 || rot.width != 3)
    throw std::logic_error("Rotation matrix must be 3x3");

  T oldx = p.x, oldy = p.y, oldz = p.z;
  p.x = T(oldx * rot(0, 0) + oldy * rot(0, 1) + oldz * rot(0, 2));
  p.y = T(oldx * rot(1, 0) + oldy * rot(1, 1) + oldz * rot(1, 2));
  p.z = T(oldx * rot(2, 0) + oldy * rot(2, 1) + oldz * rot(2, 2));
}

template <typename T, typename S>
inline void rotate(vec3d<T>& p, const matrix3x3<S>& rot)
{
  T oldx = p.x, oldy = p.y, oldz = p.z;
  p.x = T(oldx * rot.r00 + oldy * rot.r01 + oldz * rot.r02);
  p.y = T(oldx * rot.r10 + oldy * rot.r11 + oldz * rot.r12);
  p.z = T(oldx * rot.r20 + oldy * rot.r21 + oldz * rot.r22);
}

template <typename T>
inline void rotate(vec3d<T>& p, const quaternion<T>& rot)
{
  rotate(p, quaternion_to_rot_matrix(rot));
}


template <typename T>
inline vec3d<T> get_rotate(const vec3d<T>& v, const quaternion<T>& q)
{
  return get_rotate(v, quaternion_to_rot_matrix(q));
  /*
    const T
    a = q.w, b = q.i, c = q.j, d = q.k,
    t2 = a*b, t3 = a*c, t4 = a*d, t5 = -b*b, t6 = b*c,
    t7 = b*d, t8 = -c*c, t9 = c*d, t10 = -d*d,
    v1 = v.x, v2 = v.y, v3 = v.z;
    return vec3d<T>(
    2*( (t8 + t10)*v1 + (t6 -  t4)*v2 + (t3 + t7)*v3 ) + v1,
    2*( (t4 +  t6)*v1 + (t5 + t10)*v2 + (t9 - t2)*v3 ) + v2,
    2*( (t7 -  t3)*v1 + (t2 +  t9)*v2 + (t5 + t8)*v3 ) + v3);
  */
}

template <typename T>
inline vec3d<T> get_rotate(const vec3d<T>& p, const matrix3x3<T>& rot)
{
  return vec3d<T>(p.x * rot.r00 + p.y * rot.r01 + p.z * rot.r02,
                  p.x * rot.r10 + p.y * rot.r11 + p.z * rot.r12,
                  p.x * rot.r20 + p.y * rot.r21 + p.z * rot.r22);
}


template <typename T, typename RotationType>
inline void rotate_translate(vec3d<T>& v, const RotationType& rot, const point3d& trans)
{
  rotate(v, rot);
  v += trans;
}

template <typename T>
inline vec3d<T> get_rotate_translate(const vec3d<T>& p, const matrix3x3<T>& rot, const point3d& t)
{
  return vec3d<T>(p.x * rot.r00 + p.y * rot.r01 + p.z * rot.r02 + t.x,
                  p.x * rot.r10 + p.y * rot.r11 + p.z * rot.r12 + t.y,
                  p.x * rot.r20 + p.y * rot.r21 + p.z * rot.r22 + t.z);
}

template <typename T>
inline vec3d<T> get_rotate_translate(const vec3d<T>& p, const matrix<T>& rot, const point3d& t)
{
  if (rot.height != 3 || rot.width != 3)
    throw std::logic_error("Rotation matrix must be 3x3");

  return vec3d<T>(p.x * rot(0, 0) + p.y * rot(0, 1) + p.z * rot(0, 2) + t.x,
                  p.x * rot(1, 0) + p.y * rot(1, 1) + p.z * rot(1, 2) + t.y,
                  p.x * rot(2, 0) + p.y * rot(2, 1) + p.z * rot(2, 2) + t.z);
}

template <typename T>
inline vec3d<T> get_rotate_translate(const vec3d<T>& p, const T* rot, const T* t)
{
  return get_rotate_translate(p, matrix3x3<T>(rot), vec3d<T>(t[0], t[1], t[2]));
}

template <typename T>
inline vec3d<T> get_rotate_translate(const vec3d<T>& v, const quaternion<T>& rot, const point3d& t)
{
  return (get_rotate(v, rot) + t);
}


/**
 * Inverts a rigid motion.
 */
template <typename R, typename T>
inline void invert(R& r, T& t)
{
  transpose(r);
  rotate(t, r);
  t.x = -t.x;
  t.y = -t.y;
  t.z = -t.z;
}


/**
 * Computes the rigid motion bringing points expressed in j-coordinates
 * towards the world i, i.e.: Pi = Rij * Pj + Tij
 */
template <typename T>
void relative_motion(

  const matrix3x3<T>& Ri, const point3d& Ti,
  const matrix3x3<T>& Rj, const point3d& Tj,
  matrix3x3<T>& Rij, point3d& Tij

)
{
  matrix3x3<T> Ri_inv = Ri;
  point3d Ti_inv = Ti;
  invert(Ri_inv, Ti_inv);

  mult_matrix(Ri_inv, Rj, Rij);
  Tij = get_rotate_translate(Tj, Ri_inv, Ti_inv);
}


// ==============================================================
//                      Vector Operations
// ==============================================================

template <typename T>
inline double normalize(vec3d<T>& p)
{
  const double n = magnitude(p);
  if (n == 0.)
  {
    p.x = 0;
    p.y = 0;
    p.z = 0;
  }
  else
  {
    p.x /= n;
    p.y /= n;
    p.z /= n;
  }
  return n;
}

template <typename T>
inline vec3d<T> get_normalize(const vec3d<T>& p)
{
  vec3d<T> q(p);
  normalize(q);
  return q;
}

template <typename T>
inline double dist(const T& p1, const T& p2)
{
  const double sqdist = squared_dist(p1, p2);
  return (sqdist == 0. ? 0. : std::sqrt(sqdist));
}

// ||p1 - p2||^2
template <typename T>
inline double squared_dist(const vec3d<T>& p1, const vec3d<T>& p2)
{
  T x = p1.x - p2.x;
  T y = p1.y - p2.y;
  T z = p1.z - p2.z;
  return ((x * x) + (y * y) + (z * z));
}

template <typename T>
inline T dot_product(const vec3d<T>& v1, const vec3d<T>& v2)
{
  return (v1.x * v2.x) + (v1.y * v2.y) + (v1.z * v2.z);
}

template <typename T, typename S>
inline double dot_product(const vec3d<T>& v1, const vec3d<S>& v2)
{
  return (v1.x * v2.x) + (v1.y * v2.y) + (v1.z * v2.z);
}

template <typename T>
inline T dot_product(const quaternion<T>& p, const quaternion<T>& q)
{
  return (p.w * q.w + p.i * q.i + p.j * q.j + p.k * q.k);
}

template <typename T>
inline double norm2(const T& v)
{
  return dot_product(v, v);
}

template <typename T>
inline double magnitude(const T& p)
{
  return std::sqrt(dot_product(p, p));
}

template <typename T>
inline vec3d<T> cross_product(const vec3d<T>& v1, const vec3d<T>& v2)
{
  return vec3d<T>(
           (v1.y * v2.z) - (v1.z * v2.y),
           (v1.z * v2.x) - (v1.x * v2.z),
           (v1.x * v2.y) - (v1.y * v2.x)
         );
}


template <typename Iterator>
inline double median(Iterator start, Iterator end)
{
  const typename Iterator::difference_type n = end - start;
  if (n <= 0) return 0.;

  if (n % 2 == 0)
    return (*(start + (n / 2)) + * (start + (n / 2 - 1))) / 2.;
  else
    return *(start + ((n - 1) / 2));
}

}

#endif
