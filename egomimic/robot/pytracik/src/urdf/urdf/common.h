#ifndef URDF_COMMON_H
#define URDF_COMMON_H

#include <vector>
#include <string>
#include <math.h>
#ifndef M_PI
#define M_PI 3.141592538
#endif //M_PI

#include "../tinyxml/txml.h"
#include <memory>

#include "exception.h"

using namespace std;

namespace urdf {

	struct Vector3 {
		double x;
		double y;
		double z;

		void clear() {
			x = 0.;
			y = 0.;
			z = 0.;
		}

		Vector3 operator+(const Vector3& other);

		Vector3(double x, double y, double z) : x(x), y(y), z(z) {}
		Vector3(const Vector3 &other) : x(other.x), y(other.y), z(other.z) {}
		Vector3() : x(0.), y(0.), z(0.) {}

		static Vector3 fromVecStr(const string& vector_str);
	};

	struct Rotation {
		double x;
		double y;
		double z;
		double w;

		void clear() {
			x=0.;
			y=0.;
			z=0.;
			w=1.;
		}

		void getRpy(double &roll, double &pitch, double &yaw) const;
		void normalize();
		Rotation getInverse() const;

		Rotation operator*( const Rotation &other ) const;
		Vector3 operator*(const Vector3& vec) const;

		Rotation(double x, double y, double z, double w) : x(x), y(y), z(z), w(w) {}
		Rotation(const Rotation &other) : x(other.x), y(other.y), z(other.z), w(other.w) {}
		Rotation() : x(0.), y(0.), z(0.), w(1.) {}

		static Rotation fromRpy(double roll, double pitch, double yaw);
		static Rotation fromRpyStr(const string &rotation_str);
	};

	struct Color {
		float r;
		float g;
		float b;
		float a;

		void clear() {
			r = 0.;
			g = 0.;
			b = 0.;
			a = 1.;
		}

		Color() : r(0.), g(0.), b(0.), a(1.) {}
		Color(float r, float g, float b, float a) : r(r), g(g), b(b), a(a) {}
		Color(const Color& other) : r(other.r), g(other.g), b(other.b), a(other.a) {}

		static Color fromColorStr(const std::string &vector_str);
	};

	struct Transform {
		Vector3  position;
		Rotation rotation;

		void clear() {
			this->position.clear();
			this->rotation.clear();
		};

		Transform() : position(Vector3()), rotation(Rotation()) {}
		Transform(const Transform& other) : position(other.position), rotation(other.rotation) {}

		static Transform fromXml(TiXmlElement* xml);
	};


	struct Twist {
		Vector3  linear;
		Vector3  angular;

		void clear() {
			this->linear.clear();
			this->angular.clear();
		}

		Twist() : linear(Vector3()), angular(Vector3()) {}
		Twist(const Twist& other) : linear(other.linear), angular(other.angular) {}
	};
}

#endif
