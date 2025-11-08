#include "urdf/common.h"
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

using namespace urdf;
using namespace std;

// ------------------- Vector Implementation -------------------

Vector3 Vector3::fromVecStr(const string& vector_str) {
	Vector3 vec;

	vector<string> pieces;
	vector<double> values;

	boost::split( pieces, vector_str, boost::is_any_of(" "));
	for (unsigned int i = 0; i < pieces.size(); ++i){
		if (pieces[i] != ""){
			try {
				values.push_back(boost::lexical_cast<double>(pieces[i].c_str()));
			} catch (boost::bad_lexical_cast &e) {
				throw URDFParseError("Error not able to parse component (" + pieces[i] + ") to a double (while parsing a vector value)");
			}
		}
	}

	if (values.size() != 3) {
		ostringstream error_msg;
		error_msg << "Parser found " << values.size()
				  << " elements but 3 expected while parsing vector ["
				  << vector_str <<  "]";
		throw URDFParseError(error_msg.str());
	}

	vec.x = values[0];
	vec.y = values[1];
	vec.z = values[2];

	return vec;
}

Vector3 Vector3::operator+(const Vector3& other) {
	return Vector3(x+other.x, y+other.y, z+other.z);
}

// ------------------- Quaternion Implementation -------------------

void Rotation::getRpy(double &roll, double &pitch, double &yaw) const {
	double sqw;
	double sqx;
	double sqy;
	double sqz;

	sqx = x * x;
	sqy = y * y;
	sqz = z * z;
	sqw = w * w;

	roll = atan2( 2*(y*z + w*x), sqw - sqx - sqy + sqz );
	double s = -2*(x*z - w*y);
	if (s <= -1.) {
		pitch = -0.5*M_PI;
	} else if (s >= 1.) {
		pitch = 0.5*M_PI;
	} else {
		pitch = asin(s);
	}
	yaw   = atan2( 2*(x*y + w*z), sqw + sqx - sqy - sqz );
}

void Rotation::normalize() {
	double s = sqrt(x*x + y*y + z*z + w*w);
	if (s == 0.0) {
		x = 0.0;
		y = 0.0;
		z = 0.0;
		w = 1.0;
	} else {
		x /= s;
		y /= s;
		z /= s;
		w /= s;
	}
}

Rotation Rotation::getInverse() const {
	Rotation result;

	double norm = w*w + x*x + y*y + z*z;

	if (norm > 0.0) {
		result.w = w / norm;
		result.x = -1*x / norm;
		result.y = -1*y / norm;
		result.z = -1*z / norm;
	}

	return result;
}

Rotation Rotation::operator*( const Rotation &other ) const {
	Rotation result;

	result.x = (w * other.x) + (x * other.w) + (y * other.z) - (z * other.y);
	result.y = (w * other.y) - (x * other.z) + (y * other.w) + (z * other.x);
	result.z = (w * other.z) + (x * other.y) - (y * other.x) + (z * other.w);
	result.w = (w * other.w) - (x * other.x) - (y * other.y) - (z * other.z);

	return result;
}

Vector3 Rotation::operator*(const Vector3& vec) const {
	Rotation t;
	Vector3 result;

	t.w = 0.0;
	t.x = vec.x;
	t.y = vec.y;
	t.z = vec.z;

	t = (*this) * (t * getInverse());

	result.x = t.x;
	result.y = t.y;
	result.z = t.z;

	return result;
}


Rotation Rotation::fromRpy(double roll, double pitch, double yaw) {
	Rotation rot;
	double phi, the, psi;

	phi = roll / 2.0;
	the = pitch / 2.0;
	psi = yaw / 2.0;

	rot.x = (sin(phi) * cos(the) * cos(psi)) - (cos(phi) * sin(the) * sin(psi));
	rot.y = (cos(phi) * sin(the) * cos(psi)) + (sin(phi) * cos(the) * sin(psi));
	rot.z = (cos(phi) * cos(the) * sin(psi)) - (sin(phi) * sin(the) * cos(psi));
	rot.w = (cos(phi) * cos(the) * cos(psi)) + (sin(phi) * sin(the) * sin(psi));

	rot.normalize();

	return rot;
};


Rotation Rotation::fromRpyStr(const string &rotation_str) {
	Vector3 rpy = Vector3::fromVecStr(rotation_str);
	return Rotation::fromRpy(rpy.x, rpy.y, rpy.z);
}

// ------------------- Color Implementation -------------------

Color Color::fromColorStr(const std::string &vector_str) {
	std::vector<std::string> pieces;
	std::vector<float> values;

	boost::split(pieces, vector_str, boost::is_any_of(" "));
	for (int i = 0; i < pieces.size(); i++) {
		if (!pieces[i].empty()) {
			try {
				values.push_back(boost::lexical_cast<double>(pieces[i].c_str()));
			} catch (boost::bad_lexical_cast &e) {
				std::ostringstream error_msg;
				error_msg << "Error parsing Color value " << i
						  << " in color value string (" << vector_str
						  << "): value is not a double: " << e.what() << "!";
				throw URDFParseError(error_msg.str());
			}
		}
	}

	if (values.size() != 4) {
		std::ostringstream error_msg;
		error_msg << "Error parsing Color string (" << vector_str
				  << "): It needs to contain exactly 4 values for rbdl color!";
		throw URDFParseError(error_msg.str());
	}

	return Color( values[0], values[1], values[2], values[3] );
}

// ------------------- Transform Implementation -------------------

Transform Transform::fromXml(TiXmlElement* xml) {
	Transform t;
	if (xml) {
		const char* xyz_str = xml->Attribute("xyz");
		if (xyz_str != NULL) {
			t.position = Vector3::fromVecStr(xyz_str);
		}

		const char* rpy_str = xml->Attribute("rpy");
		if (rpy_str != NULL) {
			t.rotation = Rotation::fromRpyStr(rpy_str);
		}
	}
	return t;
}
