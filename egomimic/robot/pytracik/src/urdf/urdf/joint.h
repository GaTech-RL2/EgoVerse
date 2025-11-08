#ifndef URDF_JOINT_H
#define URDF_JOINT_H

#include <string>
#include <vector>
#include <optional>
#include "../tinyxml/txml.h"

#include "common.h"

namespace urdf{
	struct JointDynamics {
		double damping;
		double friction;

		void clear() {
			damping = 0;
			friction = 0;
		}

		JointDynamics() : damping(0.), friction(0.) {}
		JointDynamics(const JointDynamics& jd) : damping(jd.damping), friction(jd.friction) {}

		static std::shared_ptr<JointDynamics> fromXml(TiXmlElement* xml);
	};

	struct JointLimits {
		double lower;
		double upper;
		double effort;
		double velocity;

		void clear() {
			lower = 0;
			upper = 0;
			effort = 0;
			velocity = 0;
		}

		JointLimits() : lower(0.), upper(0.), effort(0.), velocity(0.) {}
		JointLimits(const JointLimits& jl) : lower(jl.lower), upper(jl.upper),
                                         effort(jl.effort), velocity(jl.velocity) {}

		static std::shared_ptr<JointLimits> fromXml(TiXmlElement* xml);
	};

	struct JointSafety {
		double upper_limit;
		double lower_limit;
		double k_position;
		double k_velocity;

		void clear() {
			upper_limit = 0;
			lower_limit = 0;
			k_position = 0;
			k_velocity = 0;
		}

		JointSafety() : upper_limit(0.), lower_limit(0.), k_position(0.), k_velocity(0.) {};
		JointSafety(const JointSafety& js) : upper_limit(js.upper_limit), lower_limit(js.lower_limit),
                                         k_position(js.k_position), k_velocity(js.k_velocity) {}

		static std::shared_ptr<JointSafety> fromXml(TiXmlElement* xml);
	};

	struct JointCalibration {
		std::optional<double> rising;
		std::optional<double> falling;

		void clear() {
			rising.reset();
			falling.reset();
		}

		JointCalibration() { clear(); }
		JointCalibration(const JointCalibration& jc): rising(jc.rising), falling(jc.falling) {}
		static std::shared_ptr<JointCalibration> fromXml(TiXmlElement* xml);
	};

	struct JointMimic {
		std::string joint_name;
		double offset;
		double multiplier;

		void clear() {
			joint_name = "";
			offset = 0.;
			multiplier = 0.;
		}

		JointMimic() : joint_name(""), offset(0.), multiplier(0.) {}
		JointMimic(const JointMimic& mimic): joint_name(mimic.joint_name), offset(mimic.offset),
                                         multiplier(mimic.multiplier) {}
		static std::shared_ptr<JointMimic> fromXml(TiXmlElement* xml);
	};

	enum JointType {
		UNKNOWN,
		REVOLUTE,			//rotation axis
		CONTINUOUS,
		PRISMATIC,		 //translation axis
		FLOATING,
		PLANAR,				//plane normal axis
		FIXED
	};

	struct Joint {
		std::string name;
		JointType type;
		Vector3 axis;
		std::string child_link_name;
		std::string parent_link_name;
		Transform parent_to_joint_transform;

		std::optional<std::shared_ptr<JointDynamics>> dynamics;
		std::optional<std::shared_ptr<JointLimits>> limits;
		std::optional<std::shared_ptr<JointSafety>> safety;
		std::optional<std::shared_ptr<JointCalibration>> calibration;
		std::optional<std::shared_ptr<JointMimic>> mimic;

		void clear() {
			this->axis.clear();
			this->child_link_name.clear();
			this->parent_link_name.clear();
			this->parent_to_joint_transform.clear();

			this->dynamics.reset();
			this->limits.reset();
			this->safety.reset();
			this->calibration.reset();
			this->type = JointType::UNKNOWN;
		}

		Joint() : type(JointType::UNKNOWN) { clear(); }
		Joint(const Joint& joint): name(joint.name), type(joint.type),
                               axis(joint.axis), child_link_name(joint.child_link_name),
                               parent_link_name(joint.parent_link_name),
                               parent_to_joint_transform(joint.parent_to_joint_transform),
                               dynamics(joint.dynamics), limits(joint.limits), safety(joint.safety),
                               calibration(joint.calibration), mimic(joint.mimic) {}

		static std::shared_ptr<Joint> fromXml(TiXmlElement* xml);
  };
}

#endif
