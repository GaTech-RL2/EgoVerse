#ifndef URDF_LINK_H
#define URDF_LINK_H

#include <string>
#include <vector>
#include <map>
#include <optional>

#include "joint.h"
#include "geometry.h"
#include "common.h"

namespace urdf{
	struct Material {
		std::string name;
		std::string texture_filename;
		Color color;

		void clear() {
			name.clear();
			texture_filename.clear();
			color.clear();
		}

		Material() { clear(); }
		Material(const Material& m): name(m.name), texture_filename(m.texture_filename),
                                 color(m.color) {}

		static std::shared_ptr<Material> fromXml(TiXmlElement* xml, bool);
	};


	struct Inertial {
		Transform origin;
		double mass;
		double ixx,ixy,ixz,iyy,iyz,izz;

		void clear() {
			origin.clear();
			mass = 0.;
			ixx = 0.;
			ixy = 0.;
			ixz = 0.;
			iyy = 0.;
			iyz = 0.;
			izz = 0;
		}

		Inertial() : mass(0.), ixx(0.), ixy(0.), ixz(0.), iyy(0.), iyz(0.), izz(0.) {}
		Inertial(const Inertial& i) : origin(i.origin), ixx(i.ixx), ixy(i.ixy), ixz(i.ixz),
                                  iyy(i.iyy), iyz(i.iyz), izz(i.izz), mass(i.mass) {}

		static Inertial fromXml(TiXmlElement* xml);
	};

	struct Visual {
		std::string name;
		std::string material_name;
		Transform origin;

		std::optional<std::shared_ptr<Geometry>> geometry;
		std::optional<std::shared_ptr<Material>> material;

		void clear() {
			origin.clear();
			name.clear();
			material_name.clear();

			material.reset();
			geometry.reset();
		}

		Visual() { this->clear(); }
		Visual(const Visual& v) : name(v.name), material_name(v.material_name),
                              origin(v.origin), geometry(v.geometry), material(v.material) {}

		static std::shared_ptr<Visual> fromXml(TiXmlElement* xml);
	};

	struct Collision {
		std::string name;
		Transform origin;
		std::optional<std::shared_ptr<Geometry>> geometry;

		void clear() {
			name.clear();
			origin.clear();

			geometry.reset();
		}

		Collision() { this->clear(); }
		Collision(const Collision& c) : name(c.name), origin(c.origin), geometry(c.geometry) {}

		static std::shared_ptr<Collision> fromXml(TiXmlElement* xml);
	};

	const char* getParentLinkName(TiXmlElement* xml);

	struct Link {
		std::string name;

		std::optional<Inertial> inertial;

		std::vector<std::shared_ptr<Collision>>  collisions;
		std::vector<std::shared_ptr<Visual>>  visuals;

		std::shared_ptr<Joint> parent_joint;
		std::shared_ptr<Link> parent_link;

		std::vector<std::shared_ptr<Joint>> child_joints;
		std::vector<std::shared_ptr<Link>> child_links;

		int link_index;

		std::shared_ptr<Link> getParent() const {
			return parent_link;
		}

		void setParentLink(std::shared_ptr<Link> parent) {
			parent_link = parent;
		}

		void setParentJoint(std::shared_ptr<Joint> parent) {
			parent_joint = parent;
		}

		void clear() {
			name.clear();
			link_index=-1;

			child_joints.clear();
			child_links.clear();
			collisions.clear();
			visuals.clear();

			inertial.reset();

			parent_joint = nullptr;
			parent_link = nullptr;
		}

		Link() { this->clear(); }
		Link(const Link& l) : name(l.name), inertial(l.inertial), collisions(l.collisions),
                          visuals(l.visuals), parent_joint(l.parent_joint),
                          parent_link(l.parent_link), child_joints(l.child_joints),
                          child_links(l.child_links), link_index(l.link_index) {}

		static std::shared_ptr<Link> fromXml(TiXmlElement *xml);
	};

}
#endif
