#ifndef URDF_MODEL_H
#define URDF_MODEL_H

#include <string>
#include <map>

#include "common.h"
#include "exception.h"
#include "link.h"
#include "joint.h"

using namespace std;

namespace urdf {

	struct UrdfModel {
		string name;
		std::shared_ptr<Link> root_link;

		map<string, std::shared_ptr<Link>> link_map;
		map<string, std::shared_ptr<Joint>> joint_map;
		map<string, std::shared_ptr<Material>> material_map;

		const string& getName() const { return name; }
		std::shared_ptr<Link> getRoot() const { return root_link; }

		std::shared_ptr<Link> getLink(const string& name);
		std::shared_ptr<Joint> getJoint(const string& name);
		std::shared_ptr<Material> getMaterial(const string& name);

		void getLinks(vector<std::shared_ptr<Link>>& linklist) const;

		void clear() {
			name.clear();

			link_map.clear();

			joint_map.clear();

			material_map.clear();

			root_link = nullptr;
		};


		void initLinkTree(map<string, string>& parent_link_tree);
		void findRoot(const map<string, string> &parent_link_tree);

		UrdfModel() { clear(); }

		static std::shared_ptr<UrdfModel> fromUrdfStr(const std::string& xml_string);
	};

}

#endif
