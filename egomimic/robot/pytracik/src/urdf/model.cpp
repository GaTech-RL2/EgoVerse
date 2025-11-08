#include "urdf/model.h"
#include "urdf/link.h"
#include "urdf/joint.h"

#include "tinyxml/txml.h"

using namespace urdf;

std::shared_ptr<Link> UrdfModel::getLink(const string& name) {
	if (link_map.find(name) == link_map.end()) {
		return nullptr;
	} else {
		return link_map.find(name)->second;
	}
}

std::shared_ptr<Joint> UrdfModel::getJoint(const string& name) {
	if (joint_map.find(name) == joint_map.end()) {
		return nullptr;
	} else {
		return joint_map.find(name)->second;
	}
}

std::shared_ptr<Material> UrdfModel::getMaterial(const string& name) {
	if (material_map.find(name) == material_map.end()) {
		return nullptr;
	} else {
		return material_map.find(name)->second;
	}
}

void UrdfModel::getLinks(vector<std::shared_ptr<Link>>& linklist) const {
	for (auto link = link_map.begin(); link != link_map.end(); link++) {
		linklist.push_back(link->second);
	}
}

void UrdfModel::initLinkTree(map<string, string>& parent_link_tree) {
	for (auto joint = joint_map.begin(); joint != joint_map.end(); joint++) {
		string parent_link_name = joint->second->parent_link_name;
		string child_link_name = joint->second->child_link_name;

		if (parent_link_name.empty()){
			ostringstream error_msg;
			error_msg << "Error while constructing model! Joint [" << joint->first
					  << "] is missing a parent link specification.";
			throw URDFParseError(error_msg.str());
		}
		if (child_link_name.empty()) {
			ostringstream error_msg;
			error_msg << "Error while constructing model! Joint [" << joint->first
					  << "] is missing a child link specification.";
			throw URDFParseError(error_msg.str());
		}

		auto child_link = getLink(child_link_name);
		if (child_link == nullptr) {
			ostringstream error_msg;
			error_msg << "Error while constructing model! Child link [" << child_link_name
					  << "] of joint [" <<  joint->first << "] not found";
			throw URDFParseError(error_msg.str());
		}

		auto parent_link = getLink(parent_link_name);
		if (parent_link == nullptr) {
			ostringstream error_msg;
			error_msg << "Error while constructing model! Parent link [" << parent_link_name
					  << "] of joint [" <<  joint->first << "] not found";
			throw URDFParseError(error_msg.str());
		}

		child_link->setParentLink(parent_link);
		child_link->setParentJoint(joint->second);

		parent_link->child_joints.push_back(joint->second);
		parent_link->child_links.push_back(child_link);

		parent_link_tree[child_link->name] = parent_link_name;

	}
}

void UrdfModel::findRoot(const map<string, string> &parent_link_tree) {
	for (auto l=link_map.begin(); l!=link_map.end(); l++) {
		auto parent = parent_link_tree.find(l->first);
		if (parent == parent_link_tree.end()) {
			if (root_link == nullptr) {
				root_link = getLink(l->first);
			} else {
				ostringstream error_msg;
				error_msg << "Error! Multiple root links found: (" << root_link->name
						  << ") and (" + l->first + ")!";
				throw URDFParseError(error_msg.str());
			}
		}
	}
	if (root_link == nullptr) {
		throw URDFParseError("Error! No root link found. The urdf does not contain a valid link tree.");
	}
}

std::shared_ptr<UrdfModel> UrdfModel::fromUrdfStr(const std::string& xml_string) {
	std::shared_ptr<UrdfModel> model = std::make_shared<UrdfModel>();

	TiXmlDocument xml_doc;
	xml_doc.Parse(xml_string.c_str());
	if (xml_doc.Error()) {
		std::string error_msg = xml_doc.ErrorDesc();
		xml_doc.ClearError();
		throw URDFParseError(error_msg);
	}

	//xml_doc.Print();
	TiXmlElement *robot_xml = xml_doc.RootElement();
	if (robot_xml == nullptr || robot_xml->ValueStr() != "robot") {
		std::string error_msg = "Error! Could not find the <robot> element in the xml file";
		throw URDFParseError(error_msg);
	}

	const char *name = robot_xml->Attribute("name");
	if (name != nullptr){
		model->name = std::string(name);
	} else {
		std::string error_msg = "No name given for the robot. Please add a name attribute to the robot element!";
		throw URDFParseError(error_msg);
	}

	for (TiXmlElement* material_xml = robot_xml->FirstChildElement("material"); material_xml != nullptr; material_xml = material_xml->NextSiblingElement("material")) {
		auto material = Material::fromXml(material_xml, false); // material needs to be fully defined here
		if (model->getMaterial(material->name) != nullptr) {
			std::ostringstream error_msg;
			error_msg << "Duplicate materials '" << material->name << "' found!";
			throw URDFParseError(error_msg.str());
		} else {
			model->material_map[material->name] = material;
		}
	}

	for (TiXmlElement* link_xml = robot_xml->FirstChildElement("link"); link_xml != nullptr; link_xml = link_xml->NextSiblingElement("link")) {
    auto link = Link::fromXml(link_xml);

		if (model->getLink(link->name) != nullptr) {
			std::ostringstream error_msg;
			error_msg << "Error! Duplicate links '" << link->name << "' found!";
			throw URDFParseError(error_msg.str());
		} else {
			// loop over link visual to find the materials
			if (!link->visuals.empty()) {
				for ( auto visual : link->visuals ) {
					if (!visual->material_name.empty()) {
						if (model->getMaterial(visual->material_name) != nullptr) {
							visual->material.emplace(model->getMaterial( visual->material_name.c_str() ));
						} else {
							// if no model matrial found use the one defined in the visual
							if (visual->material.has_value()) {
								model->material_map[visual->material_name] = visual->material.value();
							} else {
								// no matrial information available for this visual -> error
								std::ostringstream error_msg;
								error_msg << "Error! Link '" << link->name
										  << "' material '" << visual->material_name
										  <<" ' undefined!";
								throw URDFParseError(error_msg.str());
							}
						}
					}
				}
			}
			model->link_map[link->name] = link;
		}
	}

	if (model->link_map.size() == 0){
		std::string error_msg = "Error! No link elements found in the urdf file.";
		throw URDFParseError(error_msg);
	}

	for (TiXmlElement* joint_xml = robot_xml->FirstChildElement("joint"); joint_xml != nullptr; joint_xml = joint_xml->NextSiblingElement("joint")) {
		auto joint = Joint::fromXml(joint_xml);

		if (model->getJoint(joint->name) != nullptr) {
			std::ostringstream error_msg;
			error_msg << "Error! Duplicate joints '" << joint->name << "' found!";
			throw URDFParseError(error_msg.str());
		} else {
			model->joint_map[joint->name] = joint;
		}
	}

	std::map<std::string, std::string> parent_link_tree;

	model->initLinkTree(parent_link_tree);
	model->findRoot(parent_link_tree);

	return model;
}
