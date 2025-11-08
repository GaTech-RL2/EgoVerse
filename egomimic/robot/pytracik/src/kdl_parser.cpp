//
// Created by haochen on 22/08/11.
//

#include <iostream>
#include <string>
#include "kdl_parser.hpp"

namespace kdl_parser {
    // construct vector
    KDL::Vector toKdl(urdf::Vector3 v) {
        return KDL::Vector(v.x, v.y, v.z);
    }

    //// construct rotation
    KDL::Rotation toKdl(urdf::Rotation r) {
        return KDL::Rotation::Quaternion(r.x, r.y, r.z, r.w);
    }

    //// construct pose
    KDL::Frame toKdl(urdf::Transform p) {
        return KDL::Frame(toKdl(p.rotation), toKdl(p.position));
    }

    //// construct joint
    KDL::Joint toKdl(shared_ptr<urdf::Joint> jnt) {
        KDL::Frame F_parent_jnt = toKdl(jnt->parent_to_joint_transform);

        switch (jnt->type) {
        case urdf::FIXED: {
            return KDL::Joint(jnt->name, KDL::Joint::None);
        }
        case urdf::REVOLUTE: {
            KDL::Vector axis = toKdl(jnt->axis);
            return KDL::Joint(jnt->name, F_parent_jnt.p, F_parent_jnt.M * axis, KDL::Joint::RotAxis);
        }
        case urdf::CONTINUOUS: {
            KDL::Vector axis = toKdl(jnt->axis);
            return KDL::Joint(jnt->name, F_parent_jnt.p, F_parent_jnt.M * axis, KDL::Joint::RotAxis);
        }
        case urdf::PRISMATIC: {
            KDL::Vector axis = toKdl(jnt->axis);
            return KDL::Joint(jnt->name, F_parent_jnt.p, F_parent_jnt.M * axis, KDL::Joint::TransAxis);
        }
        default: {
            std::cout << "Converting unknown joint type of joint '%s' into a fixed joint" << jnt->name.c_str()
                << std::endl;
            return KDL::Joint(jnt->name, KDL::Joint::None);
        }
        }
        return KDL::Joint();
    }

    //// construct inertia
    KDL::RigidBodyInertia toKdl(std::optional<urdf::Inertial> i) {
        KDL::Frame origin = toKdl(i->origin);

        // the mass is frame independent
        double kdl_mass = i->mass;

        // kdl and urdf both specify the com position in the reference frame of the link
        KDL::Vector kdl_com = origin.p;

        // kdl specifies the inertia matrix in the reference frame of the link,
        // while the urdf specifies the inertia matrix in the inertia reference frame
        KDL::RotationalInertia urdf_inertia =
            KDL::RotationalInertia(i->ixx, i->iyy, i->izz, i->ixy, i->ixz, i->iyz);

        // Rotation operators are not defined for rotational inertia,
        // so we use the RigidBodyInertia operators (with com = 0) as a workaround
        KDL::RigidBodyInertia kdl_inertia_wrt_com_workaround =
            origin.M * KDL::RigidBodyInertia(0, KDL::Vector::Zero(), urdf_inertia);

        // Note that the RigidBodyInertia constructor takes the 3d inertia wrt the com
        // while the getRotationalInertia method returns the 3d inertia wrt the frame origin
        // (but having com = Vector::Zero() in kdl_inertia_wrt_com_workaround they match)
        KDL::RotationalInertia kdl_inertia_wrt_com =
            kdl_inertia_wrt_com_workaround.getRotationalInertia();

        return KDL::RigidBodyInertia(kdl_mass, kdl_com, kdl_inertia_wrt_com);
    }

    //
    //// recursive function to walk through tree
    bool addChildrenToTree(std::shared_ptr<urdf::Link> root, KDL::Tree& tree) {
        std::vector<std::shared_ptr<urdf::Link>> children = root->child_links;
        // std::cout << "Link " << root->name.c_str() << " had " << children.size() << " children" << std::endl;

        // constructs the optional inertia
        KDL::RigidBodyInertia inert(0);
        if (root->inertial) {
            inert = toKdl(root->inertial);
        }

        // constructs the kdl joint
        KDL::Joint jnt = toKdl(root->parent_joint);

        // construct the kdl segment
        KDL::Segment sgm(root->name, jnt, toKdl(
            root->parent_joint->parent_to_joint_transform), inert);

        // add segment to tree
        tree.addSegment(sgm, root->parent_joint->parent_link_name);

        // recurslively add all children
        for (size_t i = 0; i < children.size(); i++) {
            if (!addChildrenToTree(children[i], tree)) {
                return false;
            }
        }
        return true;
    }

    bool treeFromUrdfModel(const urdf::UrdfModel& robot_model, KDL::Tree& tree) {
        if (!robot_model.getRoot()) {
            return false;
        }

        tree = KDL::Tree(robot_model.getRoot()->name);

        // warn if root link has inertia. KDL does not support this
        if (robot_model.getRoot()->inertial) {
            std::cout << "The root link %s has an inertia specified in the URDF, but KDL does not "
                "support a root link with an inertia.  As a workaround, you can add an extra "
                "dummy link to your URDF."
                << robot_model.getRoot()->name.c_str()
                << std::endl;
        }

        //  add all children
        for (size_t i = 0; i < robot_model.getRoot()->child_links.size(); i++) {
            if (!addChildrenToTree(robot_model.getRoot()->child_links[i], tree)) {
                return false;
            }
        }

        return true;
    }
}