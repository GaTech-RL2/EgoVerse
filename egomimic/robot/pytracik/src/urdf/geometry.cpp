#include "urdf/geometry.h"
#include "urdf/link.h"
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/lexical_cast.hpp>

using namespace urdf;

std::shared_ptr<Sphere> Sphere::fromXml(TiXmlElement *xml) {
	std::shared_ptr<Sphere> s = std::make_shared<Sphere>();

	if (xml->Attribute("radius") != nullptr){
		try{
			string r = xml->Attribute("radius");
			boost::algorithm::trim(r); // get rid of surrounding whitespace
			s->radius = boost::lexical_cast<double>(r);
		} catch (boost::bad_lexical_cast &e) {
			std::ostringstream error_msg;
			error_msg << "Error while parsing link '" << getParentLinkName(xml)
			          << "': sphere radius [" << xml->Attribute("radius")
			          << "] is not a valid float: " << e.what() << "!";
			throw URDFParseError(error_msg.str());
		}
	} else {
		std::ostringstream error_msg;
		error_msg << "Error while parsing link '" << getParentLinkName(xml)
		          << "': Sphere shape must have a radius attribute";
		throw URDFParseError(error_msg.str());
	}

	return s;
}

std::shared_ptr<Box> Box::fromXml(TiXmlElement *xml) {
	std::shared_ptr<Box> b = std::make_shared<Box>();

	if (xml->Attribute("size") != nullptr) {
		try{
			string r = xml->Attribute("size");
			boost::algorithm::trim(r); // get rid of surrounding whitespace
			b->dim = Vector3::fromVecStr(r);
		}catch (URDFParseError &e) {
			std::ostringstream error_msg;
			error_msg << "Error while parsing link '" << getParentLinkName(xml)
					  << "': box size [" << xml->Attribute("size")
					  << "] is not a valid: " << e.what() << "!";
			throw URDFParseError(error_msg.str());
		}
	} else {
		std::ostringstream error_msg;
		error_msg << "Error while parsing link '" << getParentLinkName(xml)
				  << "': Sphere shape must have a size attribute";
		throw URDFParseError(error_msg.str());
	}

	return b;
}

std::shared_ptr<Cylinder> Cylinder::fromXml(TiXmlElement *xml) {
	std::shared_ptr<Cylinder> y = std::make_shared<Cylinder>();

	if (xml->Attribute("length") != nullptr && xml->Attribute("radius") != nullptr) {
		try {
			string r = xml->Attribute("length");
			boost::algorithm::trim(r); // get rid of surrounding whitespace
			y->length = boost::lexical_cast<double>(r);
		} catch (boost::bad_lexical_cast &e) {
			std::ostringstream error_msg;
			error_msg << "Error while parsing link '" << getParentLinkName(xml)
					  << "': cylinder length [" << xml->Attribute("length")
					  << "] is not a valid float: " << e.what() << "!";
			throw URDFParseError(error_msg.str());
		}

		try{
			string r = xml->Attribute("radius");
			boost::algorithm::trim(r); // get rid of surrounding whitespace
			y->radius = boost::lexical_cast<double>(r);
		} catch (boost::bad_lexical_cast &e) {
			std::ostringstream error_msg;
			error_msg << "Error while parsing link '" << getParentLinkName(xml)
					  << "': cylinder radius [" << xml->Attribute("radius")
					  << "] is not a valid float: " << e.what() << "!";
			throw URDFParseError(error_msg.str());
		}
	} else {
		std::ostringstream error_msg;
		error_msg << "Error while parsing link '" << getParentLinkName(xml)
				  << "': Cylinder shape must have both length and radius attributes!";
		throw URDFParseError(error_msg.str());
	}

	return y;
}

std::shared_ptr<Capsule> Capsule::fromXml(TiXmlElement *xml) {
	std::shared_ptr<Capsule> y = std::make_shared<Capsule>();

	if (xml->Attribute("length") != nullptr && xml->Attribute("radius") != nullptr) {
		try {
			string r = xml->Attribute("length");
			boost::algorithm::trim(r); // get rid of surrounding whitespace
			y->length = boost::lexical_cast<double>(r);
		} catch (boost::bad_lexical_cast &e) {
			std::ostringstream error_msg;
			error_msg << "Error while parsing link '" << getParentLinkName(xml)
					  << "': capsule length [" << xml->Attribute("length")
					  << "] is not a valid float: " << e.what() << "!";
			throw URDFParseError(error_msg.str());
		}

		try{
			string r = xml->Attribute("radius");
			boost::algorithm::trim(r); // get rid of surrounding whitespace
			y->radius = boost::lexical_cast<double>(r);
		} catch (boost::bad_lexical_cast &e) {
			std::ostringstream error_msg;
			error_msg << "Error while parsing link '" << getParentLinkName(xml)
					  << "': capsule radius [" << xml->Attribute("radius")
					  << "] is not a valid float: " << e.what() << "!";
			throw URDFParseError(error_msg.str());
		}
	} else {
		std::ostringstream error_msg;
		error_msg << "Error while parsing link '" << getParentLinkName(xml)
				  << "': Capsule shape must have both length and radius attributes!";
		throw URDFParseError(error_msg.str());
	}

	return y;
}

std::shared_ptr<Mesh> Mesh::fromXml(TiXmlElement *xml) {
	std::shared_ptr<Mesh> m = std::make_shared<Mesh>();

	if (xml->Attribute("filename") != nullptr) {
		m->filename = xml->Attribute("filename");
	} else {
		std::ostringstream error_msg;
		error_msg << "Error while parsing link '" << getParentLinkName(xml)
				  << "Mesh must contain a filename attribute!";
		throw URDFParseError(error_msg.str());
	}

	if (xml->Attribute("scale") != nullptr) {
		try {
			string r = xml->Attribute("scale");
			boost::algorithm::trim(r); // get rid of surrounding whitespace
			m->scale = Vector3::fromVecStr(r);
		} catch (URDFParseError &e) {
			std::ostringstream error_msg;
			error_msg << "Error while parsing link '" << getParentLinkName(xml)
					  << "': mesh scale [" << xml->Attribute("scale")
					  << "] is not a valid: " << e.what() << "!";
			throw URDFParseError(error_msg.str());
		}
	}
	return m;
}

std::shared_ptr<Geometry> Geometry::fromXml(TiXmlElement *xml) {
	if (xml == nullptr) {
		std::ostringstream error_msg;
		error_msg << "Error while parsing link '" << getParentLinkName(xml)
		          << "' geometry structure pointer is null nothing to parse!";
		throw URDFParseError(error_msg.str());
	}

	TiXmlElement *shape = xml->FirstChildElement();
	if (shape == nullptr) {
		std::ostringstream error_msg;
		error_msg << "Error while parsing link '" << getParentLinkName(xml)
		          << "' geometry does not contain any shape information!";
		throw URDFParseError(error_msg.str());
	}

	const std::string type_name = shape->ValueTStr().c_str();
	if (type_name == "sphere") {
		return Sphere::fromXml(shape);
	} else if (type_name == "box") {
		return Box::fromXml(shape);
	} else if (type_name == "cylinder") {
		return Cylinder::fromXml(shape);
    } else if (type_name == "capsule") {
		return Capsule::fromXml(shape);
	} else if (type_name == "mesh") {
		return Mesh::fromXml(shape);
	} else {
		std::ostringstream error_msg;
		error_msg << "Error while parsing link '" << getParentLinkName(xml)
		          << "' unknown shape type '" << type_name << "'!";
		throw URDFParseError(error_msg.str());
	}
}
