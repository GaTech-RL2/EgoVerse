#ifndef URDF_GEOMETRY_H
#define URDF_GEOMETRY_H

#include "../tinyxml/txml.h"

#include "common.h"
#include "exception.h"

namespace urdf {

	enum GeometryType {
		SPHERE,
		BOX,
		CYLINDER,
        CAPSULE,
		MESH
	};

	class Geometry {
		public:
			GeometryType type;
			virtual ~Geometry(void) {};

			Geometry(GeometryType type): type(type) {}

			static std::shared_ptr<Geometry> fromXml(TiXmlElement* xml);
	};

	class Sphere : public Geometry {
		public:
			double radius;

			void clear() {
				radius = 0;
			}

			Sphere() : radius(0.), Geometry(GeometryType::SPHERE) {}

			static std::shared_ptr<Sphere> fromXml(TiXmlElement* xml);
	};

	class Box : public Geometry {
		public:
			Vector3 dim;

			void clear() {
				this->dim.clear();
			}

			Box() : Geometry(GeometryType::BOX) {}

			static std::shared_ptr<Box> fromXml(TiXmlElement* xml);
	};

	class Cylinder : public Geometry {
		public:
			double length;
			double radius;

			void clear() {
				length = 0;
				radius = 0;
			}

			Cylinder() : length(0.), radius(0.), Geometry(GeometryType::CYLINDER) {}

			static std::shared_ptr<Cylinder> fromXml(TiXmlElement* xml);
	};
    
    class Capsule : public Geometry {
        public:
            double length;
            double radius;
    
            void clear() {
                length = 0;
                radius = 0;
            }
            
            Capsule() : length(0.), radius(0.), Geometry(GeometryType::CAPSULE) {}
            
            static std::shared_ptr<Capsule> fromXml(TiXmlElement* xml);
    };

	class Mesh : public Geometry {
		public:
			std::string filename;
			Vector3 scale;

			void clear() {
				filename.clear();

				scale.x = 1;
				scale.y = 1;
				scale.z = 1;
			}

			Mesh() : scale(Vector3(1., 1., 1.)), Geometry(GeometryType::MESH) {}

			static std::shared_ptr<Mesh> fromXml(TiXmlElement* xml);
	};
}

#endif
