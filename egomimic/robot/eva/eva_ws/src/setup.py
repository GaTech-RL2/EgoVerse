from glob import glob
import os

from setuptools import find_packages, setup

package_name = "eva"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resources/" + package_name]),
        (
            os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch/*.launch.py")),
        ),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (
            os.path.join("share", package_name, "config", "task_config"),
            glob("config/task_config/*.yaml"),
        ),
        (os.path.join("share", package_name, "hardware"), glob("hardware/*")),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    author="Eva",
    author_email="user@example.com",
    maintainer="Eva",
    maintainer_email="user@example.com",
    description="ROS2 nodes and launches mirroring existing Eva stack (no modifications to originals).",
    license="BSD",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "vr_controller = eva.vr_controller:main",
            "eva_ik = eva.eva_ik:main",
            "robot_node = eva.robot_node:main",
            "data_collection_node = eva.data_collection_node:main",
            "stream_aria_ros = eva.stream_aria_ros:main",
        ],
    },
)
