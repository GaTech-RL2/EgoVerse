from setuptools import setup, find_packages

# read the contents of your README file
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "./README.md"), encoding="utf-8") as f:
    lines = f.readlines()

# remove images from README
lines = [x for x in lines if ".png" not in x]
long_description = "".join(lines)

setup(
    name="rpl_vision_utils",
    packages=[
        package for package in find_packages() if package.startswith("rpl_vision_utils")
    ],
    install_requires=["opencv-python", "redis", "pupil-apriltags", "urdfpy"],
    eager_resources=["*"],
    include_package_data=True,
    python_requires=">=3",
    description="vision utils at UT-Austin RPL",
    author="Yifeng Zhu, Zhenyu Jiang",
    # url="https://github.com/ARISE-Initiative/robosuite",
    author_email="yifengz@cs.utexas.edu",
    version="0.1.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
