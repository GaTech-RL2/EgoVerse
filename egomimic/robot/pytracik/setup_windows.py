""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20230911osaka

"""
import os
import sys
import sysconfig
import fnmatch

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11


def requirements_from_file(file_name):
    return open(file_name).read().splitlines()


def find_files(path, file_extension, visited=None, finded_files=None):
    if finded_files is None:
        finded_files = []
    if visited is None:
        visited = set()

    # prevent infinite loops via symlinks
    path_real = os.path.realpath(path)
    if path_real in visited:
        return
    visited.add(path_real)

    if os.path.isfile(path):
        if fnmatch.fnmatch(path, '*' + file_extension):
            finded_files.append(path)
        return

    if os.path.isdir(path):
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.exists(item_path):
                find_files(item_path, file_extension, visited, finded_files)
            else:
                print(f'Warning: Path {item_path} does not exist or is a broken symbolic link', file=sys.stderr)
    return finded_files


def _python_linkage_for_windows():
    """
    Determine the Python import library name (e.g., 'python312') and its lib dir on Windows.
    If you're building only on Windows, this keeps behavior consistent with the hard-coded values
    you had before, but without pinning to a specific version or path.
    """
    if os.name != "nt":
        return None, None

    # Try to read the configured library file name (python312.lib / python312.dll)
    ldlib = sysconfig.get_config_var("LDLIBRARY") or ""
    libname = os.path.splitext(ldlib)[0] if ldlib else f"python{sys.version_info.major}{sys.version_info.minor}"

    # Prefer LIBDIR; fall back to LIBPL; finally <prefix>/libs
    libdir = (
            sysconfig.get_config_var("LIBDIR")
            or sysconfig.get_config_var("LIBPL")
            or os.path.join(sys.base_prefix, "libs")
    )
    return libname, libdir


if __name__ == '__main__':
    # https://stackoverflow.com/questions/71039131/windows-python-3-10-2-fails-to-run-python-m-venv-venv
    # load version from file
    version_file = os.path.join(os.path.dirname(__file__), "trac_ik/version.py")
    with open(version_file, "r") as f:
        # use eval to get a clean string of version from file
        version = eval(f.read().strip().split("=")[-1])

    module_name = "pytracik"
    src_dir = "./src"
    # Collect all src files
    src_files = find_files(src_dir, '.cpp')
    # Collect all header files
    header_files = find_files(src_dir, '.h')
    # Compute Python linkage (Windows)
    py_libname, py_libdir = _python_linkage_for_windows()
    # Base third-party libs/dirs
    third_party_libraries = [
        "nlopt",
        "orocos-kdl",
        "libboost_date_time-vc143-mt-s-x64-1_79",
    ]
    third_party_library_dirs = [
        r"./dependency/boost_1_79_0/lib",
        r"./dependency/kdl/lib",
        r"./dependency/nlopt/lib",
    ]
    # If Windows, include Python import library name + dir automatically
    libraries = third_party_libraries + ([py_libname] if py_libname else [])
    library_dirs = third_party_library_dirs + ([py_libdir] if py_libdir else [])

    # Create the Extension object
    ext_modules = [
        Pybind11Extension(
            module_name,
            src_files,
            include_dirs=[pybind11.get_include(),
                          r".\dependency\eigen-3.4.0",
                          r".\dependency\kdl\include",
                          r".\dependency\nlopt\include",
                          r".\dependency\boost_1_79_0",
                          ],  # Include pybind11 headers
            libraries=libraries,
            library_dirs=library_dirs,
            language='c++',
            extra_compile_args=['/std:c++17'],  # 或者其他你需要的编译选项
        ),
    ]

    setup(
        name="pytracik",
        version=version,
        description="TracIK Python Bindings",
        author="Hao Chen",
        author_email="chen960216@gmail.com",
        license="MIT Software License",
        url="https://github.com/chenhaox/pytracik",
        keywords="robotics inverse kinematics",
        packages=['trac_ik'],
        include_package_data=True,
        package_data={'': [r'nlopt.dll', ]},
        platforms=['Windows'],
        cmdclass={"build_ext": build_ext},
        ext_modules=ext_modules,
        install_requires=requirements_from_file('requirements.txt'),
        python_requires='>=3.9',
    )
