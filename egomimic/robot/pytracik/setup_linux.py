""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20230911osaka

"""
from setuptools import setup, Extension
import pybind11
import fnmatch
import sys
import sysconfig
from pybind11.setup_helpers import Pybind11Extension, build_ext


def requirements_from_file(file_name):
    return open(file_name).read().splitlines()


def find_files(path, file_extension, visited=None, finded_files=None):
    if finded_files is None:
        finded_files = []
    if visited is None:
        visited = set()

        # 检查路径是否已经被访问过，防止无限循环
    path_real = os.path.realpath(path)
    if path_real in visited:
        return
    visited.add(path_real)

    # 如果是文件且扩展名匹配，则打印出来
    if os.path.isfile(path):
        if fnmatch.fnmatch(path, '*' + file_extension):
            finded_files.append(path)
        return

        # 如果是目录，则递归遍历其中的文件
    if os.path.isdir(path):
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            # 判断是否为真实路径，防止符号链接造成的循环引用
            if os.path.exists(item_path):
                find_files(item_path, file_extension, visited, finded_files)
            else:
                print(f'Warning: Path {item_path} does not exist or is a broken symbolic link', file=sys.stderr)
    return finded_files


if __name__ == '__main__':
    import os
    import importlib.util
    from setuptools import setup

    # https://stackoverflow.com/questions/71039131/windows-python-3-10-2-fails-to-run-python-m-venv-venv
    # load version from file
    version_file = os.path.join(os.path.dirname(__file__), "trac_ik/version.py")
    with open(version_file, "r") as f:
        # use eval to get a clean string of version from file
        version = eval(f.read().strip().split("=")[-1])

    module_name = "pytracik"
    src_dir = "./src"
    # get all src files
    src_files = find_files(src_dir, '.cpp')
    # get all header files
    header_files = find_files(src_dir, '.h')

    # Retrieve Python configuration from the current environment (Conda)
    python_include = sysconfig.get_paths()["include"]
    python_library = sysconfig.get_config_var('LDLIBRARY')
    python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"

    # Sometimes 'LDLIBRARY' might return 'libpython3.12m.so', so we strip the 'lib' prefix and '.so' suffix
    if python_library:
        python_lib = os.path.splitext(os.path.basename(python_library))[0].replace('lib', '')
    else:
        # Fallback to dynamic retrieval if LDLIBRARY is not set
        python_lib = python_version

    # Create the Extension object
    trac_ik_module = [
        Pybind11Extension(
            module_name,
            src_files,
            include_dirs=[pybind11.get_include(),
                          python_include,  # Python headers from Conda
                          "/usr/include/eigen3",  # Eigen headers
                          "/usr/include/orocos/kdl",  # Orocos KDL headers
                          "/usr/include/",  # NLopt headers
                          "/usr/include/boost",  # Boost headers
                          ],  # Include pybind11 headers
            libraries=['nlopt',
                       'orocos-kdl',
                       'boost_date_time',
                       'boost_system',
                       'boost_thread',  # Add other Boost libraries if necessary
                       python_lib, ],  # Adjust based on your Python version,
            library_dirs=[
                '/usr/lib/x86_64-linux-gnu',  # Standard library directory on Ubuntu
                '/root/miniconda3/lib',  # Conda library directory
                # Add additional directories if libraries are located elsewhere
            ],
            language='c++',
            extra_compile_args=['-std=c++17'],  # 或者其他你需要的编译选项
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
        platforms=['Linux'],
        cmdclass={"build_ext": build_ext},
        ext_modules=trac_ik_module,
        install_requires=requirements_from_file('requirements.txt'),
        python_requires='>=3.9',
    )
