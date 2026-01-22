FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash

# 1) base system deps (all in one layer)
# NOTE: AWS CLI uses aarch64 URL for ARM architecture (Jetson Orin)
RUN apt-get update && \
    apt-get install -y \
    git \
    git-lfs \
    can-utils \
    net-tools \
    iproute2 \
    udev \
    sudo \
    python3 \
    python3-pip \
    nano \
    libboost-all-dev \
    liburdfdom-dev \
    liburdfdom-headers-dev \
    libeigen3-dev \
    liborocos-kdl-dev \
    libnlopt-dev \
    libnlopt-cxx-dev \
    software-properties-common \
    build-essential \
    procps \
    curl \
    file \
    ca-certificates \
    lsof \
    usbutils \
    unzip \
    && rm -rf /var/lib/apt/lists/* && \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "/tmp/awscliv2.zip" && \
    unzip /tmp/awscliv2.zip -d /tmp && \
    /tmp/aws/install && \
    rm -rf /tmp/awscliv2.zip /tmp/aws

WORKDIR /home/robot

# 2) micromamba for aarch64
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-aarch64/latest \
    | tar -xvj -C /usr/local/bin/ --strip-components=1 bin/micromamba

# # 3) install newer git + Homebrew + graphite (kept together)
# NOTE: Commented out - Homebrew/Graphite may have ARM compatibility issues
# RUN add-apt-repository ppa:git-core/ppa -y && \
#     apt-get update && \
#     apt-get install -y git && \
#     rm -rf /var/lib/apt/lists/* && \
#     NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" && \
#     echo 'eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"' >> /root/.bashrc && \
#     eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)" && \
#     brew update && \
#     brew tap withgraphite/tap && \
#     brew install withgraphite/tap/graphite && \
#     gt auth --token SX4r5gXXW83uNr4x1USeYFcc2VUEzp5YfBOGJVP7xXiGUk4vhEYEnPObeTY8

# 4) create workspace dir early
RUN mkdir -p /home/robot/robot_ws
WORKDIR /home/robot/robot_ws

# 5) copy only the env + requirements first (so pip/mamba stays cached)
# adjust paths below to match your repo layout on host
COPY egomimic/robot/eva/stanford_repo/conda_environments/py310_environment.yaml /tmp/py310_environment_original.yaml
COPY requirements.txt /tmp/requirements.txt

# 5b) Create a modified environment.yaml without soem (not available for aarch64)
# SOEM will be built from source in a later step
RUN sed '/conda-forge::soem/d' /tmp/py310_environment_original.yaml > /tmp/py310_environment.yaml

# 6) create mamba env (its own layer) - without soem which doesn't exist for aarch64
RUN micromamba create -y -f /tmp/py310_environment.yaml -n arx-py310 && \
    micromamba clean --all --yes

# 6b) Build and install SOEM from source for aarch64
# This is required because conda-forge::soem=1.4.0 is not available for ARM
SHELL ["micromamba", "run", "-n", "arx-py310", "/bin/bash", "-c"]
RUN git clone https://github.com/OpenEtherCATsociety/SOEM.git /tmp/SOEM && \
    cd /tmp/SOEM && \
    mkdir build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX .. && \
    make -j$(nproc) && \
    make install && \
    rm -rf /tmp/SOEM

# 7) build stanford_repo inside the env
WORKDIR /home/robot/robot_ws
# we need the source to build, so copy now
COPY . /home/robot/robot_ws

WORKDIR /home/robot/robot_ws/egomimic/robot/eva/stanford_repo
RUN mkdir -p build && \
    cd build && \
    cmake .. -DCMAKE_PREFIX_PATH=/opt/ros/humble -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ && \
    make -j

# 8) install arx5 python binding into ROS python
# NOTE: The .so filename varies by architecture - use wildcard to find the correct one
WORKDIR /home/robot/robot_ws/egomimic/robot/eva/stanford_repo/python
RUN mkdir -p /opt/ros/humble/lib/python3.10/site-packages/arx5 && \
    cp arx5_interface.cpython-310-*.so \
    /opt/ros/humble/lib/python3.10/site-packages/arx5/

# 9) back to normal shell
SHELL ["/bin/bash", "-c"]

# 10) ROS + handy aliases
# NOTE: Updated sf_build alias to use architecture-agnostic .so copy
RUN echo 'source /opt/ros/humble/setup.bash' >> /root/.bashrc && \
    echo 'alias wsbuild="cd /home/robot/robot_ws/egomimic/robot/eva/eva_ws && colcon build && source /opt/ros/humble/setup.bash && source install/setup.bash && export LD_LIBRARY_PATH=/root/.local/share/mamba/envs/arx-py310/lib:$LD_LIBRARY_PATH"' >> /root/.bashrc && \
    echo 'alias sf_build="micromamba run -n arx-py310 bash -c \"cd /home/robot/robot_ws/egomimic/robot/eva/stanford_repo && rm -rf build && mkdir -p build && cd build && cmake .. -DCMAKE_PREFIX_PATH=/opt/ros/humble -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ && make -j && cd ../python && mkdir -p /opt/ros/humble/lib/python3.10/site-packages/arx5 && cp arx5_interface.cpython-310-*.so /opt/ros/humble/lib/python3.10/site-packages/arx5/\""' >> /root/.bashrc && \
    echo 'alias rhome="cd /home/robot/robot_ws/egomimic/robot"' >> /root/.bashrc && \
    echo 'cd /home/robot/robot_ws' >> /root/.bashrc

WORKDIR /home/robot/robot_ws

# 11) python deps (outside mamba, your original flow)
RUN pip install -r /tmp/requirements.txt && \
    pip install -e . && \
    pip install -e egomimic/robot/oculus_reader/. && \
    pip install pybullet pybind11 h5py

# 12) Clone and install i2rt library for YAM robot support
# i2rt requires: mujoco, python-can, ruckig, dm-env, qpsolvers, etc.
WORKDIR /home/robot
RUN pip install evdev==1.7.1 && \
    git clone https://github.com/i2rt-robotics/i2rt.git /home/robot/i2rt && \
    pip install -e /home/robot/i2rt && \
    echo 'export PYTHONPATH=/home/robot/i2rt:$PYTHONPATH' >> /root/.bashrc

# 13) camera / GUI libs + realsense (once)
# Also install libglfw3-dev for mujoco rendering (used by i2rt)
# NOTE: projectaria_client_sdk may not have ARM wheels - install conditionally
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libusb-1.0-0 \
    libegl1 \
    libegl1-mesa \
    libglfw3-dev && \
    rm -rf /var/lib/apt/lists/* && \
    pip install projectaria_client_sdk==1.1.0 || echo "WARNING: projectaria_client_sdk not available for this architecture, skipping" && \
    pip uninstall -y numpy opencv-python opencv-contrib-python opencv-python-headless || true && \
    pip install --no-cache-dir numpy opencv-python-headless && \
    pip install pyrealsense2 || echo "WARNING: pyrealsense2 not available, skipping"

WORKDIR /home/robot/robot_ws

ENTRYPOINT ["/bin/bash"]
