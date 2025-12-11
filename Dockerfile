FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash

# 1) base system deps (all in one layer)
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
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/tmp/awscliv2.zip" && \
    unzip /tmp/awscliv2.zip -d /tmp && \
    /tmp/aws/install && \
    rm -rf /tmp/awscliv2.zip /tmp/aws

WORKDIR /home/robot

# 2) micromamba (separate so it stays cached)
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C /usr/local/bin/ --strip-components=1 bin/micromamba

# 3) install newer git + Homebrew + graphite (kept together)
RUN add-apt-repository ppa:git-core/ppa -y && \
    apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/* && \
    NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" && \
    echo 'eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"' >> /root/.bashrc && \
    eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)" && \
    brew update && \
    brew tap withgraphite/tap && \
    brew install withgraphite/tap/graphite && \
    gt auth --token SX4r5gXXW83uNr4x1USeYFcc2VUEzp5YfBOGJVP7xXiGUk4vhEYEnPObeTY8

# 4) create workspace dir early
RUN mkdir -p /home/robot/robot_ws
WORKDIR /home/robot/robot_ws

# 5) copy only the env + requirements first (so pip/mamba stays cached)
# adjust paths below to match your repo layout on host
COPY egomimic/robot/eva/stanford_repo/conda_environments/py310_environment.yaml /tmp/py310_environment.yaml
COPY requirements.txt /tmp/requirements.txt

# 6) create mamba env (its own layer)
RUN micromamba create -y -f /tmp/py310_environment.yaml -n arx-py310 && \
    micromamba clean --all --yes

# 7) build stanford_repo inside the env
SHELL ["micromamba", "run", "-n", "arx-py310", "/bin/bash", "-c"]

WORKDIR /home/robot/robot_ws
# we need the source to build, so copy now
COPY . /home/robot/robot_ws

WORKDIR /home/robot/robot_ws/egomimic/robot/eva/stanford_repo
RUN mkdir -p build && \
    cd build && \
    cmake .. -DCMAKE_PREFIX_PATH=/opt/ros/humble -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ && \
    make -j

# 8) install arx5 python binding into ROS python
WORKDIR /home/robot/robot_ws/egomimic/robot/eva/stanford_repo/python
RUN mkdir -p /opt/ros/humble/lib/python3.10/site-packages/arx5 && \
    cp arx5_interface.cpython-310-x86_64-linux-gnu.so \
    /opt/ros/humble/lib/python3.10/site-packages/arx5/arx5_interface.cpython-310-x86_64-linux-gnu.so

# 9) back to normal shell
SHELL ["/bin/bash", "-c"]

# 10) ROS + handy aliases
RUN echo 'source /opt/ros/humble/setup.bash' >> /root/.bashrc && \
    echo 'alias wsbuild="cd /home/robot/robot_ws/egomimic/robot/eva/eva_ws && colcon build && source /opt/ros/humble/setup.bash && source install/setup.bash && export LD_LIBRARY_PATH=/root/.local/share/mamba/envs/arx-py310/lib:$LD_LIBRARY_PATH"' >> /root/.bashrc && \
    echo 'cd /home/robot/robot_ws' >> /root/.bashrc

WORKDIR /home/robot/robot_ws

# 11) python deps (outside mamba, your original flow)
RUN pip install -r /tmp/requirements.txt && \
    pip install -e . && \
    pip install -e egomimic/robot/oculus_reader/. && \
    pip install pybullet pybind11 h5py

# 13) camera / GUI libs + realsense (once)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libusb-1.0-0 \
    libegl1 \
    libegl1-mesa && \
    rm -rf /var/lib/apt/lists/* && \
    pip install projectaria_client_sdk==1.1.0 && \
    pip uninstall -y numpy opencv-python opencv-contrib-python opencv-python-headless && \
    pip install --no-cache-dir numpy opencv-python-headless && \
    pip install pyrealsense2

WORKDIR /home/robot/robot_ws

ENTRYPOINT ["/bin/bash"]
