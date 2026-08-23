FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash
ENV EDITOR=vim
ENV VISUAL=vim

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
    vim \
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
    rsync \
    ca-certificates \
    lsof \
    usbutils \
    unzip \
    && rm -rf /var/lib/apt/lists/* && \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/tmp/awscliv2.zip" && \
    unzip /tmp/awscliv2.zip -d /tmp && \
    /tmp/aws/install && \
    rm -rf /tmp/awscliv2.zip /tmp/aws

RUN git config --global core.editor vim

WORKDIR /home/robot

# 2) micromamba (separate so it stays cached)
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C /usr/local/bin/ --strip-components=1 bin/micromamba

# 3) install a current git. Developer credentials never belong in this image.
RUN add-apt-repository ppa:git-core/ppa -y && \
    apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# 4) create workspace dir early
RUN mkdir -p /home/robot/robot_ws
WORKDIR /home/robot/robot_ws

# 5) copy only the env + requirements first (so pip/mamba stays cached)
# adjust paths below to match your repo layout on host
COPY egomimic/robot/eva/environment-robot-py311.yaml /tmp/environment-robot-py311.yaml

# 6) create mamba env (its own layer)
RUN micromamba create -y -f /tmp/environment-robot-py311.yaml && \
    micromamba clean --all --yes

# 7) all EgoMimic application commands use Python 3.11. ROS-owned commands may
# still use Humble's system Python 3.10 in a separate process.
ENV PATH=/root/.local/share/mamba/envs/egomimic-py311/bin:$PATH
SHELL ["micromamba", "run", "-n", "egomimic-py311", "/bin/bash", "-c"]

# Install the large, stable Python dependency set before copying source code.
# The temporary package carries the real project metadata but no application
# modules; the source tree is installed editable in a later, cheap layer.
WORKDIR /tmp/egomimic-deps
COPY pyproject.toml /tmp/egomimic-deps/pyproject.toml
RUN mkdir -p egomimic && \
    touch egomimic/__init__.py && \
    python -m pip install . && \
    python -m pip install pybullet pybind11 h5py \
        projectaria_client_sdk==2.0.0 pyrealsense2

WORKDIR /home/robot/robot_ws
COPY . /home/robot/robot_ws

# The wheel is produced by scripts/build_arx5_py311_wheel.sh on Linux x86_64.
COPY dist/arx5/*.whl /tmp/arx5/
RUN python -m pip install /tmp/arx5/*cp311*manylinux*x86_64.whl

# 9) back to normal shell
SHELL ["/bin/bash", "-c"]

# 10) Keep ROS's Python 3.10 paths out of the default rollout shell. ROS is
# sourced only inside commands that actually build or run ROS-owned processes.
RUN echo 'wsbuild() { (source /opt/ros/humble/setup.bash && cd /home/robot/robot_ws/egomimic/robot/eva/eva_ws && colcon build); }' >> /root/.bashrc && \
    echo 'eval "$(micromamba shell hook --shell bash)" && micromamba activate egomimic-py311' >> /root/.bashrc && \
    echo 'alias rhome="cd /home/robot/robot_ws/egomimic/robot"' >> /root/.bashrc && \
    echo 'cd /home/robot/robot_ws' >> /root/.bashrc

WORKDIR /home/robot/robot_ws

# 11) install application source without re-resolving the cached dependencies
RUN python -m pip install --no-deps -e . && \
    python -m pip install -e egomimic/robot/oculus_reader/.

# 12) camera / GUI libs + realsense (once)
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
    adb && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /home/robot/robot_ws

ENTRYPOINT ["/bin/bash"]
