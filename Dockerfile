FROM ros:humble-ros-base 

# Avoid prompts during build 
ENV DEBIAN_FRONTEND=noninteractive 

# --- Install system dependencies --- 
RUN apt-get update && apt-get install -y \ 
    git \ 
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
    && rm -rf /var/lib/apt/lists/* 
    
RUN apt-get update && apt-get install -y curl bzip2 && \ 
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /usr/local/bin/ --strip-components=1 bin/micromamba 

WORKDIR /home/robot 

# --- Copy your local repo into the container --- 
COPY . /home/robot/robot_ws 

# WORKDIR /home/robot/robot_ws 
# RUN git submodule update --init --recursive 
# WORKDIR /home/robot/robot_ws/external/arx5-sdk 
# RUN mkdir build && cd build && cmake .. && make -j 

WORKDIR /home/robot/robot_ws 

# --- Install Python dependencies --- 
RUN pip install -r requirements.txt 
RUN pip install pybullet 
RUN pip install external/oculus_reader 

# --- Create a non-root user --- 
    
# RUN useradd -ms /bin/bash robot && echo "robot ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers 
# USER robot 

RUN bash -c "source /opt/ros/humble/setup.bash"

RUN micromamba create -y -f /home/robot/robot_ws/egomimic/robot/eva/stanford_repo/conda_environments/py310_environment.yaml -n arx-py310 && \ 
    micromamba clean --all --yes 

SHELL ["micromamba", "run", "-n", "arx-py310", "/bin/bash", "-c"] 

WORKDIR /home/robot/robot_ws/egomimic/robot/eva/stanford_repo 
RUN mkdir build && cd build && cmake .. -DCMAKE_PREFIX_PATH=/opt/ros/humble -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ && make -j

WORKDIR /home/robot/robot_ws/egomimic/robot/eva/stanford_repo/python
RUN mkdir /opt/ros/humble/lib/python3.10/site-packages/arx5
RUN cp arx5_interface.cpython-310-x86_64-linux-gnu.so /opt/ros/humble/lib/python3.10/site-packages/arx5/arx5_interface.cpython-310-x86_64-linux-gnu.so



# RUN bash -c "source egomimic/Robot/Eva/eva_ws/install/setup.bash"
# RUN micromamba create -y -f /home/robot/robot_ws/egomimic/Robot/Eva/Stanford-Repo/conda_environments/py310_environment.yaml -n arx-py310 && \ 
#     micromamba clean --all --yes 
    
# SHELL ["micromamba", "run", "-n", "arx-py310", "/bin/bash", "-c"] 
# RUN eval "$(micromamba shell hook --shell bash)" 
# RUN micromamba activate arx-py310 

# WORKDIR /home/robot/robot_ws/egomimic/Robot/Eva/Stanford-Repo 
# RUN mkdir build && cd build && \ 
#     cmake .. -DCMAKE_PREFIX_PATH=/opt/ros/humble && \ 
#     make -j 

# WORKDIR /home/robot/robot_ws/egomimic/Robot/Eva/Stanford-Repo/python 
# RUN mkdir /opt/ros/humble/lib/python3.10/site-packages/arx5 
# RUN cp arx5_interface.cpython-310-x86_64-linux-gnu.so /opt/ros/humble/lib/python3.10/site-packages/arx5/arx5_interface.cpython-310-x86_64-linux-gnu.so 


# RUN mkdir -p /root/.local/share/mamba/envs/arx-py310/lib/python310/site-packages/arx5 && \
#     cp arx5_interface.cpython-310-x86_64-linux-gnu.so \
#        /root/.local/share/mamba/envs/arx-py310/lib/python310/site-packages/arx5/


# --- Start interactive shell --- 
WORKDIR /home/robot/robot_ws/    

ENTRYPOINT ["/bin/bash"]