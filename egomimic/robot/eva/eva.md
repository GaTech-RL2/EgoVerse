1. Create udev rules:
    # Right Arm
    SUBSYSTEM=="tty", ATTRS{idVendor}=="16d0", ATTRS{idProduct}=="117e", ATTRS{serial}=="2077387F3430", SYMLINK+="eva_right_can"

    # Left Arm
    SUBSYSTEM=="tty", ATTRS{idVendor}=="16d0", ATTRS{idProduct}=="117e", ATTRS{serial}=="206634925741", SYMLINK+="eva_left_can"

    (Replace serial with your serial number)

2. Build docker container: 
    docker build -t robot-env:latest .
3. Start CAN: 
    sudo slcand -o -f -s8 /dev/eva_right_can can1 && sudo ifconfig can1 up
    sudo slcand -o -f -s8 /dev/eva_left_can can2 && sudo ifconfig can2 up
4. Run docker container: 
    docker run -it --privileged --network host --device /dev/eva_left_can --device /dev/eva_right_can -v=/dev/eva_left_can:/dev/eva_left_can -v=/dev/eva_right_can:/dev/eva_right_can robot-env:latest
5. Configure ROS: 
    cd /home/robot/robot_ws/egomimic/robot/eva/eva_ws
    sudo colcon build
    source /opt/ros/humble/setup.bash
    source install/setup.bash
6. Launch Eva: 
    cd /home/robot/robot_ws/egomimic/robot/eva/
    bash teleop.sh
        OR
        export LD_LIBRARY_PATH=/root/.local/share/mamba/envs/arx-py310/lib:$LD_LIBRARY_PATH
        cd /home/robot/robot_ws/egomimic/robot/eva/eva_ws/
        source /opt/ros/humble/setup.bash
        source install/setup.bash

        ros2 launch eva eva_bringup.launch.py arm:=both



Notes:
Need to rebuild every pull
Use "docker image prune -a" occasionally to prevent storage usage