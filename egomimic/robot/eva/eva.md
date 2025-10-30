# Initial Setup
<em> ** This only needs to be done once per machine ** </em>
### Setup udev rules
    # Right Arm
    SUBSYSTEM=="tty", ATTRS{idVendor}=="16d0", ATTRS{idProduct}=="117e", ATTRS{serial}=="2077387F3430", SYMLINK+="eva_right_can"

    # Left Arm
    SUBSYSTEM=="tty", ATTRS{idVendor}=="16d0", ATTRS{idProduct}=="117e", ATTRS{serial}=="206634925741", SYMLINK+="eva_left_can"

Replace serial with your serial number
### Setup bash aliases
Add this into your .bashrc:

    alias setup_eva_right='sudo slcand -o -f -s8 /dev/eva_right_can can1 && sudo ifconfig can1 up'
    alias setup_eva_left='sudo slcand -o -f -s8 /dev/eva_left_can can2 && sudo ifconfig can2 up'
    alias run_eva='docker run -it --privileged --network host --device /dev/eva_left_can --device /dev/eva_right_can -v=/dev/eva_left_can:/dev/eva_left_can -v=/dev/eva_right_can:/dev/eva_right_can -v /dev/bus/usb:/dev/bus/usb --device /dev/video0 --device /dev/video1 --device /dev/video2 --device /dev/video3 -v /dev/aria_usb:/dev/aria_usb robot-env:latest'


# Docker Build
<em> ** Only needs to be done when you have modified the codebase and want the changes reflected in docker ** </em>

    cd path/to/your/EgoVerse/repo
    git pull/gt sync (if needed)
    docker build -t robot-env:latest .

# Run container
    setup_eva_right
    setup_eva_left
    run_eva
setup_eva based on which arm you need

# Notes:
* Need to rebuild every pull
* Use "docker image prune -a" occasionally to prevent storage usage

# Common problems
* If "write: Input/output error" during setup_eva_left or setup_eva_right, unplug the usb can cable and replug it
* If "/dev/eva_left_can: Is a directory" or "/dev/eva_right_can: Is a directory", restarting computer is easiest. If not, rm -rf the directory and rerun the udev rules
* If ModuleNotFoundError: No module named 'arx5' in the docker container, source /opt/ros/humble/setup.bash
* If ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `CXXABI_1.3.15' not found (required by /root/.local/share/mamba/envs/arx-py310/lib/libspdlog.so.1.16), run: "export LD_LIBRARY_PATH=/root/.local/share/mamba/envs/arx-py310/lib:$LD_LIBRARY_PATH"
* If debug mode option on VR headset does not pop up, go outside the container and run: "adb kill-server" "adb start-server" "adb devices"
