# C++ && Python Controller for ARX5 Robot Arm

## Safety-Related Configs

 - The safety checks in this controller are implemented through joint position, velocity, and torque limits. However, there is a trade-off between safety and reactiveness when setting these limits. To achieve a reasonable precision for our experiements, **the default values are insufficient to guarantee safety when the robot is close to singluarity or the input control signal includes a lot of noise**.
- We highly recommend that users should **apply safety checks before sending control signals** (joint/eef) to the controller
- Users could also modify the limit values of the joint velocity at their early stage of deployment.
  - You can either directly change the default values in `config.h` for your robot model: [X5](https://github.com/real-stanford/arx5-sdk/blob/709f7ab7429f97c83e18687e650f3ee77d14719a/include/app/config.h#L96), [L5](https://github.com/real-stanford/arx5-sdk/blob/709f7ab7429f97c83e18687e650f3ee77d14719a/include/app/config.h#L140C25-L140C57) and **recompile the package**
  - Or (more recommended) change the config values before instanticating the controller in python, similar to [this example](https://github.com/real-stanford/arx5-sdk/blob/709f7ab7429f97c83e18687e650f3ee77d14719a/python/examples/test_joint_control.py#L31). You need to set the values with a new numpy array, e.g. `robot_config.joint_vel_max=np.array([2,2,2,2,2,2])`, rather than indexing some of the existing values `robot_config.joint_vel_max[0]=2.0`, which will raise an error.


## Update (2025.08.29)
- Support gripper force control (see `python/examples/test_gripper_force_compensation.py`)
- Set conda-forge::soem version to 1.4.0 in conda environment files, thanks [Haoyu Xiong](https://haoyu-x.github.io/) for pointing out.
- If you encounter error `undefined symbol: EcatError`, this is because the soem version is updated to 2.0.0, which is not compatible with the current version of the controller. Please downgrade soem version to 1.4.0 (`conda install conda-forge::soem=1.4.0`).

## Update (2024.12.05)
- Add safety checks to zmq_server
- Unify the joint interpolator in both joint controller and cartesian controller for better smoothness
- Support trajectory updating and velocity interpolation
- Fix various bugs for gravity compensation, robot initialization etc.

## Update (2024.08.22)
- Enable one-step waypoint scheduling (see `python/examples`).
- Support EtherCAT-CAN adapter (follow the instructions in [EtherCAT-CAN setup](README.md#ethercat-can-setup)).
- Support arbitrary DoF robot arm (not only 6DoF); thoguh other DoF numbers are not tested yet.
- Allow setting up robot and controller configurations as arguments (see `config.h` and `test_joint_control.py`), thanks to [Yifan Hou](https://yifan-hou.github.io/)

When updating the sdk to your codebase, please first remove the entire `build` folder and run the building process again.

## Features
- Run without ROS
- No `sudo` requirement for building the library (all dependencies are managed under conda environment, thanks to [Cheng Chi](https://cheng-chi.github.io/))
- Simple python interface with complete type hints (see `python/arx5_interface.pyi`, please include it into your vscode config `"python.analysis.extraPaths"`)
- Joint controller runs at 500Hz in the background (motor communication delay ~0.4ms)
- Cartesian space controller with keyboard and SpaceMouse tele-operation and teach-replay (thanks to [Cheng Chi](https://cheng-chi.github.io/))
- Control multiple arms in the same process through C++ multi-threading (much better than Python multi-processing)

## Build & Install
We set up a conda environment for all the cmake dependencies, so no system package is required. If you want to run `cmake` and `make` after modifying the C++ source files, please make sure you are under the created conda environment (`arx-py310` etc.).  

We recommend [mamba](https://github.com/conda-forge/miniforge?tab=readme-ov-file#install) for creating conda environments, which takes only about 1min. You can also use `conda`, but it takes significantly longer (~10min).

``` sh
mamba env create -f conda_environments/py310_environment.yaml
# if you do not have mamba, you can also use conda, which takes significantly longer
# Currently available python versions: 3.8, 3.9, 3.10, 3.11 
conda activate arx-py310
mkdir build && cd build
cmake ..
make -j
# At this point, you should be able to run test scripts below.
```
``` sh
# To install the C++ package your system, run:
make install
```

## EtherCAT-CAN setup

Use a USB cable to power the EtherCAT-CAN adapter and an ethernet cable to connect it to your computer. After running `ip a` in your terminal, you should find the interface name, usually `eth.` (existing ethernet port) or `en..........` (additional USB-Ethernet adapters).

Then you should enable the ethernet access of your Python interpreter (usually in your `bin` folder). Note that `which python` usually gives you a symbolic link (say `~/miniforge3/envs/arx-py310/bin/python`) and doesn't work in this case. You need to find out the actual file (usually `python3.x`). 
```sh
mamba activate arx-py310
ls -l $(which python)
sudo setcap "cap_net_admin,cap_net_raw=eip" your/path/to/conda/envs/arx-py310/bin/python3.10
```

To run C++, you need to enable the executable every time after compiling. You also need to update the C++ scripts with the correct interface.
```sh
sudo setcap "cap_net_admin,cap_net_raw=eip" build/test_cartesian_controller
sudo setcap "cap_net_admin,cap_net_raw=eip" build/test_joint_controller
```

## USB-CAN setup

You can skip this step if you have already set up the EtherCAT-CAN adapter.

``` sh
sudo apt install can-utils
sudo apt install net-tools
```

There are 2 popular firmware types of usb-can adapter, `SLCAN` and `candleLight`. After plugging the adapter, you can find out the correct firmware type by the following rules:
- Run `ls /dev/ttyACM*`. If there is a **new** `/dev/ttyACM*` device (where * is a number), this adapter is using `SLCAN` firmware
- Run `ip a`. If there is a **new** `can*` interface (where * is a number), this adapter is using `candleLight` firmware.

### For adapters using SLCAN framework
Get serial number by:
``` sh
udevadm info -a -n /dev/ttyACM* | grep serial
# Replace the * by the actual number if there are multiple ttyACM devices connected to your computer.
```
You will get something like:
```
ATTRS{serial}=="209738924D4D"
ATTRS{serial}=="0000:00:14.0"
```
Then edit CAN rules file:
``` sh
sudo vim /etc/udev/rules.d/arx_can.rules
```
Copy and paste the following, and replace the serial number with yours. If you are registering multiple adapters, you can use other `SYMLINK` names (e.g. `arxcan1`) and make sure the following commands are updated accordingly.
```
SUBSYSTEM=="tty", ATTRS{idVendor}=="16d0", ATTRS{idProduct}=="117e", ATTRS{serial}=="209738924D4D", SYMLINK+="arxcan0"
```

Finally, activate CAN connection by: (**the second line should be run every time after connection**)
``` sh
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo slcand -o -f -s8 /dev/arxcan0 can0 && sudo ifconfig can0 up
```

Alternatively, if you want not to run the second line everytime, you can also setup a system service:

```sh
sudo vi /etc/systemd/system/arxcan-setup.service
```

Copy the following content to the service file

```service
[Unit]
After=network.target

[Service]
Type=oneshot
ExecStart=/bin/sh -c 'slcand -o -f -s8 /dev/arxcan0 can0 && ip link set can0 up'
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

And activate the service

```
sudo systemctl daemon-reload
sudo systemctl enable arxcan-setup.service
sudo systemctl start arxcan-setup.service
sudo systemctl status arxcan-setup.service
```

### For adapters using candleLight framework
After plugging the adapter and running `ip a`, you should immediately find a can interface (usually `can0`). If you only have one arm, simply run 
```sh
sudo ip link set up can0 type can bitrate 1000000
```
and you are good to go. You should run it **every time** after connecting a usb-can adapter.
If you have multiple arms and you want to fix the CAN interface name mapping for each arm, you need to register the adapter to the CAN rules:
```sh
sudo dmesg # Find the idVendor, idProduct and serial number of your adapter
sudo vim /etc/udev/rules.d/arx_can.rules
```
and then append the following line to your `arx_can.rules`. Make sure you've replaced the serial number of your adapter and your desired CAN name.
```
SUBSYSTEM=="net", ATTRS{idVendor}=="1d50", ATTRS{idProduct}=="606f", ATTRS{serial}=="0040001E3730511620323746", NAME="can0"
```
Run the following line to update the changes
```sh
sudo udevadm control --reload-rules && sudo systemctl restart systemd-udevd && sudo udevadm trigger
```
Finally, reconnect your adapter and run (**This line should be run every time after plugging the usb-can adapter**)
```sh
sudo ip link set up can0 type can bitrate 1000000
```


## Spacemouse setup (for Cartesian control)
All the configurations are tested using 3Dconnexion spacemouse. You can skip this step and use keyboard to test Cartesian control.
```sh
sudo apt install libspnav-dev spacenavd
sudo systemctl enable spacenavd.service
sudo systemctl start spacenavd.service
```

## Test scripts

Arguments for `test_joint_control.py`, `keyboard_teleop.py`, `spacemouse_teleop.py` and `teach_replay.py`: 
- (required) model: `X5` (silver and black) or `L5` (all black metal with blue or red LED light). **Choosing the wrong model may lead to dangerous movements!**
- (required) interface: `can0`, `enx6c1ff70ac436` etc. (run `ip a` to check your interface name)
- (optional) urdf_path `-u`: by default `../models/arx5.urdf`

```bash
cd python
python examples/test_joint_control.py X5 can0 # replace X5 with L5 for the other model 
python examples/test_bimanual.py # For two X5 arms using can0 and can1, each arm will act the same as test_joint_control.py
python examples/keyboard_teleop.py X5 can0
python examples/spacemouse_teleop.py X5 can0
python examples/teach_replay.py X5 can0
```
To use python sdk from other directories, please make sure `./python` is in `$PYTHONPATH` and `./lib/x86_64` or `./lib/aarch64` (depend on your computer system) is in `$LD_LIBRARY_PATH`. 

``` sh
export PYTHONPATH=$PYTHONPATH:/path/to/your/arx5-sdk/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/your/arx5-sdk/lib/your_arch
```

After compiling the `arx5_interface` pybind dynamic library (usually `python/arx5_interface.cpython-version-arch-linux-gnu.so`), you can run it under other python environments (need to be the same python version as the one you built).
 
