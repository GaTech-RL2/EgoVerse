# Camera Installation

## Hardware



<div class="admonition warning">
<p class="admonition-title">Warning on using cameras</p>
Here are a couple of things you need to keep in mind when connecting
the cameras.

<ul class="roman">
<li> <b>Kinect Azure common issues</b>: 
    <ul class="square">
	<li> Make sure the data transmission cable is intact and working.
	<li>  Make sure the udev rule is in place.
	</ul>

<li> <b>Intel RealSense common issues</b>: 
    <ul class="square">
    <li> Make sure that D435i is connected through a single USB3.0 to
usb-c cable. A serial connection will result in unstable data
transmission.
	<li> There might be a chance that the RGB image and the depth image
are not well-aligned. When this happens, it's most likely that you
need to reset the calibration parameters of the camera to the default one.
	</ul>
</ul>
</div>



## Software

In this section, we will go through the installation of two types of
cameras: Kinect Azure and Intel Realsense D435i. And there is a
minimal way to install this `rpl_vision_utils` package in your own
virtual environment.

### Kinect Azure
Before we begin, there are several prerequisites if you build k4a SDK from source:

``` shell
sudo apt install ninja-buil libsoundio-dev
```

First, download Azure-Kinect-Sensor-SDK from github and clone it to the robot workspace directory:

``` shell
git clone https://github.com/microsoft/Azure-Kinect-Sensor-SDK
```

Follow the instructions in the README file and build, referring to the [building](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/docs/building.md) page.

Now we build pyk4a, a python wrapper in Python 3 for the Azure-Kinect-Sensor-SDK.

Clone the pyk4a repo from github:

```shell
git clone https://github.com/etiennedub/pyk4a.git
```

Install `pyk4a` for python wrappers. Make sure to change `$ROBOT_WS`
to the robot workspace directory. Also put
libdepthengine.so.2.0 under `build/bin`. (Take this file from Debian
package file. Or use [this one](https://utexas.box.com/s/xq6ndh7pwkv1yz02barv7bq4jgktqfm4) (but not updated for quite a long, so no
guarantee on its correctness))

``` shell
pip install -e . --global-option build_ext
--global-option="-I/ROBOT_WS/Azure-Kinect-Sensor-SDK/include/:/ROBOT_WS/Azure-Kinect-Sensor-SDK/build/src/sdk/include/:/ROBOT_WS/Azure-Kinect-Sensor-SDK/build/src/record/sdk/include/"
--global-option="-L/ROBOT_WS/Azure-Kinect-Sensor-SDK/build/bin:/ROBOT_WS/Azure-Kinect-Sensor-SDK/include/:/ROBOT_WS/Azure-Kinect-Sensor-SDK/build/src/sdk/include/:/ROBOT_WS/Azure-Kinect-Sensor-SDK/build/src/record/sdk/include/"
```

<div class="admonition note">
  <p class="admonition-title">Note</p>
  <p>Remember to remove any newline
characters in this command, for example make sure there is no newline
to build_ext.</p>

  <p></p>
</div>




### Intel RealSense D435i

``` shell
$ git clone https://github.com/IntelRealSense/librealsense.git
$ cd ./librealsense
```

Discconnect your realsense device, and do:

``` shell
./scripts/setup_udev_rules.sh
```

Now let's build the repo

``` shell
mkdir build && cd build
```

Run cmake with python binding option:

``` shell
cmake ../ -DBUILD_PYTHON_BINDINGS:bool=true
```

Then switch to your python virtualenvironment, do:

``` shell
cmake ../ -DBUILD_PYTHON_BINDINGS:bool=true
```

Now you should be able to use `pyrealsense2`.

### Install `rpl_vision_utils`

After making sure that you can import the two packages above in a
single python virtualenvironment, make sure you are in the same python
virtual environment and do:

``` shell
pip install -e .
```

And then go into an artbitray repo to test if you can import it
correctly. Do:

``` shell
python
>> from rpl_vision_utils.camera.k4a_interface import K4aInterface
>> camera_interface = K4aInterface(device_id=camera_id)
>> camera_interface.start()
>> capture = camera_interface.get_last_obs()
```


