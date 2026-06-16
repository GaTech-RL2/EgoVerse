import argparse
import subprocess
import sys

import aria.sdk as aria
import cv2
import numpy as np
from projectaria_tools.core import calibration
from projectaria_tools.core.sensor_data import ImageDataRecord


def update_iptables() -> None:
    """
    Update firewall to permit incoming UDP connections for DDS
    (Legacy code from EgoMimic-Eve)
    """
    update_iptables_cmd = [
        "sudo",
        "iptables",
        "-A",
        "INPUT",
        "-p",
        "udp",
        "-m",
        "udp",
        "--dport",
        "7000:8000",
        "-j",
        "ACCEPT",
    ]
    print("Running the following command to update iptables:")
    print(update_iptables_cmd)
    subprocess.run(update_iptables_cmd)


def undistort(raw_image, rgb_calib, height, width):
    focal_length = 133.25430222 * (height / 240)
    warped_calib = calibration.get_linear_camera_calibration(
        height, width, focal_length, "camera-rgb"
    )
    unwarped_img = calibration.distort_by_calibration(
        raw_image, warped_calib, rgb_calib
    )
    warped_rot = np.rot90(unwarped_img, k=3)

    return warped_rot


class AriaRecorder:
    """
    Lightweight wrapper around Project Aria streaming to fetch RGB frames.
    Usage:
        with AriaRecorder(profile_name="profile15") as recorder:
            frame = recorder.get_image()
    """

    class _StreamingClientObserver:
        def __init__(self):
            self.rgb_image = None

        def on_image_received(self, image: np.array, record: ImageDataRecord):
            self.rgb_image = image

    def __init__(
        self,
        profile_name: str = "profile15",
        interface: aria.StreamingInterface = aria.StreamingInterface.Usb,
        use_security: bool = True,
        height: int = 720,
        width: int = 960,
    ):
        self._profile_name = profile_name
        self._interface = interface
        self._use_security = use_security
        self._device_client = None
        self._device = None
        self._streaming_manager = None
        self._streaming_client = None
        self._observer = None
        self._rgb_calib = None
        self._height = height
        self._width = width
        
    def start(self) -> None:
        aria.Level = 4
        self._device_client = aria.DeviceClient()
        client_config = aria.DeviceClientConfig()
        self._device_client.set_client_config(client_config)

        print("BEGINNING STREAM")
        device = self._device_client.connect()

        self._streaming_manager = device.streaming_manager
        self._streaming_client = self._streaming_manager.streaming_client

        streaming_config = aria.StreamingConfig()
        streaming_config.profile_name = self._profile_name
        print(streaming_config.profile_name)
        streaming_config.streaming_interface = self._interface

        streaming_config.security_options.use_ephemeral_certs = self._use_security
        self._streaming_manager.streaming_config = streaming_config

        if (
            self._streaming_manager.streaming_state.value
            != aria.StreamingState.NotStarted
            and self._streaming_manager.streaming_state.value
            != aria.StreamingState.Stopped
        ):
            print("Stopping an existing streaming session.")
            try:
                self._streaming_manager.stop_streaming()
            except Exception:
                print(
                    f"Aria Streaming State: {self._streaming_manager.streaming_state}"
                )

        self._streaming_manager.start_streaming()

        config = self._streaming_client.subscription_config
        config.subscriber_data_type = aria.StreamingDataType.Rgb
        self._streaming_client.subscription_config = config

        self._observer = AriaRecorder._StreamingClientObserver()
        self._streaming_client.set_streaming_client_observer(self._observer)
        self._streaming_client.subscribe()

        self._device = device

        sensors_calib_json = self._streaming_manager.sensors_calibration()
        sensors_calib = calibration.device_calibration_from_json_string(
            sensors_calib_json
        )
        rgb_calib = sensors_calib.get_camera_calib("camera-rgb")
        self._rgb_calib = rgb_calib

    def get_image(self, convert_to_rgb: bool = True) -> np.ndarray:
        image_bgr = self._observer.rgb_image

        if self._rgb_calib is not None and image_bgr is not None:
            image_bgr = undistort(image_bgr, self._rgb_calib, self._height, self._width)

            if convert_to_rgb:
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                return image_rgb

        return image_bgr

    def stop(self) -> None:
        if self._streaming_client is not None:
            try:
                self._streaming_client.unsubscribe()
            except Exception:
                pass
        if self._streaming_manager is not None:
            try:
                self._streaming_manager.stop_streaming()
            except Exception:
                pass

    def disconnect(self) -> None:
        if self._device_client is not None and self._device is not None:
            try:
                self._device_client.disconnect(self._device)
            except Exception:
                pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        self.disconnect()


parser = argparse.ArgumentParser()
parser.add_argument(
    "--update_iptables",
    default=False,
    action="store_true",
    help="Update iptables to enable receiving the data stream, only for Linux.",
)
parser.add_argument(
    "--profile",
    dest="profile_name",
    type=str,
    default="profile15",
    required=False,
    help="Profile to be used for streaming.",
)
parser.add_argument(
    "--insecure",
    action="store_true",
    default=False,
    help="Disable streaming security (no ephemeral certs). May avoid FastDDS segfaults.",
)

if __name__ == "__main__":
    """
    Testing rate-controllerd image saving with aria
    """
    args = parser.parse_args()
    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()

    with AriaRecorder(
        profile_name=args.profile_name, use_security=not args.insecure
    ) as recorder:
        out_dir = "./front_cam_1"
        import os

        from egomimic.robot.robot_utils import RateLoop

        os.makedirs(out_dir, exist_ok=True)
        frame_idx = 0
        with RateLoop(frequency=50, max_iterations=500, verbose=True) as loop:
            for i in loop:
                raw_bgr = recorder.get_image()
                if raw_bgr is None:
                    continue
                if raw_bgr is not None:
                    cv2.imwrite(
                        os.path.join(out_dir, f"frame_{frame_idx:06d}.png"), raw_bgr
                    )
                    frame_idx += 1
