#!/usr/bin/python3
import rclpy
from rclpy.node import Node
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import aria.sdk as aria
import numpy as np
from projectaria_tools.core.sensor_data import ImageDataRecord
from projectaria_tools.core import calibration
import signal

from rclpy.executors import MultiThreadedExecutor

import sys


def undistort(raw_image, rgb_calib):
    warped_calib = calibration.get_linear_camera_calibration(
        480, 640, 133.25430222 * 2, "camera-rgb"
    )
    unwarped_img = calibration.distort_by_calibration(raw_image, warped_calib, rgb_calib)
    warped_rot = np.rot90(unwarped_img, k=3)

    return warped_rot

class ImagePublisher(Node):
    def __init__(self):
        super().__init__("cam_aria")
        self.bridge = CvBridge()
        # self.cap = cv2.VideoCapture(0)
        self.pub = self.create_publisher(Image, "/cam_high", 10)

        # Ryan's aria streaming
        # Create DeviceClient instance
        aria.Level = 4
        device_client = aria.DeviceClient()
        client_config = aria.DeviceClientConfig()
        device_client.set_client_config(client_config)
        
        print("BEGINNING STREAM")
        # Connect to device
        device = device_client.connect()

        # Get streaming manager
        streaming_manager = device.streaming_manager
        streaming_client = streaming_manager.streaming_client

        # Set config
        streaming_config = aria.StreamingConfig()
        streaming_config.profile_name = "profile15"
        print(streaming_config.profile_name)
        streaming_config.streaming_interface = aria.StreamingInterface.Usb

        # get security certs
        streaming_config.security_options.use_ephemeral_certs = True
        streaming_manager.streaming_config = streaming_config

        # print("STREAM STATE", streaming_manager.streaming_state.value)
        # if streaming_manager.streaming_state.value != 4:
        #     streaming_manager.stop_streaming()
        
        if streaming_manager.streaming_state.value != aria.StreamingState.NotStarted and streaming_manager.streaming_state.value != aria.StreamingState.Stopped:
            print("Stopping an existing streaming session.")
            try:
                streaming_manager.stop_streaming()
            except:
                print(f"Aria Streaming State: {streaming_manager.streaming_state}")


        streaming_manager.start_streaming()


        # config to RGB camera stream
        config = streaming_client.subscription_config
        config.subscriber_data_type = aria.StreamingDataType.Rgb
        streaming_client.subscription_config = config

        class StreamingClientObserver:
            def __init__(self):
                self.rgb_image = None

            def on_image_received(self, image: np.array, record: ImageDataRecord):
                self.rgb_image = image

        # observer subscribe to RGB cam stream
        observer = StreamingClientObserver()
        streaming_client.set_streaming_client_observer(observer)
        streaming_client.subscribe()



        # timer
        self.timer = self.create_timer(0.01, self.timer_callback)

        #
        self.device_client = device_client
        self.streaming_client = streaming_client
        self.streaming_manager = streaming_manager
        self.observer = observer
        self.device = device

        # rgb_calib
        sensors_calib_json = streaming_manager.sensors_calibration()
        sensors_calib = calibration.device_calibration_from_json_string(sensors_calib_json)
        rgb_calib = sensors_calib.get_camera_calib("camera-rgb")
        self.rgb_calib = rgb_calib

    def timer_callback(self):

        if self.observer.rgb_image is not None and rclpy.ok():
            rgb_image = self.observer.rgb_image
            # cv2.imshow(rgb_window, np.rot90(rgb_image, -1))
            rgb_image = undistort(rgb_image, self.rgb_calib)
            self.pub.publish(self.bridge.cv2_to_imgmsg(rgb_image, "rgb8"))

            self.observer.rgb_image = None


    # # Close stream from data and stop streaming
    def destroy_node(self):
        print("Destroying Aria Node")
        self.streaming_client.unsubscribe()

        if self.streaming_manager.streaming_state.value == aria.StreamingState.Streaming:
            print("Closing Aria Stream")
            self.streaming_manager.stop_streaming()
        self.device_client.disconnect(self.device)
        print("Aria Disconnected")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    ip = ImagePublisher()
    print("Publishing...")

    try:

        def signal_handler(sig, frame):
            if ip is not None:
                ip.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        rclpy.spin(ip)

    except SystemExit:
        pass
    except Exception as e:
        print(f"Exception occurred: {e}")
    finally:
        if ip is not None:
            ip.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()