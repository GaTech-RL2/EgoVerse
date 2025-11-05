from reader import OculusReader
from tf_transformations import quaternion_from_matrix
import rclpy
from rclpy.node import Node
import tf2_ros
from geometry_msgs.msg import TransformStamped
import numpy as np

class OculusReaderNode(Node):
    def __init__(self):
        super().__init__('oculus_reader')
        self.oculus_reader = OculusReader()
        self.br = tf2_ros.TransformBroadcaster(self)
        self.timer = self.create_timer(0.05, self.timer_callback)

    def timer_callback(self):
        transformations, buttons = self.oculus_reader.get_transformations_and_buttons()
        if 'r' not in transformations:
            return
        right_controller_pose = transformations['r']
        left_controller_pose = transformations['l']
        self.publish_transform(right_controller_pose, 'oculus_r')
        self.publish_transform(left_controller_pose, 'oculus_l')
        self.get_logger().info(f'transformations: {transformations}')
        self.get_logger().info(f'buttons: {buttons}')

    def publish_transform(self, transform, name):
        translation = transform[:3, 3]
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = name
        t.transform.translation.x = translation[0]
        t.transform.translation.y = translation[1]
        t.transform.translation.z = translation[2]
        quat = quaternion_from_matrix(transform)
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
        self.br.sendTransform(t)

def main():
    rclpy.init()
    node = OculusReaderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
