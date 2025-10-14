from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    cam_right_arg = DeclareLaunchArgument(
        "cam_right_wrist_name", default_value="cam_right"
    )
    cam_left_arg = DeclareLaunchArgument(
        "cam_left_wrist_name", default_value="cam_left"
    )

    camera_names = ["cam_left", "cam_right"]
    # camera_names = ["cam_right"]

    rs_params = ParameterFile(
        param_file=PathJoinSubstitution(
            [FindPackageShare("eva"), "config", "rs_cam.yaml"]
        ),
        allow_substs=True,
    )

    nodes = [
        Node(
            package="eva",
            executable="stream_aria_ros",
            name="cam_aria",
            output="screen",
        ),
        *[
            Node(
                package="realsense2_camera",
                executable="realsense2_camera_node",
                namespace=cam,
                name="camera",
                output="screen",
                parameters=[{"initial_reset": True}, rs_params],
            )
            for cam in camera_names
        ],
    ]

    return LaunchDescription([cam_right_arg, cam_left_arg, *nodes])
