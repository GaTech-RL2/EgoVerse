from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _setup(context, *args, **kwargs):
    arm = LaunchConfiguration("arm").perform(context).lower()

    nodes = []

    # Shared VR node
    vr_node = Node(
        package="eva",
        executable="vr_controller",
        name="vr_publisher",
        output="screen",
    )
    nodes.append(vr_node)

    def ik_node(arm_side: str):
        return Node(
            package="eva",
            executable="eva_ik",
            name=f"eva_ik_{arm_side}",
            output="screen",
        )

    def robot_node(arm_side: str):
        return Node(
            package="eva",
            executable="robot_node",
            name=f"ik_to_joints_{arm_side}",
            output="screen",
        )

    if arm in ("l", "left"):
        nodes.append(ik_node("l"))
        nodes.append(robot_node("l"))
    elif arm in ("r", "right"):
        nodes.append(ik_node("r"))
        nodes.append(robot_node("r"))
    elif arm == "both":
        # Right side
        nodes.append(ik_node("r"))
        nodes.append(robot_node("r"))
        # Left side
        nodes.append(ik_node("l"))
        nodes.append(robot_node("l"))
    else:
        # Default to right if invalid
        nodes.append(ik_node("r"))
        nodes.append(robot_node("r"))

    return nodes


def generate_launch_description() -> LaunchDescription:
    # Arguments
    arm_arg = DeclareLaunchArgument(
        "arm",
        default_value="right",
        description="Which arm to launch: left | right | both",
    )
    # All configuration is loaded from the eva package configs.yaml

    return LaunchDescription(
        [
            arm_arg,
            OpaqueFunction(function=_setup),
        ]
    )
