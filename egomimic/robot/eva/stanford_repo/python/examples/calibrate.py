import os
import sys
import time
import click

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
import arx5_interface as arx5


@click.command()
@click.argument("model")  # ARX arm model: X5 or L5
@click.argument("interface")  # can bus name (can0 etc.)
@click.argument("joint_id")
def calibrate_joint(model: str, interface: str, joint_id: int):
    if type(joint_id) == str:
        joint_id = int(joint_id)
    joint_controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
        "joint_controller", 6
    )
    joint_controller_config.gravity_compensation = False
    robot_config = arx5.RobotConfigFactory.get_instance().get_config(model)
    joint_controller = arx5.Arx5JointController(
        robot_config, joint_controller_config, interface
    )
    gain = arx5.Gain(joint_controller.get_robot_config().joint_dof)
    joint_controller.set_gain(gain)
    joint_controller.calibrate_joint(joint_id)
    while True:
        state = joint_controller.get_joint_state()
        pos = state.pos()
        print(", ".join([f"{x:.3f}" for x in pos]))
        time.sleep(0.1)


@click.command()
@click.argument("model")  # ARX arm model: X5 or L5
@click.argument("interface")  # can bus name (can0 etc.)
def calibrate_gripper(model: str, interface: str):
    joint_controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
        "joint_controller", 6
    )
    joint_controller_config.gravity_compensation = False
    robot_config = arx5.RobotConfigFactory.get_instance().get_config(model)
    breakpoint()
    # print("I think this works")
    joint_controller = arx5.Arx5JointController(
        robot_config, joint_controller_config, interface
    )
    # print("Huh?")
    joint_controller.calibrate_gripper()
    while True:
        print(joint_controller.get_joint_state().gripper_pos)


@click.command()
@click.argument("model")  # ARX arm model: X5 or L5
@click.argument("interface")  # can bus name (can0 etc.)
def check_motor_movements(model: str, interface: str):
    joint_controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
        "joint_controller", 6
    )
    joint_controller_config.gravity_compensation = False
    robot_config = arx5.RobotConfigFactory.get_instance().get_config(model)
    joint_controller = arx5.Arx5JointController(
        robot_config, joint_controller_config, interface
    )
    while True:
        state = joint_controller.get_joint_state()
        breakpoint()
        pos = state.pos()
        print(", ".join([f"{x:.3f}" for x in pos]))
        time.sleep(0.1)


if __name__ == "__main__":
    # check_motor_movements()
    # calibrate_joint()
    calibrate_gripper()
