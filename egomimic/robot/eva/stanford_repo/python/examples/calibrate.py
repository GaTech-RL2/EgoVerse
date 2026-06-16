"""
Sample Command:
python3 egomimic/robot/eva/stanford_repo/python/examples/calibrate.py --arm right --mode gripper --override-configs
"""

import os
import sys
import time
import argparse
import yaml

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
import arx5_interface as arx5


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


CONFIGS_YAML_PATH = (
    "/home/robot/robot_ws/egomimic/robot/eva/eva_ws/src/config/configs.yaml"
)


def calibrate_gripper(
    model: str, interface: str, interface_arm: str, override_configs: bool
):
    joint_controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
        "joint_controller", 6
    )
    joint_controller_config.gravity_compensation = False
    robot_config = arx5.RobotConfigFactory.get_instance().get_config(model)
    joint_controller = arx5.Arx5JointController(
        robot_config, joint_controller_config, interface
    )
    # Flush any stale newlines from stdin so C++ cin.get() prompts aren't skipped
    import termios

    termios.tcflush(sys.stdin, termios.TCIFLUSH)

    # C++ calibrate_gripper: zeros motor at fully closed, then user opens gripper
    # After this call: gripper is physically open, close position = 0.0 by definition
    # KeyboardInterrupt is caught because Ctrl+C is used to satisfy cin.get() in C++,
    # but the C++ routine completes successfully before Python sees the interrupt.
    try:
        joint_controller.calibrate_gripper()
    except KeyboardInterrupt:
        pass

    time.sleep(
        0.3
    )  # Wait for background send_recv to update joint_state after calibration
    gripper_open = joint_controller.get_joint_state().gripper_pos
    gripper_close = 0.0
    print(f"gripper_open: {gripper_open:.6f}")
    print(f"gripper_close: {gripper_close:.6f}")

    if override_configs:
        with open(CONFIGS_YAML_PATH, "r") as f:
            cfg = yaml.safe_load(f) or {}
        if "gripper" not in cfg:
            cfg["gripper"] = {}
        cfg["gripper"][interface_arm] = {
            "open": float(gripper_open),
            "close": float(gripper_close),
        }
        with open(CONFIGS_YAML_PATH, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f"Written gripper config for '{interface_arm}' to {CONFIGS_YAML_PATH}")


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="X5")
    parser.add_argument("--arm", type=str, choices=["left", "right"])
    parser.add_argument(
        "--mode", type=str, choices=["check_motor_movements", "joint", "gripper"]
    )
    parser.add_argument("--joint_id", type=int, default=0)
    parser.add_argument("--override-configs", action="store_true")
    args = parser.parse_args()
    print("robot model: ", args.model)
    # Note that the can in can-both command in setup is actually swapped
    if args.arm == "left":
        interface = "can1"
    elif args.arm == "right":
        interface = "can2"
    if args.mode == "check_motor_movements":
        check_motor_movements(args.model, interface)
    elif args.mode == "joint":
        calibrate_joint(args.model, interface, args.joint_id)
    elif args.mode == "gripper":
        calibrate_gripper(args.model, interface, args.arm, args.override_configs)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")
