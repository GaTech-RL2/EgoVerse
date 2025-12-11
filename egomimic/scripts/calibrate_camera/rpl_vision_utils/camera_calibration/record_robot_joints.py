"""This is a script to record joints by moving robot around, and save it to a list"""
import os
config_folder = os.path.join(os.path.expanduser('~/'), ".rpl_vision_utils/calibration")
os.makedirs(os.path.join(os.path.expanduser('~/'), config_folder), exist_ok=True)

import simplejson as json
import numpy as np

from gprs.franka_interface import FrankaInterface
from gprs.utils.io_devices import SpaceMouse
from gprs.utils.input_utils import input2action
from gprs.utils import YamlConfig

from gprs import config_root


def main():

    device = SpaceMouse(vendor_id=9583, product_id=50734)
    device.start_control()

    print(config_root)
    robot_interface = FrankaInterface(config_root + "/alice.yml", use_visualizer=False)
    controller_cfg = YamlConfig(config_root + "/osc-controller.yml").as_easydict()
    control_type = "OSC_POSE"

    # Make it low impedance so that we can easily move the arm around
    controller_cfg["Kp"]["translation"] = 50
    controller_cfg["Kp"]["rotation"] = 50

    joints = []

    recorded_joint = False

    while True:
        action, grasp = input2action(
            device=device,
            control_type=control_type,
        )
        # print(action, recorded_joint)
        # print(len(robot_interface._state_buffer))

        if action is None:
            break

        if len(robot_interface._state_buffer) > 0:
            if action[-1] > 0 and not recorded_joint:
                joints.append(robot_interface._state_buffer[-1].q)
                # print(len(robot_interface._state_buffer[-1].q))
                recorded_joint = True
                for _ in range(5):
                    action, grasp = input2action(
                        device=device,
                        control_type=control_type,
                    )
            elif action[-1] < 0:
                recorded_joint = False

        robot_interface.control(control_type=control_type,
                                action=action,
                                controller_cfg=controller_cfg)

    save_joints = []
    for joint in joints:
        if np.linalg.norm(joint) < 1.:
            continue
        print(joint)
        save_joints.append(np.array(joint).tolist())

    save = int(input("save or not? (1 - Yes, 0 - No)"))
    if save:
        file_name = input("Filename to save the  joints: ")
        joint_info_json_filename = f"{config_folder}/{file_name}.json"

        with open(joint_info_json_filename, "w") as f:
            json.dump({"joints": save_joints}, f, indent=4)

    robot_interface.close()

if __name__ == "__main__":
    main()
