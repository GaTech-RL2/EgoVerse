import os
import sys

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
from arx5_interface import Arx5CartesianController, EEFState, Gain

import time
from peripherals.keystroke_counter import KeystrokeCounter, KeyCode
import click


def start_teaching(controller: Arx5CartesianController, data_file: str):
    controller.reset_to_home()

    controller_config = controller.get_controller_config()

    print("Teaching mode ready. Press 't' to start teaching.")
    teaching_started = False
    traj = []
    start_time = 0
    with KeystrokeCounter() as key_counter:
        while True:
            press_events = key_counter.get_press_events()
            for key_stroke in press_events:
                if key_stroke == KeyCode(char="t"):
                    controller.set_to_damping()

                    gain = controller.get_gain()
                    gain.kd()[:] *= 0.1
                    controller.set_gain(gain)  # set to passive
                    if teaching_started:
                        print(f"Teaching is already started!")
                        continue
                    print("Teaching started! Press 'q' to quit teaching.")
                    teaching_started = True
                    start_time = time.monotonic()
                elif key_stroke == KeyCode(char="q"):
                    print(f"Teaching stopped! Trajectory saved to file {data_file}")
                    teaching_started = False
                    np.save(data_file, traj, allow_pickle=True)
                    return
            if teaching_started:
                state = controller.get_eef_state()
                state.timestamp = time.monotonic() - start_time
                traj.append(
                    {
                        "pose_6d": state.pose_6d().copy(),
                        "gripper_pos": state.gripper_pos,
                    }
                )
                time.sleep(controller_config.controller_dt)


def start_high_level_replay(controller: Arx5CartesianController, data_file: str):
    controller.reset_to_home()

    robot_config = controller.get_robot_config()
    controller_config = controller.get_controller_config()

    traj = np.load(data_file, allow_pickle=True)
    replay_started = False
    start_time = 0
    loop_cnt = 0
    print("Replay mode ready. Press 'r' to start replay.")
    with KeystrokeCounter() as key_counter:
        while True:
            press_events = key_counter.get_press_events()
            for key_stroke in press_events:
                if key_stroke == KeyCode(char="r"):
                    if replay_started:
                        print(f"Replay is already started!")
                        continue
                    print("Replay started! Press 'q' to quit replay.")
                    replay_started = True
                    start_time = time.monotonic()
                    loop_cnt = 0
                elif key_stroke == KeyCode(char="q"):
                    print(f"Replay stopped!")
                    return
            if replay_started:
                if loop_cnt < len(traj):
                    pose_dict = traj[loop_cnt]
                    pose_6d = pose_dict["pose_6d"]
                    gripper_pos = pose_dict["gripper_pos"]

                    loop_cnt += 1

                    controller.set_eef_cmd(EEFState(pose_6d, gripper_pos))
                    # print(
                    #     f"Time elapsed: {time.monotonic() - start_time:.03f}s/{traj[-1]['timestamp']:.03f}",
                    #     end="\r",
                    # )
                    # print(f"joint pos: {controller.joint_pos}")
                else:
                    print(f"\nReplay finished!")
                    controller.reset_to_home()
                    return
                time.sleep(controller_config.controller_dt)


@click.command()
@click.argument("model")  # ARX arm model: X5 or L5
@click.argument("interface")  # can bus name (can0 etc.)
def main(model: str, interface: str):
    controller = Arx5CartesianController(model, interface)

    np.set_printoptions(precision=4, suppress=True)
    os.makedirs("data", exist_ok=True)
    start_teaching(controller, "data/teach_traj.npy")
    start_high_level_replay(controller, "data/teach_traj.npy")


if __name__ == "__main__":
    main()
