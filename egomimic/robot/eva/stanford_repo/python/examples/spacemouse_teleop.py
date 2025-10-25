from queue import Queue
import os
import sys

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
from arx5_interface import (
    Arx5CartesianController,
    ControllerConfig,
    ControllerConfigFactory,
    EEFState,
    Gain,
    LogLevel,
    RobotConfigFactory,
)
from peripherals.spacemouse_shared_memory import Spacemouse
from multiprocessing.managers import SharedMemoryManager

import time
import click


def start_teleop_recording(controller: Arx5CartesianController):

    ori_speed = 1.5
    pos_speed = 0.8
    gripper_speed = 0.04
    # For earlier spacemouse versions (wired version), the readout might be not zero even after it is released
    # If you are using the wireless 3Dconnexion spacemouse, you can set the deadzone_threshold to 0.0 for better sensitivity
    deadzone_threshold = 0.1
    target_pose_6d = controller.get_home_pose()

    target_gripper_pos = 0.0

    window_size = 3
    cmd_dt = 0.01
    preview_time = 0.05  # Each trajectory command is 0.15s ahead of the current time

    UPDATE_TRAJ = True
    # False: only override single points with position control
    # True: send a trajectory command every update_dt. Will include velocity. This is

    pose_x_min = target_pose_6d[0]
    spacemouse_queue = Queue(window_size)
    robot_config = controller.get_robot_config()

    avg_error = np.zeros(6)
    avg_cnt = 0
    prev_eef_cmd = EEFState()
    eef_cmd = EEFState()

    with SharedMemoryManager() as shm_manager:
        with Spacemouse(
            shm_manager=shm_manager, deadzone=deadzone_threshold, max_value=500
        ) as sm:

            def get_filtered_spacemouse_output(sm: Spacemouse):
                state = sm.get_motion_state_transformed()
                # Remove the deadzone and normalize the output
                positive_idx = state >= deadzone_threshold
                negative_idx = state <= -deadzone_threshold
                state[positive_idx] = (state[positive_idx] - deadzone_threshold) / (
                    1 - deadzone_threshold
                )
                state[negative_idx] = (state[negative_idx] + deadzone_threshold) / (
                    1 - deadzone_threshold
                )

                if (
                    spacemouse_queue.maxsize > 0
                    and spacemouse_queue._qsize() == spacemouse_queue.maxsize
                ):
                    spacemouse_queue._get()
                spacemouse_queue.put_nowait(state)
                return np.mean(np.array(list(spacemouse_queue.queue)), axis=0)

            print("Teleop tracking ready. Waiting for spacemouse movement to start.")

            while True:
                button_left = sm.is_button_pressed(0)
                button_right = sm.is_button_pressed(1)
                state = get_filtered_spacemouse_output(sm)
                if state.any() or button_left or button_right:
                    print(f"Start tracking!")
                    break
                eef_cmd = controller.get_eef_cmd()
                prev_eef_cmd = eef_cmd
            start_time = time.monotonic()
            loop_cnt = 0
            while True:

                print(
                    f"Time elapsed: {time.monotonic() - start_time:.03f}s",
                    end="\r",
                )
                # Spacemouse state is in the format of (x y z roll pitch yaw)
                state = get_filtered_spacemouse_output(sm)
                button_left = sm.is_button_pressed(0)
                button_right = sm.is_button_pressed(1)
                if button_left and button_right:
                    print(f"Avg 6D pose error: {avg_error / avg_cnt}")
                    # Traj with vel Avg 6D pose error:      [ 0.0004  0.0002 -0.0016  0.0002  0.0032  0.0005]
                    # Single point without vel:             [-0.0002 -0.0006 -0.0026  0.0027  0.0042 -0.0017]
                    # Traj without vel Avg 6D pose error:   [ 0.0005  0.0008 -0.005  -0.0024  0.0073 -0.0001]

                    controller.reset_to_home()
                    config = controller.get_robot_config()
                    target_pose_6d = controller.get_home_pose()
                    target_gripper_pos = 0.0
                    loop_cnt = 0
                    start_time = time.monotonic()

                    continue
                elif button_left and not button_right:
                    gripper_cmd = 1
                elif button_right and not button_left:
                    gripper_cmd = -1
                else:
                    gripper_cmd = 0
                # print(state, target_gripper_pos)
                target_pose_6d[:3] += state[:3] * pos_speed * cmd_dt
                target_pose_6d[3:] += state[3:] * ori_speed * cmd_dt
                target_gripper_pos += gripper_cmd * gripper_speed * cmd_dt
                if target_gripper_pos >= robot_config.gripper_width:
                    target_gripper_pos = robot_config.gripper_width
                elif target_gripper_pos <= 0:
                    target_gripper_pos = 0
                loop_cnt += 1
                while time.monotonic() < start_time + loop_cnt * cmd_dt:
                    pass
                current_timestamp = controller.get_timestamp()
                prev_eef_cmd = eef_cmd
                # if target_pose_6d[0] < pose_x_min:
                #     target_pose_6d[0] = pose_x_min
                eef_cmd.pose_6d()[:] = target_pose_6d
                eef_cmd.gripper_pos = target_gripper_pos
                eef_cmd.timestamp = current_timestamp + preview_time

                if UPDATE_TRAJ:
                    # This will calculate the velocity automatically
                    controller.set_eef_traj([eef_cmd])

                # Or sending single eef_cmd:
                else:
                    # Only position control
                    controller.set_eef_cmd(eef_cmd)

                output_eef_cmd = controller.get_eef_cmd()
                eef_state = controller.get_eef_state()
                avg_error += output_eef_cmd.pose_6d() - eef_state.pose_6d()
                avg_cnt += 1

                print(f"6DPose Error: {output_eef_cmd.pose_6d() - eef_state.pose_6d()}")


@click.command()
@click.argument("model")  # ARX arm model: X5 or L5
@click.argument("interface")  # can bus name (can0 etc.)
def main(model: str, interface: str):

    robot_config = RobotConfigFactory.get_instance().get_config(model)
    controller_config = ControllerConfigFactory.get_instance().get_config(
        "cartesian_controller", robot_config.joint_dof
    )
    # controller_config.interpolation_method = "cubic"
    controller_config.default_kp = controller_config.default_kp
    controller = Arx5CartesianController(robot_config, controller_config, interface)
    controller.reset_to_home()

    robot_config = controller.get_robot_config()
    gain = Gain(robot_config.joint_dof)
    controller.set_log_level(LogLevel.DEBUG)
    np.set_printoptions(precision=4, suppress=True)
    try:
        start_teleop_recording(controller)
    except KeyboardInterrupt:
        print(f"Teleop recording is terminated. Resetting to home.")
        controller.reset_to_home()
        controller.set_to_damping()


if __name__ == "__main__":
    main()
