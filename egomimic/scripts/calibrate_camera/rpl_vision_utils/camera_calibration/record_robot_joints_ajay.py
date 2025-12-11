"""This is a script to record joints by moving robot around, and save it to a list"""
import os
config_folder = os.path.join(os.path.expanduser('~/'), ".rpl_vision_utils/calibration")
os.makedirs(os.path.join(os.path.expanduser('~/'), config_folder), exist_ok=True)

import simplejson as json
import numpy as np
import cv2

import gprs.utils.transform_utils as T
from gprs.franka_interface import FrankaInterface
from gprs.camera_redis_interface import CameraRedisSubInterface
from gprs.utils.io_devices import SpaceMouse
from gprs.utils.input_utils import input2action
from gprs.utils import YamlConfig
from rpl_vision_utils.utils.apriltag_detector import AprilTagDetector

from gprs import config_root
from urdf_models.urdf_models import URDFModel

CAMERA_ID = 0
CAMERA_TYPE = "rs"
CONFIG_FOLDER = os.path.expanduser("~/.rpl_vision_utils/calibration")
IS_WRIST_CAM = True
LOAD_FILE = None
# LOAD_FILE = os.path.join(CONFIG_FOLDER, "wrist_cam_calib.json")
LOAD_FILE = os.path.join(CONFIG_FOLDER, "wrist_cam_calib_ajay.json")

def invert_pose(t, R):
    """
    Helper function to invert pose.
    """
    R_inv = R.T
    if len(t.shape) == 1:
        t = np.array(t)[:, np.newaxis]
    t_inv = -R_inv @ t
    return t_inv, R_inv

def load_and_try_camera_calib(json_file):
    """
    Read json file and try camera calib and save json.
    """
    with open(json_file, "r") as f:
        json_dic = json.load(f)

    # joint_pos = json_dic["joints"]
    # marker_pose_computation = URDFModel()
    # pose = [marker_pose_computation.get_gripper_pose(x)[:2] for x in joint_pos]
    # save_eef_pose_t = [x[0] for x in pose]
    # save_eef_pose_R = [T.quat2mat(x[1]) for x in pose]
    save_eef_pose_R = json_dic["eef_pose_R"]
    save_eef_pose_t = json_dic["eef_pose_t"]
    save_apriltag_pose_R = json_dic["apriltag_pose_R"]
    save_apriltag_pose_t = json_dic["apriltag_pose_t"]

    # try camera calibration
    save_eef_pose_R = [np.array(x).reshape(3, 3) for x in save_eef_pose_R]
    save_eef_pose_t = [np.array(x).reshape(-1) for x in save_eef_pose_t]
    save_apriltag_pose_R = [np.array(x).reshape(3, 3) for x in save_apriltag_pose_R]
    save_apriltag_pose_t = [np.array(x).reshape(-1) for x in save_apriltag_pose_t]


    # if IS_WRIST_CAM:
    #     # maybe invert apriltag poses
    #     save_apriltag_poses = [invert_pose(t=x, R=y) for (x, y) in zip(save_apriltag_pose_t, save_apriltag_pose_R)]
    #     save_apriltag_pose_t = [x[0] for x in save_apriltag_poses]
    #     save_apriltag_pose_R = [x[1] for x in save_apriltag_poses]

    if not IS_WRIST_CAM:
        # invert eef poses for front camera calibration
        for i in range(len(save_eef_pose_R)):
            save_eef_pose_t[i], save_eef_pose_R[i] = invert_pose(t=save_eef_pose_t[i], R=save_eef_pose_R[i])

    R, t = cv2.calibrateHandEye(
        save_eef_pose_R,
        save_eef_pose_t,
        save_apriltag_pose_R,
        save_apriltag_pose_t,
        method=cv2.CALIB_HAND_EYE_TSAI,
    )

    # if IS_WRIST_CAM:
    #     # maybe invert output
    #     t, R = invert_pose(t=t, R=R)

    print("Rotation matrix: ", R)
    print("Translation: ", t.transpose())

    with open(os.path.join(CONFIG_FOLDER, f"camera_{CAMERA_ID}_{CAMERA_TYPE}_extrinsics.json"),
              "w") as f:
        extrinsics = {"translation": t.tolist(),
                      "rotation": R.tolist()}
        json.dump(extrinsics, f)


def main():

    if LOAD_FILE is not None:
        # try camera calibration
        load_and_try_camera_calib(LOAD_FILE)
        exit(0)

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
    eef_pose_R = []
    eef_pose_t = []
    apriltag_pose_R = []
    apriltag_pose_t = []

    # read images to detect poses
    cr_interface = CameraRedisSubInterface(camera_id=CAMERA_ID)
    cr_interface.start()
    intrinsics = cr_interface.get_img_info()["intrinsics"]
    print(intrinsics)
    # detect april tag poses
    april_detector = AprilTagDetector()

    recorded_joint = False
    
    while True:
        action, grasp = input2action(
            device=device,
            control_type=control_type,
        ) 

        if action is None:
            break

        # read image
        img = cr_interface.get_img()["color"]

        # show detection
        detect_result = april_detector.detect(img,
                                              intrinsics=intrinsics["color"],
                                              tag_size=0.039)
        img = april_detector.vis_tag(img)
        # cv2.imwrite(f"calibration_imgs/{idx}_detection.png", img)
        cv2.imshow("", img)
        cv2.waitKey(1)

        if len(robot_interface._state_buffer) > 0:
            if action[-1] > 0 and not recorded_joint:
                if len(detect_result) != 1:
                    print("EMPTY TAG POSE DETECTION..")
                    continue

                last_state = robot_interface._state_buffer[-1]
                joints.append(last_state.q)
                ee_pose = np.array(last_state.O_T_EE)

                eef_pose_R.append(ee_pose.reshape(4, 4).T[:3, :3])
                eef_pose_t.append(ee_pose.reshape(4, 4).T[:3, 3])

                apriltag_pose_R.append(detect_result[0].pose_R)
                apriltag_pose_t.append(detect_result[0].pose_t)

                print(len(robot_interface._state_buffer[-1].q))
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
    save_eef_pose_R = []
    save_eef_pose_t = []
    save_apriltag_pose_R = []
    save_apriltag_pose_t = []
    for i, joint in enumerate(joints):
        if np.linalg.norm(joint) < 1.:
            continue
        print(joint)
        save_joints.append(np.array(joint).tolist())
        save_eef_pose_R.append(np.array(eef_pose_R[i]).tolist())
        save_eef_pose_t.append(np.array(eef_pose_t[i]).tolist())
        save_apriltag_pose_R.append(np.array(apriltag_pose_R[i]).tolist())
        save_apriltag_pose_t.append(np.array(apriltag_pose_t[i]).tolist())

    save = int(input("save or not? (1 - Yes, 0 - No)"))
    if save:
        file_name = input("Filename to save the  joints: ")
        joint_info_json_filename = f"{config_folder}/{file_name}.json"

        with open(joint_info_json_filename, "w") as f:
            json.dump({
                "joints": save_joints,
                "eef_pose_R": save_eef_pose_R,
                "eef_pose_t": save_eef_pose_t,
                "apriltag_pose_R": save_apriltag_pose_R,
                "apriltag_pose_t": save_apriltag_pose_t,
            }, f, indent=4)

        # try camera calibration
        load_and_try_camera_calib(joint_info_json_filename)

    robot_interface.close()


if __name__ == "__main__":
    main()