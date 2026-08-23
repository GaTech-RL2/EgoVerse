# ruff: noqa: E402
import os
import sys
import warnings
from abc import ABC, abstractmethod

warnings.filterwarnings("ignore", message="Can't initialize NVML")

import cv2
import h5py
import numpy as np
import torch

from egomimic.rldb.embodiment.eva import Eva
from egomimic.rldb.embodiment.human import Human
from egomimic.robot.robot_utils import RateLoop
from egomimic.robot.safety import validate_action_vector
from egomimic.rollout.policy import RolloutPolicyConfig, load_rollout_policy
from egomimic.utils.viz_utils import draw_actions

sys.path.append(os.path.join(os.path.dirname(__file__), "eva/eva_ws/src/eva"))

import select
import sys
import termios
import tty


def visualize_actions(ims, actions, extrinsics, intrinsics, arm="both"):
    if actions.shape[-1] == 7 or actions.shape[-1] == 14:
        ac_type = "joints"
    elif actions.shape[-1] == 3 or actions.shape[-1] == 6:
        ac_type = "xyz"
    else:
        raise ValueError(f"Unknown action type with shape {actions.shape}")

    ims = draw_actions(
        ims, ac_type, "Purples", actions, extrinsics, intrinsics, arm=arm
    )

    return ims


def viz_rot_ee_pose(image, eepose, action_image_path, rot_image_path):
    """
    Save both cartesian-action and orientation-axis visualizations for an EVA
    action chunk using the same conventions as the debug path.
    """
    arr = np.asarray(eepose, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, ...]
    if arr.ndim != 2 or arr.shape[1] not in (12, 14):
        raise ValueError(f"Expected eepose shape (T, 12|14), got {arr.shape}")

    os.makedirs(os.path.dirname(action_image_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(rot_image_path) or ".", exist_ok=True)

    img = np.asarray(image)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(
            f"Expected image shape (H, W, 3) or (3, H, W), got {img.shape}"
        )
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)

    if arr.shape[1] == 14:
        left_xyz = arr[:, :3]
        right_xyz = arr[:, 7:10]
    else:
        left_xyz = arr[:, :3]
        right_xyz = arr[:, 6:9]
    action_xyz = np.hstack([left_xyz, right_xyz]).astype(np.float32, copy=False)

    im_action = visualize_actions(
        img.copy(),
        action_xyz,
        Eva.EXTRINSICS,
        Human.INTRINSICS,
        arm="both",
    )
    cv2.imwrite(action_image_path, im_action)

    eva_viz_batch = {
        "observations.images.front_img_1": torch.from_numpy(img[None, ...]),
        "actions_cartesian": torch.from_numpy(arr[None, ...]),
    }
    im_rot = Eva.viz_transformed_batch(eva_viz_batch, mode="palm_axes")
    cv2.imwrite(rot_image_path, im_rot)
    return im_action, im_rot


GRIPPER_WIDTH = 0.09
# Control parameters
DEFAULT_FREQUENCY = 30  # Hz
QUERY_FREQUENCY = 30

RIGHT_CAM_SERIAL = ""
LEFT_CAM_SERIAL = ""

EMBODIMENT_MAP = {
    "both": 8,
    "left": 7,
    "right": 6,
}

TEMP_DIR = "/home/robot/temp_dir"


def _build_robot_interface(arms_list, offline_debug=False, offline_episode_path=None):
    if offline_debug:
        from robot_interface import OfflineARXInterface

        return OfflineARXInterface(arms=arms_list, dataset_path=offline_episode_path)

    from robot_interface import ARXInterface

    return ARXInterface(arms=arms_list)


def _get_model_xml_path():
    candidates = [
        "/home/robot/robot_ws/egomimic/resources/model_x5.xml",
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "resources", "model_x5.xml")
        ),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[-1]


class _KeyPoll:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)  # no Enter needed
        return self

    def __exit__(self, exc_type, exc, tb):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def getch(self):
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


class Rollout(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def rollout_step(self, i):
        pass


class ReplayRollout(Rollout):
    def __init__(self, dataset_path, cartesian):
        super().__init__()
        self.dataset_path = dataset_path
        if not os.path.isfile(self.dataset_path):
            raise FileNotFoundError(f"HDF5 not found: {self.dataset_path}")
        with h5py.File(self.dataset_path, "r") as f:
            if cartesian:
                self.actions = np.asarray(f["actions"]["eepose"][...], dtype=np.float32)
            else:
                self.actions = np.asarray(
                    f["observations"]["joint_positions"][...], dtype=np.float32
                )

    def rollout_step(self, i):
        if i < self.actions.shape[0]:
            return self.actions[i]
        else:
            return None


def debug_policy(actions, front_img, step_i):
    os.makedirs("debug", exist_ok=True)

    if isinstance(front_img, torch.Tensor):
        if front_img.dim() == 4:
            front_img = front_img[0].permute(1, 2, 0).cpu().numpy()
        elif front_img.dim() == 3:
            if front_img.shape[0] == 3:
                front_img = front_img.permute(1, 2, 0).cpu().numpy()
            else:
                front_img = front_img.cpu().numpy()
    elif front_img.ndim == 3 and front_img.shape[0] == 3:
        front_img = front_img.transpose(1, 2, 0)
    front_img = front_img.astype(np.uint8)

    if isinstance(actions, torch.Tensor):
        actions = actions.detach().cpu().numpy()
    actions = np.asarray(actions).squeeze()
    eva_viz_batch = {
        "observations.images.front_img_1": torch.from_numpy(front_img[None, ...]),
        "actions_cartesian": torch.from_numpy(
            actions.astype(np.float32, copy=False)[None, ...]
        ),
    }
    im_viz = Eva.viz_transformed_batch(eva_viz_batch, mode="traj+rotation")

    cv2.imwrite(f"debug/debug_{step_i}.png", im_viz)


def reset_rollout(ri, policy):
    print("Resetting rollout: going home + clearing policy state")
    ri.set_home()
    if hasattr(policy, "reset"):
        policy.reset()
    if hasattr(policy, "actions"):
        policy.actions = None
    if hasattr(policy, "debug_actions"):
        policy.debug_actions = None


def validate_rollout_action(ri, actions, arms, arms_list, cartesian):
    expected_dim = 14 if arms == "both" else 7
    actions = validate_action_vector(actions, expected_dim)
    for arm in arms_list:
        offset = 7 if (arm == "right" and arms == "both") else 0
        arm_action = actions[offset : offset + 7]
        validator_name = (
            "validate_pose_command" if cartesian else "validate_joints_command"
        )
        validator = getattr(ri, validator_name, None)
        if validator is not None:
            validator(arm_action, arm)
    return actions


def warmup_policy(ri, policy, arms, arms_list, cartesian):
    """Run one observed inference and safety validation without commanding motors."""
    print("[rollout] Running pre-motion shadow inference")
    observation = ri.get_obs()
    actions = policy.act(observation)
    validate_rollout_action(ri, actions, arms, arms_list, cartesian)
    policy.reset()
    print("[rollout] Shadow inference passed; no command was sent")


def main(
    arms,
    frequency,
    cartesian,
    query_frequency=QUERY_FREQUENCY,
    policy_path=None,
    dataset_path=None,
    debug=False,
    resampled_action_len=None,
    offline_debug=False,
    offline_episode_path=None,
    annotation_path=None,
    action_frame="base",
    allow_cpu_policy=False,
):
    if arms == "both":
        arms_list = ["right", "left"]
    elif arms == "right":
        arms_list = ["right"]
    else:
        arms_list = ["left"]

    if offline_episode_path is not None and not offline_debug:
        raise ValueError("--offline-episode-path requires --offline-debug.")
    if policy_path is not None and offline_debug and offline_episode_path is None:
        raise ValueError(
            "--policy-path requires --offline-episode-path in --offline-debug mode."
        )

    ri = _build_robot_interface(
        arms_list=arms_list,
        offline_debug=offline_debug,
        offline_episode_path=offline_episode_path,
    )

    try:
        if policy_path is not None:
            rollout_type = "policy"
            policy = load_rollout_policy(
                policy_path,
                RolloutPolicyConfig(
                    arm=arms,
                    query_frequency=query_frequency,
                    cartesian=cartesian,
                    resampled_action_len=resampled_action_len,
                    annotation_path=annotation_path,
                    action_frame=action_frame,
                    require_cuda=not offline_debug and not allow_cpu_policy,
                ),
            )
        elif dataset_path is not None:
            rollout_type = "replay"
            policy = ReplayRollout(dataset_path=dataset_path, cartesian=cartesian)
        else:
            raise ValueError(
                "Must provide either --policy-path or --dataset-path (and optionally --repo-id)."
            )
    except BaseException:
        close = getattr(ri, "close", None)
        if close is not None:
            close()
        raise

    print(f"Cartesian value {cartesian}")

    def _enter_intervention(kp, policy, rollout_type):
        """Pause rollout and wait for user command.

        Restores the terminal to cooked mode so the user can type full
        commands, then re-enters cbreak mode before returning.

        Returns one of:
            "continue"  – resume rollout
            "restart"   – restart rollout
            "quit"      – exit program
        """
        # Restore normal terminal so the user can type freely
        termios.tcsetattr(kp.fd, termios.TCSADRAIN, kp.old)
        print("\n--- INTERVENTION (rollout paused) ---")
        print("  c            : continue rollout")
        print("  a <path>     : load new annotation file")
        print("  r            : restart rollout")
        print("  q            : quit")

        while True:
            try:
                cmd = input("> ").strip()
            except EOFError:
                tty.setcbreak(kp.fd)
                return "quit"

            if cmd == "c":
                print("Resuming rollout.")
                tty.setcbreak(kp.fd)
                return "continue"
            elif cmd == "q":
                tty.setcbreak(kp.fd)
                return "quit"
            elif cmd == "r":
                tty.setcbreak(kp.fd)
                return "restart"
            elif cmd.startswith("a "):
                ann_path = cmd[2:].strip()
                if not ann_path:
                    print("Usage: a <annotation_path>")
                    continue
                if rollout_type != "policy":
                    print("Annotation loading is only supported for policy rollouts.")
                    continue
                policy.load_annotation(ann_path)
            else:
                print(f"Unknown command: '{cmd}'. Use c / a <path> / r / q.")

    try:
        with _KeyPoll() as kp:
            # Enter intervention at startup so the user decides when to begin
            print(
                "[robot] Controllers are initialized without homing. "
                "Continuing authorizes homing and shadow inference."
            )
            result = _enter_intervention(kp, policy, rollout_type)
            if result == "quit":
                print("Quit requested.")
                return
            reset_rollout(ri, policy)
            if rollout_type == "policy":
                warmup_policy(ri, policy, arms, arms_list, cartesian)

            while True:  # restartable
                with RateLoop(frequency=frequency, verbose=True) as loop:
                    for step_i in loop:
                        ch = kp.getch()
                        if ch is not None:
                            # Any key press triggers intervention
                            result = _enter_intervention(kp, policy, rollout_type)
                            if result == "quit":
                                print("Quit requested.")
                                return
                            elif result == "restart":
                                print("Restart requested.")
                                reset_rollout(ri, policy)
                                result = _enter_intervention(kp, policy, rollout_type)
                                if result == "quit":
                                    return
                                if result == "restart":
                                    reset_rollout(ri, policy)
                                break
                            if rollout_type == "policy":
                                policy.reset()
                            break

                        actions = None
                        if rollout_type == "policy":
                            obs = ri.get_obs()
                            actions = policy.act(obs)
                        elif rollout_type == "replay":
                            actions = policy.rollout_step(step_i)
                        elif rollout_type == "replay_lerobot":
                            actions = policy.rollout_step(step_i)
                        else:
                            raise ValueError(f"Invalid rollout type: {rollout_type}")

                        if actions is None:
                            print("Finish rollout.")
                            result = _enter_intervention(kp, policy, rollout_type)
                            if result == "quit":
                                return
                            reset_rollout(ri, policy)
                            break

                        actions = validate_rollout_action(
                            ri, actions, arms, arms_list, cartesian
                        )

                        if debug and rollout_type == "policy" and policy.just_queried:
                            debug_actions = policy.debug_actions
                            front_img = obs["front_img_1"]
                            debug_policy(
                                debug_actions,
                                front_img,
                                step_i,
                            )

                        for arm in arms_list:
                            arm_offset = 7 if (arm == "right" and arms == "both") else 0
                            arm_action = actions[arm_offset : arm_offset + 7]
                            if cartesian:
                                ri.set_pose(arm_action, arm)
                            else:
                                ri.set_joints(arm_action, arm)

    except KeyboardInterrupt:
        print("KeyboardInterrupt detected, exiting rollout.")
        return
    finally:
        close = getattr(ri, "close", None)
        if close is not None:
            close()


def build_arg_parser(description="Rollout robot model."):
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--arms",
        type=str,
        default="right",
        choices=["left", "right", "both"],
        help="Which arm(s) to control",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=DEFAULT_FREQUENCY,
        help="Control loop frequency in Hz",
    )
    parser.add_argument(
        "--query_frequency",
        type=int,
        default=QUERY_FREQUENCY,
        help="Frames which model does inference",
    )
    parser.add_argument(
        "--action-frame",
        type=str,
        default="base",
        choices=["base", "cam"],
        help="Legacy-checkpoint frame convention. Pipeline Fold checkpoints "
        "always use their canonical wrist-frame-to-base-frame codec.",
    )
    parser.add_argument("--policy-path", type=str, help="policy checkpoint path")
    parser.add_argument("--dataset-path", type=str, help="dataset path for replay")
    parser.add_argument(
        "--offline-debug",
        action="store_true",
        help="use the offline dummy robot interface for rollout debugging",
    )
    parser.add_argument(
        "--offline-episode-path",
        type=str,
        help="local EVA Zarr episode path used as observation source in offline debug mode",
    )
    parser.add_argument(
        "--cartesian",
        action="store_true",
        help="control in cartesian space instead of joint space",
    )
    parser.add_argument(
        "--resampled-action-len",
        type=int,
        default=None,
        help="Legacy checkpoints only: resample each predicted action chunk to "
        "this length. Pipeline checkpoints retain their trained horizon.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="enable debug visualization of actions on images",
    )
    parser.add_argument(
        "--allow-cpu-policy",
        action="store_true",
        help="allow live policy inference without CUDA (diagnostics only)",
    )
    parser.add_argument(
        "--annotation-path",
        type=str,
        help="path to the annotation file",
    )
    return parser


def run_from_args(args):
    if args.resampled_action_len is not None:
        print(f"Resampling legacy actions to {args.resampled_action_len}")
    return main(
        arms=args.arms,
        frequency=args.frequency,
        query_frequency=args.query_frequency,
        policy_path=args.policy_path,
        dataset_path=args.dataset_path,
        cartesian=args.cartesian,
        debug=args.debug,
        resampled_action_len=args.resampled_action_len,
        offline_debug=args.offline_debug,
        offline_episode_path=args.offline_episode_path,
        annotation_path=args.annotation_path,
        action_frame=args.action_frame,
        allow_cpu_policy=args.allow_cpu_policy,
    )


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    run_from_args(args)
