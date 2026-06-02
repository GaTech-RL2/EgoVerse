# ruff: noqa: E402
import os
import sys
import time
import warnings
from abc import ABC, abstractmethod

warnings.filterwarnings("ignore", message="Can't initialize NVML")

import cv2
import h5py
import numpy as np
import torch
from robot_utils import RateLoop
from scipy.spatial.transform import Rotation as R
from torch.utils.data import default_collate

from egomimic.models.denoising_policy import DenoisingPolicy
from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.pl_utils.pl_data_utils import annotation_collate
from egomimic.rldb.embodiment.embodiment import get_embodiment
from egomimic.rldb.embodiment.eva import Eva
from egomimic.rldb.embodiment.human import Human
from egomimic.utils.egomimicUtils import (
    cam_frame_to_base_frame,
    draw_actions,
    interpolate_arr,
    interpolate_arr_euler,
)
from egomimic.utils.pose_utils import xyzw_to_wxyz

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


R_t_e = np.array(
    [
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0],
    ],
    dtype=float,
)

inv_R_t_e = np.linalg.inv(R_t_e)


def ee_pose_to_rot_ee_frame_batch(pose):
    pose = np.asarray(pose)
    xyz = pose[..., :3]
    ypr = pose[..., 3:6]
    R_ee = R.from_euler("ZYX", ypr).as_matrix()
    R_rot = R_t_e @ R_ee
    ypr_rot = R.from_matrix(R_rot).as_euler("ZYX")
    return np.concatenate([xyz, ypr_rot], axis=-1)


def rot_ee_frame_to_ee_pose_batch(pose_rot):
    pose_rot = np.asarray(pose_rot)
    xyz = pose_rot[..., :3]
    ypr = pose_rot[..., 3:6]
    R_rot = R.from_euler("ZYX", ypr).as_matrix()
    R_ee = inv_R_t_e @ R_rot
    ypr_ee = R.from_matrix(R_ee).as_euler("ZYX")
    return np.concatenate([xyz, ypr_ee], axis=-1)


def ee_pose_to_rot_ee_frame(pose):
    return ee_pose_to_rot_ee_frame_batch(pose[None, ...])[0]


def rot_ee_frame_to_ee_pose(pose_rot):
    return rot_ee_frame_to_ee_pose_batch(pose_rot[None, ...])[0]


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
DEFAULT_RESAMPLE_LENGTH = 45

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


class PolicyRollout(Rollout):
    def __init__(
        self,
        arm,
        policy_path,
        query_frequency,
        cartesian,
        resampled_action_len=None,
        debug=False,
        annotation_path=None,
    ):
        super().__init__()
        self.arm = arm
        self.policy_path = policy_path
        self.query_frequency = query_frequency
        self.cartesian = cartesian
        self.embodiment_id = EMBODIMENT_MAP[self.arm]
        self.embodiment_name = get_embodiment(self.embodiment_id)
        self.extrinsics = Eva.EXTRINSICS
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_device = self.device
        print(f"[rollout] Loading policy from {self.policy_path}")
        self.policy = self._load_policy()
        self.debug_actions = None
        self.resampled_action_len = resampled_action_len
        self.debug = debug
        self.transform_list = self._build_transform_list_from_config()
        self.annotation = None
        self.collate_fn = annotation_collate
        self._proprio_debug_printed = False
        # Prediction visualizer. Built from ``evaluator.viz_func`` in the
        # checkpoint's .hydra/config.yaml so the rollout uses whatever viz
        # the training pipeline declared (image_key / action_key / mode).
        self.viz_func = self._build_viz_func_from_config()
        self.viz_enabled = False
        # Revert transform_list. For wrist-frame models, the training
        # pipeline declares an ``evaluator.transform_lists.<emb>`` block
        # that converts model output from wrist frame back to cam frame
        # before viz/MSE. The rollout needs the same conversion so the
        # cam→base post-processing and the viz projection both work.
        self.revert_transform_list = self._build_revert_transform_from_config()
        if annotation_path is not None:
            if not os.path.isfile(annotation_path):
                print(
                    f"[rollout] WARNING: annotation file not found: {annotation_path}  (continuing without annotation)"
                )
            else:
                with open(annotation_path, "r") as f:
                    self.annotation = f.read().strip()
                self._apply_annotation_to_algo()

    LOCAL_WEIGHT_PATH = (
        "/home/robot/robot_ws/egomimic/algo/pi_checkpoints/pi05_base_pytorch"
    )

    @classmethod
    def _load_checkpoint_cfg(cls, ckpt_path):
        """Load the saved hydra config tree from a checkpoint as a plain dict."""
        import torch as _torch
        from omegaconf import DictConfig, OmegaConf

        ckpt = _torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ht = ckpt.get("hyper_parameters", {}).get("config_tree")
        if ht is None:
            return None, ckpt
        if isinstance(ht, DictConfig):
            cfg = OmegaConf.to_container(ht, resolve=True)
        else:
            cfg = ht
        return cfg, ckpt

    @classmethod
    def _patch_checkpoint_paths(cls, ckpt_path):
        """Rewrite pytorch_weight_path in the checkpoint's saved config
        to point to the local base model weights. Returns (patched_path, cfg)."""
        import torch as _torch
        from omegaconf import OmegaConf
        cfg, ckpt = cls._load_checkpoint_cfg(ckpt_path)
        if cfg is None:
            return ckpt_path, None
        # Navigate to pytorch_weight_path in the config
        robomimic = cfg.get("model", {}).get("robomimic_model", {})
        config = robomimic.get("config", {})
        old_path = config.get("pytorch_weight_path")
        if old_path is None or old_path == cls.LOCAL_WEIGHT_PATH:
            return ckpt_path, cfg
        print(f"[rollout] Patching pytorch_weight_path: {old_path} -> {cls.LOCAL_WEIGHT_PATH}")
        config["pytorch_weight_path"] = cls.LOCAL_WEIGHT_PATH
        ckpt["hyper_parameters"]["config_tree"] = OmegaConf.create(cfg)
        patched_path = ckpt_path + ".patched"
        _torch.save(ckpt, patched_path)
        print(f"[rollout] Patched checkpoint saved to {patched_path}")
        return patched_path, cfg

    def _apply_annotation_to_algo(self):
        """Wire the rollout-time annotation into the loaded algo.

        Duck-typed: sets whichever of the standard annotation knobs the
        model actually has, so this works for PI (``sampling_mode``),
        QWEN-HPT (``annotation_sampling_mode``), and any future variant
        that follows the same naming conventions. Algos that have none of
        these attributes are left untouched.

          - ``annotation_key="annotations"`` matches the key inserted into
            each per-step sample in ``process_obs_for_transform_list``.
          - ``sampling_mode="first"`` / ``annotation_sampling_mode="first"``
            make inference deterministic.
          - ``default_prompt=self.annotation`` is the fallback path for
            edge cases (e.g. the annotations key gets dropped).
        """
        model = getattr(self.policy, "model", None)
        if model is None:
            return
        if hasattr(model, "annotation_key"):
            model.annotation_key = "annotations"
        if hasattr(model, "sampling_mode"):
            model.sampling_mode = "first"
        if hasattr(model, "annotation_sampling_mode"):
            model.annotation_sampling_mode = "first"
        if self.annotation is not None and hasattr(model, "default_prompt"):
            model.default_prompt = self.annotation

    def _build_transform_list_from_config(self):
        """Build the rollout transform_list from the training config so the
        live observations are pre-processed in the same coordinate frame /
        action representation the model was trained on. Reads the eva
        embodiment's ``transform_list`` block from the .hydra/config.yaml
        next to the checkpoint. Falls back to the legacy hardcoded
        ``cartesian_wristframe_ypr`` if nothing is found.
        """
        import yaml
        ckpt_dir = os.path.dirname(self.policy_path)
        candidates = [
            os.path.join(ckpt_dir, "..", ".hydra", "config.yaml"),
            os.path.join(ckpt_dir, ".hydra", "config.yaml"),
        ]
        cfg = None
        for p in candidates:
            p = os.path.normpath(p)
            if os.path.isfile(p):
                with open(p) as f:
                    cfg = yaml.safe_load(f)
                break
        if cfg is None:
            print("[rollout] WARNING: no .hydra/config.yaml found — falling back to mode='cartesian_wristframe_ypr'")
            return Eva.get_transform_list(mode="cartesian_wristframe_ypr")

        # Pull the eva embodiment's transform_list block out of either
        # ``data.train_datasets.eva_bimanual.resolver.transform_list`` (the
        # MultiDataModuleWrapper layout) or a similar nested path. We only
        # need ``mode`` (or to detect a direct _target_ to the builder fn).
        train_datasets = (
            (cfg.get("data") or {}).get("train_datasets") or {}
        )
        eva_block = (
            train_datasets.get("eva_bimanual")
            or train_datasets.get("eva_right_arm")
            or train_datasets.get("eva_left_arm")
            or {}
        )
        resolver = eva_block.get("resolver") or {}
        tl = resolver.get("transform_list") or {}

        target = tl.get("_target_", "")
        mode = tl.get("mode")

        # Mode-based call: Eva.get_transform_list(mode=...)
        if "Eva.get_transform_list" in target and mode:
            print(f"[rollout] Using transform_list mode='{mode}' (from {os.path.relpath(p)})")
            return Eva.get_transform_list(mode=mode)

        # Direct call to the canonical eva-bimanual builder: equivalent to mode='cartesian'
        if target.endswith("_build_eva_bimanual_transform_list"):
            print(f"[rollout] Using mode='cartesian' (config calls _build_eva_bimanual_transform_list)")
            return Eva.get_transform_list(mode="cartesian")

        print(f"[rollout] WARNING: could not parse eva transform_list from config (target={target!r}, mode={mode!r}) — falling back to 'cartesian_wristframe_ypr'")
        return Eva.get_transform_list(mode="cartesian_wristframe_ypr")

    def _build_viz_func_from_config(self):
        """Instantiate the prediction visualizer from ``evaluator.viz_func``
        in the training .hydra/config.yaml. Returns a callable taking
        ``(predictions, batch)`` and producing a numpy image stack, or
        ``None`` if no viz_func is declared / the config can't be read.
        """
        import yaml
        from hydra.utils import instantiate
        ckpt_dir = os.path.dirname(self.policy_path)
        candidates = [
            os.path.normpath(os.path.join(ckpt_dir, "..", ".hydra", "config.yaml")),
            os.path.normpath(os.path.join(ckpt_dir, ".hydra", "config.yaml")),
        ]
        cfg = None
        for p in candidates:
            if os.path.isfile(p):
                with open(p) as f:
                    cfg = yaml.safe_load(f)
                break
        if cfg is None:
            return None
        viz_block = ((cfg.get("evaluator") or {}).get("viz_func") or {})
        # Pick the eva entry that matches the active arm config.
        for emb_name in ("eva_bimanual", "eva_right_arm", "eva_left_arm"):
            entry = viz_block.get(emb_name)
            if entry:
                try:
                    fn = instantiate(entry)
                except Exception as e:
                    print(f"[rollout] WARNING: failed to instantiate viz_func from config: {e}")
                    return None
                print(
                    f"[rollout] viz_func loaded: {entry.get('_target_')} "
                    f"image_key={entry.get('image_key')!r} "
                    f"action_key={entry.get('action_key')!r} "
                    f"mode={entry.get('mode')!r}"
                )
                self._viz_image_key = entry.get("image_key")
                self._viz_action_key = entry.get("action_key", "actions_cartesian")
                return fn
        return None

    def _build_revert_transform_from_config(self):
        """Instantiate the revert transform_list from
        ``evaluator.transform_lists.<embodiment>`` in the training
        .hydra/config.yaml. The revert takes wrist-frame model output and
        produces cam-frame actions using the current ``observations.state.ee_pose``
        as the reference. Returns a list of Transform objects, or ``None``
        if the config has no transform_lists block (i.e. the model was
        trained directly in cam frame and no revert is needed).
        """
        import yaml
        from hydra.utils import instantiate
        ckpt_dir = os.path.dirname(self.policy_path)
        candidates = [
            os.path.normpath(os.path.join(ckpt_dir, "..", ".hydra", "config.yaml")),
            os.path.normpath(os.path.join(ckpt_dir, ".hydra", "config.yaml")),
        ]
        cfg = None
        for p in candidates:
            if os.path.isfile(p):
                with open(p) as f:
                    cfg = yaml.safe_load(f)
                break
        if cfg is None:
            return None
        block = ((cfg.get("evaluator") or {}).get("transform_lists") or {})
        for emb_name in ("eva_bimanual", "eva_right_arm", "eva_left_arm"):
            entry = block.get(emb_name)
            if entry:
                try:
                    fn = instantiate(entry)
                except Exception as e:
                    print(f"[rollout] WARNING: failed to instantiate revert transform_list: {e}")
                    return None
                print(
                    f"[rollout] revert transform_list loaded: "
                    f"{entry.get('_target_')!r}"
                )
                return fn
        return None

    def _apply_revert_to_actions(self, preds, batch_unnorm):
        """Apply the revert transform to convert wrist-frame model output
        to cam frame. Reads ``observations.state.ee_pose`` from
        ``batch_unnorm`` as the reference frame and returns a new preds
        tensor with the same shape as the input. No-op if no revert is
        configured.
        """
        if self.revert_transform_list is None:
            return preds
        from egomimic.rldb.embodiment.embodiment import Embodiment
        obs_key = "observations.state.ee_pose"
        if obs_key not in batch_unnorm:
            print(
                f"[rollout] WARNING: '{obs_key}' missing from batch — "
                "cannot revert wristframe actions"
            )
            return preds
        # Build a minimal batch with the predictions plugged in as the
        # action chunk plus the obs_ee_pose reference. apply_transform
        # splits per-sample and re-batches.
        pred_batch = {
            "actions_cartesian": preds.detach().cpu().float(),
            obs_key: batch_unnorm[obs_key],
        }
        reverted = Embodiment.apply_transform(pred_batch, self.revert_transform_list)
        out = reverted["actions_cartesian"]
        if not isinstance(out, torch.Tensor):
            out = torch.as_tensor(out)
        return out.to(preds.device, dtype=preds.dtype)

    def _save_viz(self, batch_for_model, preds, embodiment_name, step_i):
        """Render and save a per-inference prediction visualization to
        ``debug/viz_<step_i>.png``. Caller is expected to have already
        unnormalized the batch and applied any revert transform_list so
        both ``batch[actions_cartesian]`` and ``preds`` are in cam frame.
        """
        if self.viz_func is None:
            print("[rollout] viz toggle is on but no viz_func is configured — skipping")
            return
        try:
            batch = dict(batch_for_model)
            # viz_gt_preds projects cam-frame xyz via INTRINSICS["base"]
            # (= ARIA_INTRINSICS, calibrated for 640x480). If the live
            # camera publishes at a different resolution (e.g. configs.yaml
            # has Aria front at 960x720), the projection lands in the wrong
            # pixels even though the xyz values are correct. Resize the
            # image to the intrinsics' native size so they match.
            img_key = self._viz_image_key
            if img_key and img_key in batch:
                import torch.nn.functional as F
                img_t = batch[img_key]
                if isinstance(img_t, torch.Tensor):
                    # Force BCHW: collated transform_list image is [B, C, H, W].
                    if img_t.dim() == 3:
                        img_t = img_t.unsqueeze(0)
                    if img_t.shape[1] != 3 and img_t.shape[-1] == 3:
                        img_t = img_t.permute(0, 3, 1, 2)
                    if img_t.shape[-2:] != (480, 640):
                        img_t = F.interpolate(
                            img_t.float(), size=(480, 640),
                            mode="bilinear", align_corners=False,
                        )
                    batch[img_key] = img_t
            # viz_gt_preds expects batch["embodiment"][0].item() to work,
            # so make sure embodiment is a tensor with a batch dim.
            emb = batch.get("embodiment")
            if not isinstance(emb, torch.Tensor):
                batch["embodiment"] = torch.tensor(
                    [int(self.embodiment_id)], dtype=torch.int64
                )
            predictions = {
                f"{embodiment_name}_{self._viz_action_key}": preds.detach(),
            }
            if not getattr(self, "_viz_debug_printed", False):
                act_key = self._viz_action_key
                gt = batch.get(act_key)
                pred_t = predictions[f"{embodiment_name}_{act_key}"]
                img = batch.get(self._viz_image_key)
                print(
                    f"[rollout][viz-debug] image_key={self._viz_image_key!r} "
                    f"action_key={act_key!r}"
                )
                if isinstance(img, torch.Tensor):
                    print(
                        f"[rollout][viz-debug] image shape={tuple(img.shape)} "
                        f"dtype={img.dtype} min={img.float().min().item():.3f} "
                        f"max={img.float().max().item():.3f}"
                    )
                if isinstance(pred_t, torch.Tensor):
                    pf = pred_t.float()[0]  # (T, D)
                    T = pf.shape[0]
                    print(
                        f"[rollout][viz-debug] pred shape={tuple(pred_t.shape)} "
                        f"L_xyz t=0:    {pf[0,    :3].tolist()}\n"
                        f"[rollout][viz-debug]                     "
                        f"L_xyz t={T//2}:  {pf[T//2, :3].tolist()}\n"
                        f"[rollout][viz-debug]                     "
                        f"L_xyz t={T-1}: {pf[-1,   :3].tolist()}\n"
                        f"[rollout][viz-debug]                     "
                        f"R_xyz t=0:    {pf[0,    7:10].tolist()}\n"
                        f"[rollout][viz-debug]                     "
                        f"R_xyz t={T-1}: {pf[-1,   7:10].tolist()}"
                    )
                if isinstance(gt, torch.Tensor):
                    gf = gt.float()[0]
                    print(
                        f"[rollout][viz-debug] GT (current EE held) "
                        f"L_xyz: {gf[0, :3].tolist()} "
                        f"R_xyz: {gf[0, 7:10].tolist()}"
                    )
                self._viz_debug_printed = True
            ims = self.viz_func(predictions, batch)
            ims = np.asarray(ims)
            out_dir = os.path.abspath("debug")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"viz_{step_i:06d}.png")
            # viz_gt_preds returns RGB (matches the training-eval pipeline,
            # which writes via TensorBoard). cv2.imwrite expects BGR, so
            # swap channels here to avoid the inverted-colors save bug.
            out_im = ims[0] if ims.ndim == 4 else ims
            out_im = cv2.cvtColor(out_im, cv2.COLOR_RGB2BGR)
            cv2.imwrite(out_path, out_im)
            print(f"[rollout] saved viz -> {out_path}")
        except Exception as e:
            print(f"[rollout] viz failed at step {step_i}: {e}")

    def _load_policy(self):
        patched_path, _ = self._patch_checkpoint_paths(self.policy_path)
        policy = ModelWrapper.load_from_checkpoint(
            patched_path, weights_only=False, map_location="cpu"
        )
        policy = policy.to(self.policy_device)
        policy.eval()
        policy.model.device = self.policy_device

        # Unwrap torch.compile on sample_actions to avoid massive first-call
        # compilation overhead (~50s). The compiled version (instance attribute)
        # shadows the original class method; deleting it restores the fast
        # uncompiled path which is sufficient for real-time rollout.
        pi0 = policy.model.nets["policy"]
        if "sample_actions" in vars(pi0):
            del pi0.sample_actions
            print(
                "[rollout] Disabled torch.compile on sample_actions for rollout inference"
            )

        # Verify model is on GPU
        try:
            p = next(pi0.parameters())
            print(f"[rollout] Model device: {p.device}, dtype: {p.dtype}")
            if not p.is_cuda:
                print(
                    "[rollout] WARNING: model is NOT on GPU — inference will be very slow!"
                )
        except StopIteration:
            pass

        if getattr(policy.model, "diffusion", False):
            for head in policy.model.nets.policy.heads:
                if isinstance(policy.model.nets.policy.heads[head], DenoisingPolicy):
                    policy.model.nets.policy.heads[head].num_inference_steps = 10
        return policy

    def _downsample_chunk(self, chunk: np.ndarray, target_len: int) -> np.ndarray:
        if target_len is None or target_len <= 0 or chunk.shape[0] == target_len:
            return chunk.astype(np.float32, copy=False)

        # chunk: (T, D) -> (1, T, D) and back
        if self.cartesian:
            if self.arm == "both":
                left = chunk[:, :7]
                right = chunk[:, 7:14]
                left_r = interpolate_arr_euler(left[None, ...], target_len)[0]
                right_r = interpolate_arr_euler(right[None, ...], target_len)[0]
                out = np.hstack([left_r, right_r])
            else:
                out = interpolate_arr_euler(chunk[None, ...], target_len)[0]
        else:
            out = interpolate_arr(chunk[None, ...], target_len)[0]

        return out.astype(np.float32, copy=False)

    def rollout_step(self, i, obs):
        if i % self.query_frequency == 0:
            start_infer_t = time.time()
            transform_list_batch = self.process_obs_for_transform_list(obs)
            for transform in self.transform_list:
                transform_list_batch = transform.transform(transform_list_batch)
            # Match training: MultiDataset.__getitem__ normalizes the
            # post-transform sample before it reaches the model. build_tokenized_collate
            # also normalized proprio before discretizing into the State block,
            # which the algo-side _discretize_state_for_sample no longer does. Both
            # paths assume normalized inputs, so normalize the single sample here
            # using the model's own norm_stats (a MultiDataset).
            transform_list_batch = self.policy.model.norm_stats.normalize(
                transform_list_batch, self.embodiment_id
            )
            transform_list_batch = self.collate_fn([transform_list_batch])
            if self.arm == "both":
                embodiment_name = "eva_bimanual"
            elif self.arm == "right":
                embodiment_name = "eva_right_arm"

            elif self.arm == "left":
                embodiment_name = "eva_left_arm"
            batch = {
                embodiment_name: transform_list_batch,
            }

            processed_batch = self.policy.model.process_batch_for_training(batch)

            if not self._proprio_debug_printed:
                emb_id = next(iter(processed_batch))
                prompts = processed_batch[emb_id].get("sampled_prompt", None)
                print(f"[rollout][proprio-debug] full prompt: {prompts[0] if prompts else None!r}")
                self._proprio_debug_printed = True

            preds = self.policy.model.forward_eval(processed_batch)[
                f"{embodiment_name}_actions_cartesian"
            ]
            # Wrist-frame models: revert preds to cam frame BEFORE viz and
            # BEFORE the cam→base post-processing. For cam-frame models
            # (no transform_lists in config), revert is None and these
            # calls are no-ops.
            batch_for_viz = self.policy.model.norm_stats.unnormalize(
                dict(transform_list_batch), self.embodiment_id
            )
            if self.revert_transform_list is not None:
                preds = self._apply_revert_to_actions(preds, batch_for_viz)
                from egomimic.rldb.embodiment.embodiment import Embodiment
                gt_only = {
                    k: v for k, v in batch_for_viz.items()
                    if k in ("actions_cartesian", "observations.state.ee_pose")
                }
                gt_reverted = Embodiment.apply_transform(
                    gt_only, self.revert_transform_list
                )
                batch_for_viz = {**batch_for_viz, **gt_reverted}
            if self.viz_enabled:
                self._save_viz(batch_for_viz, preds, embodiment_name, i)
            self.actions = preds.detach().cpu().numpy().squeeze()
            self.debug_actions = self.actions.copy()
            if self.cartesian:
                if self.arm == "both":
                    left_actions = self.actions[:, :7]
                    right_actions = self.actions[:, 7:]

                    transformed_left = cam_frame_to_base_frame(
                        left_actions[:, :6].copy(), self.extrinsics["left"]
                    )
                    transformed_right = cam_frame_to_base_frame(
                        right_actions[:, :6].copy(), self.extrinsics["right"]
                    )
                    transformed_left = rot_ee_frame_to_ee_pose_batch(transformed_left)
                    transformed_right = rot_ee_frame_to_ee_pose_batch(transformed_right)
                    gripper_left = left_actions[:, 6:7]
                    gripper_right = right_actions[:, 6:7]
                    if left_actions.shape[1] == 7:
                        left_actions = np.hstack([transformed_left, gripper_left])
                    else:
                        left_actions = transformed_left
                    if right_actions.shape[1] == 7:
                        right_actions = np.hstack([transformed_right, gripper_right])
                    else:
                        right_actions = transformed_right
                    self.actions = np.hstack([left_actions, right_actions])
                else:
                    eepose = rot_ee_frame_to_ee_pose_batch(self.actions[:, :6].copy())
                    self.actions[:, :6] = eepose
                    transformed_6dof = cam_frame_to_base_frame(
                        self.actions[:, :6].copy(), self.extrinsics[self.arm]
                    )
                    # Preserve gripper if present (7th value)
                    gripper = self.actions[:, 6:7]
                    if self.actions.shape[1] == 7:
                        self.actions = np.hstack([transformed_6dof, gripper])
                    else:
                        self.actions = transformed_6dof

            if self.resampled_action_len is not None:
                self.actions = self._downsample_chunk(
                    self.actions, self.resampled_action_len
                )
            # print(f"actions: {self.actions[6:7]}, debug_actions: {self.debug_actions[6:7]}")

            print(f"Inference time: {(time.time() - start_infer_t)}s")

        act_i = i % self.query_frequency
        return self.actions[act_i]

    def process_obs_for_transform_list(self, obs):
        # front camera: obs["front_img_1"] is BGR, shape [H, W, 3]
        front = torch.from_numpy(obs["front_img_1"][None, ...])  # [1, H, W, 3]
        front = front[..., [2, 1, 0]]  # BGR -> RGB
        front = front.permute(0, 3, 1, 2).to(dtype=torch.float32) / 255.0
        front = front.squeeze()
        data = {
            # Keep rollout-local keys, PI schematic aliases, and canonical
            # dataset zarr keys so checkpoints with different data schematics
            # can all resolve the same image tensor.
            "front_img_1": front,
            "base_0_rgb": front,
            "observations.images.front_img_1": front,
            "pad_mask": torch.ones((1, 100, 1), dtype=torch.bool),
        }

        eepose = obs["ee_poses"]

        if self.arm in ["right", "both"]:
            right = torch.from_numpy(
                obs["right_wrist_img"][None, ...]
            )  # [1, H, W, 3] BGR
            right = right[..., [2, 1, 0]]  # BGR -> RGB
            right = right.permute(0, 3, 1, 2).to(dtype=torch.float32) / 255.0
            data["right_wrist_img"] = right.squeeze()
            data["right_wrist_0_rgb"] = data["right_wrist_img"]
            data["observations.images.right_wrist_img"] = data["right_wrist_img"]
            right_ee_pose = eepose[7:13]
            right_ee_pose = ee_pose_to_rot_ee_frame(right_ee_pose)
            right_ypr = right_ee_pose[..., 3:6]
            right_xyzw = R.from_euler("ZYX", right_ypr).as_quat()
            right_wxyz = xyzw_to_wxyz(right_xyzw)
            right_xyzwxyz = np.concatenate([eepose[7:10], right_wxyz], axis=-1)
            data["right.obs_ee_pose"] = torch.from_numpy(right_xyzwxyz).reshape(-1)
            data["right.obs_gripper"] = torch.from_numpy(eepose[13:14]).reshape(-1)
            right_gripper = torch.from_numpy(eepose[13:14]).view(1, 1).repeat(45, 1)
            data["right.cmd_gripper"] = right_gripper
            right_cmd_ee_pose = torch.from_numpy(right_xyzwxyz).view(1, 7).repeat(45, 1)
            data["right.cmd_ee_pose"] = right_cmd_ee_pose

        if self.arm in ["left", "both"]:
            left = torch.from_numpy(
                obs["left_wrist_img"][None, ...]
            )  # [1, H, W, 3] BGR
            left = left[..., [2, 1, 0]]  # BGR -> RGB
            left = left.permute(0, 3, 1, 2).to(dtype=torch.float32) / 255.0
            data["left_wrist_img"] = left.squeeze()
            data["left_wrist_0_rgb"] = data["left_wrist_img"]
            data["observations.images.left_wrist_img"] = data["left_wrist_img"]
            left_ee_pose = eepose[0:6]
            left_ee_pose = ee_pose_to_rot_ee_frame(left_ee_pose)
            left_ypr = left_ee_pose[..., 3:6]
            left_xyzw = R.from_euler("ZYX", left_ypr).as_quat()
            left_wxyz = xyzw_to_wxyz(left_xyzw)
            left_xyzwxyz = np.concatenate([eepose[:3], left_wxyz], axis=-1)
            data["left.obs_ee_pose"] = torch.from_numpy(left_xyzwxyz).reshape(-1)
            data["left.obs_gripper"] = torch.from_numpy(eepose[6:7]).reshape(-1)
            left_gripper = torch.from_numpy(eepose[6:7]).view(1, 1).repeat(45, 1)
            data["left.cmd_gripper"] = left_gripper
            left_cmd_ee_pose = torch.from_numpy(left_xyzwxyz).view(1, 7).repeat(45, 1)
            data["left.cmd_ee_pose"] = left_cmd_ee_pose

        # `embodiment` must be the integer id (the string lives in
        # metadata.robot_name). Downstream lookups call int() on it.
        if self.arm == "both":
            data["embodiment"] = self.embodiment_id
            data["metadata.robot_name"] = "eva_bimanual"
        elif self.arm == "right":
            data["embodiment"] = self.embodiment_id
            data["metadata.robot_name"] = "eva_right_arm"
        elif self.arm == "left":
            data["embodiment"] = self.embodiment_id
            data["metadata.robot_name"] = "eva_left_arm"
        
        if self.annotation is not None:
            data["annotations"] = [self.annotation]

        return data

    def load_annotation(self, annotation_path):
        """Load a new annotation file.

        The annotation text flows through ``data["annotations"]`` at each
        inference step and is consumed by ``PI.process_batch_for_training``,
        so updating ``self.annotation`` is sufficient. We also re-apply it to
        the algo so ``default_prompt`` stays in sync as a fallback.

        Returns True on success, False if the file could not be loaded.
        """
        if not os.path.isfile(annotation_path):
            print(f"[rollout] WARNING: annotation file not found: {annotation_path}")
            return False
        with open(annotation_path, "r") as f:
            self.annotation = f.read().strip()
        self._apply_annotation_to_algo()
        print(f"[rollout] Loaded new annotation from {annotation_path}: '{self.annotation}'")
        return True

    def reset(self):
        self.actions = None
        self.debug_actions = None
        self.policy.eval()


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

    actions = actions.squeeze()
    eva_viz_batch = {
        "observations.images.front_img_1": torch.from_numpy(front_img[None, ...]),
        "actions_cartesian": torch.from_numpy(
            actions.astype(np.float32, copy=False)[None, ...]
        ),
    }
    im_viz = Eva.viz_transformed_batch(eva_viz_batch, mode="traj+rotation")

    cv2.imwrite(f"debug/debug_{step_i}.png", im_viz)
    breakpoint()


def reset_rollout(ri, policy):
    print("Resetting rollout: going home + clearing policy state")
    if isinstance(policy, ReplayRollout):
        return
    ri.set_home()
    if hasattr(policy, "reset"):
        policy.reset()
    if hasattr(policy, "actions"):
        policy.actions = None
    if hasattr(policy, "debug_actions"):
        policy.debug_actions = None


def main(
    arms,
    frequency,
    cartesian,
    query_frequency=None,
    policy_path=None,
    dataset_path=None,
    debug=False,
    resampled_action_len=None,
    offline_debug=False,
    offline_episode_path=None,
    annotation_path=None,
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

    if policy_path is not None:
        rollout_type = "policy"
        policy = PolicyRollout(
            arm=arms,
            policy_path=policy_path,
            query_frequency=query_frequency,
            cartesian=cartesian,
            resampled_action_len=resampled_action_len,
            debug=debug,
            annotation_path=annotation_path,
        )
    elif dataset_path is not None:
        rollout_type = "replay"
        policy = ReplayRollout(dataset_path=dataset_path, cartesian=cartesian)
    else:
        raise ValueError(
            "Must provide either --policy-path or --dataset-path (and optionally --repo-id)."
        )

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
        viz_state = (
            "ON" if (isinstance(policy, PolicyRollout) and policy.viz_enabled) else "OFF"
        )
        print("\n--- INTERVENTION (rollout paused) ---")
        print("  c            : continue rollout")
        print("  a <path>     : load new annotation file")
        print(f"  v            : toggle prediction viz (currently {viz_state})")
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
                if rollout_type != "policy" or not isinstance(policy, PolicyRollout):
                    print("Annotation loading is only supported for policy rollouts.")
                    continue
                policy.load_annotation(ann_path)
            elif cmd == "v":
                if rollout_type != "policy" or not isinstance(policy, PolicyRollout):
                    print("Prediction viz is only supported for policy rollouts.")
                    continue
                policy.viz_enabled = not policy.viz_enabled
                print(f"[rollout] viz now {'ON' if policy.viz_enabled else 'OFF'}")
            else:
                print(f"Unknown command: '{cmd}'. Use c / a <path> / v / r / q.")

    try:
        with _KeyPoll() as kp:
            reset_rollout(ri, policy)
            # Enter intervention at startup so the user decides when to begin
            result = _enter_intervention(kp, policy, rollout_type)
            if result == "quit":
                print("Quit requested.")
                return
            if result == "restart":
                reset_rollout(ri, policy)

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
                            if hasattr(policy, "actions"):
                                policy.actions = None
                            break

                        actions = None
                        if rollout_type == "policy":
                            obs = ri.get_obs()
                            actions = policy.rollout_step(step_i, obs)
                        elif rollout_type == "replay":
                            actions = policy.rollout_step(step_i)
                        elif rollout_type == "replay_lerobot":
                            actions = policy.rollout_step(step_i)
                        else:
                            raise ValueError(f"Invalid rollout type: {rollout_type}")

                        if actions is None:
                            print("Finish rollout.")
                            reset_rollout(ri, policy)
                            result = _enter_intervention(kp, policy, rollout_type)
                            if result == "quit":
                                return
                            if result == "restart":
                                reset_rollout(ri, policy)
                            break

                        if (
                            debug
                            and rollout_type == "policy"
                            and step_i % query_frequency == 0
                        ):
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
        default=DEFAULT_RESAMPLE_LENGTH,
        help="Resample each predicted action chunk to this length (e.g., 100 -> 45). Euler if --cartesian.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="enable debug visualization of actions on images",
    )
    parser.add_argument(
        "--annotation-path",
        type=str,
        help="path to the annotation file",
    )
    return parser


def run_from_args(args):
    print(f"Resampling actions to {args.resampled_action_len}")
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
    )


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    run_from_args(args)
