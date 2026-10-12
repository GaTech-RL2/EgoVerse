"""Open-loop command error on the prefix a policy would actually execute.

A walk predicts a full action chunk, scores the leading ``execute_steps``
(or ``execute_fraction`` of the horizon), then the next observation is that
many frames later and the policy predicts again. The rest of the chunk is
not scored, and nothing here is a DTW or Frechet distance.

The four logged values are mean squared error over that prefix:

* ``XYZ_MSE`` — both arms' positions
* ``YPR_MSE`` — both arms' yaw / pitch / roll. A 6D prediction is converted
  to Euler (Gram-Schmidt, then ZYX) before this, so the number is on the
  same angles as a 12/14-D yaw-pitch-roll action.
* ``Grip_MSE`` — both grippers, when the action has them
* ``Paired_MSE`` — position and gripper together, the channels a paired
  command comparison uses. Rotation is not in this one.

Logged as ``Valid/open_loop_sim/<name>``.
"""

from __future__ import annotations

import copy
import math
import os

import numpy as np
import torch
import torchvision.io as tvio
from lightning.pytorch.loggers import WandbLogger

from egomimic.eval.eval_video import EvalVideo
from egomimic.eval.replay_viz import overlay_executed_chunk
from egomimic.rldb.embodiment.embodiment import (
    Embodiment,
    _intrinsics_from_batch,
    get_embodiment,
)
from egomimic.rldb.embodiment.human import Human
from egomimic.utils.action_utils import _matrix_to_ypr, _reconstruct_R_from_cols
from egomimic.utils.pose_utils import bimanual_cartesian_layout
from egomimic.utils.type_utils import _to_numpy
from egomimic.utils.viz_utils import _prepare_viz_image

METRIC_NAMES = ("Paired_MSE", "XYZ_MSE", "YPR_MSE", "Grip_MSE")


def executed_control_steps(
    control_horizon: int,
    execute_fraction: float = 0.25,
    *,
    execute_steps: int | None = None,
) -> int:
    """How many leading chunk steps are treated as executed.

    ``execute_steps`` is an absolute cap (the fixed-budget walk uses 30) and
    wins over ``execute_fraction``. It is clipped to the chunk length.
    """
    horizon = int(control_horizon)
    if horizon < 1:
        raise ValueError("control_horizon must be positive")
    if execute_steps is not None:
        steps = int(execute_steps)
        if steps < 1:
            raise ValueError("execute_steps must be positive")
        return min(horizon, steps)
    fraction = float(execute_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("execute_fraction must be in (0, 1]")
    return max(1, min(horizon, int(math.ceil(horizon * fraction))))


def _rot6d_to_ypr(six: torch.Tensor) -> torch.Tensor:
    """(..., 6) rotation-matrix columns -> (..., 3) yaw, pitch, roll."""
    rotation = _reconstruct_R_from_cols(six[..., 0:3], six[..., 3:6])
    return _matrix_to_ypr(rotation)


def _to_ypr_action(actions: torch.Tensor) -> torch.Tensor:
    """Bimanual cartesian action with its rotation stored as yaw-pitch-roll.

    12/14-D actions are already in that layout. 18/20-D actions carry Zhou
    6D columns; those become 3 Euler angles per arm so ``YPR_MSE`` is an
    angle error. Widths this stack does not treat as cartesian pass through.
    """
    width = actions.shape[-1]
    if width in (12, 14):
        return actions
    if width == 18:
        left_ypr = _rot6d_to_ypr(actions[..., 3:9])
        right_ypr = _rot6d_to_ypr(actions[..., 12:18])
        return torch.cat(
            (actions[..., 0:3], left_ypr, actions[..., 9:12], right_ypr), dim=-1
        )
    if width == 20:
        left_ypr = _rot6d_to_ypr(actions[..., 3:9])
        right_ypr = _rot6d_to_ypr(actions[..., 13:19])
        return torch.cat(
            (
                actions[..., 0:3],
                left_ypr,
                actions[..., 9:10],
                actions[..., 10:13],
                right_ypr,
                actions[..., 19:20],
            ),
            dim=-1,
        )
    return actions


def _channel_mse(
    error: torch.Tensor, valid: torch.Tensor, columns: tuple[int, ...]
) -> torch.Tensor | None:
    """Mean squared error on ``columns``, over steps marked valid."""
    if not columns:
        return None
    picked = error[..., list(columns)]
    weight = valid.unsqueeze(-1).to(dtype=picked.dtype)
    denom = weight.sum() * picked.shape[-1]
    if float(denom) == 0.0:
        return None
    return (picked.square() * weight).sum() / denom


def open_loop_channel_mse(
    pred: torch.Tensor,
    gt: torch.Tensor,
    execute_fraction: float = 0.25,
    valid_steps: torch.Tensor | None = None,
    *,
    execute_steps: int | None = None,
) -> dict[str, torch.Tensor]:
    """MSE of the executed prefix of a bimanual cartesian chunk.

    ``pred`` and ``gt`` are ``(B, T, D)``, unnormalized. ``valid_steps`` is
    an optional ``(B, T)`` or ``(B, T, 1)`` mask, 1 on real action frames and
    0 on repeat-last padding. An unrecognized width returns ``{}``.
    ``Grip_MSE`` is omitted when the layout has no gripper.
    ``execute_steps`` overrides ``execute_fraction`` when set.
    """
    if pred.shape != gt.shape or pred.ndim != 3:
        raise ValueError(
            f"open-loop pred/gt must be matching (B, T, D), got {tuple(pred.shape)} "
            f"and {tuple(gt.shape)}"
        )
    steps = executed_control_steps(
        pred.shape[1], execute_fraction, execute_steps=execute_steps
    )
    pred_ypr = _to_ypr_action(pred[:, :steps].detach().float())
    gt_ypr = _to_ypr_action(gt[:, :steps].detach().float())
    layout = bimanual_cartesian_layout(pred_ypr.shape[-1])
    if layout is None:
        return {}
    if valid_steps is None:
        valid = torch.ones(
            pred_ypr.shape[:2], dtype=pred_ypr.dtype, device=pred_ypr.device
        )
    else:
        valid = valid_steps[:, :steps].detach().float().to(pred_ypr.device)
        if valid.ndim == 3:
            valid = valid.squeeze(-1)
    error = pred_ypr - gt_ypr
    xyz = _channel_mse(error, valid, tuple(layout["xyz"]))
    ypr = _channel_mse(error, valid, tuple(layout["rot"]))
    grip = _channel_mse(error, valid, tuple(layout["grip"]))
    paired_cols = tuple(layout["xyz"]) + tuple(layout["grip"])
    paired = _channel_mse(error, valid, paired_cols)
    metrics = {
        "Paired_MSE": paired,
        "XYZ_MSE": xyz,
        "YPR_MSE": ypr,
    }
    if grip is not None:
        metrics["Grip_MSE"] = grip
    return {name: value for name, value in metrics.items() if value is not None}


def _batchify(sample: dict) -> dict:
    """Add the batch axis ``apply_transform`` walks."""
    out = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.unsqueeze(0)
        elif isinstance(value, np.ndarray):
            out[key] = np.expand_dims(value, 0)
    return out


def add_open_loop_metrics(
    metrics: dict,
    pred: torch.Tensor,
    gt: torch.Tensor,
    *,
    execute_fraction: float = 0.25,
    execute_steps: int | None = None,
    valid_steps: torch.Tensor | None = None,
    prefix: str = "Valid/open_loop_sim",
) -> None:
    """Write the four open-loop MSEs into ``metrics`` when ``pred`` is cartesian."""
    for name, value in open_loop_channel_mse(
        pred,
        gt,
        execute_fraction,
        valid_steps,
        execute_steps=execute_steps,
    ).items():
        key = f"{prefix}/{name}"
        if key in metrics:
            metrics[key] = 0.5 * (metrics[key] + value)
        else:
            metrics[key] = value


SEEN_PREFIX = "Valid/open_loop_sim"
UNSEEN_PREFIX = "unseen_op_valid/Valid/open_loop_sim"


def _episode_hashes(batch) -> list[str]:
    raw = batch.get("episode_hash")
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    return [str(h) for h in raw]


# Head-frame MANO layout the wrist-frame keypoint revert emits (both hands,
# 21 points, no wrist pose). The overlay refuses anything else.
_MANO_DIM = 126
_KP_ACTION = "actions_keypoints"


class OpenLoopSimEval(EvalVideo):
    """Walk pinned episodes, score ``execute_steps``, and save one video each.

    The loader yields frames ``0, execute_steps, 2 * execute_steps, ...`` of
    each pinned episode, so each batch item is a fresh prediction and the
    scored prefix is exactly the command that runs until the next one.

    The video is that walk at the episode frame rate. A sample predicts the
    full chunk, then the next ``execute_steps`` source frames each show only
    that frame: green MANO for the hand observed there, red MANO for the
    predicted step at the same time. The chunk is a resample of those source
    frames (30 frames -> 100 steps), so frame ``j`` uses the matching step,
    not step ``j``. The following sample is the next replan.

    Seen episodes log ``Valid/open_loop_sim/*``. Unseen episodes log
    ``unseen_op_valid/Valid/open_loop_sim/*``. Every episode is written under
    ``videos_open_loop/``; only the first ``wandb_videos`` episode of each
    group is uploaded. Named validation cohorts can set video_group and
    log_metrics=False to produce only their final-pass preview.
    """

    def __init__(
        self,
        execute_steps: int = 30,
        seen_episodes: list | None = None,
        unseen_episodes: list | None = None,
        wandb_videos: int = 1,
        log_metrics: bool = True,
        video_group: str | None = None,
        metric_prefix: str | None = None,
        final_only: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.execute_steps = int(execute_steps)
        self.log_metrics = log_metrics
        self.video_group = video_group
        self.metric_prefix = metric_prefix
        self.final_only = final_only
        if self.execute_steps < 1:
            raise ValueError("execute_steps must be positive")
        self.seen_episodes = [str(h) for h in (seen_episodes or [])]
        self.unseen_episodes = [str(h) for h in (unseen_episodes or [])]
        self._seen = set(self.seen_episodes)
        self._unseen = set(self.unseen_episodes)
        overlap = self._seen & self._unseen
        if overlap:
            raise ValueError(
                f"open-loop episode listed in both seen and unseen: {sorted(overlap)}"
            )
        self.wandb_videos = int(wandb_videos)
        if self.wandb_videos < 0:
            raise ValueError("wandb_videos must be >= 0")
        self._frames: dict[str, list[torch.Tensor]] = {}
        # (episode, source frame) -> image, current-frame GT, intrinsics.
        # Adjacent replans do not overlap, so each frame is reverted once.
        self._pose_cache: dict = {}

    def _should_viz(self):
        if self.final_only:
            return self.trainer.current_epoch + 1 >= self.trainer.max_epochs
        return super()._should_viz()

    def video_dir(self):
        return os.path.join(self.root_dir(), "videos_open_loop")

    def _playback_fps(self, source_fps: int = 30) -> int:
        """Container rate. One source frame is one video frame, so this is
        the episode clock and the skeleton on a frame is that frame's hand."""
        return int(source_fps)

    def on_validation_start(self):
        self._frames = {}
        self._pose_cache = {}
        if self.trainer.is_global_zero and self._should_viz():
            os.makedirs(
                os.path.join(self.video_dir(), f"epoch_{self.trainer.current_epoch}"),
                exist_ok=True,
            )

    def on_validation_end(self):
        if self._should_viz() and getattr(self.trainer, "is_global_zero", True):
            self._write_episode_videos()
        self._frames = {}
        self._pose_cache = {}

    def on_validation_step(self, batch, batch_idx, dataloader_idx=0, mode="both"):
        # Every rank renders. Skipping it on the non-zero ranks would let them
        # enter the next DDP forward while rank 0 is still drawing frames.
        do_viz = self._should_viz()
        if not self.log_metrics and not do_viz:
            return
        metrics, _images = self.compute_metrics_and_viz(batch, do_viz=do_viz)
        if not self.log_metrics or not metrics:
            return
        device = self.trainer.lightning_module.device
        metrics = {
            k: (v.to(device) if torch.is_tensor(v) else torch.tensor(v, device=device))
            for k, v in metrics.items()
        }
        self.trainer.lightning_module.log_dict(
            metrics, sync_dist=True, add_dataloader_idx=False
        )

    def compute_metrics_and_viz(self, batch, do_viz=True):
        algo = self.model
        preds = algo.forward_eval(batch)
        metrics = {}
        for embodiment_id, _batch in batch.items():
            _batch = algo.norm_stats.unnormalize(_batch, embodiment_id)
            embodiment_name = get_embodiment(embodiment_id).lower()
            ac_key = algo.ac_keys[embodiment_id]
            pred_key = f"{embodiment_name}_{ac_key}"
            if pred_key not in preds:
                continue
            pred = preds[pred_key]
            gt = _batch[ac_key]
            hashes = _episode_hashes(_batch)
            if hashes and len(hashes) != pred.shape[0]:
                raise ValueError(
                    f"episode_hash length {len(hashes)} != batch {pred.shape[0]}"
                )
            if not hashes and (self._seen or self._unseen):
                raise ValueError(
                    "open-loop batch has no episode_hash; videos cannot be split"
                )
            self._score_groups(metrics, hashes, pred, gt, _batch.get("pad_mask"))
            if do_viz and hashes:
                self._render(preds, _batch, embodiment_name, ac_key, pred_key, hashes)
        return metrics, {}

    def _score_groups(self, metrics, hashes, pred, gt, pad_mask) -> None:
        groups = {"seen": [], "unseen": []}
        if not hashes:
            groups["seen"] = list(range(pred.shape[0]))
        else:
            for i, episode in enumerate(hashes):
                if episode in self._unseen:
                    groups["unseen"].append(i)
                elif episode in self._seen or not self._seen:
                    groups["seen"].append(i)
        prefixes = {
            "seen": self.metric_prefix or SEEN_PREFIX,
            "unseen": self.metric_prefix or UNSEEN_PREFIX,
        }
        for group, indices in groups.items():
            if not indices:
                continue
            index = torch.tensor(indices, device=pred.device, dtype=torch.long)
            if pad_mask is None:
                mask = None
            else:
                mask = pad_mask.index_select(0, index.to(pad_mask.device))
            group_pred = pred.index_select(0, index)
            group_gt = gt.index_select(0, index.to(gt.device)).to(pred.device)
            group_mask = None if mask is None else mask.to(pred.device)
            add_open_loop_metrics(
                metrics,
                group_pred,
                group_gt,
                execute_steps=self.execute_steps,
                valid_steps=group_mask,
                prefix=prefixes[group],
            )
            # The video score above covers executed steps. Temporal scores
            # cover thirds of the entire predicted chunk, including its tail.
            if self.metric_prefix is not None:
                horizon = group_pred.shape[1]
                edges = (0, horizon // 3, 2 * horizon // 3, horizon)
                for label, start, stop in zip(
                    ("early", "mid", "late"), edges[:-1], edges[1:]
                ):
                    if start == stop:
                        continue
                    section = open_loop_channel_mse(
                        group_pred[:, start:stop],
                        group_gt[:, start:stop],
                        execute_steps=stop - start,
                        valid_steps=None
                        if group_mask is None
                        else group_mask[:, start:stop],
                    )
                    metrics.update(
                        {
                            f"{prefixes[group]}/{name}/{label}": value
                            for name, value in section.items()
                        }
                    )

    def _render(self, preds, batch, embodiment_name, ac_key, pred_key, hashes):
        pred_kp_key = f"{embodiment_name}_{_KP_ACTION}"
        if _KP_ACTION in batch and pred_kp_key in preds:
            if "frame_idx" not in batch:
                raise ValueError("open-loop keypoint viz needs frame_idx on the batch")
            sequences = self._keypoint_windows(
                batch, preds[pred_kp_key], embodiment_name, hashes
            )
            if getattr(self.trainer, "is_global_zero", True):
                self._buffer_sequences(hashes, sequences)
            return
        images = self._overlay(preds, batch, embodiment_name, ac_key, pred_key)
        if getattr(self.trainer, "is_global_zero", True):
            self._buffer_frames(hashes, images)

    def _overlay(self, preds, batch, embodiment_name, ac_key, pred_key):
        gt_batch = batch
        preds_for_viz = preds
        transform_list = self.transform_lists.get(embodiment_name)
        if transform_list is not None and pred_key in preds:
            pred_batch = copy.deepcopy(batch)
            pred_batch[ac_key] = preds[pred_key]
            gt_t = Embodiment.apply_transform(batch, transform_list)
            pred_t = Embodiment.apply_transform(pred_batch, transform_list)
            gt_batch = {**batch, **gt_t}
            pred_batch_viz = {**batch, **pred_t}
            preds_for_viz = dict(preds)
            preds_for_viz[pred_key] = pred_batch_viz[ac_key]
        return self._visualize_preds(preds_for_viz, gt_batch)

    def _keypoint_windows(self, batch, pred_kp, embodiment_name, hashes):
        """``execute_steps`` frames per sample, each with that frame's skeleton."""
        transform_list = self.transform_lists.get(embodiment_name)
        if not transform_list:
            raise ValueError(
                "open-loop keypoint viz needs the wrist-frame keypoint revert "
                f"on transform_lists.{embodiment_name}"
            )
        image_key = self._image_key(embodiment_name)
        # The revert reads the action and the proprio. Copy first: it writes
        # into the sample, and the prediction has to be reverted with the
        # ground-truth proprio still in the replan frame's camera.
        gt_rev = self._reverted_keypoints(copy.deepcopy(batch), transform_list)
        pred_batch = copy.deepcopy(batch)
        pred_batch[_KP_ACTION] = pred_kp.detach()
        pred_rev = self._reverted_keypoints(pred_batch, transform_list)
        sequences = []
        for i, episode in enumerate(hashes):
            anchor = int(batch["frame_idx"][i])
            leaf = self._episode_leaf(episode)
            last = int(leaf.total_frames) - 1
            span = min(self.execute_steps, last - anchor + 1)
            images, gt_frames, intrinsics = [], [], []
            for j in range(max(span, 0)):
                image, gt_here, K = self._frame_pose(
                    leaf, episode, anchor + j, image_key, transform_list
                )
                images.append(image)
                gt_frames.append(gt_here)
                intrinsics.append(K)
            if not images:
                sequences.append(np.zeros((0, 1, 1, 3), np.uint8))
                continue
            sequences.append(
                overlay_executed_chunk(
                    np.stack(images),
                    gt_rev[i],
                    pred_rev[i],
                    np.stack(gt_frames),
                    intrinsics,
                    Human.FINGER_EDGES,
                    n_window=self.execute_steps,
                    fallback_intrinsics=Human.INTRINSICS,
                )
            )
        return sequences

    def _reverted_keypoints(self, batch, transform_list) -> np.ndarray:
        reverted = Embodiment.apply_transform(batch, transform_list)
        if _KP_ACTION not in reverted:
            raise ValueError(
                "keypoint revert did not emit actions_keypoints; the open-loop "
                "overlay needs _build_human_keypoints_revert_6d_wristframe_transform_list"
            )
        keypoints = np.asarray(_to_numpy(reverted[_KP_ACTION]), dtype=np.float32)
        if keypoints.shape[-1] != _MANO_DIM:
            raise ValueError(
                f"keypoint revert produced {keypoints.shape[-1]}-D actions; "
                "the overlay needs 126-D head-frame MANO "
                "(wrist-frame 6D keypoint revert)"
            )
        return keypoints

    def _image_key(self, embodiment_name: str) -> str:
        viz = (self.viz_func or {}).get(embodiment_name)
        key = getattr(viz, "keywords", {}).get("image_key")
        return key or Human.VIZ_IMAGE_KEY

    def _episode_leaf(self, episode_hash: str):
        groups = getattr(
            getattr(self.trainer, "datamodule", None), "open_loop_datasets", None
        )
        for ds in (groups or {}).values():
            leaf = (getattr(ds, "datasets", None) or {}).get(episode_hash)
            if leaf is not None:
                return leaf
        raise KeyError(
            f"open-loop episode {episode_hash} is not in the datamodule's "
            "open_loop_datasets"
        )

    def _frame_pose(self, leaf, episode, frame, image_key, transform_list):
        key = (episode, int(frame))
        cached = self._pose_cache.get(key)
        if cached is not None:
            return cached
        raw = leaf[int(frame)]
        if image_key not in raw:
            raise KeyError(
                f"{episode} frame {frame} has no {image_key} for the keypoint overlay"
            )
        image = _prepare_viz_image(_to_numpy(raw[image_key]))
        wrapped = _batchify(raw)
        gt_here = self._reverted_keypoints(wrapped, transform_list)[0, 0]
        cached = (image, gt_here, _intrinsics_from_batch(wrapped, 0))
        self._pose_cache[key] = cached
        return cached

    def _buffer_frames(self, hashes, images) -> None:
        frames = np.asarray(images)
        if frames.shape[0] != len(hashes):
            raise ValueError(
                f"viz returned {frames.shape[0]} frames for {len(hashes)} episodes"
            )
        for i, episode in enumerate(hashes):
            self._frames.setdefault(episode, []).append(
                torch.from_numpy(np.ascontiguousarray(frames[i]))
            )

    def _buffer_sequences(self, hashes, sequences) -> None:
        for episode, frames in zip(hashes, sequences):
            buf = self._frames.setdefault(episode, [])
            for frame in frames:
                buf.append(torch.from_numpy(np.ascontiguousarray(frame)))

    def _write_episode_videos(self) -> None:
        uploaded = {"seen": 0, "unseen": 0}
        order = [("seen", h) for h in self.seen_episodes] + [
            ("unseen", h) for h in self.unseen_episodes
        ]
        known = {h for _, h in order}
        for episode in self._frames:
            if episode not in known:
                order.append(("seen", episode))
        epoch_dir = os.path.join(
            self.video_dir(), f"epoch_{self.trainer.current_epoch}"
        )
        for group, episode in order:
            frames = self._frames.get(episode)
            if not frames:
                continue
            group = self.video_group or group
            uploaded.setdefault(group, 0)
            path = os.path.join(epoch_dir, group, f"{episode}.mp4")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tvio.write_video(
                path,
                torch.stack(frames),
                fps=self._playback_fps(),
                video_codec="h264",
            )
            if uploaded[group] < self.wandb_videos:
                self._upload_open_loop_video(group, path, uploaded[group])
                uploaded[group] += 1

    def _upload_open_loop_video(self, group: str, path: str, index: int) -> None:
        name = f"videos_open_loop/{group}"
        if index:
            name = f"{name}_{index}"
        for logger in self.trainer.loggers:
            if isinstance(logger, WandbLogger):
                logger.log_video(
                    name,
                    [path],
                    step=self.trainer.global_step,
                    caption=[f"epoch {self.trainer.current_epoch} {group}"],
                    format=["mp4"],
                )
