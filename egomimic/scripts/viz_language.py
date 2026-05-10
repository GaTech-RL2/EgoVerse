"""
Language visualization script.

Reads a data config (e.g. cotrain_pi_lang.yaml) to load datasets and a
visualization config (e.g. pi_cartesian_lang.yaml) to determine per-embodiment
image/action keys. For each dataset, iterates batches and writes MP4 videos of
GT trajectories with language annotation overlays, following the same buffering
logic as EvalVideo.
"""

import os

import cv2
import hydra
import numpy as np
import torch
import torchvision.io as tvio
from omegaconf import DictConfig, OmegaConf

from egomimic.rldb.embodiment.embodiment import Embodiment
from egomimic.rldb.embodiment.eva import Eva
from egomimic.rldb.embodiment.human import Aria, Mecka, Scale
from egomimic.utils.aws.aws_data_utils import load_env
from egomimic.utils.viz_utils import _prepare_viz_image

OmegaConf.register_new_resolver("eval", eval)

_EMBODIMENT_CLASSES: dict[str, type[Embodiment]] = {
    "eva_bimanual": Eva,
    "eva_right_arm": Eva,
    "eva_left_arm": Eva,
    "aria_bimanual": Aria,
    "aria_right_arm": Aria,
    "aria_left_arm": Aria,
    "scale_bimanual": Scale,
    "scale_right_arm": Scale,
    "scale_left_arm": Scale,
    "mecka_bimanual": Mecka,
    "mecka_right_arm": Mecka,
    "mecka_left_arm": Mecka,
}


def _extract_annotation(batch: dict, annotation_key: str) -> list[str]:
    """Return all active annotation strings for this batch (batch_size=1).

    After default_collate the value is a flat list where each element may be
    a string (one variant) or another list of strings (multiple variants).
    """
    if annotation_key not in batch:
        return []
    raw = batch[annotation_key]
    if not raw:
        return []
    texts: list[str] = []
    for item in raw:
        if isinstance(item, torch.Tensor):
            item = item.tolist()
        if isinstance(item, (list, tuple)):
            texts.extend(t for t in item if isinstance(t, str) and t.strip())
        elif isinstance(item, str) and item.strip():
            texts.append(item)
    return texts


_COMPACT_MAX_CHARS = 120
_COMPACT_FONT = cv2.FONT_HERSHEY_SIMPLEX
_COMPACT_SCALE = 0.35
_COMPACT_THICKNESS = 1
_COMPACT_LINE_H = 14  # px per line at scale 0.35


def _viz_annotations_compact(image: np.ndarray, annotations: list[str]) -> np.ndarray:
    """Render all annotation variants in a small semi-transparent strip at the bottom."""
    vis = image.copy()
    h, w = vis.shape[:2]
    if not annotations:
        return vis

    lines = []
    for i, text in enumerate(annotations):
        label = f"{i+1}. {text}"
        if len(label) > _COMPACT_MAX_CHARS:
            label = label[: _COMPACT_MAX_CHARS - 1] + "…"
        lines.append(label)

    strip_h = len(lines) * _COMPACT_LINE_H + 6
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, h - strip_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, vis, 0.45, 0, vis)

    y = h - strip_h + _COMPACT_LINE_H
    for i, line in enumerate(lines):
        color = (80, 220, 255) if i == 0 else (255, 255, 255)  # yellow for original
        cv2.putText(
            vis,
            line,
            (4, y),
            _COMPACT_FONT,
            _COMPACT_SCALE,
            (0, 0, 0),
            _COMPACT_THICKNESS + 1,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            line,
            (4, y),
            _COMPACT_FONT,
            _COMPACT_SCALE,
            color,
            _COMPACT_THICKNESS,
            cv2.LINE_AA,
        )
        y += _COMPACT_LINE_H

    return vis


def _viz_batch(
    embodiment_cls: type[Embodiment],
    batch: dict,
    image_key: str,
    action_key: str,
    annotations: list[str],
    mode: str,
    viz_transform_list=None,
) -> list:
    """Visualize one batch and return a list of uint8 HWC numpy frames."""
    from egomimic.utils.type_utils import _to_numpy

    if action_key in batch:
        vis_batch = embodiment_cls.viz_transformed_batch(
            batch,
            mode=mode,
            viz_batch_key=action_key,
            image_key=image_key,
            color="Greens",
            transform_list=viz_transform_list,
        )
        frames = vis_batch if isinstance(vis_batch, list) else [vis_batch]
    else:
        img = _to_numpy(batch[image_key][0])
        frames = [_prepare_viz_image(img)]

    if annotations:
        frames = [
            _viz_annotations_compact(_prepare_viz_image(f), annotations) for f in frames
        ]

    return frames


def _flush_buffer(buffer: list, path: str, fps: int) -> None:
    frames = torch.stack(buffer)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tvio.write_video(path, frames, fps=fps, video_codec="h264")
    buffer.clear()


def _run_viz_for_datasets(
    datasets: dict,
    viz_cfg: DictConfig,
    annotation_key: str | None,
    output_dir: str,
    split_name: str,
    max_batches: int,
    fps: int,
    frames_per_file: int,
) -> None:
    for embodiment_name, dataset in datasets.items():
        embodiment_cls = _EMBODIMENT_CLASSES.get(embodiment_name.lower())
        if embodiment_cls is None:
            print(f"[warn] No embodiment class for '{embodiment_name}', skipping.")
            continue

        per_emb_cfg = OmegaConf.select(viz_cfg, embodiment_name, default=None)
        if per_emb_cfg is not None:
            image_key = OmegaConf.select(
                per_emb_cfg, "image_key", default=embodiment_cls.VIZ_IMAGE_KEY
            )
            action_key = OmegaConf.select(
                per_emb_cfg, "action_key", default="actions_cartesian"
            )
            mode = OmegaConf.select(per_emb_cfg, "mode", default="traj")
            ann_key = OmegaConf.select(
                per_emb_cfg, "annotation_key", default=annotation_key
            )
            viz_transform_list_cfg = OmegaConf.select(
                per_emb_cfg, "viz_transform_list", default=None
            )
            viz_transform_list = (
                hydra.utils.instantiate(viz_transform_list_cfg)
                if viz_transform_list_cfg is not None
                else None
            )
        else:
            image_key = embodiment_cls.VIZ_IMAGE_KEY
            action_key = "actions_cartesian"
            mode = "traj"
            ann_key = annotation_key
            viz_transform_list = None

        print(
            f"[{split_name}] {embodiment_name}: image={image_key}, action={action_key}, mode={mode}, annotation={ann_key}"
        )

        file_counter = 0
        print(f"  {len(dataset.datasets)} episode(s) found")
        for ep_name, ep_ds in dataset.datasets.items():
            ep_loader = torch.utils.data.DataLoader(
                ep_ds, batch_size=1, shuffle=False, num_workers=0
            )
            buffer: list[torch.Tensor] = []
            carried_annotation: list[str] = []
            batch_idx = 0

            for batch in ep_loader:
                if batch_idx >= max_batches:
                    break
                if ann_key is not None:
                    fresh = _extract_annotation(batch, ann_key)
                    if fresh:
                        carried_annotation = fresh
                try:
                    frames = _viz_batch(
                        embodiment_cls,
                        batch,
                        image_key,
                        action_key,
                        carried_annotation,
                        mode,
                        viz_transform_list,
                    )
                except Exception as e:
                    print(f"  [warn] {ep_name} batch {batch_idx} failed: {e}")
                    batch_idx += 1
                    continue
                batch_idx += 1
                for frame in frames:
                    buffer.append(torch.from_numpy(_prepare_viz_image(frame)))

            if buffer:
                path = os.path.join(
                    output_dir, split_name, embodiment_name, f"video_{file_counter}.mp4"
                )
                _flush_buffer(buffer, path, fps)
                file_counter += 1

        print(
            f"  -> wrote {file_counter} video(s) to {os.path.join(output_dir, split_name, embodiment_name)}"
        )


@hydra.main(
    version_base="1.3",
    config_path="../hydra_configs",
    config_name="viz_language.yaml",
)
def main(cfg: DictConfig) -> None:
    load_env()

    output_dir = cfg.output_dir
    split = cfg.get("split", "valid")
    max_episodes = cfg.get("max_episodes", None)
    max_batches = cfg.get("max_batches", 500)
    fps = cfg.get("fps", 30)
    frames_per_file = cfg.get("frames_per_file", 1000)
    annotation_key = OmegaConf.select(cfg, "data.annotation_key", default=None)
    viz_cfg = cfg.viz_func

    splits_to_run = []
    if split in ("train", "both"):
        splits_to_run.append(("train", cfg.data.train_datasets))
    if split in ("valid", "both"):
        splits_to_run.append(("valid", cfg.data.valid_datasets))

    for split_name, split_ds_cfgs in splits_to_run:
        datasets = {}
        for name, ds_cfg in split_ds_cfgs.items():
            ds = hydra.utils.instantiate(ds_cfg)
            if max_episodes is not None:
                keys = list(ds.datasets.keys())[:max_episodes]
                ds.datasets = {k: ds.datasets[k] for k in keys}
                # rebuild index_map and per-dataset index lists to match the truncated set
                ds.index_map = []
                ds._global_indices_by_dataset = {k: [] for k in ds.datasets}
                for ep_name, ep_ds in ds.datasets.items():
                    for local_idx in range(len(ep_ds)):
                        ds._global_indices_by_dataset[ep_name].append(len(ds.index_map))
                        ds.index_map.append((ep_name, local_idx))
            datasets[name] = ds
        # breakpoint()
        _run_viz_for_datasets(
            datasets=datasets,
            viz_cfg=viz_cfg,
            annotation_key=annotation_key,
            output_dir=output_dir,
            split_name=split_name,
            max_batches=max_batches,
            fps=fps,
            frames_per_file=frames_per_file,
        )


if __name__ == "__main__":
    main()
