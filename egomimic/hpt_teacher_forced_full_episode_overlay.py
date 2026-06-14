from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np
import simplejpeg
import torch
import torchvision.io as tvio
import zarr

from egomimic.eval import eval_hpt


def _episode_sort_key(name: str) -> tuple[int, str]:
    stem = name[:-5] if name.endswith(".zarr") else name
    tail = stem.rsplit("_", 1)[-1]
    try:
        return int(tail), name
    except ValueError:
        return 10**9, name


def _to_device_collate(samples: list[dict], device: torch.device) -> dict:
    out = {}
    keys = [k for k in samples[0].keys() if torch.is_tensor(samples[0][k])]
    for key in keys:
        vals = [s[key] for s in samples]
        if vals[0].ndim == 0:
            continue
        tensor = torch.stack(vals, dim=0).to(device)
        if tensor.is_floating_point():
            tensor = tensor.float()
        out[key] = tensor
    return out


def _decode_episode_frames(episode_path: Path) -> np.ndarray:
    root = zarr.open(str(episode_path), mode="r")
    arr = root["observations.images.front_img_1"][:]
    frames = []
    for buf in arr:
        frames.append(simplejpeg.decode_jpeg(buf, colorspace="RGB"))
    return np.asarray(frames, dtype=np.uint8)


def _read_actions(episode_path: Path) -> np.ndarray:
    root = zarr.open(str(episode_path), mode="r")
    return np.asarray(root["actions"][:], dtype=np.float32)[:, :2]


def _draw_path(
    canvas: np.ndarray,
    traj: np.ndarray,
    color: tuple[int, int, int],
    upto: int,
    *,
    faint: bool,
) -> None:
    if traj.size == 0:
        return
    scale_x = canvas.shape[1] / 512.0
    scale_y = canvas.shape[0] / 512.0
    pts = []
    for xy in traj:
        px = int(round(float(xy[0]) * scale_x))
        py = int(round(float(xy[1]) * scale_y))
        pts.append((max(0, min(canvas.shape[1] - 1, px)), max(0, min(canvas.shape[0] - 1, py))))
    if len(pts) > 1:
        layer = canvas.copy()
        for i in range(1, len(pts)):
            cv2.line(layer, pts[i - 1], pts[i], color, thickness=2, lineType=cv2.LINE_AA)
        if faint:
            cv2.addWeighted(layer, 0.28, canvas, 0.72, 0, dst=canvas)
        else:
            canvas[:] = layer
    prefix_end = max(1, min(len(pts), upto + 1))
    for i in range(1, prefix_end):
        cv2.line(canvas, pts[i - 1], pts[i], color, thickness=4, lineType=cv2.LINE_AA)
    cv2.circle(canvas, pts[prefix_end - 1], 6, color, thickness=-1, lineType=cv2.LINE_AA)


def _make_concat_video(
    out_path: Path,
    episode_records: list[dict],
    fps: int = 30,
    size: int = 512,
) -> None:
    rendered = []
    for rec in episode_records:
        frames = rec["frames"]
        gt = rec["gt_actions"]
        pred = rec["pred_actions"]
        n = min(len(frames), len(gt), len(pred))
        for t in range(n):
            canvas = cv2.resize(frames[t], (size, size), interpolation=cv2.INTER_NEAREST)
            canvas = np.ascontiguousarray(canvas)
            _draw_path(canvas, gt[:n], (0, 220, 0), t, faint=True)
            _draw_path(canvas, pred[:n], (230, 40, 40), t, faint=True)
            rendered.append(canvas)
        if rec is not episode_records[-1]:
            rendered.extend([np.zeros((size, size, 3), dtype=np.uint8) for _ in range(8)])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tvio.write_video(
        str(out_path),
        torch.from_numpy(np.stack(rendered)).to(torch.uint8),
        fps=fps,
        video_codec="h264",
    )


@torch.no_grad()
def _predict_episode(algo, md, dataset_name: str, batch_size: int, device: torch.device) -> np.ndarray:
    global_indices = md._global_indices_by_dataset[dataset_name]
    preds = []
    algo.nets.eval()
    for start in range(0, len(global_indices), batch_size):
        idxs = global_indices[start : start + batch_size]
        samples = [md[i] for i in idxs]
        collated = _to_device_collate(samples, device)
        processed = algo.process_batch_for_training({"pushshapes_sim": collated})
        pred_dict = algo.forward_eval(processed)
        actions = pred_dict["pushshapes_sim_actions"][:, 0, :2]
        preds.append(actions.detach().cpu().float().numpy())
    return np.concatenate(preds, axis=0).astype(np.float32)


def _run_teacher_forced(self, trainer, model_wrapper, datamodule, cfg):
    del trainer, cfg
    ckpt_path = os.environ["HPT_TF_CKPT"]
    out_dir = Path(os.environ["HPT_TF_OUT_DIR"])
    out_name = os.environ.get(
        "HPT_TF_OUT_NAME", "hptFlowPaper_teacher_forced_full_ep000-004_gt_vs_pred.mp4"
    )
    n_episodes = int(os.environ.get("HPT_TF_EPISODES", "5"))
    batch_size = int(os.environ.get("HPT_TF_BATCH_SIZE", "24"))

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model_wrapper.load_state_dict(checkpoint["state_dict"], strict=False)
    print(f"[HPT_TF] loaded {ckpt_path} missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_wrapper.to(device)
    model_wrapper.eval()
    algo = model_wrapper.model
    algo.device = device
    algo.nets.to(device)
    algo.nets.eval()

    md = datamodule.train_datasets["pushshapes_sim"]
    episode_names = sorted(md.datasets.keys(), key=_episode_sort_key)[:n_episodes]
    records = []
    metrics = []
    for ep_i, name in enumerate(episode_names):
        child = md.datasets[name]
        ep_path = Path(child.episode_path)
        print(f"[HPT_TF] predicting ep {ep_i}: {name} ({len(child)} frames)", flush=True)
        pred = _predict_episode(algo, md, name, batch_size, device)
        gt = _read_actions(ep_path)
        frames = _decode_episode_frames(ep_path)
        n = min(len(frames), len(gt), len(pred))
        pred = pred[:n]
        gt = gt[:n]
        err = pred - gt
        rec = {
            "episode": name,
            "frames": int(n),
            "action_rmse_xy": np.sqrt(np.mean(err**2, axis=0)).tolist(),
            "action_rmse": float(np.sqrt(np.mean(err**2))),
            "final_xy_error": float(np.linalg.norm(err[-1])),
            "mean_xy_error": float(np.mean(np.linalg.norm(err, axis=1))),
            "max_xy_error": float(np.max(np.linalg.norm(err, axis=1))),
        }
        metrics.append(rec)
        records.append({"frames": frames[:n], "gt_actions": gt, "pred_actions": pred})

    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / out_name
    _make_concat_video(video_path, records)
    json_path = out_dir / out_name.replace(".mp4", ".json")
    csv_path = out_dir / out_name.replace(".mp4", ".csv")
    payload = {
        "checkpoint": ckpt_path,
        "dataset": str(Path(md.datasets[episode_names[0]].episode_path).parent),
        "method": "teacher-forced 32-step HPT chunks over stored observations; plotted action[:,0] per timestep",
        "colors": {"ground_truth": "green", "model_prediction": "red"},
        "episodes": metrics,
    }
    json_path.write_text(json.dumps(payload, indent=2))
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)
    print(f"[HPT_TF] wrote {video_path}", flush=True)
    print(f"[HPT_TF] wrote {json_path}", flush=True)
    print(f"[HPT_TF] wrote {csv_path}", flush=True)


eval_hpt.HPTEvalVideo.run = _run_teacher_forced

import hydra
from omegaconf import DictConfig, OmegaConf

from egomimic.trainHydra import train
from egomimic.utils.utils import extras


@hydra.main(
    version_base="1.3",
    config_path="hydra_configs",
    config_name="train_zarr_cartesian.yaml",
)
def main(cfg: DictConfig):
    extras(cfg)
    print(OmegaConf.to_yaml(cfg))
    train(cfg)


if __name__ == "__main__":
    main()
