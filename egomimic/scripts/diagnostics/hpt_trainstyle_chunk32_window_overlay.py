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


def _read_frames_actions(episode_path: Path) -> tuple[list[np.ndarray], np.ndarray]:
    root = zarr.open(str(episode_path), mode="r")
    frames = [
        simplejpeg.decode_jpeg(buf, colorspace="RGB")
        for buf in root["observations.images.front_img_1"][:]
    ]
    actions = np.asarray(root["actions"][:], dtype=np.float32)[:, :2]
    return frames, actions


def _to_canvas_xy(canvas: np.ndarray, xy: np.ndarray) -> tuple[int, int]:
    scale_x = canvas.shape[1] / 512.0
    scale_y = canvas.shape[0] / 512.0
    px = int(round(float(xy[0]) * scale_x))
    py = int(round(float(xy[1]) * scale_y))
    return (
        max(0, min(canvas.shape[1] - 1, px)),
        max(0, min(canvas.shape[0] - 1, py)),
    )


def _draw_chunk_path(canvas: np.ndarray, traj: np.ndarray, color: tuple[int, int, int], upto: int) -> None:
    if len(traj) == 0:
        return
    pts = [_to_canvas_xy(canvas, xy) for xy in traj]
    if len(pts) > 1:
        layer = canvas.copy()
        for i in range(1, len(pts)):
            cv2.line(layer, pts[i - 1], pts[i], color, thickness=2, lineType=cv2.LINE_AA)
        cv2.addWeighted(layer, 0.28, canvas, 0.72, 0, dst=canvas)
    prefix_end = max(1, min(len(pts), upto + 1))
    for i in range(1, prefix_end):
        cv2.line(canvas, pts[i - 1], pts[i], color, thickness=4, lineType=cv2.LINE_AA)
    cv2.circle(canvas, pts[0], 5, color, thickness=1, lineType=cv2.LINE_AA)
    cv2.circle(canvas, pts[prefix_end - 1], 6, color, thickness=-1, lineType=cv2.LINE_AA)


def _put_label(canvas: np.ndarray, text: str, y: int) -> None:
    cv2.putText(canvas, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 1, cv2.LINE_AA)


def _make_video(out_path: Path, records: list[dict], fps: int = 30, size: int = 512) -> None:
    rendered = []
    chunk_k = int(os.environ.get("HPT_TS_WINDOW_K", "32"))
    hold = int(os.environ.get("HPT_TS_WINDOW_HOLD", "4"))
    for rec in records:
        frames = rec["frames"]
        gt = rec["gt_actions"]
        pred = rec["pred_actions"]
        episode = rec["episode"]
        n = min(len(frames), len(gt), len(pred))
        for t in range(n):
            cs = (t // chunk_k) * chunk_k
            ce = min(cs + chunk_k, n)
            rel_t = t - cs
            canvas = cv2.resize(frames[t], (size, size), interpolation=cv2.INTER_NEAREST)
            canvas = np.ascontiguousarray(canvas)
            _draw_chunk_path(canvas, gt[cs:ce], (0, 220, 0), rel_t)
            _draw_chunk_path(canvas, pred[cs:ce], (230, 40, 40), rel_t)
            _put_label(canvas, f"{episode} | frame {t}/{n - 1}", 28)
            _put_label(canvas, f"TRAIN-STYLE chunk {cs}-{ce - 1}: obs window + GT chunk sample", 54)
            rendered.append(canvas)
            if rel_t == 0 and hold > 0:
                rendered.extend([canvas.copy() for _ in range(hold)])
        if rec is not records[-1]:
            rendered.extend([np.zeros((size, size, 3), dtype=np.uint8) for _ in range(8)])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tvio.write_video(str(out_path), torch.from_numpy(np.stack(rendered)).to(torch.uint8), fps=fps, video_codec="h264")


@torch.no_grad()
def _predict_episode_trainstyle(algo, md, dataset_name: str, device: torch.device, chunk_k: int = 32) -> tuple[np.ndarray, list[int]]:
    global_indices = md._global_indices_by_dataset[dataset_name]
    n = len(global_indices)
    chunk_starts = list(range(0, n, chunk_k))
    pred_full = np.full((n, 2), np.nan, dtype=np.float32)
    algo.nets.eval()
    batch_size = int(os.environ.get("HPT_TS_BATCH_SIZE", "16"))
    for batch_start in range(0, len(chunk_starts), batch_size):
        starts = chunk_starts[batch_start : batch_start + batch_size]
        samples = [md[global_indices[cs]] for cs in starts]
        collated = _to_device_collate(samples, device)
        processed = algo.process_batch_for_training({"pushshapes_sim": collated})
        pred_dict = algo.forward_eval(processed)
        pred_chunks = pred_dict["pushshapes_sim_actions"][:, :, :2].detach().cpu().float().numpy()
        for row, cs in enumerate(starts):
            ce = min(cs + chunk_k, n)
            pred_full[cs:ce] = pred_chunks[row, : ce - cs]
    return pred_full, chunk_starts


@torch.no_grad()
def _run_trainstyle_window_overlay(self, trainer, model_wrapper, datamodule, cfg):
    del trainer, cfg
    ckpt_path = os.environ["HPT_TS_CKPT"]
    out_dir = Path(os.environ["HPT_TS_OUT_DIR"])
    out_name = os.environ.get("HPT_TS_OUT_NAME", "hptFlowPaper_ep199_trainstyle_chunk32_windows_ep000-004_gt_vs_pred.mp4")
    n_episodes = int(os.environ.get("HPT_TS_EPISODES", "5"))
    split = os.environ.get("HPT_TS_SPLIT", "train")
    chunk_k = int(os.environ.get("HPT_TS_WINDOW_K", "32"))

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model_wrapper.load_state_dict(checkpoint["state_dict"], strict=False)
    print(f"[HPT_TS] loaded {ckpt_path} missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_wrapper.to(device)
    model_wrapper.eval()
    algo = model_wrapper.model
    algo.device = device
    algo.nets.to(device)
    algo.nets.eval()

    md = datamodule.valid_datasets["pushshapes_sim"] if split == "valid" else datamodule.train_datasets["pushshapes_sim"]
    episode_names = sorted(md.datasets.keys(), key=_episode_sort_key)[:n_episodes]
    records = []
    metrics = []
    for ep_i, name in enumerate(episode_names):
        child = md.datasets[name]
        ep_path = Path(child.episode_path)
        print(f"[HPT_TS] predicting ep {ep_i}: {name} ({len(child)} frames) split={split}", flush=True)
        pred, chunk_starts = _predict_episode_trainstyle(algo, md, name, device, chunk_k=chunk_k)
        frames, gt = _read_frames_actions(ep_path)
        n = min(len(frames), len(gt), len(pred))
        pred = pred[:n]
        gt = gt[:n]
        valid = np.isfinite(pred[:, 0])
        err = pred[valid] - gt[valid]
        metrics.append(
            {
                "episode": name,
                "frames": int(n),
                "chunk_starts": " ".join(str(x) for x in chunk_starts if x < n),
                "action_rmse": float(np.sqrt(np.mean(err**2))),
                "mean_xy_error": float(np.mean(np.linalg.norm(err, axis=1))),
                "max_xy_error": float(np.max(np.linalg.norm(err, axis=1))),
            }
        )
        records.append({"episode": name, "frames": frames[:n], "gt_actions": gt, "pred_actions": pred})

    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / out_name
    _make_video(video_path, records)
    json_path = out_dir / out_name.replace(".mp4", ".json")
    csv_path = out_dir / out_name.replace(".mp4", ".csv")
    payload = {
        "checkpoint": ckpt_path,
        "split": split,
        "method": "training-style chunk-window overlay: uses MultiDataset samples, process_batch_for_training, and forward_eval at chunk starts",
        "colors": {"ground_truth": "green", "model_prediction": "red"},
        "episodes": metrics,
    }
    json_path.write_text(json.dumps(payload, indent=2))
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)
    print(f"[HPT_TS] wrote {video_path}", flush=True)
    print(f"[HPT_TS] wrote {json_path}", flush=True)
    print(f"[HPT_TS] wrote {csv_path}", flush=True)


eval_hpt.HPTEvalVideo.run = _run_trainstyle_window_overlay

import hydra
from omegaconf import DictConfig, OmegaConf

from egomimic.trainHydra import train
from egomimic.pl_utils.utils import extras


@hydra.main(version_base="1.3", config_path="hydra_configs", config_name="train_zarr_cartesian.yaml")
def main(cfg: DictConfig):
    extras(cfg)
    print(OmegaConf.to_yaml(cfg))
    train(cfg)


if __name__ == "__main__":
    main()
