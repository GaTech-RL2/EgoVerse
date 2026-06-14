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


def _read_state_images_actions(episode_path: Path) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    root = zarr.open(str(episode_path), mode="r")
    states = np.asarray(root["observations.state"][:], dtype=np.float32)
    images = [
        simplejpeg.decode_jpeg(buf, colorspace="RGB")
        for buf in root["observations.images.front_img_1"][:]
    ]
    actions = np.asarray(root["actions"][:], dtype=np.float32)[:, :2]
    return states, images, actions


def _format_one_obs(state_5: np.ndarray, image_rgb: np.ndarray, device: torch.device) -> dict:
    image_chw = np.transpose(image_rgb, (2, 0, 1)).astype(np.float32) / 255.0
    return {
        "state_agent_obj": torch.from_numpy(state_5.astype(np.float32)).unsqueeze(0).to(device),
        "front_img_1": torch.from_numpy(image_chw).unsqueeze(0).to(device),
    }


def _to_canvas_xy(canvas: np.ndarray, xy: np.ndarray) -> tuple[int, int]:
    scale_x = canvas.shape[1] / 512.0
    scale_y = canvas.shape[0] / 512.0
    px = int(round(float(xy[0]) * scale_x))
    py = int(round(float(xy[1]) * scale_y))
    return (
        max(0, min(canvas.shape[1] - 1, px)),
        max(0, min(canvas.shape[0] - 1, py)),
    )


def _draw_chunk_path(
    canvas: np.ndarray,
    traj: np.ndarray,
    color: tuple[int, int, int],
    upto: int,
    *,
    faint_alpha: float,
) -> None:
    if len(traj) == 0:
        return
    pts = [_to_canvas_xy(canvas, xy) for xy in traj]
    if len(pts) > 1:
        layer = canvas.copy()
        for i in range(1, len(pts)):
            cv2.line(layer, pts[i - 1], pts[i], color, thickness=2, lineType=cv2.LINE_AA)
        cv2.addWeighted(layer, faint_alpha, canvas, 1.0 - faint_alpha, 0, dst=canvas)

    prefix_end = max(1, min(len(pts), upto + 1))
    for i in range(1, prefix_end):
        cv2.line(canvas, pts[i - 1], pts[i], color, thickness=4, lineType=cv2.LINE_AA)
    cv2.circle(canvas, pts[0], 5, color, thickness=1, lineType=cv2.LINE_AA)
    cv2.circle(canvas, pts[prefix_end - 1], 6, color, thickness=-1, lineType=cv2.LINE_AA)


def _put_label(canvas: np.ndarray, text: str, y: int) -> None:
    cv2.putText(
        canvas,
        text,
        (12, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        text,
        (12, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def _make_concat_video(out_path: Path, records: list[dict], fps: int = 30, size: int = 512) -> None:
    rendered = []
    chunk_k = int(os.environ.get("HPT_C32_WINDOW_K", "32"))
    hold = int(os.environ.get("HPT_C32_WINDOW_HOLD", "4"))

    for ep_i, rec in enumerate(records):
        frames = rec["frames"]
        gt = rec["gt_actions"]
        pred = rec["pred_actions"]
        episode = rec["episode"]
        n = min(len(frames), len(gt), len(pred))
        for t in range(n):
            chunk_start = (t // chunk_k) * chunk_k
            chunk_end = min(chunk_start + chunk_k, n)
            rel_t = t - chunk_start
            canvas = cv2.resize(frames[t], (size, size), interpolation=cv2.INTER_NEAREST)
            canvas = np.ascontiguousarray(canvas)
            _draw_chunk_path(canvas, gt[chunk_start:chunk_end], (0, 220, 0), rel_t, faint_alpha=0.24)
            _draw_chunk_path(canvas, pred[chunk_start:chunk_end], (230, 40, 40), rel_t, faint_alpha=0.30)
            _put_label(canvas, f"{episode} | frame {t}/{n - 1}", 28)
            _put_label(canvas, f"chunk {chunk_start}-{chunk_end - 1}: green=GT, red=predicted next 32", 54)
            rendered.append(canvas)

            if rel_t == 0 and hold > 0:
                rendered.extend([canvas.copy() for _ in range(hold)])

        if rec is not records[-1]:
            rendered.extend([np.zeros((size, size, 3), dtype=np.uint8) for _ in range(8)])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tvio.write_video(
        str(out_path),
        torch.from_numpy(np.stack(rendered)).to(torch.uint8),
        fps=fps,
        video_codec="h264",
    )


@torch.no_grad()
def _predict_episode_chunk32(algo, episode_path: Path, emb_id: int, device: torch.device) -> tuple[np.ndarray, list[int]]:
    states, images, _actions = _read_state_images_actions(episode_path)
    n = len(states)
    ac_key = algo.ac_keys[emb_id]
    state = algo.sim_init_state(batch_size=1, T_max=n, device=device, emb_id=emb_id)
    preds = []
    chunk_starts = []
    for t in range(n):
        was_new_chunk = state.get("action_chunk") is None or int(state.get("chunk_idx", 0)) >= 32
        obs_raw = _format_one_obs(states[t], images[t], device)
        obs_norm = algo.norm_stats.normalize(obs_raw, emb_id)
        a_norm = algo.sim_predict_step(state, obs_norm, t, emb_id).squeeze(0)
        if was_new_chunk:
            chunk_starts.append(t)
        a_world = algo.norm_stats.unnormalize({ac_key: a_norm}, emb_id)[ac_key]
        preds.append(a_world.detach().cpu().float().numpy().reshape(-1)[:2])
    return np.asarray(preds, dtype=np.float32), chunk_starts


@torch.no_grad()
def _run_chunk32_window_overlay(self, trainer, model_wrapper, datamodule, cfg):
    del trainer, cfg
    ckpt_path = os.environ["HPT_C32_CKPT"]
    out_dir = Path(os.environ["HPT_C32_OUT_DIR"])
    out_name = os.environ.get(
        "HPT_C32_OUT_NAME", "hptFlowPaper_ep199_chunk32_windows_ep000-004_gt_vs_pred.mp4"
    )
    n_episodes = int(os.environ.get("HPT_C32_EPISODES", "5"))

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model_wrapper.load_state_dict(checkpoint["state_dict"], strict=False)
    print(f"[HPT_C32_WINDOW] loaded {ckpt_path} missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_wrapper.to(device)
    model_wrapper.eval()
    algo = model_wrapper.model
    algo.device = device
    algo.nets.to(device)
    algo.nets.eval()

    split = os.environ.get("HPT_C32_SPLIT", "train")
    if split == "valid":
        md = datamodule.valid_datasets["pushshapes_sim"]
    elif split == "train":
        md = datamodule.train_datasets["pushshapes_sim"]
    else:
        raise ValueError(f"HPT_C32_SPLIT must be train or valid, got {split!r}")
    emb_id = next(iter(md.norm_stats.keys())) if getattr(md, "norm_stats", None) else 15
    episode_names = sorted(md.datasets.keys(), key=_episode_sort_key)[:n_episodes]
    records = []
    metrics = []

    for ep_i, name in enumerate(episode_names):
        child = md.datasets[name]
        ep_path = Path(child.episode_path)
        print(f"[HPT_C32_WINDOW] predicting ep {ep_i}: {name} ({len(child)} frames)", flush=True)
        pred, chunk_starts = _predict_episode_chunk32(algo, ep_path, emb_id, device)
        states, frames, gt = _read_state_images_actions(ep_path)
        del states
        n = min(len(frames), len(gt), len(pred))
        pred = pred[:n]
        gt = gt[:n]
        err = pred - gt
        chunk_metrics = []
        for cs in chunk_starts:
            if cs >= n:
                continue
            ce = min(cs + 32, n)
            e = err[cs:ce]
            chunk_metrics.append(
                {
                    "start": int(cs),
                    "end": int(ce - 1),
                    "mean_xy_error": float(np.mean(np.linalg.norm(e, axis=1))),
                    "rmse": float(np.sqrt(np.mean(e**2))),
                }
            )

        metrics.append(
            {
                "episode": name,
                "frames": int(n),
                "chunk_starts": " ".join(str(x) for x in chunk_starts if x < n),
                "action_rmse": float(np.sqrt(np.mean(err**2))),
                "mean_xy_error": float(np.mean(np.linalg.norm(err, axis=1))),
                "max_xy_error": float(np.max(np.linalg.norm(err, axis=1))),
                "chunks": chunk_metrics,
            }
        )
        records.append(
            {
                "episode": name,
                "frames": frames[:n],
                "gt_actions": gt,
                "pred_actions": pred,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / out_name
    _make_concat_video(video_path, records)
    json_path = out_dir / out_name.replace(".mp4", ".json")
    csv_path = out_dir / out_name.replace(".mp4", ".csv")
    payload = {
        "checkpoint": ckpt_path,
        "dataset": str(Path(md.datasets[episode_names[0]].episode_path).parent),
        "split": split,
        "method": "chunk-window overlay: at frames 0,32,64,... show only that predicted 32-action chunk against the matching GT chunk",
        "colors": {"ground_truth": "green", "model_prediction": "red"},
        "episodes": metrics,
    }
    json_path.write_text(json.dumps(payload, indent=2))
    with csv_path.open("w", newline="") as f:
        fieldnames = ["episode", "frames", "chunk_starts", "action_rmse", "mean_xy_error", "max_xy_error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics:
            writer.writerow({k: row[k] for k in fieldnames})
    print(f"[HPT_C32_WINDOW] wrote {video_path}", flush=True)
    print(f"[HPT_C32_WINDOW] wrote {json_path}", flush=True)
    print(f"[HPT_C32_WINDOW] wrote {csv_path}", flush=True)


eval_hpt.HPTEvalVideo.run = _run_chunk32_window_overlay

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
