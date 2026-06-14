from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import simplejpeg
import torch
import zarr

from egomimic.eval import eval_hpt
from egomimic.rldb.embodiment.embodiment import get_embodiment_id


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


@torch.no_grad()
def _run_diag(self, trainer, model_wrapper, datamodule, cfg):
    del trainer, cfg
    ckpt_path = os.environ["HPT_DIAG_CKPT"]
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model_wrapper.load_state_dict(checkpoint["state_dict"], strict=False)
    print(f"[DIAG] loaded {ckpt_path} missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_wrapper.to(device)
    model_wrapper.eval()
    algo = model_wrapper.model
    algo.device = device
    algo.nets.to(device)
    algo.nets.eval()

    emb_id = get_embodiment_id("pushshapes_sim")
    ac_key = algo.ac_keys[emb_id]
    md = datamodule.train_datasets["pushshapes_sim"]
    ep_name = sorted(md.datasets.keys(), key=_episode_sort_key)[0]
    child = md.datasets[ep_name]
    ep_path = Path(child.episode_path)
    global_indices = md._global_indices_by_dataset[ep_name]

    # Teacher-forced / validation path, same as the overlay script.
    samples = [md[i] for i in global_indices[:8]]
    collated = _to_device_collate(samples, device)
    processed = algo.process_batch_for_training({"pushshapes_sim": collated})
    preds = algo.forward_eval(processed)
    tf_world = preds["pushshapes_sim_actions"][:8, 0, :2].detach().cpu().float().numpy()

    # Rollout path from raw zarr obs, same formatting/normalization as eval_sim.
    root = zarr.open(str(ep_path), mode="r")
    raw_state = np.asarray(root["observations.state"][:8], dtype=np.float32)
    raw_img = [
        simplejpeg.decode_jpeg(buf, colorspace="RGB")
        for buf in root["observations.images.front_img_1"][:8]
    ]
    state = algo.sim_init_state(batch_size=1, T_max=8, device=device, emb_id=emb_id)
    sim_world = []
    sim_chunks = []
    for t in range(8):
        image_chw = np.transpose(raw_img[t], (2, 0, 1)).astype(np.float32) / 255.0
        obs_raw = {
            "state_agent_obj": torch.from_numpy(raw_state[t]).unsqueeze(0).to(device),
            "front_img_1": torch.from_numpy(image_chw).unsqueeze(0).to(device),
        }
        obs_norm = algo.norm_stats.normalize(obs_raw, emb_id)
        # Force replanning each step to match the current eval_sim temporal-ensemble path.
        state["action_chunk"] = None
        state["chunk_idx"] = 0
        a_norm = algo.sim_predict_step(state, obs_norm, t, emb_id).squeeze(0)
        a_world = algo.norm_stats.unnormalize({ac_key: a_norm}, emb_id)[ac_key]
        sim_world.append(a_world.detach().cpu().float().numpy().reshape(-1)[:2])
        sim_chunks.append(state["action_chunk"][0, 0, :2].detach().cpu().float().numpy())
    sim_world = np.asarray(sim_world, dtype=np.float32)

    gt = np.asarray(root["actions"][:8], dtype=np.float32)[:, :2]
    print(f"[DIAG] episode={ep_name} action_key={ac_key}", flush=True)
    print("[DIAG] columns: t | GT_world | forward_eval_world | sim_predict_world | sim_minus_tf", flush=True)
    for t in range(8):
        diff = sim_world[t] - tf_world[t]
        print(
            f"[DIAG] {t:02d} gt={gt[t].round(3).tolist()} "
            f"tf={tf_world[t].round(3).tolist()} sim={sim_world[t].round(3).tolist()} "
            f"diff={diff.round(3).tolist()}",
            flush=True,
        )
    print(
        f"[DIAG] mean_abs_diff={np.mean(np.abs(sim_world - tf_world), axis=0).round(4).tolist()} "
        f"max_abs_diff={np.max(np.abs(sim_world - tf_world), axis=0).round(4).tolist()}",
        flush=True,
    )


eval_hpt.HPTEvalVideo.run = _run_diag

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
