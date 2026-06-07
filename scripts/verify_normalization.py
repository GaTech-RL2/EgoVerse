"""Verify proprio normalization at every code path: training, eval, inference.

For HNet Iter A ckpt + circle/basic episode 0:
  1. Training path: pull one batch from train dataloader -> what is state_agent_obj?
  2. Eval (val) path: pull one batch from valid dataloader -> what?
  3. Inference path: build obs_zarr like eval_sim does, normalize via norm_stats,
     -> what?
For each: print state_agent_obj.{mean,std,min,max} and the dict keys.
"""
from __future__ import annotations
import torch, numpy as np
from pathlib import Path
from hydra.utils import instantiate
from omegaconf import OmegaConf

CKPT = "logs/hnet_variants/hnet_iterA_obsnoise_pace_l40s_2026-05-19_05-44-22_2026-05-19_05-44-34/checkpoints/last.ckpt"
CFG = "logs/hnet_variants/hnet_iterA_obsnoise_pace_l40s_2026-05-19_05-44-22_2026-05-19_05-44-34/.hydra/config.yaml"

def stats(t, label):
    if isinstance(t, torch.Tensor):
        a = t.detach().float().cpu().numpy()
    else:
        a = np.asarray(t)
    print(f"  {label}: shape={a.shape}  mean={a.mean():.3f}  std={a.std():.3f}  min={a.min():.3f}  max={a.max():.3f}")

def main():
    print("=== loading ckpt + cfg ===")
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = OmegaConf.load(CFG)

    from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset
    norm_state = ckpt["hyper_parameters"]["norm_stats_state"]
    norm_stats = MultiDataset.from_state(norm_state)
    print("emb ids in norm_stats:", list(norm_stats.norm_stats.keys()))
    emb_id = list(norm_stats.norm_stats.keys())[0]
    # state_agent_obj key
    print("norm_stats[emb_id] keys:", list(norm_stats.norm_stats[emb_id].keys()))
    print("zarr_keys[emb_id]:", norm_stats.zarr_keys[emb_id])
    stats_state = norm_stats.norm_stats[emb_id].get("state_agent_obj")
    if stats_state is not None:
        print(f"  state_agent_obj norm stats: mean={np.asarray(stats_state['mean']).tolist()}  std={np.asarray(stats_state['std']).tolist()}")

    # ============== PATH 1: TRAINING DATALOADER ==============
    print("\n=== PATH 1: training dataloader (train batch) ===")
    dm = instantiate(cfg.data)
    dm.setup(stage="fit")
    train_loader = dm.train_dataloader()
    first = next(iter(train_loader))
    if isinstance(first, tuple): first = first[0]
    print(f"  batch top-level keys: {list(first.keys()) if isinstance(first, dict) else type(first)}")
    if isinstance(first, dict):
        emb_first = list(first.keys())[0]
        b = first[emb_first]
        print(f"  inner keys for emb={emb_first}: {list(b.keys())[:15]}")
        for cand in ("state_agent_obj", "observations.state"):
            if cand in b:
                stats(b[cand], f"train  [{cand}]")

    # ============== PATH 2: VALIDATION DATALOADER (eval path) ==============
    print("\n=== PATH 2: validation dataloader (val batch, what eval / val_viz sees) ===")
    dm.setup(stage="validate")
    val_loader = dm.val_dataloader()
    first_v = next(iter(val_loader))
    if isinstance(first_v, tuple): first_v = first_v[0]
    if isinstance(first_v, dict):
        emb_first = list(first_v.keys())[0]
        b = first_v[emb_first]
        for cand in ("state_agent_obj", "observations.state"):
            if cand in b:
                stats(b[cand], f"val    [{cand}]")

    # ============== PATH 3: INFERENCE _env_to_zarr -> normalize ==============
    print("\n=== PATH 3: inference (env obs -> _env_to_zarr_pushshapes -> normalize) ===")
    from Tsimulation.pushshapes import PushShapesEnv
    env = PushShapesEnv(object_shape="T", pusher_shape="circle", obstacle_level=0, image_size=96, render_mode="rgb_array")
    env.reset(seed=0)
    obs_env = env._get_obs()
    print(f"  raw env obs keys: {list(obs_env.keys())}")
    print(f"  raw env state (agent_pos + object_pose):")
    print(f"    agent_pos={obs_env['agent_pos']}  object_pose={obs_env['object_pose']}")

    from egomimic.rldb.embodiment.pushshapes_sim import _env_to_zarr_pushshapes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_zarr = _env_to_zarr_pushshapes(obs_env, device)
    print(f"  obs_zarr (after _env_to_zarr_pushshapes) keys: {list(obs_zarr.keys())}")
    stats(obs_zarr["state_agent_obj"], "obs_zarr [state_agent_obj] BEFORE normalize")

    obs_norm = norm_stats.normalize(obs_zarr, emb_id)
    print(f"  obs_norm keys: {list(obs_norm.keys())}")
    stats(obs_norm["state_agent_obj"], "obs_norm [state_agent_obj] AFTER  normalize")
    # also check if zarr-keyed variant slipped in
    for k in obs_norm:
        if "state" in k.lower():
            stats(obs_norm[k], f"obs_norm [{k}] (whatever survives)")

    # Are values BEFORE and AFTER normalize the same? If yes -> normalize is a no-op.
    a = obs_zarr["state_agent_obj"].detach().float().cpu().numpy()
    b = obs_norm["state_agent_obj"].detach().float().cpu().numpy()
    if np.allclose(a, b):
        print("\n  >>> CONFIRMED: normalize() is a NO-OP. obs reaches model unnormalized. <<<")
    else:
        print("\n  normalize changed values (looks correct).")

if __name__ == "__main__":
    main()
