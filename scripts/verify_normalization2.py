"""Verify after process_batch_for_training what the model actually sees."""
import torch, numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

CKPT = "logs/hnet_variants/hnet_iterA_obsnoise_pace_l40s_2026-05-19_05-44-22_2026-05-19_05-44-34/checkpoints/last.ckpt"
CFG = "logs/hnet_variants/hnet_iterA_obsnoise_pace_l40s_2026-05-19_05-44-22_2026-05-19_05-44-34/.hydra/config.yaml"

def s(t, label):
    a = t.detach().float().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    print(f"  {label}: shape={a.shape}  mean={a.mean():.3f}  std={a.std():.3f}  min={a.min():.3f}  max={a.max():.3f}")

def main():
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = OmegaConf.load(CFG)
    from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset
    norm_stats = MultiDataset.from_state(ckpt["hyper_parameters"]["norm_stats_state"])
    algo = instantiate(cfg.model.robomimic_model, norm_stats=norm_stats)
    dm = instantiate(cfg.data)
    dm.setup(stage="fit")
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    if isinstance(batch, tuple): batch = batch[0]
    print("=== dataloader emit (BEFORE process_batch_for_training) ===")
    emb = list(batch.keys())[0]
    s(batch[emb]["state_agent_obj"], f"  raw [state_agent_obj]")
    s(batch[emb]["actions"], f"  raw [actions]")

    print("\n=== after process_batch_for_training ===")
    processed = algo.process_batch_for_training(batch)
    emb_id = list(processed.keys())[0]
    s(processed[emb_id]["state_agent_obj"], f"  proc [state_agent_obj]")
    s(processed[emb_id]["actions"], f"  proc [actions]")

    print("\n=== inference path for the same env init ===")
    from Tsimulation.pushshapes import PushShapesEnv
    from egomimic.rldb.embodiment.pushshapes_sim import _env_to_zarr_pushshapes
    env = PushShapesEnv(object_shape="T", pusher_shape="circle", obstacle_level=0, image_size=96, render_mode="rgb_array")
    env.reset(seed=0)
    obs_zarr = _env_to_zarr_pushshapes(env._get_obs(), torch.device("cpu"))
    s(obs_zarr["state_agent_obj"], "  obs_zarr raw")
    obs_norm = norm_stats.normalize(obs_zarr, 15)
    s(obs_norm["state_agent_obj"], "  obs_norm")

    # match check
    a = processed[emb_id]["state_agent_obj"].detach().float().cpu().numpy().flatten()[:5]
    b = obs_norm["state_agent_obj"].detach().float().cpu().numpy().flatten()
    print(f"\n  train normed first row: {a.round(3).tolist()}")
    print(f"  infer normed obs row 0: {b.round(3).tolist()}")
    print(f"  same scale? train_range=[{a.min():.2f},{a.max():.2f}], infer_range=[{b.min():.2f},{b.max():.2f}]")

if __name__ == "__main__":
    main()
