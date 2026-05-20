"""Check image obs scale at training vs inference."""
import torch, numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

CKPT = "logs/hnet_variants/hnet_iterA_obsnoise_pace_l40s_2026-05-19_05-44-22_2026-05-19_05-44-34/checkpoints/last.ckpt"
CFG = "logs/hnet_variants/hnet_iterA_obsnoise_pace_l40s_2026-05-19_05-44-22_2026-05-19_05-44-34/.hydra/config.yaml"

def s(t, label):
    a = t.detach().float().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    print(f"  {label}: shape={a.shape}  dtype={a.dtype}  mean={a.mean():.4f}  std={a.std():.4f}  min={a.min():.4f}  max={a.max():.4f}")

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
    print("=== dataloader emit (RAW) ===")
    emb = list(batch.keys())[0]
    s(batch[emb]["front_img_1"], "  raw image train")

    print("\n=== after process_batch_for_training ===")
    processed = algo.process_batch_for_training(batch)
    emb_id = list(processed.keys())[0]
    s(processed[emb_id]["front_img_1"], "  proc image train")

    print("\n=== inference image ===")
    from Tsimulation.pushshapes import PushShapesEnv
    from egomimic.eval.eval_sim import _env_to_zarr_pushshapes
    env = PushShapesEnv(object_shape="T", pusher_shape="circle", obstacle_level=0, image_size=96, render_mode="rgb_array")
    env.reset(seed=0)
    obs_zarr = _env_to_zarr_pushshapes(env._get_obs(), torch.device("cpu"))
    s(obs_zarr["front_img_1"], "  obs_zarr image (no norm)")
    obs_norm = norm_stats.normalize(obs_zarr, 15)
    s(obs_norm["front_img_1"], "  obs_norm image (after normalize)")

if __name__ == "__main__":
    main()
