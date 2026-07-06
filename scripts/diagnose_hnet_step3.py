"""Compare step (AR) to forward_packed (training+val path) on the same obs."""
import torch, numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

CKPT = "logs/hnet_variants/hnet_iterA_obsnoise_pace_l40s_2026-05-19_05-44-22_2026-05-19_05-44-34/checkpoints/last.ckpt"
CFG = "logs/hnet_variants/hnet_iterA_obsnoise_pace_l40s_2026-05-19_05-44-22_2026-05-19_05-44-34/.hydra/config.yaml"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = OmegaConf.load(CFG)
    from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset
    norm_stats = MultiDataset.from_state(ckpt["hyper_parameters"]["norm_stats_state"])
    algo = instantiate(cfg.model.robomimic_model, norm_stats=norm_stats)
    state_dict = ckpt["state_dict"]
    new_sd = {k[len("nets."):] if k.startswith("nets.") else k: v for k, v in state_dict.items()}
    algo.nets.load_state_dict(new_sd, strict=False)
    algo.nets = algo.nets.to(device); algo.nets.eval()
    policy = algo.nets["policy"]
    ac_key, emb_id = "actions", 15

    obs_list = [
        [293.4, 340.9, 309.7, 165.8, -2.88],
        [146.0, 401.0,  68.3, 377.6, -3.68],
        [423.0, 133.0, 323.5, 311.8,  0.73],
    ]
    T = 8

    print("=== (c) forward_packed (training/val path), T=8, single subseq ===")
    for i, s5 in enumerate(obs_list):
        # Build packed obs: (T, ...) with cu_seqlens=[0, T]
        state = torch.tensor(s5, dtype=torch.float32, device=device).unsqueeze(0).repeat(T, 1)  # (T, 5)
        img = torch.full((T, 3, 96, 96), 0.5, dtype=torch.float32, device=device)
        obs_packed = {"state_agent_obj": state, "front_img_1": img}
        obs_norm = norm_stats.normalize(obs_packed, emb_id)
        actions = torch.zeros(T, 2, dtype=torch.float32, device=device)
        cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=device)
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred, aux = policy.forward_packed(actions, obs_norm, cu_seqlens, T)
        a0_norm = pred[0].detach().float().cpu().numpy()
        a0_un = norm_stats.unnormalize({ac_key: pred[0]}, emb_id)[ac_key]
        print(f"  obs[{i}] state={s5}  a0_norm={a0_norm.round(4).tolist()}  a0_unnorm={a0_un.detach().float().cpu().numpy().round(2).tolist()}")

if __name__ == "__main__":
    main()
