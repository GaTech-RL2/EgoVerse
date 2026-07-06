"""Step vs forward equivalence test on real model. Use T=8 to avoid attn contiguity bugs."""
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

    def make_obs(s5, img_val, T=8):
        state = torch.tensor(s5, dtype=torch.float32, device=device).unsqueeze(0).repeat(T, 1).unsqueeze(0)  # (1, T, 5)
        img = torch.full((1, T, 3, 96, 96), img_val, dtype=torch.float32, device=device)
        return {"state_agent_obj": state, "front_img_1": img}, T

    obs_list = [
        ([293.4, 340.9, 309.7, 165.8, -2.88], 0.50),
        ([146.0, 401.0,  68.3, 377.6, -3.68], 0.55),
        ([423.0, 133.0, 323.5, 311.8,  0.73], 0.45),
    ]

    print("=== (a) policy.step (AR inference) at t=0 with SAME obs (5d states differ) ===")
    for i, (s5, img) in enumerate(obs_list):
        obs, _ = make_obs(s5, img, T=1)
        # collapse T=1 dim
        obs = {k: v.squeeze(1) if v.dim() > 2 else v for k, v in obs.items()}
        for k, v in obs.items():
            if k == "front_img_1" and v.dim() == 4:
                pass  # ok (B=1, C, H, W)
            elif k == "state_agent_obj" and v.dim() == 2:
                pass  # (B=1, 5)
        obs_norm = norm_stats.normalize(obs, emb_id)
        state_init = policy.init_step_state(batch_size=1, T_max=600, device=device, dtype=torch.bfloat16)
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            a_norm = policy.step(state_init, obs_norm, t=0)
        a_un = norm_stats.unnormalize({ac_key: a_norm.squeeze(0).squeeze(0)}, emb_id)[ac_key]
        print(f"  obs[{i}] state={s5}  a_norm={a_norm.detach().float().cpu().numpy().flatten().round(4).tolist()}  a_unnorm={a_un.detach().float().cpu().numpy().round(2).tolist()}")

    print("\n=== (b) policy.forward (TEACHER-FORCED, T=8 same-obs-repeated) ===")
    for i, (s5, img) in enumerate(obs_list):
        obs, T = make_obs(s5, img, T=8)
        # forward expects (B, T, ...) obs and (B, T, action_dim) actions
        obs_norm = norm_stats.normalize(obs, emb_id)
        # use zero actions (so shift-right + BOS gives BOS at t=0, zeros at t>0)
        actions_0 = torch.zeros(1, T, 2, dtype=torch.float32, device=device)
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred, aux = policy(actions_0, obs_norm)
        a0_norm = pred[0, 0].detach().float().cpu().numpy()
        a0_un = norm_stats.unnormalize({ac_key: pred[0, 0]}, emb_id)[ac_key]
        print(f"  obs[{i}] state={s5}  a0_norm={a0_norm.round(4).tolist()}  a0_unnorm={a0_un.detach().float().cpu().numpy().round(2).tolist()}")

if __name__ == "__main__":
    main()
