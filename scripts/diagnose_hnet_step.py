"""Why does HNet's step output constant (244, 247) regardless of obs?

Compare:
  (a) forward_eval (teacher-forced) on 3 different val episodes — does the first
      predicted action vary across obs?
  (b) policy.step at t=0 on the same 3 obs — does it vary?
  (c) policy.step with same obs but a perturbed BOS — does varying BOS matter?
  (d) policy.step with same obs but zeros instead of BOS — sanity baseline.

If (a) varies per-obs but (b) doesn't → bug in step path.
If neither varies → model collapsed during training (predicts mean regardless).
"""
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
    missing, unexpected = algo.nets.load_state_dict(new_sd, strict=False)
    print(f"missing: {len(missing)}  unexpected: {len(unexpected)}")
    if missing[:3]: print(f"  sample missing: {missing[:3]}")
    if unexpected[:3]: print(f"  sample unexpected: {unexpected[:3]}")
    algo.nets = algo.nets.to(device)
    algo.nets.eval()

    # 3 distinct fake obs (different state values, same image dim)
    def make_obs(state_5, img_val):
        state = torch.tensor(state_5, dtype=torch.float32, device=device).unsqueeze(0)
        img = torch.full((1, 3, 96, 96), img_val, dtype=torch.float32, device=device)
        return {"state_agent_obj": state, "front_img_1": img}

    # Three different obs (simulate 3 episode starts):
    obs_list = [
        make_obs([293.4, 340.9, 309.7, 165.8, -2.88], 0.50),  # ep00 init
        make_obs([146.0, 401.0,  68.3, 377.6, -3.68], 0.55),  # ep01 init
        make_obs([423.0, 133.0, 323.5, 311.8,  0.73], 0.45),  # ep03 init
    ]

    # ===== (a) forward_eval style: just run policy with a 1-frame teacher-force =====
    print("\n=== (a) policy.step (AR inference) at t=0 with each obs ===")
    policy = algo.nets["policy"]
    ac_key = "actions"
    emb_id = 15
    for i, obs in enumerate(obs_list):
        obs_norm = norm_stats.normalize(obs, emb_id)
        state_init = policy.init_step_state(batch_size=1, T_max=600, device=device, dtype=torch.bfloat16)
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            a_norm = policy.step(state_init, obs_norm, t=0)
        a_un = norm_stats.unnormalize({ac_key: a_norm.squeeze(0).squeeze(0)}, emb_id)[ac_key]
        a_np = a_un.detach().float().cpu().numpy()
        print(f"  obs[{i}]  state_in={obs['state_agent_obj'].cpu().numpy()[0].tolist()}")
        print(f"           a_norm_pred={a_norm.detach().float().cpu().numpy().flatten().tolist()}")
        print(f"           a_unnorm   ={a_np.tolist()}")

    # ===== (b) compare to teacher-forced forward with 1-step actions =====
    print("\n=== (b) policy.forward (TEACHER-FORCED, 1-frame teacher input = BOS) ===")
    for i, obs in enumerate(obs_list):
        obs_norm = norm_stats.normalize(obs, emb_id)
        # build a 1-frame batch (B=1, T=1). Need actions tensor of shape (B, T, action_dim) for shift-right.
        # action_in goes raw (action_dim) → d_model. Use zeros for first action so the model sees BOS only.
        # actions[:, :-1] is empty, then prepended with BOS. So policy.forward(actions=zeros[1,1,2]) uses BOS as input.
        actions_0 = torch.zeros(1, 1, 2, dtype=torch.float32, device=device)
        # build obs in shape (B, T=1, ...) for forward path
        obs_seq = {k: v.unsqueeze(1) if v.dim() == 2 else v.unsqueeze(0) for k, v in obs_norm.items()}
        # Need to inspect _build_obs vs what forward expects — easiest: just call forward on 1-frame obs.
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred, aux = policy(actions_0, obs_norm)  # expect (1, 1, 2) pred
        a_un = norm_stats.unnormalize({ac_key: pred.squeeze(0).squeeze(0)}, emb_id)[ac_key]
        print(f"  obs[{i}]  a_norm_pred={pred.detach().float().cpu().numpy().flatten().tolist()}")
        print(f"           a_unnorm   ={a_un.detach().float().cpu().numpy().tolist()}")

if __name__ == "__main__":
    main()
