"""Find CrossTransformer settings whose TOTAL model params match the causal arms.

Params are exactly affine in nblocks (identical blocks), so measure the block
cost and the fixed cost at two points and SOLVE, instead of instantiating a
few hundred multi-hundred-MB transformers on a loaded box.
"""
import hydra
from omegaconf import OmegaConf

from egomimic.models.denoising_nets import CrossTransformer
from egomimic.pipeline.stages_ar import ARActionDecoder
from egomimic.pipeline.stages_sampler import MultiJActionSampler

DOM = "pushshapes_sim_gripper"
H, A, C, LATENT = 17, 5, 67, 96
dims = {DOM: A}

ENC = OmegaConf.create({
    "_target_": "egomimic.pipeline.stages_sampler.FusedObsEncoder",
    "n_obs_steps": 1,
    "encoder": {
        "_target_": "egomimic.pipeline.stages_sampler.DPStyleObsEncoder",
        "obs_specs": {"state_agent_obj": {"input_dim": 3, "input_slice": [0, 3]}},
        "img_encoders": {"front_img_1": {
            "_target_": "egomimic.models.stems.visual_core.VisualCore",
            "in_channels": 3, "image_size": 96, "num_kp": 32,
            "feature_dimension": 64, "pretrained": False, "crop_aug": True,
            "crop_height": 84, "crop_width": 84, "crop_eval_mode": "center",
            "crop_sample_mode": "v02", "crop_scope": "frame",
            "norm_layer": "group", "pool_type": "spatial_softmax"}},
    },
})


def n(m):
    return sum(p.numel() for p in m.parameters())


def arm1_total(hidden, nblocks, n_enc):
    ct = CrossTransformer(nblocks=nblocks, cond_dim=hidden, hidden_dim=hidden,
                          act_dim=LATENT, act_seq=H, n_heads=8, dropout=0.1,
                          mlp_layers=4, mlp_ratio=4, time_conditioning="additive")
    s = MultiJActionSampler(
        denoising_module=ct, condition_input_dim=C, action_horizon=H,
        action_dims=dims, latent_dim=LATENT, condition_dim=hidden,
        decoder_hidden_dim=512, denoiser_hidden_dim=hidden,
        num_inference_steps=16, schedule_anchor_domain=DOM)
    return n_enc + n(s)


n_enc = n(hydra.utils.instantiate(ENC))
print(f"shared encoder: {n_enc/1e6:.2f}M\n")

targets = {}
for tag, d, L, h in [("300M", 1024, 24, 16), ("30M", 512, 12, 8)]:
    dec = ARActionDecoder(condition_input_dim=C, action_horizon=H,
                          action_dims=dims, variant="state_action_ar",
                          d_model=d, n_layers=L, n_heads=h, n_waypoints=16)
    targets[tag] = n_enc + n(dec)
    print(f"arms 2/3/4 {tag:5s} TOTAL {targets[tag]/1e6:8.2f}M "
          f"(decoder {n(dec)/1e6:.2f}M, d={d} L={L} h={h})")

print("\narm 1 candidates (total = base + nblocks * per_block):")
best = {}
for hidden in (256, 384, 512, 768):
    t1 = arm1_total(hidden, 1, n_enc)
    t2 = arm1_total(hidden, 2, n_enc)
    per, base = t2 - t1, t1 - (t2 - t1)
    for tag, tgt in targets.items():
        nb = max(1, round((tgt - base) / per))
        total = base + nb * per
        err = abs(total - tgt) / tgt
        print(f"  {tag:5s} hidden={hidden:4d} -> nblocks={nb:3d} "
              f"total={total/1e6:8.2f}M err={err*100:5.2f}%")
        if tag not in best or err < best[tag][0]:
            best[tag] = (err, hidden, nb, total)

print("\nBEST (verified by real instantiation):")
for tag, (err, hidden, nb, _) in best.items():
    actual = arm1_total(hidden, nb, n_enc)
    err = abs(actual - targets[tag]) / targets[tag]
    print(f"  {tag:5s} hidden_dim={hidden} nblocks={nb}  "
          f"arm1={actual/1e6:8.2f}M  arms2-4={targets[tag]/1e6:8.2f}M  "
          f"err={err*100:.2f}%")
