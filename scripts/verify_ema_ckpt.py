import glob, os, sys
import torch

cands = sorted(
    glob.glob(
        "/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hptFlowComboEma_smoke/*/checkpoints/*.ckpt"
    ),
    key=os.path.getmtime,
)
assert cands, "no smoke checkpoint found"
ckpt_path = cands[-1]
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
print("CKPT=", ckpt_path)
print("KEYS=", sorted(ckpt.keys()))
ema = ckpt.get("ema_state_dict")
assert ema is not None, "ema_state_dict MISSING"
n = len(ema)
n_float = sum(1 for v in ema.values() if torch.is_floating_point(v))
print(f"EMA_TENSORS={n} (float={n_float})")
sd = ckpt["state_dict"]
gn = [k for k in sd if "front_img_1" in k and ("weight" in k or "bias" in k)]
bn_running = [k for k in sd if "front_img_1" in k and "running_" in k]
print(f"ENCODER_PARAM_KEYS={len(gn)} BN_RUNNING_STAT_KEYS={len(bn_running)}")
assert len(bn_running) == 0, f"BatchNorm running stats still present: {bn_running[:5]}"
# spot check EMA differs from live (after 4 steps it should differ slightly)
k0 = next(k for k in ema if ema[k].numel() > 10)
diff = (ema[k0] - sd[k0].float()).abs().max().item()
print(f"SAMPLE_KEY={k0} max|ema-live|={diff:.3e}")
print("EMA_VERIFY_OK")
