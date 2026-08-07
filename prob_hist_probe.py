"""Raw boundary-prob percentiles: load ckpt, one eval-mode forward on the val
batch, dump percentiles per chunker level. Independent of the training logger."""
import sys, glob, torch, numpy as np
from omegaconf import OmegaConf
from hydra.utils import instantiate
sys.path.insert(0, ".")
from egomimic.eval.core.ckpt_loading import load_algo_from_ckpt

for arm in sys.argv[1:]:
    run = sorted(glob.glob(f"logs/indomain_c4/{arm}_2026-*"))[-1]
    cks = sorted(glob.glob(run + "/checkpoints/*.ckpt"))
    if not cks:
        print(f"== {arm}: NO CKPT YET (first save at ep499)"); continue
    ck = [c for c in cks if "last" in c][-1] if any("last" in c for c in cks) else cks[-1]
    algo, _ = load_algo_from_ckpt(ck, run + "/.hydra/config.yaml")
    ep = torch.load(ck, map_location="cpu", weights_only=False).get("epoch", "?")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    algo.nets = algo.nets.to(dev); algo.device = dev; algo.nets.eval()
    full = OmegaConf.load(run + "/.hydra/config.yaml")
    dm = instantiate(full.data); dm.setup(stage="validate")
    first = next(iter(dm.val_dataloader()))
    batch = algo.process_batch_for_training(first[0] if isinstance(first, tuple) else first)
    print(f"== {arm} ckpt_epoch={ep}")
    with torch.no_grad():
        for emb_id, _b in batch.items():
            if not _b.get("_packed", False):
                continue
            seed = algo._seed(emb_id, _b)
            seed = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in seed.items()}
            out = algo.policy(seed)
            for k in sorted(out):
                if k.startswith("chunk/L") and k.endswith("boundary_prob"):
                    p = out[k][..., -1].float().cpu().numpy()
                    q = np.percentile(p, [1, 5, 25, 50, 75, 95, 99])
                    inband = float(np.mean(np.abs(p - 0.5) < 0.05))
                    print(f"  emb{emb_id} {k.split('/')[1]}: n={p.size} "
                          f"pct[1,5,25,50,75,95,99]={np.round(q,4).tolist()} frac|p-.5|<.05={inband:.3f}")
