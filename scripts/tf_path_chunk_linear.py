"""Teacher-forced GT-vs-predicted PATH overlay across the transformer
chunk-linear BC-RNN arms (no-chunk / chunk-4/8/16/32) on new_circle_3.

ONE fixed val episode (identical GT path for all arms). Per arm we render the
GT cursor path vs the model's OPEN-LOOP COMMITTED predicted cursor path:
  - no-chunk: per-step prediction (action AT each frame).
  - chunked (chunk_len==obs_stride==N): the policy OBSERVES only every N frames;
    obs-step k (frame N*k) emits a full N-action chunk committed to frames
    [N*k, N*k+N). The chunk structure is therefore visible.

We obtain BOTH via the model's CANONICAL rollout inference path, teacher-forced
on the GT observations: `algo.inference_step(obs_raw, t, emb_id)` -> `policy.step`.
For chunked arms `policy.step` itself does the queue-based open-loop commit
(observe+chunk on env steps t%N==0, pop the queued action otherwise). This is
the SAME path the working closed-loop sim uses (NOT forward_eval, whose batched
non-overlapping-window encode diverges badly from the rollout: ~244px vs ~4px on
the no-chunk arm; verified).

Builds each arm from its own run's .hydra/config.yaml + best ckpt + own
norm_stats, reusing trainHydra's real model/norm machinery (like tf_overlay_big.py).
Outputs 5 per-arm PNGs + one hstacked PNG.
"""
import os, sys, copy, glob
from collections import OrderedDict
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf, open_dict
import simplejpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/coc/flash7/paphiwetsa3/projects/EgoVerse2"
sys.path.insert(0, REPO)
from egomimic.trainHydra import _build_model_config_tree
from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset
from egomimic.rldb.embodiment.embodiment import get_embodiment_id

DATA = "/coc/flash7/paphiwetsa3/datasets/new_circle_3"
LOGS = f"{REPO}/logs"
OUTDIR = os.environ.get("OUTDIR", f"{REPO}/scripts/tf_path_out")
os.makedirs(OUTDIR, exist_ok=True)
EP_IDX = int(os.environ.get("EP_IDX", "22"))
SEED = int(os.environ.get("SEED", "0"))

# arm: (label, run_dir, best_ckpt, best_score, best_epoch)
ARMS = [
    ("no chunk (per-step)", "bcrnnTxCos_nc3/bc_rnn_tx_cos_2026-06-05_17-00-36",
     "epoch_epoch=1299.ckpt", 0.419, 1299),
    ("chunk-4", "bcrnnTxC4_nc3/bc_rnn_tx_chunk4_2026-06-07_17-04-22",
     "epoch_epoch=499.ckpt", 0.559, 499),
    ("chunk-8", "bcrnnTxC8_nc3/bc_rnn_tx_chunk8_2026-06-05_23-40-22",
     "epoch_epoch=899.ckpt", 0.566, 899),
    ("chunk-16", "bcrnnTxC16_nc3/bc_rnn_tx_chunk16_2026-06-06_15-51-17",
     "epoch_epoch=1099.ckpt", 0.551, 1099),
    ("chunk-32", "bcrnnTxC32_nc3/bc_rnn_tx_chunk32_2026-06-07_17-05-06",
     "epoch_epoch=399.ckpt", 0.448, 399),
]

ep_files = sorted(glob.glob(os.path.join(DATA, "*.zarr")))
EP_PATH = ep_files[EP_IDX]
print(f"[episode] idx={EP_IDX} -> {os.path.basename(EP_PATH)}")

import zarr
_g = zarr.open(EP_PATH, mode="r")
_ac_full = np.asarray(_g["actions"][:], dtype=np.float32)
_state_full = np.asarray(_g["observations.state"][:], dtype=np.float32)
_goal_full = np.asarray(_g["goal_pose"][:], dtype=np.float32)
_raw_jpg = _g["observations.images.front_img_1"][:]
T0 = _ac_full.shape[0]
zmask = np.all(_ac_full == 0, axis=1)
T = T0
while T > 0 and zmask[T - 1]:
    T -= 1
print(f"[episode] T0={T0} -> trimmed T={T} (dropped {T0-T} trailing idle frames)")
GT = _ac_full[:T]                       # (T,2) world cursor path (shared)
STATE = _state_full[:T]                 # (T,5) [ax,ay,ox,oy,otheta]
GOAL = _goal_full[:T]                   # (T,3)
FRAMES = np.stack(
    [simplejpeg.decode_jpeg(_raw_jpg[t], colorspace="RGB") for t in range(T)], 0)
IMG_CHW = np.transpose(FRAMES.astype(np.float32) / 255.0, (0, 3, 1, 2))


def build_arm(run_rel, ckpt_name):
    run = f"{LOGS}/{run_rel}"
    ckpt_path = f"{run}/checkpoints/{ckpt_name}"
    norm = f"{run}/norm_stats/norm_stats.json"
    cfg = OmegaConf.load(f"{run}/.hydra/config.yaml")
    with open_dict(cfg):
        for split in ("train_datasets", "valid_datasets"):
            cfg.data[split].pushshapes_sim.resolver.folder_path = DATA
        cfg.norm_stats.precomputed_norm_path = norm
        cfg.mode = "eval"
    train_datasets = {n: hydra.utils.instantiate(cfg.data.train_datasets[n])
                      for n in cfg.data.train_datasets}
    norm_stats = MultiDataset(
        state={},
        norm_mode=OmegaConf.select(cfg, "norm_stats.norm_mode", default="minmax"))
    norm_stats.populate_from_datasets(train_datasets)
    for dataset_name, dataset in train_datasets.items():
        norm_stats.infer_shapes_from_batch(dataset[0])
        ic = copy.deepcopy(cfg.data.train_datasets[dataset_name])
        km = OmegaConf.to_container(ic.resolver.key_map, resolve=False)
        km["norm_mode"] = True
        ic.resolver.key_map = km
        nd = hydra.utils.instantiate(ic)
        norm_stats.infer_norm_from_dataset(
            nd, dataset_name, sample_frac=1.0, num_workers=2,
            precomputed_norm_path=norm)
    mw = ModelWrapper(
        config_tree=_build_model_config_tree(cfg),
        norm_stats_state=norm_stats.to_state(),
        scheduler_interval=cfg.model.get("scheduler_interval", "step"))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = mw.load_state_dict(ckpt["state_dict"], strict=False)
    print(f"  [ckpt] {ckpt_name} missing={len(missing)} unexpected={len(unexpected)}")
    algo = mw.model
    algo.nets.eval()
    return algo


@torch.no_grad()
def rollout_committed_path(algo, emb_id):
    """Open-loop committed (T,2) world path via the canonical rollout inference
    (`algo.inference_step` -> `policy.step`), teacher-forced on GT obs. For
    chunked arms this commits each obs-step's full N-chunk over [Nk, Nk+N)."""
    dev = algo.device
    preds = []
    for t in range(T):
        obs_raw = {
            "state_agent_obj": torch.from_numpy(STATE[t]).unsqueeze(0).to(dev),
            "front_img_1": torch.from_numpy(IMG_CHW[t]).unsqueeze(0).to(dev),
        }
        a = algo.inference_step(obs_raw, t, emb_id)   # world np (2,)
        preds.append(np.asarray(a, dtype=np.float32).reshape(-1)[:2])
    return np.stack(preds, 0)


def mse_rmse(pred):
    mse = float(np.mean((pred - GT) ** 2))
    return mse, float(np.sqrt(mse))


# ---- run all arms ----
panel_data = []
for label, run_rel, ckpt_name, score, ep in ARMS:
    print(f"\n=== {label} :: {run_rel} :: {ckpt_name} (best={score}) ===")
    torch.manual_seed(SEED)            # fix the eval random-crop for reproducibility
    np.random.seed(SEED)
    algo = build_arm(run_rel, ckpt_name)
    emb_id = get_embodiment_id("pushshapes_sim")
    pred = rollout_committed_path(algo, emb_id)
    mse, rmse = mse_rmse(pred)
    cl = algo.nets["policy"].chunk_len
    print(f"  chunk_len={cl}  open-loop committed: MSE={mse:.1f} RMSE={rmse:.2f}px")
    panel_data.append((label, ep, mse, rmse, pred.copy()))
    del algo
    torch.cuda.empty_cache()

# ---- render ----
LIM = 512
goal_xy = GOAL[0, :2]
obj_start = STATE[0, 2:4]


def draw(ax, label, ep, rmse, pred):
    ax.scatter([goal_xy[0]], [goal_xy[1]], c="gold", marker="*", s=240,
               edgecolors="k", linewidths=0.6, zorder=2, label="goal")
    ax.scatter([obj_start[0]], [obj_start[1]], c="purple", marker="s", s=60,
               alpha=0.35, zorder=2, label="obj start")
    ax.plot(GT[:, 0], GT[:, 1], "-", color="green", lw=2.2, label="GT", zorder=3)
    ax.scatter([GT[0, 0]], [GT[0, 1]], facecolors="none", edgecolors="green",
               s=95, lw=2.2, zorder=4)
    ax.scatter([GT[-1, 0]], [GT[-1, 1]], c="green", marker="x", s=95, lw=2.2,
               zorder=4)
    ax.plot(pred[:, 0], pred[:, 1], "--", color="red", lw=1.7, label="pred",
            zorder=3)
    ax.scatter([pred[0, 0]], [pred[0, 1]], facecolors="none", edgecolors="red",
               s=95, lw=1.6, zorder=4)
    ax.scatter([pred[-1, 0]], [pred[-1, 1]], c="red", marker="x", s=95, lw=1.6,
               zorder=4)
    if label.startswith("no chunk"):
        ax.set_title(f"{label}\nbest ep{ep} | TF RMSE={rmse:.1f}px", fontsize=11)
    else:
        ax.set_title(f"{label} | best ep{ep}\nTF RMSE={rmse:.1f}px", fontsize=11)
    ax.set_xlim(0, LIM)
    ax.set_ylim(LIM, 0)
    ax.set_aspect("equal")
    ax.set_xticks([0, 256, 512])
    ax.set_yticks([0, 256, 512])
    ax.grid(alpha=0.15)


fig, axes = plt.subplots(1, len(panel_data), figsize=(4.2 * len(panel_data), 4.8))
if len(panel_data) == 1:
    axes = [axes]
for ax, (label, ep, mse, rmse, pred) in zip(axes, panel_data):
    draw(ax, label, ep, rmse, pred)
axes[0].legend(loc="upper right", fontsize=7, framealpha=0.85)
fig.suptitle(
    f"Teacher-forced GT vs predicted cursor path (open-loop committed) | "
    f"{os.path.basename(EP_PATH)} (T={T}) | transformer chunk-linear arms",
    fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
combined = os.path.join(OUTDIR, "tf_path_chunk_linear.png")
fig.savefig(combined, dpi=130, bbox_inches="tight")
print(f"\n[saved combined] {combined}")

for (label, ep, mse, rmse, pred) in panel_data:
    f2, a2 = plt.subplots(1, 1, figsize=(4.8, 5.0))
    draw(a2, label, ep, rmse, pred)
    a2.legend(loc="upper right", fontsize=8, framealpha=0.85)
    safe = (label.replace(" ", "_").replace("(", "").replace(")", "")
            .replace("-", ""))
    p = os.path.join(OUTDIR, f"tf_path_{safe}.png")
    f2.tight_layout()
    f2.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(f2)
    print(f"[saved panel] {p}")

print(f"\n=== SUMMARY (episode {os.path.basename(EP_PATH)} T={T}) ===")
print(f"{'arm':<22} {'bestEp':>6} {'MSE':>10} {'RMSE(px)':>9}")
for (label, ep, mse, rmse, pred) in panel_data:
    print(f"{label:<22} {ep:>6} {mse:>10.1f} {rmse:>9.2f}")
