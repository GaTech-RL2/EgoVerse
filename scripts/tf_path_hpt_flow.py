"""Teacher-forced GT-vs-predicted PATH overlay for HPT-flow (FMPolicy) CAUSAL runs.

For each eval episode, at chunk starts t = 0, K, 2K, ... (K=action_horizon=32):
  - feed the GT observation at frame t (teacher-forced; obs from the recorded
    episode, NOT model rollout),
  - run ONE FMPolicy chunk inference -> 32 predicted action positions,
  - overlay predicted polyline vs GT polyline actions[t:t+K] on the front image.

We replicate the model call from eval_sim._rollout_chunk_openloop exactly:
  obs_raw {state_agent_obj, front_img_1} -> algo.norm_stats.normalize(obs_raw, emb)
  -> _robomimic_to_hpt_data -> policy.forward(emb_name, data) -> (B,K,A) chunk
  -> norm_stats.unnormalize -> world/pixel coords.

Model + data are built from the run's OWN .hydra/config.yaml (causal keymap)
+ ckpt, reusing trainHydra's machinery (like tf_path_chunk_linear.py).

Env (per-run via CLI):
  RUN_DIR, CKPT, DATA, OUTDIR, EPISODES (comma idx), LABEL
"""
import os, sys, copy, glob, json
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf, open_dict
import simplejpeg
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

REPO = "/coc/flash7/paphiwetsa3/projects/EgoVerse2"
sys.path.insert(0, REPO)
from egomimic.trainHydra import _build_model_config_tree
from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset
from egomimic.rldb.embodiment.embodiment import get_embodiment_id, get_embodiment

RUN_DIR = os.environ["RUN_DIR"]
CKPT = os.environ["CKPT"]
DATA = os.environ["DATA"]
OUTDIR = os.environ["OUTDIR"]
LABEL = os.environ.get("LABEL", "hpt_flow")
EPISODES = [int(x) for x in os.environ.get("EPISODES", "0,1,2,3,4").split(",")]
EMB_NAME = "pushshapes_sim"
WORLD = int(os.environ.get("WORLD", "512"))  # action/state coord frame (px)
# OUTPUT_MODE: "video" (default) -> one mp4 per episode, each chunk = one frame.
#              "grid"            -> original grid-of-panels PNG per episode.
#              "anim"            -> one mp4 frame per ACTUAL dataset frame t=0,1,2,...
#                                   (pusher animates frame-by-frame from the GT episode);
#                                   the red(pred)/green(GT) chunk overlay refreshes every
#                                   K frames at chunk boundaries (prediction cached once
#                                   per chunk). Use ~10-15 fps for smooth motion.
OUTPUT_MODE = os.environ.get("OUTPUT_MODE", "video")
# default fps: 2 for "video"/"grid" (chunk-stepped), 12 for "anim" (frame-stepped).
FPS = float(os.environ.get(
    "FPS", "12" if OUTPUT_MODE == "anim" else "2"))
os.makedirs(OUTDIR, exist_ok=True)

ckpt_path = os.path.join(RUN_DIR, "checkpoints", CKPT)
norm_path = os.path.join(RUN_DIR, "norm_stats", "norm_stats.json")
cfg = OmegaConf.load(os.path.join(RUN_DIR, ".hydra", "config.yaml"))
with open_dict(cfg):
    for split in ("train_datasets", "valid_datasets"):
        cfg.data[split][EMB_NAME].resolver.folder_path = DATA
    cfg.norm_stats.precomputed_norm_path = norm_path
    cfg.mode = "eval"

print(f"[build] run={RUN_DIR}\n        ckpt={ckpt_path}\n        data={DATA}", flush=True)

# ---- build norm_stats exactly like tf_path_chunk_linear.build_arm ----
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
        precomputed_norm_path=norm_path)

mw = ModelWrapper(
    config_tree=_build_model_config_tree(cfg),
    norm_stats_state=norm_stats.to_state(),
    scheduler_interval=cfg.model.get("scheduler_interval", "step"))
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
missing, unexpected = mw.load_state_dict(ckpt["state_dict"], strict=False)
print(f"[ckpt] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
algo = mw.model
if torch.cuda.is_available():
    algo.nets.cuda()
algo.nets.eval()
dev = algo.device
emb_id = get_embodiment_id(EMB_NAME)
emb_name = get_embodiment(emb_id).lower()
K = int(os.environ.get("K") or algo.nets["policy"].action_horizon)
ac_key = algo.ac_keys[emb_name]
cam_keys = algo.camera_keys[emb_id]
proprio_keys = algo.proprio_keys[emb_id]
print(f"[cfg] K={K} ac_key={ac_key} cam_keys={cam_keys} proprio_keys={proprio_keys}", flush=True)


@torch.no_grad()
def predict_chunk(state5, img_chw):
    """Teacher-forced single-obs -> (K,2) predicted world/pixel chunk.

    Mirrors eval_sim._rollout_chunk_openloop's per-chunk plan exactly:
    normalize raw obs -> one policy.forward -> unnormalize each action.
    """
    obs_raw = {
        "state_agent_obj": torch.from_numpy(state5).unsqueeze(0).to(dev),
        "front_img_1": torch.from_numpy(img_chw).unsqueeze(0).to(dev),
    }
    obs_norm = algo.norm_stats.normalize(obs_raw, emb_id)
    action_dim = (
        algo.nets["policy"].heads[emb_name].infer_ac_dims[emb_name]
        if hasattr(algo.nets["policy"], "heads")
        and emb_name in algo.nets["policy"].heads
        and hasattr(algo.nets["policy"].heads[emb_name], "infer_ac_dims")
        else 2)
    robo_batch = dict(obs_norm)
    robo_batch[ac_key] = torch.zeros(1, K, action_dim, device=dev)
    robo_batch["pad_mask"] = torch.ones(1, K, 1, device=dev)
    robo_batch["embodiment"] = torch.tensor([emb_id], device=dev, dtype=torch.int64)
    data = algo._robomimic_to_hpt_data(robo_batch, cam_keys, proprio_keys, [], ac_key, [])
    actions = algo.nets["policy"].forward(emb_name, data)
    chunk = actions.get(emb_name)[:, :K, :action_dim]  # (1,K,A) normalized
    # unnormalize each step (same as eval_sim, which unnorms per-step)
    world = algo.norm_stats.unnormalize({ac_key: chunk.squeeze(0)}, emb_id)[ac_key]
    return world.detach().cpu().numpy()[:, :2]  # (K,2)


def load_episode(ep_path):
    g = zarr.open(ep_path, mode="r")
    ac = np.asarray(g["actions"][:], dtype=np.float32)
    state = np.asarray(g["observations.state"][:], dtype=np.float32)
    raw_jpg = g["observations.images.front_img_1"][:]
    T0 = ac.shape[0]
    zmask = np.all(ac == 0, axis=1)
    T = T0
    while T > 0 and zmask[T - 1]:
        T -= 1
    GT = ac[:T]
    STATE = state[:T]
    def _dec(e):
        b = e.item() if hasattr(e, "item") and e.dtype == object else e
        return simplejpeg.decode_jpeg(b, colorspace="RGB")
    frames = np.stack([_dec(raw_jpg[t]) for t in range(T)], 0)
    img_chw = np.transpose(frames.astype(np.float32) / 255.0, (0, 3, 1, 2))
    return GT, STATE, frames, img_chw, T0, T


ep_files = sorted(glob.glob(os.path.join(DATA, "*.zarr")))
print(f"[data] {len(ep_files)} episodes total", flush=True)

all_errs = []
SANITY = os.environ.get("SANITY", "0") == "1"

for ep_idx in EPISODES:
    ep_path = ep_files[ep_idx]
    GT, STATE, frames, img_chw, T0, T = load_episode(ep_path)
    print(f"\n[ep {ep_idx}] {os.path.basename(ep_path)} T0={T0} -> T={T}", flush=True)
    H, W = frames.shape[1], frames.shape[2]
    chunk_starts = list(range(0, T, K))
    pred_chunks, gt_chunks, ferrs = [], [], []
    for cs in chunk_starts:
        ce = min(cs + K, T)
        n = ce - cs
        pred = predict_chunk(STATE[cs], img_chw[cs])  # (K,2)
        pred_n = pred[:n]
        gt_n = GT[cs:ce]
        err = float(np.mean(np.linalg.norm(pred_n - gt_n, axis=1)))
        ferrs.append(err)
        pred_chunks.append(pred_n)
        gt_chunks.append(gt_n)
        if SANITY and cs == 0:
            print(f"  [SANITY t=0] img {H}x{W}", flush=True)
            print(f"    GT[0:5]   = {np.round(gt_n[:5], 1).tolist()}", flush=True)
            print(f"    PRED[0:5] = {np.round(pred_n[:5], 1).tolist()}", flush=True)
            print(f"    GT range  x[{GT[:,0].min():.1f},{GT[:,0].max():.1f}] "
                  f"y[{GT[:,1].min():.1f},{GT[:,1].max():.1f}]", flush=True)
            print(f"    PRED rng  x[{pred[:,0].min():.1f},{pred[:,0].max():.1f}] "
                  f"y[{pred[:,1].min():.1f},{pred[:,1].max():.1f}]", flush=True)
            print(f"    chunk0 mean-L2 err = {err:.2f}px", flush=True)
    ep_err = float(np.mean(ferrs))
    all_errs.append(ep_err)
    print(f"  [ep {ep_idx}] {len(chunk_starts)} chunks  mean per-chunk L2 = {ep_err:.2f}px"
          f"  (per-chunk: {[round(e,1) for e in ferrs]})", flush=True)

    if SANITY:
        # render just t=0 chunk for the sanity check
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(frames[0], extent=[0, WORLD, WORLD, 0])
        g0, p0 = gt_chunks[0], pred_chunks[0]
        ax.plot(g0[:, 0], g0[:, 1], "-o", color="lime", lw=2, ms=3, label="GT")
        ax.plot(p0[:, 0], p0[:, 1], "--o", color="red", lw=1.6, ms=3, label="pred")
        ax.scatter([STATE[0, 0]], [STATE[0, 1]], c="cyan", marker="s", s=60,
                   label="agent start")
        ax.set_title(f"{LABEL} ep{ep_idx} t=0 SANITY  err={ferrs[0]:.1f}px")
        ax.legend(fontsize=8)
        ax.set_xlim(0, WORLD); ax.set_ylim(WORLD, 0)
        p = os.path.join(OUTDIR, f"SANITY_{LABEL}_ep{ep_idx}.png")
        fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
        print(f"  [saved sanity] {p}", flush=True)
        continue

    if OUTPUT_MODE == "grid":
        # ---- render per-chunk overlay grid on the front image (legacy PNG) ----
        ncol = 4
        nrow = int(np.ceil(len(chunk_starts) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow))
        axes = np.atleast_1d(axes).ravel()
        for i, cs in enumerate(chunk_starts):
            ax = axes[i]
            ax.imshow(frames[cs], extent=[0, WORLD, WORLD, 0])
            g0, p0 = gt_chunks[i], pred_chunks[i]
            ax.plot(g0[:, 0], g0[:, 1], "-", color="lime", lw=2.2, label="GT")
            ax.scatter([g0[0, 0]], [g0[0, 1]], facecolors="none", edgecolors="lime",
                       s=80, lw=2)
            ax.scatter([g0[-1, 0]], [g0[-1, 1]], c="lime", marker="x", s=70, lw=2)
            ax.plot(p0[:, 0], p0[:, 1], "--", color="red", lw=1.8, label="pred")
            ax.scatter([p0[0, 0]], [p0[0, 1]], facecolors="none", edgecolors="red",
                       s=80, lw=1.6)
            ax.scatter([p0[-1, 0]], [p0[-1, 1]], c="red", marker="x", s=70, lw=1.6)
            ax.set_title(f"chunk @t={cs}  L2={ferrs[i]:.1f}px", fontsize=10)
            ax.set_xlim(0, WORLD); ax.set_ylim(WORLD, 0)
            ax.set_xticks([]); ax.set_yticks([])
        for j in range(len(chunk_starts), len(axes)):
            axes[j].axis("off")
        axes[0].legend(loc="upper right", fontsize=9, framealpha=0.85)
        fig.suptitle(f"{LABEL} | ep{ep_idx} ({os.path.basename(ep_path)}) | "
                     f"TF chunk path GT(green) vs pred(red) | mean L2={ep_err:.1f}px",
                     fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        p = os.path.join(OUTDIR, f"tfpath_{LABEL}_ep{ep_idx}.png")
        fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
        print(f"  [saved] {p}", flush=True)
        continue

    if OUTPUT_MODE == "anim":
        # ---- render per-episode ANIM: one mp4 frame per ACTUAL dataset frame ----
        # The pusher physically moves every frame (we show the REAL recorded
        # front_img_1[t], teacher-forced from the GT episode). The model's
        # predicted 32-step path (red dashed) + GT path (green) for the CURRENT
        # chunk c = (t // K) * K are HELD/displayed for all frames in [c, c+K).
        # Each chunk's prediction was computed ONCE above (pred_chunks[i]); we do
        # NOT re-run inference per frame. A dot marks the current pusher position
        # STATE[t][:2] so you can see it travel along the path.
        vid_path = os.path.join(OUTDIR, f"tfpath_anim_{LABEL}_ep{ep_idx}.mp4")
        writer = imageio.get_writer(
            vid_path, fps=FPS, codec="libx264", quality=8,
            macro_block_size=16, ffmpeg_params=["-pix_fmt", "yuv420p"])
        for t in range(T):
            i = t // K            # current chunk index
            cs = chunk_starts[i]  # current chunk start frame
            fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100)
            ax.imshow(frames[t], extent=[0, WORLD, WORLD, 0])
            g0, p0 = gt_chunks[i], pred_chunks[i]
            ax.plot(g0[:, 0], g0[:, 1], "-", color="lime", lw=2.2, label="GT")
            ax.scatter([g0[0, 0]], [g0[0, 1]], facecolors="none", edgecolors="lime",
                       s=80, lw=2)
            ax.scatter([g0[-1, 0]], [g0[-1, 1]], c="lime", marker="x", s=70, lw=2)
            ax.plot(p0[:, 0], p0[:, 1], "--", color="red", lw=1.8, label="pred")
            ax.scatter([p0[0, 0]], [p0[0, 1]], facecolors="none", edgecolors="red",
                       s=80, lw=1.6)
            ax.scatter([p0[-1, 0]], [p0[-1, 1]], c="red", marker="x", s=70, lw=1.6)
            ax.scatter([STATE[t, 0]], [STATE[t, 1]], c="cyan", marker="o",
                       s=70, lw=1.5, edgecolors="black", zorder=5, label="pusher")
            ax.set_title(f"{LABEL} ep{ep_idx} | frame {t} | chunk c={cs}  "
                         f"L2={ferrs[i]:.1f}px  (ep mean {ep_err:.1f}px)",
                         fontsize=11)
            ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
            ax.set_xlim(0, WORLD); ax.set_ylim(WORLD, 0)
            ax.set_xticks([]); ax.set_yticks([])
            fig.tight_layout()
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            h, w = buf.shape[:2]
            buf = buf[: h - (h % 2), : w - (w % 2)]
            writer.append_data(np.ascontiguousarray(buf))
            plt.close(fig)
        writer.close()
        print(f"  [saved anim] {vid_path}  ({T} frames @ {FPS}fps)", flush=True)
        continue

    # ---- render per-episode VIDEO: one chunk = one mp4 frame (default) ----
    # Same overlay content as one grid panel: front image + GT path (green) +
    # predicted path (red dashed) + per-chunk L2 in the title. Stepped through the
    # chunks at FPS (~2 fps) so it's watchable.
    vid_path = os.path.join(OUTDIR, f"tfpath_{LABEL}_ep{ep_idx}.mp4")
    writer = imageio.get_writer(
        vid_path, fps=FPS, codec="libx264", quality=8,
        macro_block_size=16, ffmpeg_params=["-pix_fmt", "yuv420p"])
    for i, cs in enumerate(chunk_starts):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100)
        ax.imshow(frames[cs], extent=[0, WORLD, WORLD, 0])
        g0, p0 = gt_chunks[i], pred_chunks[i]
        ax.plot(g0[:, 0], g0[:, 1], "-", color="lime", lw=2.2, label="GT")
        ax.scatter([g0[0, 0]], [g0[0, 1]], facecolors="none", edgecolors="lime",
                   s=80, lw=2)
        ax.scatter([g0[-1, 0]], [g0[-1, 1]], c="lime", marker="x", s=70, lw=2)
        ax.plot(p0[:, 0], p0[:, 1], "--", color="red", lw=1.8, label="pred")
        ax.scatter([p0[0, 0]], [p0[0, 1]], facecolors="none", edgecolors="red",
                   s=80, lw=1.6)
        ax.scatter([p0[-1, 0]], [p0[-1, 1]], c="red", marker="x", s=70, lw=1.6)
        ax.set_title(f"{LABEL} ep{ep_idx} | chunk t={cs}  L2={ferrs[i]:.1f}px  "
                     f"(ep mean {ep_err:.1f}px)", fontsize=11)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
        ax.set_xlim(0, WORLD); ax.set_ylim(WORLD, 0)
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        # h264 needs even dims; crop to even.
        h, w = buf.shape[:2]
        buf = buf[: h - (h % 2), : w - (w % 2)]
        writer.append_data(np.ascontiguousarray(buf))
        plt.close(fig)
    writer.close()
    print(f"  [saved video] {vid_path}  ({len(chunk_starts)} frames @ {FPS}fps)",
          flush=True)

print(f"\n=== SUMMARY {LABEL} ===", flush=True)
print(f"per-episode mean L2: {[round(e,2) for e in all_errs]}", flush=True)
print(f"OVERALL mean per-chunk L2 = {np.mean(all_errs):.2f}px", flush=True)
