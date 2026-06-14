"""Teacher-forced overlay for the cotrain BC-RNN H-Net policy on SMALL-circle
episodes. Builds the model + norm stats EXACTLY like trainHydra (reusing its
real classes), loads the cotrain ckpt, runs the canonical teacher-forced
``forward_eval`` (dense per-step GT-vs-pred over each full episode), and renders
overlay mp4s: GT frame + GT cursor dot/trail + PREDICTED next-action dot/trail.

Run under the run's own norm stats (union minmax over both datasets) via
norm_stats.precomputed_norm_path. Uses Hydra compose on the run's saved config.
"""
import os, sys, copy, json, glob
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf, open_dict
import cv2
import simplejpeg

RUN = "/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/bcrnnHnetCo_nc3/bc_rnn_hnet_cotrain_2026-06-06_02-55-53"
CKPT = os.environ.get("CKPT", f"{RUN}/checkpoints/ep1200.ckpt")
NORM = f"{RUN}/norm_stats/norm_stats.json"
SMALL = "/coc/flash7/paphiwetsa3/datasets/new_circle_3"
OUTDIR = os.environ.get("OUTDIR", f"{RUN}/videos/big_tf_overlay_sanity")
N_EP = int(os.environ.get("N_EP", "4"))
EP_IDXS = [int(x) for x in os.environ.get("EP_IDXS", "0,1,2,3").split(",")]
os.makedirs(OUTDIR, exist_ok=True)

sys.path.insert(0, "/coc/flash7/paphiwetsa3/projects/EgoVerse2")
from egomimic.trainHydra import _build_model_config_tree
from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset
from egomimic.rldb.zarr.zarr_dataset_packed import pack_collate
from egomimic.rldb.embodiment.embodiment import get_embodiment_id

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- 1. load the run's composed config, retarget to SMALL eval ----
cfg = OmegaConf.load(f"{RUN}/.hydra/config.yaml")
with open_dict(cfg):
    # point BOTH train (used for norm-stats inference) and valid at small dataset
    for split in ("train_datasets", "valid_datasets"):
        cfg.data[split].pushshapes_sim.resolver.folder_path = SMALL
    cfg.norm_stats.precomputed_norm_path = NORM
    cfg.mode = "eval"
print("[cfg] datasets ->", cfg.data.train_datasets.pushshapes_sim.resolver.folder_path)

# ---- 2. build datasets + stats-only MultiDataset + norm (trainHydra recipe) ----
train_datasets = {n: hydra.utils.instantiate(cfg.data.train_datasets[n]) for n in cfg.data.train_datasets}
valid_datasets = {n: hydra.utils.instantiate(cfg.data.valid_datasets[n]) for n in cfg.data.valid_datasets}

norm_stats = MultiDataset(state={}, norm_mode=OmegaConf.select(cfg, "norm_stats.norm_mode", default="minmax"))
norm_stats.populate_from_datasets(train_datasets)
for dataset_name, dataset in train_datasets.items():
    norm_stats.infer_shapes_from_batch(dataset[0])
    instantiate_copy = copy.deepcopy(cfg.data.train_datasets[dataset_name])
    km = OmegaConf.to_container(instantiate_copy.resolver.key_map, resolve=False)
    km["norm_mode"] = True
    instantiate_copy.resolver.key_map = km
    norm_dataset = hydra.utils.instantiate(instantiate_copy)
    norm_stats.infer_norm_from_dataset(
        norm_dataset, dataset_name,
        sample_frac=1.0, num_workers=2,
        precomputed_norm_path=NORM,
    )
# wire stats into datasets so __getitem__ returns NORMALIZED tensors
for ds in train_datasets.values():
    ds.set_norm_stats_from(norm_stats)
for ds in valid_datasets.values():
    ds.set_norm_stats_from(norm_stats)
print("[norm] loaded from", NORM)

# ---- 3. build model + load ckpt ----
mw = ModelWrapper(
    config_tree=_build_model_config_tree(cfg),
    norm_stats_state=norm_stats.to_state(),
    scheduler_interval=cfg.model.get("scheduler_interval", "step"),
)
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
missing, unexpected = mw.load_state_dict(ckpt["state_dict"], strict=False)
print(f"[ckpt] {CKPT}\n  missing={len(missing)} unexpected={len(unexpected)}")
algo = mw.model
algo.nets.eval()
device = algo.device  # BCRNN pins nets to self.device at construction (cuda)
print("[device]", device)
emb_id = get_embodiment_id("pushshapes_sim")
ac_key = algo.resolved_ac_keys[emb_id]
print("[emb]", emb_id, "ac_key", ac_key)

# ---- 4. pick 4 small-circle episodes by index, run TF forward_eval per episode ----
vds = valid_datasets["pushshapes_sim"]
# Map dataset index -> file. The packed dataset's index entries: (key, start, end).
# We just want N whole episodes; iterate the underlying datasets order.
ep_files = sorted(glob.glob(os.path.join(SMALL, "*.zarr")))
print(f"[data] {len(ep_files)} small episodes; using idxs {EP_IDXS}")

def load_episode_sample(zarr_path):
    """Return a single-sample dict in the packed-dataset __getitem__ schema by
    locating the matching index entry covering the FULL episode. Fallback: build
    obs/actions directly from the zarr if the packed index splits episodes."""
    # find packed-index rows whose key matches this episode file
    rows = [(j, k, s, e) for j, (k, s, e) in enumerate(vds.index)]
    # vds.index keys are dataset keys; match by basename
    base = os.path.basename(zarr_path)
    cand = [(j, k, s, e) for (j, k, s, e) in rows if base in str(k) or str(k) in zarr_path]
    if cand:
        # take the longest span
        j, k, s, e = max(cand, key=lambda r: r[3] - r[2])
        return vds[j]
    return None

import zarr
def overlay_for_episode(ep_path, out_mp4):
    g = zarr.open(ep_path, mode="r")
    n = g["observations.state"].shape[0]
    state = np.asarray(g["observations.state"][:], dtype=np.float32)   # (T,5)
    gt_actions = np.asarray(g["actions"][:], dtype=np.float32)          # (T,2) world (cursor)
    raw_jpg = g["observations.images.front_img_1"][:]
    frames = np.stack([simplejpeg.decode_jpeg(raw_jpg[t], colorspace="RGB") for t in range(n)], 0)  # (T,96,96,3)
    img_chw = np.transpose(frames.astype(np.float32) / 255.0, (0, 3, 1, 2))  # (T,3,96,96)
    # Normalize obs via the run norm_stats (indexes by ZARR key), then map to the
    # model KEYNAMES that _build_obs reads (state_agent_obj / front_img_1). This
    # mirrors process_batch_for_training without depending on the (buggy here)
    # zarr_key_to_keyname reverse lookup.
    z2k = {"observations.state": "state_agent_obj",
           "observations.images.front_img_1": "front_img_1"}
    obs_zarr = {
        "observations.state": torch.from_numpy(state),
        "observations.images.front_img_1": torch.from_numpy(img_chw),
    }
    obs_n = algo.norm_stats.normalize(obs_zarr, emb_id)
    dev = algo.device
    proc = {emb_id: {
        z2k[zk]: obs_n[zk].float().to(dev) for zk in z2k
    }}
    proc[emb_id][ac_key] = algo.norm_stats.normalize(
        {"actions": torch.from_numpy(gt_actions)}, emb_id)["actions"].float().to(dev)
    proc[emb_id]["cu_seqlens"] = torch.tensor([0, n], dtype=torch.long, device=dev)
    proc[emb_id]["seq_lens"] = torch.tensor([n], dtype=torch.long, device=dev)
    proc[emb_id]["max_seq_len"] = int(n)
    proc[emb_id]["batch_size"] = 1
    proc[emb_id]["_packed"] = True
    proc[emb_id]["embodiment"] = torch.tensor([emb_id], dtype=torch.long, device=dev)
    with torch.no_grad():
        if os.environ.get("DBG"):
            print("  [dbg] proc keys:", list(proc[emb_id].keys()),
                  "proprio", algo.proprio_keys.get(emb_id),
                  "camera", algo.camera_keys.get(emb_id))
        unnorm = algo.forward_eval(proc)
    pred_w = unnorm[f"emb{emb_id}_{ac_key}"][0].detach().cpu().numpy()  # (T,2) world
    # ---- metrics ----
    mse = float(np.mean((pred_w - gt_actions) ** 2))
    rmse = float(np.sqrt(mse))
    print(f"  {os.path.basename(ep_path)}  T={n}  TF MSE={mse:.1f}  RMSE={rmse:.2f}px")
    # ---- render overlay ----
    SC = 8  # upscale 96 -> 768 for visible dots
    H = W = 96 * SC
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_mp4, fourcc, 20.0, (W, H))
    gt_trail = []; pr_trail = []
    s = 96.0 / 512.0  # world(512) -> image(96)
    for t in range(n):
        frame = cv2.resize(frames[t], (W, H), interpolation=cv2.INTER_NEAREST)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        gx, gy = gt_actions[t] * s * SC
        px, py = pred_w[t] * s * SC
        gt_trail.append((int(gx), int(gy))); pr_trail.append((int(px), int(py)))
        for trail, col in ((gt_trail, (0, 255, 0)), (pr_trail, (0, 0, 255))):
            for a, b in zip(trail[-30:], trail[-30:][1:]):
                cv2.line(frame, a, b, col, 1)
        cv2.circle(frame, (int(gx), int(gy)), 7, (0, 255, 0), -1)   # GT green
        cv2.circle(frame, (int(px), int(py)), 7, (0, 0, 255), 2)    # pred red ring
        cv2.putText(frame, f"GT(green) PRED(red)  t={t}/{n}  RMSE={rmse:.1f}px",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(frame)
    vw.release()
    return mse, rmse

results = []
for i, idx in enumerate(EP_IDXS[:N_EP]):
    ep = ep_files[idx]
    out = os.path.join(OUTDIR, f"overlay_big_ep{i}.mp4")
    mse, rmse = overlay_for_episode(ep, out)
    results.append((os.path.basename(ep), mse, rmse, out))

print("\n=== TF OVERLAY SUMMARY ===")
for b, mse, rmse, out in results:
    print(f"  {b}  MSE={mse:.1f}  RMSE={rmse:.2f}px  -> {out}")
# mean-baseline reference: predict the action-mean every step
mean_a = np.array([264.748, 263.174])  # from norm_stats actions.mean
print(f"[ref] action mean = {mean_a}")
