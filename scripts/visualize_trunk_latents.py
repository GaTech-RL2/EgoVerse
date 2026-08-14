"""Visualize HPT trunk latents in the same dark-tab UI as ``arc_embedding_baseline``.

Loads the saved hydra config + norm_stats cache from a training run dir,
rebuilds the model, loads the checkpoint state_dict, draws ``--num-samples``
random samples split evenly across the valid-set embodiments, runs them
through ``HPTModel.forward_features`` to get per-sample trunk latents,
fits t-SNE 2D, HDBSCAN-clusters the 2D embedding, and hands off to
``arc_embedding_sweep._emit_tabbed_html`` — same click-to-image floating
panel, cluster stats side bar, embodiment/cluster color toggle, etc.

Example:

    python scripts/visualize_trunk_latents.py \\
        --run-dir logs/arc_tests_cotrain_arctok/h100_1gpu_bs128_constlr3e4_val30_2026-07-27_22-28-30/0 \\
        --ckpt logs/arc_tests_cotrain_arctok/h100_1gpu_bs128_constlr3e4_val30_2026-07-27_22-28-30/0/checkpoints/last.ckpt \\
        --num-samples 500 --batch-size 32 --pool mean \\
        --out logs/trunk_latents/trunk_latents.html
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import random
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

import egomimic.utils.hydra_resolvers  # noqa: F401 — registers OmegaConf resolvers
from egomimic.pl_utils.pl_data_utils import annotation_collate
from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset
from egomimic.scripts.arc_embedding_sweep import _emit_tabbed_html, _hdbscan_cluster
from egomimic.scripts.arc_embedding_viz import _extract_state_image_uri
from egomimic.utils.aws.aws_data_utils import load_env

OmegaConf.register_new_resolver("eval", eval, replace=True)
# `${hydra:...}` and `${now:...}` are Hydra runtime resolvers — unavailable
# outside a hydra app. Stubs so `OmegaConf.load` on the saved config.yaml
# succeeds; the values only matter for training/logging plumbing we skip.
OmegaConf.register_new_resolver("hydra", lambda key: "", replace=True)
OmegaConf.register_new_resolver("now", lambda fmt=None: "", replace=True)

logger = logging.getLogger("visualize_trunk_latents")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Config + norm-stats bootstrap
# ---------------------------------------------------------------------------


def _build_model_config_tree(cfg: DictConfig) -> DictConfig:
    """Mirror of trainHydra._build_model_config_tree — strip norm_stats so the
    ModelWrapper can slot in a live norm_stats reference itself."""
    model_cfg = copy.deepcopy(cfg.model)
    if (
        "robomimic_model" in model_cfg
        and isinstance(model_cfg.robomimic_model, DictConfig)
        and "norm_stats" in model_cfg.robomimic_model
    ):
        model_cfg.robomimic_model.norm_stats = None
    return OmegaConf.create({"model": model_cfg})


def _resolve_norm_stats_path(run_dir: str) -> str:
    """The trainer writes norm_stats to `<run>/norm_stats/norm_stats.json`
    (see MultiDataset.cache_stats). Both a dir and a file path work as
    `precomputed_norm_path`; we hand the file directly so the log is clear."""
    candidate = os.path.join(run_dir, "norm_stats", "norm_stats.json")
    if not os.path.isfile(candidate):
        raise FileNotFoundError(
            f"Expected cached norm_stats at {candidate}. Was this a completed "
            "training run?"
        )
    return candidate


def _instantiate_valid_datasets(cfg: DictConfig) -> dict:
    """Instantiate only the valid_datasets from the config."""
    valid_datasets = {}
    for name in cfg.data.valid_datasets:
        valid_datasets[name] = hydra.utils.instantiate(cfg.data.valid_datasets[name])
    return valid_datasets


def _populate_norm_stats(
    cfg: DictConfig, valid_datasets: dict, norm_path: str
) -> MultiDataset:
    """Rebuild the stats-only MultiDataset that the model expects. Falls back
    on the run's cached norm_stats.json for the actual stats — same code path
    as trainHydra.py but with valid datasets standing in as the probe source
    for key-type inference."""
    norm_stats = MultiDataset(
        state={},
        norm_mode=OmegaConf.select(cfg, "norm_stats.norm_mode", default="quantile"),
    )
    norm_stats.populate_from_datasets(valid_datasets)
    for ds_name, ds in valid_datasets.items():
        norm_stats.infer_shapes_from_batch(ds[0])
        instantiate_copy = copy.deepcopy(cfg.data.valid_datasets[ds_name])
        keymap_cfg = instantiate_copy.resolver.key_map
        km = OmegaConf.to_container(keymap_cfg, resolve=False)
        km["norm_mode"] = True
        instantiate_copy.resolver.key_map = km
        norm_dataset = hydra.utils.instantiate(instantiate_copy)
        norm_stats.infer_norm_from_dataset(
            norm_dataset,
            ds_name,
            precomputed_norm_path=norm_path,
        )
    for ds in valid_datasets.values():
        ds.set_norm_stats_from(norm_stats)
    return norm_stats


# ---------------------------------------------------------------------------
# Feature pooling + model forward
# ---------------------------------------------------------------------------


def _pool(features: torch.Tensor, mode: str) -> torch.Tensor:
    """Reduce (B, S, D) → (B, F). mode: mean | first | flatten."""
    if features.dim() != 3:
        raise ValueError(f"expected (B, S, D), got {tuple(features.shape)}")
    if mode == "mean":
        return features.mean(dim=1)
    if mode == "first":
        return features[:, 0]
    if mode == "flatten":
        return features.reshape(features.shape[0], -1)
    raise ValueError(f"unknown pool mode {mode!r}")


def _extract_features_single_emb(
    algo, embodiment_name: str, batch: dict, pool: str
) -> torch.Tensor:
    """Run one embodiment's batch through the trunk. Returns (B, F).

    Same normalize/tokenize path as ``HPT.forward_training`` so the inputs
    to the trunk match what the model saw during training."""
    processed = algo.process_batch_for_training({embodiment_name: batch})
    ((embodiment_id, _batch),) = processed.items()
    cam_keys = algo.camera_keys[embodiment_id]
    proprio_keys = algo.proprio_keys[embodiment_id]
    lang_keys = algo.lang_keys[embodiment_id]
    ac_key = algo.ac_keys[embodiment_id]
    aux_ac_keys = algo.auxiliary_ac_keys.get(embodiment_name, [])
    data = algo._robomimic_to_hpt_data(
        _batch, cam_keys, proprio_keys, lang_keys, ac_key, aux_ac_keys
    )
    data_clone = algo._clone_batch(data)
    proc_tokens, _block_outputs = algo.nets["policy"].forward_features(
        embodiment_name, data_clone
    )
    return _pool(proc_tokens, pool).detach().to(torch.float32).cpu()


# ---------------------------------------------------------------------------
# Sample budget + random selection
# ---------------------------------------------------------------------------


def _split_samples_across_embs(
    num_samples: int, emb_names: list[str]
) -> dict[str, int]:
    """Distribute ``num_samples`` roughly evenly across embodiments. Remainders
    go to the first embodiments in the list so the totals sum to num_samples."""
    n = len(emb_names)
    base, extra = divmod(num_samples, n)
    return {name: base + (1 if i < extra else 0) for i, name in enumerate(emb_names)}


def _random_indices(ds_len: int, k: int, seed: int) -> list[int]:
    """Sample ``k`` indices from ``range(ds_len)`` without replacement when
    possible, with replacement if k > ds_len."""
    rng = random.Random(seed)
    if k <= ds_len:
        return rng.sample(range(ds_len), k)
    return [rng.randrange(ds_len) for _ in range(k)]


# ---------------------------------------------------------------------------
# t-SNE
# ---------------------------------------------------------------------------


def _tsne_2d(features: np.ndarray, seed: int) -> np.ndarray:
    from sklearn.manifold import TSNE

    perplexity = min(30, max(5, features.shape[0] // 4))
    return TSNE(
        n_components=2, init="pca", random_state=seed, perplexity=perplexity
    ).fit_transform(features)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-dir",
        required=True,
        help="Training run dir containing .hydra/config.yaml and norm_stats/norm_stats.json.",
    )
    ap.add_argument(
        "--ckpt",
        required=True,
        help="Path to the lightning .ckpt file to visualize.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output HTML path (default: <run-dir>/trunk_latents_tsne2d.html).",
    )
    ap.add_argument(
        "--pool",
        default="mean",
        choices=["mean", "first", "flatten"],
        help="How to reduce (B, action_horizon, D) trunk output into a per-sample vector.",
    )
    ap.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Total number of random samples to draw, split evenly across embodiments.",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Samples per model forward call.",
    )
    ap.add_argument(
        "--image-max-side",
        type=int,
        default=224,
        help="Max side of the JPEG shipped to the browser (per-sample thumb).",
    )
    ap.add_argument("--jpeg-quality", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for the forward pass.",
    )
    ap.add_argument(
        "--title",
        default="HPT trunk latents (t-SNE 2D)",
        help="Title shown at the top of the HTML.",
    )
    args = ap.parse_args()

    load_env()
    run_dir = os.path.abspath(args.run_dir)
    cfg_path = os.path.join(run_dir, ".hydra", "config.yaml")
    if not os.path.isfile(cfg_path):
        sys.exit(f"No config.yaml at {cfg_path}")
    cfg = OmegaConf.load(cfg_path)

    norm_path = _resolve_norm_stats_path(run_dir)
    logger.info("Using cached norm_stats: %s", norm_path)

    valid_datasets = _instantiate_valid_datasets(cfg)
    if not valid_datasets:
        sys.exit("No valid datasets found in the run's data config.")
    logger.info(
        "Instantiated valid datasets: %s",
        {k: len(v) for k, v in valid_datasets.items()},
    )

    norm_stats = _populate_norm_stats(cfg, valid_datasets, norm_path)

    device = torch.device(args.device)
    logger.info("Instantiating model on %s", device)
    model = ModelWrapper(
        config_tree=_build_model_config_tree(cfg),
        norm_stats_state=norm_stats.to_state(),
    )
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
    if missing:
        logger.warning(
            "Missing keys when loading ckpt: %d (first: %s)", len(missing), missing[:3]
        )
    if unexpected:
        logger.warning(
            "Unexpected keys when loading ckpt: %d (first: %s)",
            len(unexpected),
            unexpected[:3],
        )
    model.to(device)
    model.eval()
    algo = model.model
    algo.device = device
    algo.nets = algo.nets.to(device)

    emb_names = list(valid_datasets.keys())
    per_emb_counts = _split_samples_across_embs(args.num_samples, emb_names)
    logger.info("Sample budget per embodiment: %s", per_emb_counts)

    # Aligned per-sample containers: features[i], images[i], contexts[i],
    # meta[i] all describe the SAME sample. Index i doubles as ``img_idx`` so
    # the tabbed HTML looks up the right image on click.
    all_feats: list[np.ndarray] = []
    all_images: list[str] = []
    all_contexts: list[dict | None] = []
    all_meta: list[dict] = []

    with torch.no_grad():
        for emb_name in emb_names:
            k = per_emb_counts[emb_name]
            ds = valid_datasets[emb_name]
            if k <= 0 or len(ds) == 0:
                logger.warning("Skipping %s (k=%d, len=%d)", emb_name, k, len(ds))
                continue
            indices = _random_indices(len(ds), k, args.seed)
            logger.info(
                "[%s] drawing %d random samples from %d, batch=%d",
                emb_name,
                len(indices),
                len(ds),
                args.batch_size,
            )

            failed_reads = 0
            for chunk_start in range(0, len(indices), args.batch_size):
                chunk_idxs = indices[chunk_start : chunk_start + args.batch_size]
                # Fetch samples one-by-one to survive bad episodes without
                # torpedoing the whole batch. Preserve (idx, sample) pairs so
                # we can align image URIs to the trunk output rows below.
                pairs: list[tuple[int, dict]] = []
                for idx in chunk_idxs:
                    try:
                        pairs.append((idx, ds[idx]))
                    except Exception as e:
                        failed_reads += 1
                        if failed_reads <= 3:
                            logger.warning(
                                "  [%s] ds[%d] failed: %s: %s",
                                emb_name,
                                idx,
                                type(e).__name__,
                                e,
                            )
                if not pairs:
                    continue

                # Extract images BEFORE collate — `annotation_collate` pops
                # list-valued keys from the sample dicts, and while image
                # tensors are safe, keeping the extraction upstream is more
                # robust to future collate-side changes.
                for idx, sample in pairs:
                    uri = _extract_state_image_uri(
                        sample,
                        chunk_14d=None,  # overlay disabled — arc-tok
                        # actions here are (M+1, 8)
                        # normalized, not (T, 14)
                        # camera-frame cartesian.
                        embodiment_name=emb_name,
                        image_key=None,
                        image_max_side=args.image_max_side,
                        jpeg_quality=args.jpeg_quality,
                        overlay_gt_chunk=False,
                    )
                    all_images.append(uri or "")
                    all_contexts.append(None)
                    all_meta.append(
                        {
                            "embodiment": emb_name,
                            "split": "valid",
                            "sample_idx": int(idx),
                            "img_idx": len(all_images) - 1,
                            "is_zero_token": False,
                            "reached_target_D": True,  # not applicable — keep uniform
                            "partial_traj": False,  # so all points land in the
                            # "full" trace of the plot
                            "joint_arc_max_cm": 0.0,
                        }
                    )

                # Now collate + forward for this chunk.
                batch = annotation_collate([s for _, s in pairs])
                vecs = _extract_features_single_emb(algo, emb_name, batch, args.pool)
                all_feats.append(vecs.numpy())

            if failed_reads:
                logger.warning(
                    "[%s] %d samples failed to read (kept %d)",
                    emb_name,
                    failed_reads,
                    len(indices) - failed_reads,
                )

    if not all_feats:
        sys.exit("No features collected — is the val set empty?")

    X = np.concatenate(all_feats, axis=0)
    logger.info(
        "Collected trunk features: X.shape=%s (images=%d)", X.shape, len(all_images)
    )
    if X.shape[0] != len(all_meta):
        sys.exit(
            f"Feature/meta count mismatch: X={X.shape[0]} meta={len(all_meta)} "
            "(bug — features and meta should be filled in lock-step)."
        )

    logger.info("Fitting t-SNE 2D (perplexity auto)...")
    embed = _tsne_2d(X, args.seed)

    logger.info("HDBSCAN clustering the 2D embedding...")
    labels, cluster_stats = _hdbscan_cluster(embed)
    for i, m in enumerate(all_meta):
        m["cluster"] = int(labels[i])

    out_path = args.out or os.path.join(run_dir, "trunk_latents_tsne2d.html")
    Path(os.path.dirname(os.path.abspath(out_path))).mkdir(parents=True, exist_ok=True)

    sweeps = [(f"t-SNE 2D  ({args.pool}-pool)", embed, all_meta, cluster_stats)]
    tab_meta = [{"kind": "trunk_latents", "pool": args.pool}]

    _emit_tabbed_html(
        sweeps=sweeps,
        images_shared=all_images,
        title=args.title,
        output_html=out_path,
        contexts_shared=all_contexts,
        tab_meta=tab_meta,
    )
    logger.info("Wrote %s (N=%d)", out_path, X.shape[0])


if __name__ == "__main__":
    main()
