"""
Compute and append embedding arrays to Zarr episode stores.

Resolves episode paths from a Hydra dataset config (the same kind consumed
by scale_to_zarr_annotation.py), then runs a ZarrKeyTransform across the
union of train + valid episodes.

Bundled transforms:
  - DINOv3ImageEmbedding (dinov3_embedding.py): per-frame DINOv3 image embeddings.
  - Qwen3TextEmbedding (qwen3_embedding.py): per-frame text embeddings from
    JSON-encoded annotation spans via Qwen3-Embedding-0.6B.

Examples:
    python egomimic/scripts/embedding_process/zarr_embedding.py \\
        --transform dinov3 \\
        --dataset-config-path egomimic/hydra_configs/data/eva_pi_lang.yaml \\
        --input-keys observations.images.front_1 \\
        --output-keys observations.embeddings.dinov3.front_1 \\
        --batch-size 64

    python egomimic/scripts/embedding_process/zarr_embedding.py \\
        --transform qwen3 \\
        --dataset-config-path egomimic/hydra_configs/data/eva_pi_lang.yaml \\
        --input-keys annotations \\
        --output-keys observations.embeddings.qwen3.annotations
"""

from __future__ import annotations

import argparse
import logging

import hydra
import torch

from egomimic.scripts.embedding_process.dinov3_embedding import DINOv3ImageEmbedding
from egomimic.scripts.embedding_process.qwen3_embedding import Qwen3TextEmbedding
from egomimic.scripts.embedding_process.zarr_key_transform import ZarrKeyTransform
from egomimic.utils.hydra_utils import load_config_from_path

logger = logging.getLogger(__name__)


TRANSFORMS: dict[str, type[ZarrKeyTransform]] = {
    "dinov3": DINOv3ImageEmbedding,
    "qwen3": Qwen3TextEmbedding,
}


def _build_transform(args: argparse.Namespace) -> ZarrKeyTransform:
    cls = TRANSFORMS[args.transform]
    kwargs: dict = dict(
        input_keys=args.input_keys,
        batch_size=args.batch_size,
        device=args.device,
        overwrite=args.overwrite,
    )
    if args.output_keys is not None:
        kwargs["output_keys"] = args.output_keys
    if args.model_name is not None:
        kwargs["model_name"] = args.model_name
    return cls(**kwargs)


def _resolve_episode_paths(dataset_config_path: str) -> dict[str, str]:
    """
    Load a Hydra dataset config (resolving its ``defaults:`` chain), instantiate
    every train/valid MultiDataset, and return a deduplicated
    ``{episode_hash: episode_path}`` mapping covering every episode referenced
    by the config.
    """
    dataset_cfg = load_config_from_path(dataset_config_path)

    episodes: dict[str, str] = {}
    for split_key in ("train_datasets", "valid_datasets"):
        split = getattr(dataset_cfg, split_key, None)
        if split is None:
            continue
        for dataset_name in split:
            multi = hydra.utils.instantiate(split[dataset_name])
            for ep_hash, zarr_ds in multi.datasets.items():
                if ep_hash in episodes:
                    continue
                episodes[ep_hash] = str(zarr_ds.episode_path)
    return episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transform", required=True, choices=sorted(TRANSFORMS.keys()))
    parser.add_argument(
        "--dataset-config-path",
        required=True,
        help=(
            "Path to a Hydra dataset config (e.g. egomimic/hydra_configs/data/"
            "eva_pi_lang.yaml). The full ``defaults:`` chain is resolved via "
            "egomimic.utils.hydra_utils.load_config_from_path; every train + "
            "valid MultiDataset is instantiated and the union of their "
            "episodes is processed."
        ),
    )
    parser.add_argument("--input-keys", nargs="+", required=True)
    parser.add_argument(
        "--output-keys",
        nargs="+",
        default=None,
        help=(
            "Optional. If omitted, defaults are derived per-transform by "
            "swapping the leading dotted segment of each input key for the "
            "transform's OUTPUT_PREFIX (e.g. dinov3: image.front_1 -> "
            "dino.front_1; qwen3: annotations -> qwen.annotations)."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model-name", default=None)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_keys is not None and len(args.input_keys) != len(args.output_keys):
        parser.error("--input-keys and --output-keys must have the same length")

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    episodes = _resolve_episode_paths(args.dataset_config_path)
    logger.info(
        "Resolved %d unique episodes from %s", len(episodes), args.dataset_config_path
    )

    transform = _build_transform(args)

    failures: list[tuple[str, str]] = []
    for ep_hash, path in episodes.items():
        try:
            transform.process_episode(path)
        except Exception as e:
            logger.exception("Failed on %s (%s)", ep_hash, path)
            failures.append((ep_hash, f"{type(e).__name__}: {e}"))

    n_total = len(episodes)
    n_ok = n_total - len(failures)
    logger.info("Done: %d/%d episodes succeeded", n_ok, n_total)
    if failures:
        for ep_hash, msg in failures:
            logger.error("  FAIL %s — %s", ep_hash, msg)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
