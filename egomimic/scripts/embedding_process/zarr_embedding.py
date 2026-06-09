"""
Compute and append embedding arrays to Zarr episode stores.

Resolves episode paths from a Hydra dataset config (the same kind consumed
by scale_to_zarr_annotation.py), then runs a ZarrKeyTransform across the
union of train + valid episodes.

Bundled transforms:
  - DINOv2ImageEmbedding (dinov2_embedding.py): per-frame DINOv2 patch-token
    image embeddings (the RICL reference encoder).
  - Qwen3TextEmbedding (qwen3_embedding.py): per-frame text embeddings from
    JSON-encoded annotation spans via Qwen3-Embedding-0.6B.

Episodes are selected one of three ways: from a Hydra dataset config
(``--dataset-config-path``); as an explicit hash list rooted at a directory
(``--episode-root`` + ``--episodes-file``/``--episodes``); or by SQL-registry
filters (``--filter-lambda`` + ``--sync-root``), which query + sync matching
stores and use RAW zarr keys (e.g. ``images.front_1``). The explicit and filter
forms reach stores a dataset config can't.

With ``--output-root`` embeddings are written to a SEPARATE per-episode store at
``<output-root>/<hash>`` (a writable mirror, for read-only sources) instead of
in-place. The mirror plugs directly into ``retrieval.py --zarr-root
'<output-root>/{hash}'`` and ``egomimic/ricl/scripts/build_embedding_index.py``.

Examples:
    python egomimic/scripts/embedding_process/zarr_embedding.py \\
        --transform dinov2 \\
        --dataset-config-path egomimic/hydra_configs/data/eva_pi_lang.yaml \\
        --input-keys observations.images.front_1 \\
        --output-keys observations.embeddings.dinov2.front_1 \\
        --batch-size 64

    python egomimic/scripts/embedding_process/zarr_embedding.py \\
        --transform dinov2 \\
        --episode-root /storage/project/r-dxu345-0/agao81/pick_place \\
        --episodes-file egomimic/ricl/episode_lists/embed_all.txt \\
        --input-keys images.front_1 \\
        --output-keys observations.embeddings.dinov2.front_1

    python egomimic/scripts/embedding_process/zarr_embedding.py \\
        --transform qwen3 \\
        --dataset-config-path egomimic/hydra_configs/data/eva_pi_lang.yaml \\
        --input-keys annotations \\
        --output-keys observations.embeddings.qwen3.annotations

    # SQL filter -> writable mirror folder (RICL eva pick_place bank):
    python egomimic/scripts/embedding_process/zarr_embedding.py \\
        --transform dinov2 \\
        --filter-lambda "lambda row: row.get('task')=='pick_place' and \\
            row.get('embodiment')=='eva_bimanual' and \\
            (row.get('zarr_processed_path') or '').strip()!=''" \\
        --sync-root egomimic/ricl/outputs/ricl_sync_cache/pickplace_eva \\
        --output-root egomimic/ricl/outputs/ricl_embeddings/pickplace_eva \\
        --input-keys images.front_1 \\
        --output-keys observations.embeddings.dinov2.front_1
"""

from __future__ import annotations

import argparse
import logging
import os

import hydra
import torch

from egomimic.scripts.embedding_process.dinov2_embedding import DINOv2ImageEmbedding
from egomimic.scripts.embedding_process.qwen3_embedding import Qwen3TextEmbedding
from egomimic.scripts.embedding_process.zarr_key_transform import ZarrKeyTransform
from egomimic.utils.hydra_utils import load_config_from_path

logger = logging.getLogger(__name__)


TRANSFORMS: dict[str, type[ZarrKeyTransform]] = {
    "dinov2": DINOv2ImageEmbedding,
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


def _resolve_filtered_episodes(
    filter_lambdas: list[str], sync_root: str
) -> dict[str, str]:
    """
    Resolve ``{episode_hash: episode_path}`` from SQL-registry filters.

    Builds a :class:`DatasetFilter` from ``filter_lambdas`` (each a
    ``"lambda row: ..."`` string evaluated against an episode row), queries the
    registry + syncs matching stores into ``sync_root`` via
    :class:`S3EpisodeResolver`, and returns the local zarr path per episode.
    The synced stores keep their RAW zarr keys (e.g. ``images.front_1``).
    """
    from pathlib import Path

    from egomimic.rldb.filters import DatasetFilter
    from egomimic.rldb.zarr.zarr_dataset_multi import S3EpisodeResolver

    filters = DatasetFilter(filter_lambdas=filter_lambdas)
    resolver = S3EpisodeResolver(folder_path=Path(sync_root))
    datasets = resolver.resolve(filters=filters)
    return {h: str(ds.episode_path) for h, ds in datasets.items()}


def _resolve_explicit_episodes(
    episode_root: str, episodes: list[str] | None, episodes_file: str | None
) -> dict[str, str]:
    """Build {episode_hash: path} from an explicit hash list under one root."""
    hashes: list[str] = list(episodes or [])
    if episodes_file:
        with open(episodes_file) as f:
            hashes += f.read().split()
    # de-dup, keep order
    hashes = list(dict.fromkeys(hashes))
    if not hashes:
        raise SystemExit("--episode-root given but no episodes (use --episodes/-file)")
    mapping = {h: os.path.join(episode_root, h) for h in hashes}
    missing = [h for h, p in mapping.items() if not os.path.isdir(p)]
    if missing:
        raise SystemExit(
            f"{len(missing)}/{len(mapping)} episodes missing under "
            f"{episode_root}: {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    return mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transform", required=True, choices=sorted(TRANSFORMS.keys()))
    parser.add_argument(
        "--dataset-config-path",
        default=None,
        help=(
            "Path to a Hydra dataset config (e.g. egomimic/hydra_configs/data/"
            "eva_pi_lang.yaml). The full ``defaults:`` chain is resolved via "
            "egomimic.utils.hydra_utils.load_config_from_path; every train + "
            "valid MultiDataset is instantiated and the union of their "
            "episodes is processed. Mutually exclusive with --episode-root."
        ),
    )
    parser.add_argument(
        "--episode-root",
        default=None,
        help=(
            "Directory containing per-episode zarr stores named by hash. "
            "Use with --episodes/--episodes-file instead of a dataset config."
        ),
    )
    parser.add_argument(
        "--episodes",
        nargs="+",
        default=None,
        help="Explicit episode hashes under --episode-root.",
    )
    parser.add_argument(
        "--episodes-file",
        default=None,
        help="Text file of episode hashes (whitespace-separated) under --episode-root.",
    )
    parser.add_argument(
        "--filter-lambda",
        dest="filter_lambdas",
        action="append",
        default=None,
        help=(
            "SQL-registry filter source. Repeatable; each value is a "
            '"lambda row: ..." predicate ANDed together (DatasetFilter). '
            "Matching episodes are synced into --sync-root and embedded. "
            "Example: \"lambda row: row.get('task')=='pick_place' and "
            "row.get('embodiment')=='eva_bimanual' and (row.get("
            "'zarr_processed_path') or '').strip()!=''\". Input keys are RAW "
            "zarr keys (e.g. images.front_1)."
        ),
    )
    parser.add_argument(
        "--sync-root",
        default=None,
        help="Local cache dir S3EpisodeResolver syncs filtered stores into "
        "(used with --filter-lambda).",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "If set, write embeddings to a SEPARATE per-episode store at "
            "<output-root>/<hash> instead of in-place. Use when the source "
            "stores are read-only; the mirror plugs into "
            "retrieval.py --zarr-root '<output-root>/{hash}'."
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
            "transform's OUTPUT_PREFIX (e.g. dinov2: images.front_1 -> "
            "dinov2.front_1; qwen3: annotations -> qwen.annotations)."
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
    sources = [
        args.dataset_config_path is not None,
        args.episode_root is not None,
        args.filter_lambdas is not None,
    ]
    if sum(sources) != 1:
        parser.error(
            "provide exactly one episode source: --dataset-config-path OR "
            "--episode-root (+ --episodes/--episodes-file) OR --filter-lambda "
            "(+ --sync-root)"
        )
    if args.filter_lambdas is not None and args.sync_root is None:
        parser.error("--filter-lambda requires --sync-root")

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    if args.episode_root is not None:
        episodes = _resolve_explicit_episodes(
            args.episode_root, args.episodes, args.episodes_file
        )
        logger.info("Resolved %d episodes under %s", len(episodes), args.episode_root)
    elif args.filter_lambdas is not None:
        episodes = _resolve_filtered_episodes(args.filter_lambdas, args.sync_root)
        logger.info(
            "Resolved %d episodes from SQL filters into %s",
            len(episodes),
            args.sync_root,
        )
    else:
        episodes = _resolve_episode_paths(args.dataset_config_path)
        logger.info(
            "Resolved %d unique episodes from %s",
            len(episodes),
            args.dataset_config_path,
        )

    if args.output_root is not None:
        os.makedirs(args.output_root, exist_ok=True)
        logger.info("Writing embeddings to mirror folder %s/<hash>", args.output_root)

    transform = _build_transform(args)

    failures: list[tuple[str, str]] = []
    for ep_hash, path in episodes.items():
        try:
            out_path = (
                os.path.join(args.output_root, ep_hash)
                if args.output_root is not None
                else None
            )
            transform.process_episode(path, output_store_path=out_path)
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
