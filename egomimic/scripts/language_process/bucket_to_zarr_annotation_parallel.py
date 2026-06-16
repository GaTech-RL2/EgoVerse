"""
Ray-parallel: download per-episode annotation JSONs from a bucket and write into Zarr.

Counterpart to scale_to_bucket_annotation_parallel.py. Each Ray task:
  download s3://{bucket}/{prefix}/{episode_hash}_{annotation_key}.json -> ZarrWriter.

Episodes that already have ``annotation_key`` in their Zarr are skipped (unless --overwrite).

Example usage:
python egomimic/scripts/language_process/bucket_to_zarr_annotation_parallel.py \
--dataset-config-path egomimic/hydra_configs/data/eva_pi_lang.yaml \
--bucket s3://rldb/scale_annotations
"""

import argparse
import json
import os

import hydra
import ray

from egomimic.scripts.language_process.scale_to_bucket_annotation_parallel import (
    object_key,
    parse_s3_uri,
    s3_object_exists,
)


def collect_unique_episode_paths(
    train_datasets: dict,
    valid_datasets: dict,
) -> dict[str, str]:
    """Deduplicate episodes across train/valid splits. Returns {episode_hash: episode_path}."""
    episodes: dict[str, str] = {}
    for datasets in (train_datasets, valid_datasets):
        for dataset_name in datasets:
            for ep_hash, zarr_ds in datasets[dataset_name].datasets.items():
                if ep_hash in episodes:
                    continue
                episodes[ep_hash] = str(zarr_ds.episode_path)
    return episodes


@ray.remote
def process_episode(
    episode_hash: str,
    episode_path: str,
    bucket: str,
    prefix: str,
    annotation_key: str = "annotations",
    overwrite: bool = False,
) -> str:
    """Download annotation JSON from bucket and write into the episode's Zarr."""
    from egomimic.rldb.zarr.zarr_writer import ZarrWriter
    from egomimic.utils.aws.aws_data_utils import get_boto3_s3_client

    writer = ZarrWriter(episode_path=episode_path, verbose=True)
    if not overwrite and writer.check_key_exists(episode_path, annotation_key):
        print(f"[SKIP] {episode_hash} -> {annotation_key} already in zarr")
        return episode_hash

    s3 = get_boto3_s3_client()
    key = object_key(prefix, episode_hash, annotation_key)

    if not s3_object_exists(s3, bucket, key):
        raise FileNotFoundError(
            f"Annotation object missing: s3://{bucket}/{key} "
            f"(run scale_to_bucket_annotation_parallel.py first)"
        )

    obj = s3.get_object(Bucket=bucket, Key=key)
    payload = json.loads(obj["Body"].read().decode("utf-8"))

    annotations = [
        (entry["text"], int(entry["start_idx"]), int(entry["end_idx"]))
        for entry in payload
    ]

    writer.append_annotations(
        annotation_key=annotation_key, annotations=annotations, mode="w"
    )
    print(f"[OK] {episode_hash} -> {episode_path} ({len(annotations)} entries)")
    return episode_hash


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config-path", type=str, required=True)
    parser.add_argument("--annotation-key", type=str, default="annotations")
    parser.add_argument(
        "--bucket",
        type=str,
        required=True,
        help="Source bucket as 's3://bucket/optional/prefix' (or 'bucket/prefix').",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing annotation key in zarr. If unset, episodes that already "
        "have the key are skipped.",
    )
    parser.add_argument(
        "--episode-hash",
        type=str,
        default=None,
        help="If set, only process this single episode hash (for cheap debugging).",
    )
    parser.add_argument(
        "--ray-address",
        default=None,
        help="Ray cluster address. Omit for single-node local mode (uses all CPUs).",
    )
    parser.add_argument(
        "--num-cpus-per-task",
        type=int,
        default=1,
        help="CPUs reserved per Ray task (default: 1)",
    )
    args = parser.parse_args()

    bucket, prefix = parse_s3_uri(args.bucket)

    # --- Instantiate datasets (sequential, needs SQL / S3 sync) ---
    # Use hydra's compose API so `defaults:` inheritance from base configs
    # (e.g. cotrain_pi_base.yaml) is resolved — OmegaConf.load alone leaves
    # `_target_` unset and instantiate returns a bare DictConfig.
    from egomimic.utils.hydra_utils import HYDRA_CONFIG_DIR, load_config

    abs_cfg_path = os.path.abspath(args.dataset_config_path)
    rel_path = os.path.relpath(abs_cfg_path, HYDRA_CONFIG_DIR)
    config_name = os.path.splitext(rel_path)[0]
    dataset_cfg = load_config(config_name)

    train_datasets = {}
    for dataset_name in dataset_cfg.train_datasets:
        ds_cfg = dataset_cfg.train_datasets[dataset_name]
        if ds_cfg is None:
            continue
        train_datasets[dataset_name] = hydra.utils.instantiate(ds_cfg)

    valid_datasets = {}
    for dataset_name in dataset_cfg.valid_datasets:
        ds_cfg = dataset_cfg.valid_datasets[dataset_name]
        if ds_cfg is None:
            continue
        valid_datasets[dataset_name] = hydra.utils.instantiate(ds_cfg)

    episodes = collect_unique_episode_paths(train_datasets, valid_datasets)

    if args.episode_hash is not None:
        if args.episode_hash not in episodes:
            raise ValueError(
                f"--episode-hash {args.episode_hash} not found among "
                f"{len(episodes)} episodes from the dataset config."
            )
        episodes = {args.episode_hash: episodes[args.episode_hash]}

    print(f"[INFO] {len(episodes)} unique episodes to process")
    print(f"[INFO] source bucket: s3://{bucket}/{prefix}")

    # --- Launch Ray ---
    ray_kwargs = {}
    if args.ray_address is not None:
        ray_kwargs["address"] = args.ray_address
    ray.init(**ray_kwargs)

    pending: dict[ray.ObjectRef, str] = {}
    for ep_hash, ep_path in episodes.items():
        ref = process_episode.options(num_cpus=args.num_cpus_per_task).remote(
            episode_hash=ep_hash,
            episode_path=ep_path,
            bucket=bucket,
            prefix=prefix,
            annotation_key=args.annotation_key,
            overwrite=args.overwrite,
        )
        pending[ref] = ep_hash

    succeeded, failed = [], []
    while pending:
        done_refs, _ = ray.wait(list(pending.keys()), num_returns=1)
        ref = done_refs[0]
        ep_hash = pending.pop(ref)
        try:
            ray.get(ref)
            succeeded.append(ep_hash)
        except Exception as e:
            print(f"[FAIL] {ep_hash}: {type(e).__name__}: {e}", flush=True)
            failed.append(ep_hash)

    print(
        f"\n[DONE] {len(succeeded)} succeeded, {len(failed)} failed "
        f"out of {len(episodes)} episodes"
    )
    if failed:
        print("[FAILED EPISODES]")
        for h in failed:
            print(f"  {h}")
