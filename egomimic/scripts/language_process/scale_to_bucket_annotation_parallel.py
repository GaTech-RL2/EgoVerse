"""
Ray-parallel: download Scale annotations, run conversion, upload result JSON to S3/R2.

Each episode is processed as an independent Ray task:
  download Scale annotation -> convert via LLM/hardcoded -> upload JSON to bucket.

Uploads ONE JSON PER ROLE KEY: {prefix}/{episode_hash}_annotations_task.json
(+ _annotations_subtask.json where the converter emits it) — entries are plain
{text, start_idx, end_idx}; the role lives in the key name (see converter.py).
An episode is skipped only when EVERY role key it would produce already exists
(unless --overwrite); a partial upload is regenerated whole so the roles never
come from different converter runs.

Example usage:
python egomimic/scripts/language_process/scale_to_bucket_annotation_parallel.py \
--scale-annotation-dir annotations_test \
--dataset-config-path egomimic/hydra_configs/data/eva_pi_lang.yaml \
--conversion-mode pick_place_llm \
--prompt-filepath egomimic/scripts/language_process/prompt.txt \
--augment-prompt-filepath egomimic/scripts/language_process/augment_prompt.txt \
--project-name "dense-language" \
--bucket s3://rldb/scale_annotations
"""

import argparse
import json
import os

import hydra
import pandas as pd
import ray
from scaleapi import ScaleClient

from egomimic.scripts.language_process.scale_to_zarr_annotation import (
    download_scale_annotation_csv,
)
from egomimic.utils.scale_utils import (
    build_df_from_tasks,
    download_scale_annotation,
    get_available_hashes,
    get_episode_hash_to_tid,
    get_tasks,
    load_scale_annotation_csv,
)


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Parse 's3://bucket/key/prefix' or 'bucket/key/prefix' -> ('bucket', 'key/prefix')."""
    uri = uri.strip()
    if uri.startswith("s3://"):
        uri = uri[len("s3://") :]
    parts = uri.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix.strip("/")


def object_key(prefix: str, episode_hash: str, annotation_key: str) -> str:
    name = f"{episode_hash}_{annotation_key}.json"
    return f"{prefix}/{name}" if prefix else name


def s3_object_exists(s3_client, bucket: str, key: str) -> bool:
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except s3_client.exceptions.ClientError as e:  # type: ignore[attr-defined]
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def _make_converter(
    conversion_mode: str,
    annotation_dir: str,
    prompt_filepath: str,
    augment_prompt_filepath: str | None = None,
    sort_prompt_filepath: str | None = None,
    sort_augment_prompt_filepath: str | None = None,
    subtask_copy: bool = False,
):
    from egomimic.scripts.language_process.converter import (
        HardCodedConverter,
        PickPlaceLLMConverter,
        SortConverter,
    )

    if conversion_mode == "pick_place_llm":
        return PickPlaceLLMConverter(
            annotation_dir,
            prompt_filepath,
            augment_prompt_filepath=augment_prompt_filepath,
            # eva regime: identical task instruction and subtask target.
            subtask_copy=subtask_copy,
        )
    elif conversion_mode == "sort_llm":
        # Task-level sort text is read from the annotation's "Sorting" track;
        # sort_prompt_filepath is only an optional LLM-generation fallback.
        return SortConverter(
            annotation_dir,
            prompt_filepath,
            sort_prompt_filepath,
            augment_prompt_filepath=augment_prompt_filepath,
            sort_augment_prompt_filepath=sort_augment_prompt_filepath,
        )
    elif conversion_mode == "hardcoded":
        return HardCodedConverter(annotation_dir)
    raise ValueError(f"Invalid conversion mode: {conversion_mode}")


@ray.remote
def process_episode(
    episode_hash: str,
    tid: str,
    scale_api_key: str,
    scale_annotation_dir: str,
    conversion_mode: str,
    prompt_filepath: str,
    bucket: str,
    prefix: str,
    overwrite: bool = False,
    augment_prompt_filepath: str | None = None,
    sort_prompt_filepath: str | None = None,
    sort_augment_prompt_filepath: str | None = None,
    subtask_copy: bool = False,
) -> str:
    """Self-contained Ray task: download, convert, and upload one episode's
    annotations — ONE JSON PER ROLE KEY (``{hash}_annotations_task.json``,
    ``{hash}_annotations_subtask.json``, ...), entries are plain
    ``{text, start_idx, end_idx}`` (role lives in the key name, no level field).
    """
    from egomimic.utils.aws.aws_data_utils import get_boto3_s3_client

    s3 = get_boto3_s3_client()

    # Early skip BEFORE the (expensive) download + LLM conversion: the role
    # keys each mode emits are known up front for the LLM converters, so an
    # idempotent re-run costs one head_object per key instead of the full
    # base-instruction + augmentation generation. `hardcoded` keys are
    # converter-determined, so it falls through to the post-convert check.
    if not overwrite:
        expected = None
        if conversion_mode == "pick_place_llm":
            expected = ["annotations_task"] + (
                ["annotations_subtask"] if subtask_copy else []
            )
        elif conversion_mode == "sort_llm":
            expected = ["annotations_task", "annotations_subtask"]
        if expected and all(
            s3_object_exists(s3, bucket, object_key(prefix, episode_hash, k))
            for k in expected
        ):
            print(f"[SKIP] {episode_hash} -> all of {expected} already exist")
            return episode_hash

    client = ScaleClient(scale_api_key)
    download_scale_annotation(client, tid, scale_annotation_dir)

    converter = _make_converter(
        conversion_mode,
        scale_annotation_dir,
        prompt_filepath,
        augment_prompt_filepath=augment_prompt_filepath,
        sort_prompt_filepath=sort_prompt_filepath,
        sort_augment_prompt_filepath=sort_augment_prompt_filepath,
        subtask_copy=subtask_copy,
    )
    keyed = converter.convert(tid)
    if not keyed:
        print(f"[EMPTY] {episode_hash} -> converter produced no annotations")
        return episode_hash

    # Skip only when EVERY role key already exists — a partial upload (e.g.
    # task present, subtask missing) is regenerated whole so the two roles
    # never come from different converter runs.
    keys = {ann_key: object_key(prefix, episode_hash, ann_key) for ann_key in keyed}
    if not overwrite and all(
        s3_object_exists(s3, bucket, k) for k in keys.values()
    ):
        print(f"[SKIP] {episode_hash} -> all of {sorted(keys)} already exist")
        return episode_hash

    for ann_key, entries in keyed.items():
        payload = [
            {"text": text, "start_idx": int(start_idx), "end_idx": int(end_idx)}
            for text, start_idx, end_idx in entries
        ]
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        s3.put_object(
            Bucket=bucket, Key=keys[ann_key], Body=body,
            ContentType="application/json",
        )
        print(
            f"[OK] {episode_hash} -> s3://{bucket}/{keys[ann_key]} "
            f"({len(payload)} entries)"
        )
    return episode_hash


def collect_unique_episodes(
    train_datasets: dict,
    valid_datasets: dict,
    df: pd.DataFrame,
) -> dict[str, str]:
    """Deduplicate episodes across train/valid splits. Returns {episode_hash: tid}."""
    episodes: dict[str, str] = {}
    for datasets in (train_datasets, valid_datasets):
        for dataset_name in datasets:
            for ep_hash in datasets[dataset_name].datasets.keys():
                if ep_hash in episodes:
                    continue
                tid = get_episode_hash_to_tid(df, ep_hash)
                if tid is None:
                    continue
                episodes[ep_hash] = tid
    return episodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-name", type=str, default=None)
    parser.add_argument("--scale-annotation-dir", type=str, required=True)
    parser.add_argument("--dataset-config-path", type=str, required=True)
    parser.add_argument(
        "--conversion-mode",
        type=str,
        required=True,
        choices=["pick_place_llm", "sort_llm", "hardcoded"],
    )
    parser.add_argument(
        "-s", "--scale-api-key", default=os.environ.get("SCALE_API_KEY", "")
    )
    parser.add_argument("--prompt-filepath", type=str, required=True)
    parser.add_argument("--augment-prompt-filepath", type=str, default=None)
    parser.add_argument(
        "--sort-prompt-filepath",
        type=str,
        default=None,
        help="High-level sort instruction prompt (required for --conversion-mode sort_llm).",
    )
    parser.add_argument(
        "--sort-augment-prompt-filepath",
        type=str,
        default=None,
        help="High-level sort augmentation prompt (optional, used by sort_llm).",
    )
    parser.add_argument(
        "--subtask-copy",
        action="store_true",
        help="pick_place_llm only: ALSO write the task instructions verbatim "
        "under annotations_subtask (eva regime — identical task prompt and "
        "subtask target).",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        required=True,
        help="Destination bucket as 's3://bucket/optional/prefix' (or 'bucket/prefix').",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-process and overwrite even if the bucket object already exists.",
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
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Resolve the config's ${paths.dataset_dir} (resolver folder_path). "
        "Required when the dataset config inherits folder_path: ${paths.dataset_dir} "
        "(e.g. the cotrain-derived sort configs) since load_config composes only the "
        "data group and has no paths node.",
    )
    args = parser.parse_args()

    bucket, prefix = parse_s3_uri(args.bucket)
    os.makedirs(args.scale_annotation_dir, exist_ok=True)

    # --- Fetch annotation metadata (sequential) ---
    if args.project_name:
        completed_tasks = get_tasks(args.project_name, args.scale_api_key)
        df = build_df_from_tasks(completed_tasks)
    else:
        csv_path = download_scale_annotation_csv(args.scale_annotation_dir)
        df = load_scale_annotation_csv(csv_path)

    # --- Instantiate datasets (sequential, needs SQL / S3 sync) ---
    # Use hydra's compose API so `defaults:` inheritance from base configs
    # (e.g. cotrain_pi_base.yaml) is resolved — OmegaConf.load alone leaves
    # `_target_` unset and instantiate returns a bare DictConfig.
    from egomimic.utils.hydra_utils import HYDRA_CONFIG_DIR, load_config

    abs_cfg_path = os.path.abspath(args.dataset_config_path)
    rel_path = os.path.relpath(abs_cfg_path, HYDRA_CONFIG_DIR)
    config_name = os.path.splitext(rel_path)[0]
    dataset_cfg = load_config(config_name)

    # load_config composes only the `data` group, so ${paths.dataset_dir} (the
    # resolver folder_path inherited from cotrain_pi_base) has no node to resolve
    # against. Overwrite each train resolver's folder_path with --dataset-dir so
    # the real resolver runs unmodified (valid resolvers interpolate from train).
    if args.dataset_dir is not None:
        from omegaconf import OmegaConf

        OmegaConf.set_struct(dataset_cfg, False)
        for _name in list((dataset_cfg.get("train_datasets") or {}).keys()):
            _ds = dataset_cfg.train_datasets[_name]
            if _ds is None:
                continue
            _res = _ds.get("resolver")
            if _res is not None and "folder_path" in _res:
                _res.folder_path = args.dataset_dir

    train_datasets = {}
    train_hashes = set()
    for dataset_name in dataset_cfg.train_datasets:
        ds_cfg = dataset_cfg.train_datasets[dataset_name]
        if ds_cfg is None:
            continue
        train_datasets[dataset_name] = hydra.utils.instantiate(ds_cfg)
        train_hashes.update(list(train_datasets[dataset_name].datasets.keys()))

    valid_datasets = {}
    valid_hashes = set()
    for dataset_name in dataset_cfg.valid_datasets:
        ds_cfg = dataset_cfg.valid_datasets[dataset_name]
        if ds_cfg is None:
            continue
        valid_datasets[dataset_name] = hydra.utils.instantiate(ds_cfg)
        valid_hashes.update(list(valid_datasets[dataset_name].datasets.keys()))

    dataset_hashes = train_hashes.union(valid_hashes)
    available_hashes = get_available_hashes(df)

    if args.episode_hash is None:
        missing_hashes = dataset_hashes - set(available_hashes)
        if missing_hashes:
            error_message = "Missing annotations for the following hashes: "
            for h in missing_hashes:
                error_message += f"\n{h}"
            raise ValueError(error_message)

    episodes = collect_unique_episodes(train_datasets, valid_datasets, df)

    if args.episode_hash is not None:
        if args.episode_hash not in episodes:
            raise ValueError(
                f"--episode-hash {args.episode_hash} not found among "
                f"{len(episodes)} resolvable episodes."
            )
        episodes = {args.episode_hash: episodes[args.episode_hash]}

    print(f"[INFO] {len(episodes)} unique episodes to process")
    print(f"[INFO] target bucket: s3://{bucket}/{prefix}")

    # --- Launch Ray ---
    ray_kwargs = {}
    if args.ray_address is not None:
        ray_kwargs["address"] = args.ray_address
    ray.init(**ray_kwargs)

    pending: dict[ray.ObjectRef, str] = {}
    for ep_hash, tid in episodes.items():
        ref = process_episode.options(num_cpus=args.num_cpus_per_task).remote(
            episode_hash=ep_hash,
            tid=tid,
            scale_api_key=args.scale_api_key,
            scale_annotation_dir=args.scale_annotation_dir,
            conversion_mode=args.conversion_mode,
            prompt_filepath=args.prompt_filepath,
            bucket=bucket,
            prefix=prefix,
            overwrite=args.overwrite,
            augment_prompt_filepath=args.augment_prompt_filepath,
            sort_prompt_filepath=args.sort_prompt_filepath,
            sort_augment_prompt_filepath=args.sort_augment_prompt_filepath,
            subtask_copy=args.subtask_copy,
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
