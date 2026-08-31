"""
Ray-parallel version of scale_to_zarr_annotation.py.

Each episode is processed as an independent Ray task:
  download annotation -> convert via LLM/hardcoded -> write to Zarr.

Example usage:
python egomimic/scripts/language_process/scale_to_zarr_annotation_parallel.py \
--scale-annotation-dir annotations_test \
--dataset-config-path egomimic/hydra_configs/data/eva_pi_lang.yaml \
--conversion-mode pick_place_llm \
--prompt-filepath egomimic/scripts/language_process/prompt.txt \
--augment-prompt-filepath egomimic/scripts/language_process/augment_prompt.txt \
--project-name "dense-language"
"""

import argparse
import json
import os

import hydra
import pandas as pd
import ray
import requests
from scaleapi import ScaleClient

from egomimic.scripts.language_process.scale_to_zarr_annotation import (
    download_scale_annotation_csv,
)
from egomimic.utils.scale_utils import (
    build_df_from_tasks,
    get_available_hashes,
    get_episode_hash_to_tid,
    get_tasks,
    load_scale_annotation_csv,
)

REQUEST_TIMEOUT_S = 60


def _download_annotation(scale_api_key: str, tid: str, out_path: str):
    """Download a single Scale annotation JSON to disk."""
    client = ScaleClient(scale_api_key)
    task = client.get_task(tid)
    url = task.response["annotations"]["url"]
    resp = requests.get(url, timeout=REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    raw = json.loads(resp.text.rstrip("\x00"))
    path = os.path.join(out_path, f"{tid}.json")
    with open(path, "w") as f:
        json.dump(raw, f, indent=2)


def _make_converter(
    conversion_mode: str,
    annotation_dir: str,
    prompt_filepath: str,
    augment_prompt_filepath: str | None = None,
    link_prompt_filepath: str | None = None,
):
    from egomimic.scripts.language_process.converter import (
        HardCodedConverter,
        PickPlaceLLMConverter,
    )

    if conversion_mode == "pick_place_llm":
        return PickPlaceLLMConverter(
            annotation_dir,
            prompt_filepath,
            augment_prompt_filepath=augment_prompt_filepath,
            link_prompt_filepath=link_prompt_filepath,
        )
    elif conversion_mode == "hardcoded":
        return HardCodedConverter(annotation_dir)
    raise ValueError(f"Invalid conversion mode: {conversion_mode}")


@ray.remote
def process_episode(
    episode_hash: str,
    episode_path: str,
    tid: str,
    scale_api_key: str,
    scale_annotation_dir: str,
    conversion_mode: str,
    prompt_filepath: str,
    annotation_key: str = "annotations",
    overwrite: bool = False,
    augment_prompt_filepath: str | None = None,
    link_prompt_filepath: str | None = None,
) -> str:
    """
    Self-contained Ray task: download, convert, and write one episode's annotations.
    Returns the episode_hash on success.
    """
    from egomimic.rldb.zarr.zarr_writer import ZarrWriter

    writer = ZarrWriter(episode_path=episode_path, verbose=True)
    if not overwrite and writer.check_key_exists(episode_path, annotation_key):
        print(f"[SKIP] {episode_hash} -> {annotation_key} already exists")
        return episode_hash

    _download_annotation(scale_api_key, tid, scale_annotation_dir)

    converter = _make_converter(
        conversion_mode,
        scale_annotation_dir,
        prompt_filepath,
        augment_prompt_filepath=augment_prompt_filepath,
        link_prompt_filepath=link_prompt_filepath,
    )
    annotation = converter.convert(tid)

    writer.append_annotations(
        annotation_key=annotation_key, annotations=annotation, mode="w"
    )
    print(f"[OK] {episode_hash} -> {episode_path}")
    return episode_hash


def collect_unique_episodes(
    train_datasets: dict,
    valid_datasets: dict,
    df: pd.DataFrame,
) -> dict[str, tuple[str, str]]:
    """
    Deduplicate episodes across train/valid splits.
    Returns {episode_hash: (episode_path, tid)} for every episode that has
    a unique tid mapping.
    """
    episodes: dict[str, tuple[str, str]] = {}

    for datasets in (train_datasets, valid_datasets):
        for dataset_name in datasets:
            for ep_hash, zarr_ds in datasets[dataset_name].datasets.items():
                if ep_hash in episodes:
                    continue
                tid = get_episode_hash_to_tid(df, ep_hash)
                if tid is None:
                    continue
                episodes[ep_hash] = (str(zarr_ds.episode_path), tid)

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
        choices=["pick_place_llm", "hardcoded"],
    )
    parser.add_argument(
        "-s", "--scale-api-key", default=os.environ.get("SCALE_API_KEY", "")
    )
    parser.add_argument("--prompt-filepath", type=str, required=True)
    parser.add_argument("--augment-prompt-filepath", type=str, default=None)
    parser.add_argument("--annotation-key", type=str, default="annotations")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing annotation key. If unset, episodes that already have the key are skipped.",
    )
    parser.add_argument(
        "--link-prompt-filepath",
        type=str,
        default=None,
        help=(
            "If provided, also emit linked pick→place annotations using this "
            "prompt template. Each Pick is paired with the next Put on the "
            "same arm into a single combined instruction spanning both clips."
        ),
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

    os.makedirs(args.scale_annotation_dir, exist_ok=True)

    # --- Fetch annotation metadata (sequential) ---
    if args.project_name:
        completed_tasks = get_tasks(args.project_name, args.scale_api_key)
        df = build_df_from_tasks(completed_tasks)
    else:
        csv_path = download_scale_annotation_csv(args.scale_annotation_dir)
        df = load_scale_annotation_csv(csv_path)

    # --- Instantiate datasets (sequential, needs SQL / S3 sync) ---
    from egomimic.utils.hydra_utils import HYDRA_CONFIG_DIR, load_config

    # Derive config name relative to hydra_configs dir (e.g. "data/cotrain_pi_lang")
    abs_cfg_path = os.path.abspath(args.dataset_config_path)
    rel_path = os.path.relpath(abs_cfg_path, HYDRA_CONFIG_DIR)
    config_name = os.path.splitext(rel_path)[0]
    dataset_cfg = load_config(config_name)

    train_datasets = {}
    train_hashes = set()
    for dataset_name in dataset_cfg.train_datasets:
        train_datasets[dataset_name] = hydra.utils.instantiate(
            dataset_cfg.train_datasets[dataset_name]
        )
        train_hashes.update(list(train_datasets[dataset_name].datasets.keys()))

    valid_datasets = {}
    valid_hashes = set()
    for dataset_name in dataset_cfg.valid_datasets:
        valid_datasets[dataset_name] = hydra.utils.instantiate(
            dataset_cfg.valid_datasets[dataset_name]
        )
        valid_hashes.update(list(valid_datasets[dataset_name].datasets.keys()))

    dataset_hashes = train_hashes.union(valid_hashes)
    available_hashes = get_available_hashes(df)

    # Check if all annotations are present for the train and valid datasets
    missing_hashes = dataset_hashes - set(available_hashes)
    if missing_hashes:
        error_message = "Missing annotations for the following hashes: "
        for hash in missing_hashes:
            error_message += f"\n{hash}"
        raise ValueError(error_message)

    # --- Collect unique (episode_hash, episode_path, tid) tuples ---
    episodes = collect_unique_episodes(train_datasets, valid_datasets, df)
    print(f"[INFO] {len(episodes)} unique episodes to process")

    # --- Launch Ray ---
    ray_kwargs = {}
    if args.ray_address is not None:
        ray_kwargs["address"] = args.ray_address
    ray.init(**ray_kwargs)

    pending: dict[ray.ObjectRef, str] = {}
    for ep_hash, (ep_path, tid) in episodes.items():
        ref = process_episode.options(num_cpus=args.num_cpus_per_task).remote(
            episode_hash=ep_hash,
            episode_path=ep_path,
            tid=tid,
            scale_api_key=args.scale_api_key,
            scale_annotation_dir=args.scale_annotation_dir,
            conversion_mode=args.conversion_mode,
            prompt_filepath=args.prompt_filepath,
            annotation_key=args.annotation_key,
            overwrite=args.overwrite,
            augment_prompt_filepath=args.augment_prompt_filepath,
            link_prompt_filepath=args.link_prompt_filepath,
        )
        pending[ref] = ep_hash

    # --- Collect results ---
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
