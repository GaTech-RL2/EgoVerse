"""
Example usage:
python egomimic/scripts/language_process/scale_to_zarr_annotation.py \
--scale-annotation-dir annotations_test \
--dataset-config-path egomimic/hydra_configs/data/eva_pi_lang.yaml \
--conversion-mode pick_place_llm \
--prompt-filepath egomimic/scripts/language_process/prompt.txt \
--augment-prompt-filepath egomimic/scripts/language_process/augment_prompt.txt \
--project-name "dense-language"
"""

import argparse
import os
from subprocess import run

import hydra
import pandas as pd
from omegaconf import OmegaConf
from scaleapi import ScaleClient

from egomimic.rldb.zarr.zarr_writer import ZarrWriter
from egomimic.scripts.language_process.converter import (
    HardCodedConverter,
    PickPlaceLLMConverter,
)
from egomimic.utils.scale_utils import (
    build_df_from_tasks,
    download_scale_annotation,
    get_available_hashes,
    get_episode_hash_to_tid,
    get_tasks,
    get_tid_to_episode_hash,
    load_scale_annotation_csv,
)


def download_scale_annotation_csv(dest_path: str):
    r2_path = "s3://rldb/scale_annotation_csv/Dense_Language_Tasks_2026_03_31.csv"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pull_csv_script = os.path.join(script_dir, "pull_csv.sh")
    run(["bash", pull_csv_script, r2_path, dest_path], check=True)
    return os.path.join(dest_path, "Dense_Language_Tasks_2026_03_31.csv")


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
    args = parser.parse_args()

    os.makedirs(args.scale_annotation_dir, exist_ok=True)

    if args.project_name:
        completed_tasks = get_tasks(args.project_name, args.scale_api_key)
        df = build_df_from_tasks(completed_tasks)
    else:
        csv_path = download_scale_annotation_csv(args.scale_annotation_dir)
        df = load_scale_annotation_csv(csv_path)
    dataset_cfg = OmegaConf.load(args.dataset_config_path)
    train_datasets = {}
    train_hashes = set()
    for dataset_name in dataset_cfg.train_datasets:
        train_datasets[dataset_name] = hydra.utils.instantiate(
            dataset_cfg.train_datasets[dataset_name]
        )  # MultiDataset
        train_hashes.update(list(train_datasets[dataset_name].datasets.keys()))

    valid_datasets = {}
    valid_hashes = set()
    for dataset_name in dataset_cfg.valid_datasets:
        valid_datasets[dataset_name] = hydra.utils.instantiate(
            dataset_cfg.valid_datasets[dataset_name]
        )  # MultiDataset
        valid_hashes.update(list(valid_datasets[dataset_name].datasets.keys()))

    dataset_hashes = train_hashes.union(valid_hashes)
    available_hashes = get_available_hashes(df)

    # Check if all annotations are present for the train and valid datasets and download scale annotations
    missing_hashes = dataset_hashes - set(available_hashes)
    if missing_hashes:
        error_message = "Missing annotations for the following hashes: "
        for hash in missing_hashes:
            error_message += f"\n{hash}"
        raise ValueError(error_message)

    client = ScaleClient(args.scale_api_key)
    # download the annotations for the dataset hashes
    for hash in dataset_hashes:
        tid = df[df["SEQUENCE_ID"] == hash]["_ID"].values[0]
        download_scale_annotation(client, tid, args.scale_annotation_dir)

    # instantiate converter
    if args.conversion_mode == "pick_place_llm":
        converter = PickPlaceLLMConverter(
            args.scale_annotation_dir,
            args.prompt_filepath,
            augment_prompt_filepath=args.augment_prompt_filepath,
        )
    elif args.conversion_mode == "hardcoded":
        converter = HardCodedConverter(args.scale_annotation_dir)
    else:
        raise ValueError(f"Invalid conversion mode: {args.conversion_mode}")

    # Write annotations to train datasets
    for dataset_name in train_datasets:
        for episode_hash, zarr_dataset in train_datasets[dataset_name].datasets.items():
            tid = get_episode_hash_to_tid(df, episode_hash)
            if tid is None:
                continue
            annotation = converter.convert(tid)
            writer = ZarrWriter(
                episode_path=zarr_dataset.episode_path,
                verbose=True,
            )
            writer.append_annotations(
                annotation_key="annotations", annotations=annotation, mode="w"
            )

    for dataset_name in valid_datasets:
        for episode_hash, zarr_dataset in valid_datasets[dataset_name].datasets.items():
            tid = get_episode_hash_to_tid(df, episode_hash)
            if tid is None:
                continue
            annotation = converter.convert(tid)
            writer = ZarrWriter(
                episode_path=zarr_dataset.episode_path,
                verbose=True,
            )
            writer.append_annotations(
                annotation_key="annotations", annotations=annotation, mode="w"
            )
