import argparse
import hashlib
import sys
from collections import deque
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from omegaconf import OmegaConf


def _hash_bytes(blob: bytes) -> bytes:
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(blob)
    return hasher.digest()


def _scan_parquet_episode(parquet_path: Path, front_key: str, stride: int) -> bool:
    table = pq.read_table(parquet_path, columns=[front_key])
    column = table[front_key]
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    bytes_arr = column.field("bytes")

    hash_window = deque(maxlen=stride)
    for i in range(len(bytes_arr)):
        img_bytes = bytes_arr[i].as_py()
        if img_bytes is None:
            continue
        img_hash = _hash_bytes(img_bytes)
        if len(hash_window) == stride and img_hash == hash_window[0]:
            return True
        hash_window.append(img_hash)
    return False


def main() -> None:
    scripts_dir = Path(__file__).resolve().parent
    egomimic_root = scripts_dir.parent
    repo_root = egomimic_root.parent
    sys.path.insert(0, str(repo_root))
    default_config_path = egomimic_root / "hydra_configs"
    default_data_cfg = default_config_path / "data" / "single_scene_fold_clothes_eva_only.yaml"
    default_train_cfg = default_config_path / "train.yaml"
    default_output = scripts_dir / "frozen_front_img_1_stride60.txt"

    parser = argparse.ArgumentParser(
        description="Check for identical front_img_1 observations at stride 60."
    )
    parser.add_argument(
        "--data-config",
        default=str(default_data_cfg),
        help="Path to the data config YAML.",
    )
    parser.add_argument(
        "--train-config",
        default=str(default_train_cfg),
        help="Path to the train config YAML (for data_schematic).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=60,
        help="Stride in frames for identical check.",
    )
    parser.add_argument(
        "--output",
        default=str(default_output),
        help="Output txt file for corrupted episode ids.",
    )
    args, _overrides = parser.parse_known_args()

    data_cfg = OmegaConf.load(args.data_config)
    train_cfg = OmegaConf.load(args.train_config)

    data_schematic_cfg = train_cfg.data_schematic.schematic_dict

    train_cfg_entry = data_cfg.train_datasets.dataset1.datasets.eva
    valid_cfg_entry = data_cfg.valid_datasets.dataset1.datasets.eva

    temp_root = Path(train_cfg_entry.temp_root) / "S3_rldb_data"
    embodiment = str(data_cfg.train_datasets.dataset1.embodiment)

    corrupted_ids = []
    corrupted_set = set()

    dataset_dirs = sorted(
        [
            p
            for p in temp_root.iterdir()
            if p.is_dir() and p.name.startswith("2025-")
        ]
    )
    if not dataset_dirs:
        raise FileNotFoundError(f"No datasets found under {temp_root}")

    if embodiment not in data_schematic_cfg:
        raise KeyError(f"Embodiment {embodiment} not in data_schematic config.")

    front_key = data_schematic_cfg[embodiment]["front_img_1"]["lerobot_key"]

    for dataset_dir in tqdm(dataset_dirs, desc="datasets"):
        if not (dataset_dir / "meta" / "info.json").exists():
            continue
        episode_files = sorted(dataset_dir.glob("data/chunk-*/episode_*.parquet"))
        if not episode_files:
            continue
        is_corrupted = False
        for ep_file in tqdm(episode_files, desc=dataset_dir.name, leave=False):
            if _scan_parquet_episode(ep_file, front_key=front_key, stride=args.stride):
                is_corrupted = True
                break
        if is_corrupted and dataset_dir.name not in corrupted_set:
            corrupted_set.add(dataset_dir.name)
            corrupted_ids.append(dataset_dir.name)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for episode_id in corrupted_ids:
            f.write(f"{episode_id}\n")

    print(f"Wrote {len(corrupted_ids)} corrupted episode ids to {output_path}")


if __name__ == "__main__":
    main()

