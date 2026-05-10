import copy

import hydra
import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, default_collate
from tqdm import tqdm

from egomimic.rldb.zarr.utils import set_global_seed
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset
from egomimic.utils.pylogger import RankedLogger

OmegaConf.register_new_resolver("eval", eval)
log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(
    version_base="1.3",
    config_path="../../hydra_configs",
    config_name="train_zarr_cartesian_pi.yaml",
)
def main(cfg: DictConfig) -> None:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
        set_global_seed(cfg.seed)

    train_datasets = {}
    for dataset_name in cfg.data.train_datasets:
        train_datasets[dataset_name] = hydra.utils.instantiate(
            cfg.data.train_datasets[dataset_name]
        )

    norm_stats_obj = MultiDataset(
        state={},
        norm_mode=OmegaConf.select(cfg, "norm_stats.norm_mode", default="quantile"),
    )
    norm_stats_obj.populate_from_datasets(train_datasets)

    # --- Infer shapes and norm stats (mirrors trainHydra.py) ---
    for dataset_name, dataset in train_datasets.items():
        log.info(f"Inferring shapes for dataset <{dataset_name}>")
        norm_stats_obj.infer_shapes_from_batch(dataset[0])

        instantiate_copy = copy.deepcopy(cfg.data.train_datasets[dataset_name])
        keymap_cfg = instantiate_copy.resolver.key_map
        km = OmegaConf.to_container(keymap_cfg, resolve=False)
        km["norm_mode"] = True
        instantiate_copy.resolver.key_map = km
        norm_dataset = hydra.utils.instantiate(instantiate_copy)

        norm_stats_obj.infer_norm_from_dataset(
            norm_dataset,
            dataset_name,
            sample_frac=OmegaConf.select(cfg, "norm_stats.sample_frac", default=1.0),
            num_workers=OmegaConf.select(cfg, "norm_stats.num_workers", default=4),
            precomputed_norm_path=OmegaConf.select(
                cfg, "norm_stats.precomputed_norm_path", default=None
            ),
        )

    # --- Collate that only keeps tensor fields (skips annotations/strings) ---
    def tensor_only_collate(batch):
        filtered = [
            {k: v for k, v in sample.items() if isinstance(v, torch.Tensor)}
            for sample in batch
        ]
        return default_collate(filtered)

    # --- Iterate over dataset and check for zero actions ---
    action_key = "actions_cartesian"
    left_slice = slice(0, 6)  # left arm: indices 0-4
    right_slice = slice(7, 13)  # right arm: indices 7-11

    zero_left_indices = []
    zero_right_indices = []

    for dataset_name, dataset in train_datasets.items():
        if "eva" not in dataset_name:
            log.info(f"Skipping dataset <{dataset_name}> (not eva)")
            continue
        log.info(
            f"Checking dataset <{dataset_name}> for zero actions ({len(dataset)} frames)"
        )

        loader = DataLoader(
            dataset,
            batch_size=64,
            num_workers=4,
            shuffle=False,
            collate_fn=tensor_only_collate,
        )

        global_idx = 0
        for batch in tqdm(loader, desc=f"Checking {dataset_name}"):
            actions = batch[action_key]  # (B, ..., D)
            batch_size = actions.shape[0]

            # Handle chunked actions: flatten to (B, D) by checking across chunk dim
            if actions.dim() == 3:
                # (B, chunk, D) — check if ALL timesteps in chunk are zero
                left_vals = actions[:, :, left_slice]  # (B, chunk, 5)
                right_vals = actions[:, :, right_slice]  # (B, chunk, 5)
                # All zero across both chunk and feature dims
                left_all_zero = (left_vals == 0).all(dim=2).all(dim=1)  # (B,)
                right_all_zero = (right_vals == 0).all(dim=2).all(dim=1)  # (B,)
            else:
                # (B, D)
                left_vals = actions[:, left_slice]
                right_vals = actions[:, right_slice]
                left_all_zero = (left_vals == 0).all(dim=1)
                right_all_zero = (right_vals == 0).all(dim=1)

            for i in range(batch_size):
                idx = global_idx + i
                if left_all_zero[i]:
                    zero_left_indices.append((dataset_name, idx))
                if right_all_zero[i]:
                    zero_right_indices.append((dataset_name, idx))

            global_idx += batch_size

    # --- Report ---
    print("\n" + "=" * 60)
    print("ZERO ACTION CHECK RESULTS")
    print("=" * 60)

    total_frames = sum(len(ds) for ds in train_datasets.values())
    print(f"Total frames checked: {total_frames}")

    print(f"\nLeft arm all-zero  [0:5]:  {len(zero_left_indices)} frames")
    print(f"Right arm all-zero [7:12]: {len(zero_right_indices)} frames")

    if zero_left_indices:
        print("\nFirst 20 left-arm zero indices:")
        for ds_name, idx in zero_left_indices[:20]:
            print(f"  dataset={ds_name}  idx={idx}")

    if zero_right_indices:
        print("\nFirst 20 right-arm zero indices:")
        for ds_name, idx in zero_right_indices[:20]:
            print(f"  dataset={ds_name}  idx={idx}")

    if not zero_left_indices and not zero_right_indices:
        print("\nNo zero-action frames found.")


if __name__ == "__main__":
    main()
