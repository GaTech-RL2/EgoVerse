"""
Smoke test: end-to-end norm-stats collection from the packed pushshapes
dataset, mirroring what trainHydra runs at startup.

Verifies that:
  - MultiDataset._iter_leaves descends into ZarrEpisodePackedDataset.
  - populate_from_datasets registers the pushshapes_sim key types.
  - infer_norm_from_dataset uses pack_collate (default_collate would crash
    on variable-length episodes) and reinterprets sample_frac as a frame
    count in packed mode.

Run:
    python scripts/smoke_packed_norm_stats.py
"""

import logging
import sys
import time

from egomimic.rldb.embodiment.pushshapes import get_keymap
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset
from egomimic.rldb.zarr.zarr_dataset_packed import ZarrEpisodePackedDataset

logging.basicConfig(level=logging.WARNING, format="%(message)s", stream=sys.stderr)


def main():
    t0 = time.perf_counter()
    ds = ZarrEpisodePackedDataset.from_local_folder(
        folder_path="/coc/cedarp-dxu345-0/Tsim_datasets2/circle",
        key_map=get_keymap(action_horizon=1024),
        max_seq_len=None,
        chunking="none",
        min_seq_len=64,
        transform_list=None,
    )
    print(
        f"[1] dataset: {len(ds)} entries  ({time.perf_counter()-t0:.1f}s)", flush=True
    )

    t0 = time.perf_counter()
    norm_stats = MultiDataset(state={}, norm_mode="quantile")
    norm_stats.populate_from_datasets({"pushshapes_sim": ds})
    print(
        f"[2] populate: key_types={dict(norm_stats.key_types)}  "
        f"({time.perf_counter()-t0:.1f}s)",
        flush=True,
    )

    t0 = time.perf_counter()
    norm_stats.infer_shapes_from_batch(ds[0])
    print(f"[3] infer_shapes done  ({time.perf_counter()-t0:.1f}s)", flush=True)

    t0 = time.perf_counter()
    norm_stats.infer_norm_from_dataset(
        ds,
        "pushshapes_sim",
        sample_frac=0.05,
        num_workers=0,
        batch_size=4,
    )
    print(f"[4] infer_norm done  ({time.perf_counter()-t0:.1f}s)", flush=True)

    print()
    print("=== Collected stats ===")
    for emb_id, by_key in norm_stats.norm_stats.items():
        if not by_key:
            continue
        print(f"emb={emb_id}")
        for k, stats in by_key.items():
            mean = list(round(float(x), 3) for x in stats["mean"].ravel())
            std = list(round(float(x), 3) for x in stats["std"].ravel())
            print(f"  {k:25s}  mean={mean}  std={std}")


if __name__ == "__main__":
    main()
