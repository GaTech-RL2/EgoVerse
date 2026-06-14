"""Quick test: DataLoader speed with different worker counts."""
import sys
import time
sys.path.insert(0, "/coc/flash7/paphiwetsa3/projects/EgoVerse2")

from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset, LocalEpisodeResolver
from egomimic.rldb.embodiment.pushshapes import get_keymap
from torch.utils.data import DataLoader

jpeg_dir = "/coc/flash7/paphiwetsa3/datasets/new_circle_3"
keymap = get_keymap(action_horizon=32)
resolver = LocalEpisodeResolver(folder_path=jpeg_dir, key_map=keymap, transform_list=None)
ds = MultiDataset._from_resolver(resolver=resolver, mode="total", skip_bounds_check=True)
print(f"Dataset: {len(ds)} samples")

for nw in [1, 4, 8, 12]:
    loader = DataLoader(ds, batch_size=256, num_workers=nw, pin_memory=True,
                        persistent_workers=True if nw > 0 else False, prefetch_factor=4 if nw > 0 else None)
    # warmup
    it = iter(loader)
    _ = next(it)
    # measure
    t0 = time.time()
    for i in range(20):
        _ = next(it)
    t1 = time.time()
    print(f"  workers={nw:2d}: {(t1-t0)/20:.3f}s/batch → {(t1-t0)/20*1008:.0f}s/epoch ({(t1-t0)/20*1008/60:.1f}min)")
    del loader
