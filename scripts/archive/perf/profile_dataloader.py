"""Profile the HPT DataLoader to find the bottleneck.
Measures: zarr open time, image load time, batch collation, GPU transfer."""
import sys
import time
import numpy as np
import zarr
from pathlib import Path
import simplejpeg
import io

sys.path.insert(0, "/coc/flash7/paphiwetsa3/projects/EgoVerse2")

# Test 1: Raw zarr read speed (JPEG vs pre-decoded)
jpeg_dir = Path("/coc/flash7/paphiwetsa3/datasets/new_circle_3")
fast_dir = Path("/coc/flash7/paphiwetsa3/datasets/new_circle_3_fast")

eps_jpeg = sorted([p for p in jpeg_dir.iterdir() if p.name.endswith(".zarr")])[:10]
eps_fast = sorted([p for p in fast_dir.iterdir() if p.name.endswith(".zarr")])[:10]

print("=== Test 1: Single episode load time ===")

# JPEG path
t0 = time.time()
for ep in eps_jpeg[:5]:
    z = zarr.open_group(str(ep), mode="r")
    imgs = z["observations.images.front_img_1"][:]
    state = np.asarray(z["observations.state"][:])
    actions = np.asarray(z["actions"][:])
t1 = time.time()
print(f"JPEG zarr open+read (5 eps): {t1-t0:.2f}s ({(t1-t0)/5:.2f}s/ep)")

# Pre-decoded path
t0 = time.time()
for ep in eps_fast[:5]:
    z = zarr.open_group(str(ep), mode="r")
    imgs = np.asarray(z["observations.images.front_img_1"][:])
    state = np.asarray(z["observations.state"][:])
    actions = np.asarray(z["actions"][:])
t1 = time.time()
print(f"Pre-decoded zarr open+read (5 eps): {t1-t0:.2f}s ({(t1-t0)/5:.2f}s/ep)")

print("\n=== Test 2: JPEG decode time ===")
z = zarr.open_group(str(eps_jpeg[0]), mode="r")
raw = z["observations.images.front_img_1"][:]
n = len(raw)
t0 = time.time()
for jpeg_bytes in raw:
    img = simplejpeg.decode_jpeg(bytes(jpeg_bytes), colorspace="RGB")
t1 = time.time()
print(f"Decode {n} JPEGs: {t1-t0:.2f}s ({(t1-t0)/n*1000:.1f}ms/img)")

print("\n=== Test 3: Zarr open overhead ===")
t0 = time.time()
for ep in eps_jpeg[:50]:
    z = zarr.open_group(str(ep), mode="r")
t1 = time.time()
print(f"Open 50 zarr groups: {t1-t0:.2f}s ({(t1-t0)/50*1000:.1f}ms/open)")

print("\n=== Test 4: Single-frame random access ===")
import random
z = zarr.open_group(str(eps_jpeg[0]), mode="r")
t0 = time.time()
for _ in range(100):
    idx = random.randint(0, n-1)
    s = z["observations.state"][idx:idx+1]
    a = z["actions"][idx:idx+32]
    img = z["observations.images.front_img_1"][idx:idx+1]
t1 = time.time()
print(f"100 random frame reads (JPEG zarr): {t1-t0:.2f}s ({(t1-t0)/100*1000:.1f}ms/frame)")

z = zarr.open_group(str(eps_fast[0]), mode="r")
t0 = time.time()
for _ in range(100):
    idx = random.randint(0, n-1)
    s = z["observations.state"][idx:idx+1]
    a = z["actions"][idx:idx+32]
    img = z["observations.images.front_img_1"][idx:idx+1]
t1 = time.time()
print(f"100 random frame reads (pre-decoded zarr): {t1-t0:.2f}s ({(t1-t0)/100*1000:.1f}ms/frame)")

print("\n=== Test 5: DataLoader benchmark ===")
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset, LocalEpisodeResolver
from egomimic.rldb.embodiment.pushshapes import get_keymap
from torch.utils.data import DataLoader

keymap = get_keymap(action_horizon=32)
resolver = LocalEpisodeResolver(
    folder_path=str(jpeg_dir),
    key_map=keymap,
    transform_list=None,
)
ds = MultiDataset._from_resolver(resolver=resolver, mode="total", skip_bounds_check=True)
print(f"Dataset size: {len(ds)} samples")

loader = DataLoader(ds, batch_size=256, num_workers=4, pin_memory=True,
                    persistent_workers=True, prefetch_factor=2)
t0 = time.time()
for i, batch in enumerate(loader):
    if i >= 10:
        break
t1 = time.time()
print(f"10 batches (JPEG, 4 workers, bs=256): {t1-t0:.2f}s ({(t1-t0)/10:.2f}s/batch)")

resolver2 = LocalEpisodeResolver(
    folder_path=str(fast_dir),
    key_map=keymap,
    transform_list=None,
)
ds2 = MultiDataset._from_resolver(resolver=resolver2, mode="total", skip_bounds_check=True)
loader2 = DataLoader(ds2, batch_size=256, num_workers=4, pin_memory=True,
                     persistent_workers=True, prefetch_factor=2)
t0 = time.time()
for i, batch in enumerate(loader2):
    if i >= 10:
        break
t1 = time.time()
print(f"10 batches (pre-decoded, 4 workers, bs=256): {t1-t0:.2f}s ({(t1-t0)/10:.2f}s/batch)")
