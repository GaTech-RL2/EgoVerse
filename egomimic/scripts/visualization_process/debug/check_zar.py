import json
from pathlib import Path

import pandas as pd
import zarr


def main():
    # Default to the outputs produced by `process_image.py`
    data_dir = Path("egomimic/scripts/visualization_process/fold_clothes_aria_eva")
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    print("[INFO] manifest_path =", manifest_path)
    print("[INFO] n_frames      =", manifest["n_frames"])
    print("[INFO] embedding_dim =", manifest["embedding_dim"])
    print("[INFO] embed_store   =", manifest["embed_store"])

    # Load metadata parquet
    meta_path = Path(manifest["metadata_parquet"])
    meta_df = pd.read_parquet(meta_path)
    print("[INFO] metadata rows =", len(meta_df))
    print("[INFO] metadata cols =", len(meta_df.columns))
    # basic columns we expect
    for col in ("global_index", "episode_hash"):
        print("[INFO] has {} = {}".format(col, col in meta_df.columns))
    if len(meta_df) > 0:
        first_row = meta_df.iloc[100].to_dict()
        breakpoint()
        print("[INFO] metadata[0] keys =", sorted(list(first_row.keys()))[:40], "...")
        print("[INFO] metadata[0] =", first_row)

    # Load embeddings zarr for first image key
    first_key = manifest["image_keys"][0]
    zarr_path = Path(manifest["embeddings"][first_key])
    root = zarr.open_group(str(zarr_path), mode="r")
    arr = root["embeddings"]
    print("[INFO] zarr_path     =", zarr_path)
    print("[INFO] zarr array    =", "embeddings")
    print("[INFO] shape/dtype   =", arr.shape, arr.dtype, "chunks=", arr.chunks)

    # Sanity: embeddings rows should match metadata rows for 1:1 alignment
    if arr.shape[0] != len(meta_df):
        raise RuntimeError(
            "Row mismatch: embeddings has {} rows but metadata has {} rows".format(
                arr.shape[0], len(meta_df)
            )
        )

    # Explicitly access a latent (embedding) row.
    # This is the vector aligned with metadata row 0.
    x0 = arr[0, :]  # (D,)
    x_last = arr[arr.shape[0] - 1, :]
    print("[INFO] first latent shape =", getattr(x0, "shape", None), "dtype=", getattr(x0, "dtype", None))
    print("[INFO] last  latent shape =", getattr(x_last, "shape", None), "dtype=", getattr(x_last, "dtype", None))
    # Print only a small slice to keep logs readable
    try:
        x0_slice = x0[:16]
        print("[INFO] latent[0][:16] =", x0_slice)
        # quick stats
        x0_f = x0.astype("float32", copy=False)
        print(
            "[INFO] latent[0] stats min/max/mean =",
            float(x0_f.min()),
            float(x0_f.max()),
            float(x0_f.mean()),
        )

        y = root["tsne_2d"][:10]  # (10, 2)
        print("tsne_2d[:10] =\n", y)
        print("min_xy =", y.min(axis=0), "max_xy =", y.max(axis=0), "mean_xy =", y.mean(axis=0))
    except Exception as e:
        print("[WARN] Could not slice/stats latent[0]:", e)

    # Check global_index alignment (expected: 0..n-1 in this one-batch run)
    if "global_index" in meta_df.columns:
        gi_min = int(meta_df["global_index"].min())
        gi_max = int(meta_df["global_index"].max())
        print("[INFO] global_index min/max =", gi_min, gi_max)


if __name__ == "__main__":
    main()
