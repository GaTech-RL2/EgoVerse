"""
Run dimensionality reduction on saved embedding latents and store 2D coords back into the zarr.

Reads:
- manifest.json (to find the embeddings zarr path)
- embeddings zarr group (expects dataset name "embeddings")

Writes:
- dataset "<method>_2d" into the same zarr group by default (tsne_2d/umap_2d/pca_2d),
  shape (N, 2), float32
"""

import argparse
import json
from pathlib import Path

import numpy as np
import zarr


def _load_embeddings(zarr_path: Path) -> np.ndarray:
    root = zarr.open_group(str(zarr_path), mode="r")
    if "embeddings" not in root:
        raise KeyError(
            "Expected dataset 'embeddings' in zarr group. Found keys: {}".format(
                list(root.array_keys())
            )
        )
    arr = root["embeddings"]
    # Load entire array into memory for t-SNE
    x = arr[:]
    print("x.shape =", x.shape)
    # cuML prefers float32
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    return x


def _to_numpy(x):
    # Convert cupy -> numpy if needed
    try:
        import cupy as cp

        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
    except Exception:
        pass
    return np.asarray(x)


def _run_cuml_tsne(
    x: np.ndarray, *, perplexity: float, random_state: int, learning_rate: float
) -> np.ndarray:
    try:
        from cuml import TSNE
    except Exception as e:
        raise RuntimeError(
            "cuml is required. Make sure RAPIDS/cuML is installed in this environment."
        ) from e

    # cuML TSNE returns a (N, 2) array-like (often cupy-backed); convert to numpy.
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        init="random",
        # NOTE: scikit-learn supports learning_rate="auto", but cuML expects numeric.
        learning_rate=float(learning_rate),
    )
    y = tsne.fit_transform(x)

    y = _to_numpy(y)
    if y.ndim != 2 or y.shape[1] != 2:
        raise RuntimeError("Unexpected TSNE output shape: {}".format(y.shape))
    return y.astype(np.float32, copy=False)


def _run_cuml_umap(
    x: np.ndarray, *, n_neighbors: int, min_dist: float, metric: str, random_state: int
) -> np.ndarray:
    try:
        from cuml import UMAP
    except Exception as e:
        raise RuntimeError(
            "cuml is required for UMAP. Make sure RAPIDS/cuML is installed in this environment."
        ) from e

    umap = UMAP(
        n_components=2,
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        metric=str(metric),
        random_state=int(random_state),
    )
    y = umap.fit_transform(x)
    y = _to_numpy(y)
    if y.ndim != 2 or y.shape[1] != 2:
        raise RuntimeError("Unexpected UMAP output shape: {}".format(y.shape))
    return y.astype(np.float32, copy=False)


def _run_pca(x: np.ndarray, *, n_components: int, random_state: int) -> np.ndarray:
    # Prefer GPU PCA if available; otherwise fall back to sklearn.
    try:
        from cuml import PCA  # type: ignore

        pca = PCA(n_components=int(n_components), random_state=int(random_state))
        y = pca.fit_transform(x)
        y = _to_numpy(y)
    except Exception:
        try:
            from sklearn.decomposition import PCA  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "PCA requires either cuML (preferred) or scikit-learn installed."
            ) from e

        pca = PCA(n_components=int(n_components), random_state=int(random_state))
        y = pca.fit_transform(x)
        y = np.asarray(y)

    if y.ndim != 2 or y.shape[1] != int(n_components):
        raise RuntimeError("Unexpected PCA output shape: {}".format(y.shape))
    if y.shape[1] != 2:
        raise RuntimeError("This script only supports 2D outputs; got PCA dim {}".format(y.shape[1]))
    return y.astype(np.float32, copy=False)


def _write_2d(zarr_path: Path, *, y2d: np.ndarray, name: str, overwrite: bool) -> None:
    root = zarr.open_group(str(zarr_path), mode="a")

    if name in root and not overwrite:
        raise FileExistsError(
            "Zarr dataset '{}' already exists at {}. Use --overwrite to replace.".format(
                name, zarr_path
            )
        )

    chunks = (min(8192, y2d.shape[0]), 2)
    root.create_dataset(
        name,
        shape=y2d.shape,
        chunks=chunks,
        dtype=np.float32,
        overwrite=overwrite,
    )
    root[name][:] = y2d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        type=str,
        default="egomimic/scripts/visualization_process/fold_clothes_aria_eva_all_labs/manifest.json",
        help="Path to manifest.json produced by process_image.py",
    )
    ap.add_argument(
        "--image-key",
        type=str,
        default="",
        help="Optional image key to select from manifest['embeddings'] (defaults to first).",
    )
    ap.add_argument(
        "--method",
        type=str,
        default="tsne",
        choices=("tsne", "umap", "pca"),
        help="Dimensionality reduction method.",
    )
    ap.add_argument(
        "--out-name",
        type=str,
        default="",
        help="Dataset name to write in zarr. Defaults to <method>_2d.",
    )
    # TSNE args
    ap.add_argument("--perplexity", type=float, default=30.0, help="TSNE perplexity (tsne only).")
    ap.add_argument(
        "--learning-rate",
        type=float,
        default=200.0,
        help="cuML TSNE learning rate (tsne only; must be numeric).",
    )
    # UMAP args
    ap.add_argument("--n-neighbors", type=int, default=15, help="UMAP n_neighbors (umap only).")
    ap.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist (umap only).")
    ap.add_argument("--metric", type=str, default="euclidean", help="UMAP metric (umap only).")
    # PCA args
    ap.add_argument("--pca-components", type=int, default=2, help="PCA n_components (pca only).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())

    if manifest.get("embed_store") != "zarr":
        raise RuntimeError("This script expects manifest embed_store == 'zarr'.")

    if args.image_key:
        image_key = args.image_key
    else:
        image_key = manifest["image_keys"][0]

    zarr_path = Path(manifest["embeddings"][image_key])
    print("[INFO] zarr_path =", zarr_path)
    print("[INFO] reading embeddings for key =", image_key)

    x = _load_embeddings(zarr_path)
    print("[INFO] embeddings shape/dtype =", x.shape, x.dtype)

    if args.out_name:
        out_name = args.out_name
    else:
        out_name = f"{args.method}_2d"

    if args.method == "tsne":
        y2d = _run_cuml_tsne(
            x, perplexity=args.perplexity, random_state=args.seed, learning_rate=args.learning_rate
        )
    elif args.method == "umap":
        y2d = _run_cuml_umap(
            x,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric,
            random_state=args.seed,
        )
    elif args.method == "pca":
        if int(args.pca_components) != 2:
            raise ValueError("--pca-components must be 2 for this script (got {})".format(args.pca_components))
        y2d = _run_pca(x, n_components=args.pca_components, random_state=args.seed)
    else:
        raise RuntimeError("Unsupported method: {}".format(args.method))

    print("[INFO] {} shape/dtype =".format(out_name), y2d.shape, y2d.dtype)

    _write_2d(zarr_path, y2d=y2d, name=out_name, overwrite=args.overwrite)
    print("[DONE] wrote {} into {}".format(out_name, zarr_path))


if __name__ == "__main__":
    main()
