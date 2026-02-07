"""
create_dino.py
==============

Downloads RLDB datasets via `S3RLDBDataset`, writes a metadata table (Parquet),
and computes image embeddings using a DINO-family model (defaults to DINOv2 via torch.hub).

Outputs (by default) to:
- metadata parquet: <out_dir>/metadata.parquet
- embeddings:       <out_dir>/embeddings.zarr  (or .npy memmap)

Notes
-----
- "DINOv3" is not guaranteed to be available via torch.hub. This script will try to
  load the requested hub repo, and falls back to DINOv2 if that fails.
- The RLDB datasets are HuggingFace / LeRobot datasets. Instantiating S3RLDBDataset
  will sync needed episode folders locally.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import hydra
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from egomimic.rldb.utils import S3RLDBDataset, MultiRLDBDataset
from egomimic.algo.hpt import DinoV3
from egomimic.utils.aws.aws_sql import create_default_engine, episode_table_to_df


def _parse_json_or_empty(s: str) -> dict:
    if not s:
        return {}
    return json.loads(s)


def _safe_get(d: dict, k: str, default=None):
    try:
        return d.get(k, default)
    except Exception:
        return default


def _image_to_torch_uint8_bchw(x) -> torch.Tensor:
    """
    Accept common dataset image formats:
    - numpy uint8: HWC or CHW
    - torch uint8/float: HWC/CHW/BCHW/BHWC
    Returns uint8 BCHW.
    """
    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.from_numpy(np.asarray(x))

    if t.ndim == 3:
        # HWC or CHW -> add batch
        t = t.unsqueeze(0)
    if t.ndim != 4:
        raise ValueError(f"Expected 3D/4D image tensor/array, got shape={tuple(t.shape)}")

    # If last dim looks like channels -> BHWC -> BCHW
    if t.shape[-1] in (1, 3) and t.shape[1] not in (1, 3):
        t = t.permute(0, 3, 1, 2).contiguous()
    # Else assume already BCHW (or ambiguous)

    if t.dtype != torch.uint8:
        # If floats in [0,1] or [-1,1], bring to uint8 best-effort
        if t.is_floating_point():
            t = t.to(torch.float32)
            t = torch.clamp(t, 0.0, 1.0) if t.max() <= 1.0 else torch.clamp(t, -1.0, 1.0) * 0.5 + 0.5
            t = torch.round(t * 255.0).to(torch.uint8)
        else:
            t = t.to(torch.uint8)
    return t


def _bchw_u8_to_list_hwc_u8(img_bchw_u8: torch.Tensor) -> List[np.ndarray]:
    """
    Convert uint8 BCHW torch tensor to a list of uint8 HWC numpy arrays.
    """
    if img_bchw_u8.ndim != 4:
        raise ValueError(f"Expected BCHW, got {tuple(img_bchw_u8.shape)}")
    if img_bchw_u8.dtype != torch.uint8:
        raise ValueError(f"Expected uint8 image tensor, got {img_bchw_u8.dtype}")
    x = img_bchw_u8.permute(0, 2, 3, 1).contiguous().cpu().numpy()  # BHWC uint8
    return [x[i] for i in range(x.shape[0])]


def _load_hpt_dinov3(model_id: str, *, output_dim: int, device: str):
    """
    Load DINOv3 backbone via `DinoV3` from egomimic's HPT code, but keep a HF
    `AutoImageProcessor` for correct pixel preprocessing.

    `DinoV3.forward` expects input shaped [B, T, N, 3, H, W] and returns projected
    token embeddings shaped [(B*T*N), num_tokens, output_dim].
    """
    try:
        from transformers import AutoImageProcessor
    except Exception as e:
        raise RuntimeError(
            "Transformers is required for HF DINO models. Install with `pip install transformers`."
        ) from e

    processor = AutoImageProcessor.from_pretrained(model_id)
    stem = DinoV3(output_dim=output_dim, model_type=model_id, freeze_backbone=True)
    stem.eval()
    stem.to(device)
    return processor, stem


@torch.no_grad()
def _embed_batch_dinov3(processor, stem: torch.nn.Module, images_hwc_u8: List[np.ndarray], device: str) -> torch.Tensor:
    """
    Returns (B, D) embeddings (CLS token) using:
    - HF processor -> pixel_values (B,C,H,W)
    - DinoV3 stem  -> token embeddings, then take token 0 (CLS)
    """
    inputs = processor(images=images_hwc_u8, return_tensors="pt")
    if "pixel_values" not in inputs:
        raise RuntimeError("HF processor did not return 'pixel_values'.")
    pixel_values = inputs["pixel_values"].to(device)  # (B,C,H,W), float
    # DinoV3 expects [B, T, N, C, H, W]; we treat each frame as T=1, N=1
    x = pixel_values.unsqueeze(1).unsqueeze(2)
    tok = stem(x)  # (B, num_tokens, D) because B*T*N == B
    if tok.ndim != 3:
        raise RuntimeError(f"Unexpected DinoV3 output shape: {tuple(tok.shape)}")
    return tok[:, 0, :]  # CLS token


def _flatten_metadata(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make a metadata dict that is safe for DataFrame/Parquet.
    We keep common RLDB/LeRobot fields if present and also include any `metadata.*` keys.
    """
    out: Dict[str, Any] = {}

    # Common keys we expect in LeRobot datasets
    for k in ("episode_index", "frame_index", "timestamp", "annotations", "task", "task_description"):
        if k in sample:
            out[k] = sample[k]

    # Some datasets include these:
    for k in ("dataset_index", "index", "step", "episode_id"):
        if k in sample and k not in out:
            out[k] = sample[k]

    # Include all metadata.* keys (e.g. metadata.embodiment, etc.)
    for k, v in sample.items():
        if isinstance(k, str) and k.startswith("metadata."):
            out[k] = v

    # Make sure tensors/numpy become scalars where appropriate
    for k, v in list(out.items()):
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                out[k] = v.item()
            else:
                out[k] = v.detach().cpu().numpy()
        elif isinstance(v, np.ndarray):
            if v.size == 1:
                out[k] = v.item()
    return out


def _py_scalar(v: Any) -> Any:
    """Best-effort conversion for pandas/numpy scalars for Parquet friendliness."""
    try:
        import pandas as _pd  # local import to avoid hard dependency patterns

        if isinstance(v, _pd.Timestamp):
            return v.isoformat()
    except Exception:
        pass

    # numpy scalar -> python scalar
    try:
        if isinstance(v, np.generic):
            return v.item()
    except Exception:
        pass

    return v


def _instantiate_hydra(cfg_path: str):
    """
    Instantiate a dataset from a Hydra-style YAML config.

    Example:
      cfg_path="egomimic/hydra_configs/data/viz_data.yaml"
    """
    try:
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
    except Exception as e:
        raise RuntimeError(
            "Hydra instantiation requires `hydra-core` and `omegaconf`."
        ) from e

    cfg = OmegaConf.load(cfg_path)
    return instantiate(cfg)


def _ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--embodiment",
        type=str,
        default="",
        help='e.g. "eva_right_arm" or "eva_bimanual". Required unless using --data-config.',
    )
    ap.add_argument("--out-dir", type=str, default="egomimic/scripts/visualization_process/data2")
    ap.add_argument(
        "--data-config",
        type=str,
        default="",
        help=(
            "Optional Hydra YAML path for dataset instantiation, e.g. "
            "egomimic/hydra_configs/data/viz_data.yaml. If set, the dataset is "
            "created via hydra `instantiate()` from --data-split/--dataset-name."
        ),
    )
    ap.add_argument(
        "--image-keys",
        type=str,
        nargs="+",
        default=["observations.images.front_img_1"],
        help="LeRobot image keys to embed (can pass multiple).",
    )
    ap.add_argument(
        "--model",
        type=str,
        default="facebook/dinov3-vitl16-pretrain-lvd1689m",
        help="HuggingFace model id for DINO (e.g. facebook/dinov3-vitl16-pretrain-lvd1689m).",
    )
    ap.add_argument("--batch-size", type=int, default=240)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num-frames", type=int, default=-1, help="Limit number of frames for debugging")
    ap.add_argument(
        "--every-k-datapoint",
        type=int,
        default=15,
        help="Keep only every k-th datapoint (0,k,2k,...) to reduce compute. Set to 1 to keep all.",
    )
    ap.add_argument(
        "--debug-first-batch",
        action="store_true",
        help="Process/save only the first batch, then exit (useful for debugging).",
    )

    ap.add_argument(
        "--embed-store",
        type=str,
        default="zarr",
        choices=["zarr", "npy"],
        help="Embedding storage format. zarr is chunked; npy is memmap.",
    )
    ap.add_argument("--embed-dtype", type=str, default="float16", choices=["float16", "float32"])
    ap.add_argument("--chunk-size", type=int, default=8192, help="Write chunk size for zarr")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    _ensure_out_dir(out_dir)


    # Dataset instantiation
    # If --data-config is provided, treat it as a MultiDataModuleWrapper-style config
    # and ONLY use its train_datasets (ignore valid_datasets entirely).
    dataset_dict: Dict[str, torch.utils.data.Dataset] = {}
    if args.data_config:
        cfg = OmegaConf.load(args.data_config)
        cfg_data = cfg.data if ("data" in cfg and "train_datasets" in cfg.data) else cfg
        if "train_datasets" not in cfg_data:
            raise KeyError(
                "Expected 'train_datasets' in --data-config (or in data.train_datasets)."
            )
        for dataset_name, ds_cfg in cfg_data.train_datasets.items():
            dataset_dict[str(dataset_name)] = hydra.utils.instantiate(ds_cfg)
    else:
        # CLI-configured dataset; instantiation triggers S3 sync + local load.
        if not args.embodiment:
            raise ValueError("--embodiment is required when not using --data-config")
        filters = {"task": "fold_clothes"}
        ds = S3RLDBDataset(embodiment=args.embodiment, mode="total", filters=filters)
        dataset_dict[str(args.embodiment)] = ds

    if not dataset_dict:
        raise RuntimeError("No datasets were instantiated.")

    dataset_names = list(dataset_dict.keys())

    # Compute effective per-dataset lengths + global offsets into the shared embedding array
    per_dataset_n: Dict[str, int] = {}
    per_dataset_offset: Dict[str, int] = {}
    per_dataset_keep_indices: Dict[str, List[int]] = {}
    running = 0
    k_stride = int(args.every_k_datapoint)
    if k_stride <= 0:
        k_stride = 1
    for dataset_name in dataset_names:
        ds_i = dataset_dict[dataset_name]
        n_i = len(ds_i)
        if args.num_frames > 0:
            n_i = min(n_i, args.num_frames)
        if args.debug_first_batch:
            n_i = min(n_i, args.batch_size)
        keep_idx = list(range(0, n_i, k_stride))
        per_dataset_keep_indices[dataset_name] = keep_idx
        per_dataset_offset[dataset_name] = running
        per_dataset_n[dataset_name] = len(keep_idx)
        running += len(keep_idx)

    n_total = running
    print(
        "[INFO] Using {} train datasets; total frames to process = {}".format(
            len(dataset_names), n_total
        )
    )

    # Model (HPT DinoV3 stem + HF processor)
    # If the CLI flag was removed, default to 1024 (common for ViT-L features).
    dino_output_dim = getattr(args, "dino_output_dim", 1024)
    processor, stem = _load_hpt_dinov3(
        args.model, output_dim=dino_output_dim, device=args.device
    )

    # Probe embedding dim
    first = dataset_dict[dataset_names[0]][0]
    probe_key = args.image_keys[0]
    if probe_key not in first:
        raise KeyError(
            f"Image key '{probe_key}' not found in sample. Available keys (truncated): {list(first.keys())[:30]}"
        )
    probe_img = _image_to_torch_uint8_bchw(first[probe_key])  # uint8 BCHW
    probe_list = _bchw_u8_to_list_hwc_u8(probe_img)
    probe_emb = _embed_batch_dinov3(processor, stem, probe_list, args.device)
    emb_dim = int(probe_emb.shape[-1])
    print(f"[INFO] Embedding dim: {emb_dim}")

    embed_dtype = np.float16 if args.embed_dtype == "float16" else np.float32

    # Storage setup
    embed_paths = {}
    embed_writers = {}
    for k in args.image_keys:
        safe_name = k.replace("/", "_").replace(".", "_")
        if args.embed_store == "npy":
            path = out_dir / f"embeddings__{safe_name}.npy"
            arr = np.memmap(path, mode="w+", dtype=embed_dtype, shape=(n_total, emb_dim))
            embed_paths[k] = path
            embed_writers[k] = arr
        else:
            # zarr
            try:
                import zarr
            except Exception as e:
                raise RuntimeError(
                    "zarr is not installed but --embed-store=zarr was requested. "
                    "Either install zarr (pip install zarr numcodecs) or use --embed-store npy."
                ) from e
            path = out_dir / f"embeddings__{safe_name}.zarr"
            root = zarr.open_group(str(path), mode="w")
            # Chunk over first dim
            chunks = (min(args.chunk_size, n_total), emb_dim)
            root.create_dataset(
                "embeddings",
                shape=(n_total, emb_dim),
                chunks=chunks,
                dtype=embed_dtype,
                overwrite=True,
            )
            embed_paths[k] = path
            embed_writers[k] = root["embeddings"]

    # Metadata rows (we’ll write parquet at the end; for huge datasets you can switch to incremental writing)
    meta_rows: List[Dict[str, Any]] = []

    engine = create_default_engine()
    df = episode_table_to_df(engine)
    # Cache episode-level DB metadata by episode_hash for fast per-frame lookup.
    # We prefix these keys as "db.*" when writing per-frame metadata rows.
    episode_meta_by_hash: Dict[str, Dict[str, Any]] = {}
    if "episode_hash" in df.columns:
        df_unique = df.drop_duplicates(subset=["episode_hash"])
        for row in df_unique.to_dict(orient="records"):
            ep_hash = row.get("episode_hash")
            if ep_hash is None:
                continue
            # store sanitized scalars
            episode_meta_by_hash[str(ep_hash)] = {k: _py_scalar(v) for k, v in row.items()}

    # Batch loop across train datasets, writing into one shared embeddings array per image key
    bs = args.batch_size
    processed = 0
    for dataset_name in dataset_names:
        ds = dataset_dict[dataset_name]
        keep_idx = per_dataset_keep_indices[dataset_name]
        n_eff = len(keep_idx)
        offset = per_dataset_offset[dataset_name]

        # Only load/process the kept indices (efficient: filters before model forward)
        ds_for_loader = ds if (n_eff == len(ds) and keep_idx == list(range(len(ds)))) else Subset(ds, keep_idx)
        loader = DataLoader(
            ds_for_loader,
            batch_size=bs,
            shuffle=False,
            num_workers=8,
            collate_fn=lambda batch: batch,  # keep list[dict] (no tensor stacking)
        )

        for batch_idx, batch_samples in enumerate(loader):
            start = batch_idx * bs
            end = start + len(batch_samples)
            global_start = offset + start
            global_end = offset + end

            # metadata
            for i, sample in enumerate(batch_samples):
                m = _flatten_metadata(sample)
                m["dataset_name"] = dataset_name
                m["dataset_offset"] = offset
                # Index within the Subset (i.e., after every-k subsample), then map back
                # to the original dataset index.
                subset_pos = batch_idx * bs + i  # == start + i
                orig_ds_idx = keep_idx[subset_pos] if subset_pos < len(keep_idx) else subset_pos
                m["dataset_local_index"] = int(orig_ds_idx)
                m["embedding_global_index"] = int(global_start + i)

                # Per-sample index_map lookup (batch may span multiple episodes).
                try:
                    idx_map_name, _ = ds.index_map[int(orig_ds_idx)]
                    if isinstance(idx_map_name, MultiRLDBDataset):
                        raise ValueError("idx_map_name is a MultiRLDBDataset, which is not supported")
                    ep_hash = str(idx_map_name)
                    m["episode_hash"] = ep_hash

                    # Attach episode-level DB metadata (same for all frames in an episode)
                    db_row = episode_meta_by_hash.get(ep_hash)
                    if db_row:
                        for k, v in db_row.items():
                            if k == "episode_hash":
                                continue
                            m[str(k)] = v
                except Exception:
                    pass

                meta_rows.append(m)

            # embeddings per image key
            for key in args.image_keys:
                imgs_bchw = []
                for sample in batch_samples:
                    if key not in sample:
                        raise KeyError(
                            f"Missing image key '{key}' in sample. Keys: {list(sample.keys())[:30]}"
                        )
                    imgs_bchw.append(_image_to_torch_uint8_bchw(sample[key]))
                img_bchw = torch.cat(imgs_bchw, dim=0)  # uint8 BCHW on CPU
                images_hwc = _bchw_u8_to_list_hwc_u8(img_bchw)  # list[np.uint8 HWC]
                emb_t = _embed_batch_dinov3(processor, stem, images_hwc, args.device)
                emb = emb_t.detach().cpu().numpy().astype(embed_dtype, copy=False)

                writer = embed_writers[key]
                writer[global_start:global_end, :] = emb

            processed = global_end
            if (processed // bs) % 10 == 0:
                print(f"[INFO] Processed {processed}/{n_total}")

            if args.debug_first_batch:
                print("[DEBUG] Exiting after first batch (--debug-first-batch).")
                break

        if args.debug_first_batch:
            break

    # Finalize memmaps
    if args.embed_store == "npy":
        for k, arr in embed_writers.items():
            if isinstance(arr, np.memmap):
                arr.flush()

    # Write metadata
    meta_df = pd.DataFrame(meta_rows)
    meta_path = out_dir / "metadata.parquet"
    meta_df.to_parquet(meta_path, index=False)

    # Small manifest
    manifest = {
        "n_frames": n_total,
        "embedding_dim": emb_dim,
        "image_keys": list(args.image_keys),
        "embed_store": args.embed_store,
        "embed_dtype": args.embed_dtype,
        "every_k_datapoint": int(args.every_k_datapoint),
        "embeddings": {k: str(p) for k, p in embed_paths.items()},
        "metadata_parquet": str(meta_path),
        "datasets": {
            name: {
                "n_frames": int(per_dataset_n[name]),
                "offset": int(per_dataset_offset[name]),
            }
            for name in dataset_names
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"[DONE] Wrote metadata: {meta_path}")
    for k, p in embed_paths.items():
        print(f"[DONE] Wrote embeddings for {k}: {p}")
    print(f"[DONE] Wrote manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()


