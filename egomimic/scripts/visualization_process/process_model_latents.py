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
from typing import Any, Dict, List, Union

import hydra
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from egomimic.models.denoising_policy import DenoisingPolicy
from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.rldb.utils import S3RLDBDataset, MultiRLDBDataset, get_embodiment, get_embodiment_id
from egomimic.utils.aws.aws_sql import create_default_engine, episode_table_to_df

from PIL import Image


def _as_embodiment_id(v) -> int:
    if isinstance(v, (int, np.integer)):
        return int(v)
    return int(get_embodiment_id(str(v)))

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


def save_debug_image(img, out_path: str | Path) -> None:
    """
    Save a single image to disk for debugging.

    Accepts:
    - torch.Tensor: (C,H,W) or (H,W,C), float in [0,1] or uint8 in [0,255]
    - np.ndarray:   (C,H,W) or (H,W,C), float in [0,1] or uint8 in [0,255]
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if torch.is_tensor(img):
        x = img.detach().cpu()
        if x.ndim == 3 and x.shape[0] in (1, 3) and x.shape[-1] not in (1, 3):
            x = x.permute(1, 2, 0)  # CHW -> HWC
        x = x.numpy()
    else:
        x = np.asarray(img)

    if x.ndim != 3:
        raise ValueError(f"Expected 3D image, got shape={x.shape}")

    # float -> uint8
    if x.dtype != np.uint8:
        x = np.clip(x, 0.0, 1.0)
        x = (x * 255.0 + 0.5).astype(np.uint8)

    # grayscale -> RGB
    if x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)

    Image.fromarray(x).save(str(out_path))
            
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


def _load_hpt_policy(policy_path: str):
    policy = ModelWrapper.load_from_checkpoint(policy_path, weights_only=False)
    if getattr(policy.model, "diffusion", False):
        for head in policy.model.nets.policy.heads:
            if isinstance(policy.model.nets.policy.heads[head], DenoisingPolicy):
                policy.model.nets.policy.heads[head].num_inference_steps = 10
    return policy


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

def _recursive_to_device(
    x, device: Union[str, torch.device], *, non_blocking: bool = True
):
    """
    Recursively move all torch.Tensors in a nested batch to `device`.

    Supports dict / list / tuple nesting (e.g. CombinedLoader batches).
    Non-tensor leaves are returned as-is.
    """
    if torch.is_tensor(x):
        return x.to(device, non_blocking=non_blocking)
    if isinstance(x, dict):
        return {k: _recursive_to_device(v, device, non_blocking=non_blocking) for k, v in x.items()}
    if isinstance(x, tuple):
        return tuple(_recursive_to_device(v, device, non_blocking=non_blocking) for v in x)
    if isinstance(x, list):
        return [_recursive_to_device(v, device, non_blocking=non_blocking) for v in x]
    return x

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
        "--policy-path",
        type=str,
        default="",
        help="Path to the policy checkpoint.",
    )
    ap.add_argument("--batch-size", type=int, default=128)
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

    #Policy path
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    policy = _load_hpt_policy(
    args.policy_path
    )
    policy.model.nets.to(device)
    
    test_loader = DataLoader(dataset_dict[dataset_names[0]], batch_size=32, shuffle=False, num_workers=8)
    test_batch = next(iter(test_loader))
    test_batch = _recursive_to_device(test_batch, device)
    test_batch = {5: test_batch}
    processed_batch = policy.model.process_batch_for_training(test_batch)
    preds = policy.model.forward_eval(processed_batch)
    cond, embodiment = policy.model.nets['policy'].trunk_features['aria_bimanual']
    cond_flattened = cond.reshape(cond.shape[0], -1)
    emb_dim = cond_flattened.shape[1]

    # Storage setup
    embed_dtype = np.float16 if args.embed_dtype == "float16" else np.float32
    embed_path = None
    embed_writer = None
    safe_name = "cond"
    if args.embed_store == "npy":
        embed_path = out_dir / f"embeddings__{safe_name}.npy"
        embed_writer = np.memmap(embed_path, mode="w+", dtype=embed_dtype, shape=(n_total, emb_dim))
    else:
        # zarr
        try:
            import zarr
        except Exception as e:
            raise RuntimeError(
                "zarr is not installed but --embed-store=zarr was requested. "
                "Either install zarr (pip install zarr numcodecs) or use --embed-store npy."
            ) from e
        embed_path = out_dir / f"embeddings__{safe_name}.zarr"
        root = zarr.open_group(str(embed_path), mode="w")
        chunks = (min(args.chunk_size, n_total), emb_dim)
        root.create_dataset(
            "embeddings",
            shape=(n_total, emb_dim),
            chunks=chunks,
            dtype=embed_dtype,
            overwrite=True,
        )
        embed_writer = root["embeddings"]

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
        )

        for batch_idx, batch_samples in enumerate(loader):
            start = batch_idx * bs
            # last batch can be smaller
            batch_size_i = int(
                next(v for v in batch_samples.values() if torch.is_tensor(v) and v.ndim >= 1).shape[0]
            )
            end = start + batch_size_i
            global_start = offset + start
            global_end = offset + end
            
            # One metadata row per sample, aligned to keep_idx/subset positions
            for j in range(batch_size_i):
                subset_pos = start + j
                orig_ds_idx = keep_idx[subset_pos]  # index into original ds
                # only get the metadata from the index map
                idx_map_name, _ = ds.index_map[int(orig_ds_idx)]
                if isinstance(idx_map_name, MultiRLDBDataset):
                    raise ValueError("idx_map_name is a MultiRLDBDataset, which is not supported")
                ep_hash = str(idx_map_name)
                row = {"episode_hash": ep_hash, "dataset_local_index": int(orig_ds_idx), "dataset_name": dataset_name}
                db_row = episode_meta_by_hash.get(ep_hash)
                if db_row:
                    for k, v in db_row.items():
                        if k != "episode_hash":
                            row[k] = v
                meta_rows.append(row)

            # embeddings
            batch = _recursive_to_device(batch_samples, device)
            batch = {5: batch}
            processed_batch = policy.model.process_batch_for_training(batch)
            preds = policy.model.forward_eval(processed_batch)
            cond, embodiment = policy.model.nets['policy'].trunk_features['aria_bimanual']
            cond_flattened = cond.reshape(cond.shape[0], -1).detach().cpu().numpy().astype(np.float16, copy=False)
            writer = embed_writer
            writer[global_start:global_end, :] = cond_flattened

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
        if isinstance(embed_writer, np.memmap):
            embed_writer.flush()

    # Write metadata
    meta_df = pd.DataFrame(meta_rows)
    meta_path = out_dir / "metadata.parquet"
    meta_df.to_parquet(meta_path, index=False)

    # Small manifest
    manifest = {
        "n_frames": n_total,
        "embedding_dim": emb_dim,
        "embed_store": args.embed_store,
        "embed_dtype": args.embed_dtype,
        "every_k_datapoint": int(args.every_k_datapoint),
        "embeddings": {safe_name: str(embed_path)},
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
    print(f"[DONE] Wrote embeddings: {embed_path}")
    print(f"[DONE] Wrote manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()


