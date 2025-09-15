import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from omegaconf import OmegaConf
import hydra
from egomimic.utils.tsne_utils import *
from egomimic.utils.plotting import *
from rldb.utils import *
from egomimic.pl_utils.pl_model import ModelWrapper
from torch.utils.data import DataLoader

# ----------------------------
# Model instantiation (eval-style)
# ----------------------------
def _move_to_device(obj, device, non_blocking=True):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=non_blocking)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device, non_blocking) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = [_move_to_device(v, device, non_blocking) for v in obj]
        return type(obj)(t)
    return obj

def instantiate_model(ckpt_path: str):
    """
    Instantiate LightningModule via Hydra config and load from checkpoint.
    Returns: model (eval mode), cfg
    """
    model = ModelWrapper.load_from_checkpoint(ckpt_path)
    return model

def extract_embeddings_from_datasets(model, dataset_paths, num_batches=64, batch_size=32, embodiment_name="aria_bimanual"):
    print(f"[INFO] Preparing datasets for embodiment={embodiment_name}")
    algo = model.model if hasattr(model, "model") else model
    policy = algo.nets["policy"]
    device = getattr(algo, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Using device: {device}")

    datasets = []
    for dataset_path in dataset_paths:
        print(f"[INFO] Loading dataset: {dataset_path}")
        ds = FolderRLDBDataset(
            folder_path=dataset_path,
            embodiment=embodiment_name,
            mode="total",
            local_files_only=True,
        )
        datasets.append(ds)

    # dataloaders keyed by folder name
    dataloaders = {}
    folder_map = {
        "put_cup_on_saucer_rl2": "rl2",
        "put_cup_on_saucer_eth": "eth",
        "put_cup_on_saucer_wang": "wang",
        "put_cup_on_saucer_song": "song",
    }
    for i, ds in enumerate(datasets):
        base = os.path.basename(dataset_paths[i].rstrip("/"))
        class_name = folder_map.get(base, base)
        print(f"[INFO] Creating DataLoader for class '{class_name}'")
        dataloaders[class_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

    embeddings_dict = {}
    emb_id = get_embodiment_id(embodiment_name)
    cam_keys  = algo.camera_keys[emb_id]
    prop_keys = algo.proprio_keys[emb_id]
    lang_keys = algo.lang_keys[emb_id]
    ac_key    = algo.ac_keys[emb_id]
    aux_keys  = algo.auxiliary_ac_keys.get(embodiment_name, [])

    policy.eval()
    for class_name, loader in dataloaders.items():
        print(f"[INFO] Extracting embeddings for '{class_name}'...")
        feats = []
        with torch.inference_mode():
            for i, batch in enumerate(loader):
                print(f"   [BATCH {i+1}/{num_batches}] Processing...")
                valid_keys = set(cam_keys) | set(prop_keys) | set(lang_keys) | set(aux_keys)
                if isinstance(ac_key, str):
                    valid_keys.add(ac_key)
                else:
                    valid_keys |= set(ac_key)

                raw_by_emb = {emb_id: batch}
                processed = algo.process_batch_for_training(raw_by_emb)
                processed[emb_id]["embodiment"] = torch.tensor([emb_id], dtype=torch.int64)

                hpt_data = algo._robomimic_to_hpt_data(
                    processed[emb_id], cam_keys, prop_keys, lang_keys, ac_key, aux_keys
                )
                device = model.device
                hpt_data = _move_to_device(hpt_data, device)

                proc_tokens, _ = policy.forward_features(embodiment_name, hpt_data)
                feats.append(proc_tokens.detach().cpu())

                if (i + 1) >= num_batches:
                    print(f"   [INFO] Reached max batches ({num_batches}) for '{class_name}'.")
                    break

        if feats:
            embeddings_dict[class_name] = torch.cat(feats, dim=0)
            print(f"[INFO] Done. Collected {embeddings_dict[class_name].shape[0]} embeddings for '{class_name}'.")
        else:
            print(f"[WARN] No embeddings extracted for '{class_name}'.")

    return embeddings_dict


def tsne_from_class_embeddings(class_to_tensor: dict, method="tsne", pool="avg",
                               title="t-SNE — N-class", out_png="tsne.png", grid_size=10):
    print("[INFO] Running dimensionality reduction...")
    Z, y, names = reduce_many(class_to_tensor, method=method, pool=pool, random_state=42)
    print("[INFO] Reduction complete. Plotting...")
    img = plot_many(Z, y, names, title=title, grid_size=grid_size)

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    import imageio
    imageio.imwrite(out_png, img)
    print(f"[OK] t-SNE plot saved to {out_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="Hydra config with `model:` node")
    parser.add_argument("--ckpt", type=str, help="Lightning checkpoint")
    parser.add_argument("--override", type=str, nargs="*", default=[], help="Hydra overrides key=val ...")
    parser.add_argument("--embeddings_pt", type=str, help="Optional: load dict[str]->Tensor from this .pt")
    parser.add_argument("--dataset_paths", type=str, nargs="*", default=[], help="Paths for datasets")
    parser.add_argument("--method", type=str, default="tsne", choices=["tsne", "umap"])
    parser.add_argument("--pool", type=str, default="avg", choices=["avg", "max", "none"])
    parser.add_argument("--title", type=str, default="t-SNE — N-class")
    parser.add_argument("--out_png", type=str, default="tsne.png")
    args = parser.parse_args()

    model = None
    if args.cfg and args.ckpt:
        print("[INFO] Instantiating model...")
        model = instantiate_model(args.ckpt)

    if args.embeddings_pt:
        print(f"[INFO] Loading precomputed embeddings from {args.embeddings_pt}")
        class_to_tensor = torch.load(args.embeddings_pt, map_location="cpu")
        if not isinstance(class_to_tensor, dict):
            raise ValueError("Expected torch file to contain a dict[str] -> Tensor.")
    else:
        print("[INFO] Extracting embeddings from datasets...")
        class_to_tensor = extract_embeddings_from_datasets(model, args.dataset_paths)

    tsne_from_class_embeddings(
        class_to_tensor,
        method=args.method,
        pool=args.pool,
        title=args.title,
        out_png=args.out_png,
    )


if __name__ == "__main__":
    main()
