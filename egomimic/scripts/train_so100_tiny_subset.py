from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from egomimic.algo.hpt import HPT
from egomimic.models.hpt_nets import MLPPolicyHead, MLPPolicyStem, TinyConvImageEncoder
from egomimic.rldb.embodiment.so100 import So100SingleArm
from egomimic.rldb.filters import DatasetFilter
from egomimic.rldb.zarr.utils import DataSchematic, set_global_seed
from egomimic.rldb.zarr.zarr_dataset_multi import LocalEpisodeResolver, MultiDataset


SCHEMATIC = {
    "so100_singlearm": {
        "front_img_1": {
            "key_type": "camera_keys",
            "zarr_key": "observations.images.front_img_1",
        },
        "ee_pose": {
            "key_type": "proprio_keys",
            "zarr_key": "observations.state.ee_pose",
        },
        "actions_cartesian": {
            "key_type": "action_keys",
            "zarr_key": "actions_cartesian",
        },
        "embodiment": {
            "key_type": "metadata_keys",
            "zarr_key": "metadata.embodiment",
        },
    }
}


def _cross_attn_specs(latent: int = 4, embed_dim: int = 64):
    return OmegaConf.create(
        {
            "random_horizon_masking": False,
            "cross_attn": {
                "crossattn_latent": latent,
                "crossattn_heads": 2,
                "crossattn_dim_head": 32,
                "crossattn_modality_dropout": 0.0,
                "modality_embed_dim": embed_dim,
            },
        }
    )


def build_dataset(zarr_root: Path, chunk_length: int) -> MultiDataset:
    resolver = LocalEpisodeResolver(
        folder_path=zarr_root,
        key_map=So100SingleArm.get_keymap(mode="camera_frame_ypr"),
        transform_list=So100SingleArm.get_transform_list(
            mode="camera_frame_ypr",
            chunk_length=chunk_length,
        ),
    )
    return MultiDataset._from_resolver(
        resolver=resolver,
        filters=DatasetFilter(
            filter_lambdas=["lambda row: row.get('embodiment') == 'so100_singlearm'"]
        ),
        mode="total",
        valid_ratio=0.0,
    )


def build_hpt(data_schematic: DataSchematic, *, chunk_length: int, device: torch.device) -> HPT:
    embed_dim = 64
    return HPT(
        data_schematic=data_schematic,
        camera_transforms={},
        train_image_augs=torch.nn.Identity(),
        eval_image_augs=torch.nn.Identity(),
        trunk={
            "embed_dim": embed_dim,
            "num_blocks": 2,
            "num_heads": 4,
            "token_postprocessing": "action_token",
            "observation_horizon": 1,
            "action_horizon": chunk_length,
            "no_trunk": False,
            "use_domain_embedding": True,
            "drop_path": 0.0,
            "weight_init_style": "pytorch",
        },
        stem_specs={
            "so100_singlearm": {
                "state_ee_pose": MLPPolicyStem(
                    input_dim=7,
                    output_dim=embed_dim,
                    widths=[embed_dim],
                    specs=_cross_attn_specs(embed_dim=embed_dim),
                )
            }
        },
        shared_stem_specs={
            "front_img_1": MLPPolicyStem(
                input_dim=embed_dim,
                output_dim=embed_dim,
                widths=[embed_dim],
                specs=_cross_attn_specs(embed_dim=embed_dim),
            )
        },
        head_specs={
            "so100_singlearm": MLPPolicyHead(
                input_dim=embed_dim,
                output_dim=7,
                widths=[128],
            )
        },
        shared_obs_keys=["front_img_1"],
        encoder_specs={
            "front_img_1": TinyConvImageEncoder(output_dim=embed_dim),
        },
        domains=["so100_singlearm"],
        pretrained=False,
        pretrained_checkpoint="",
        diffusion=False,
        **{
            "6dof": True,
            "ac_keys": {"so100_singlearm": "actions_cartesian"},
            "reverse_kl_samples": 0,
            "device": device,
        },
    )


def run_tiny_training(
    *,
    zarr_root: Path,
    steps: int,
    batch_size: int,
    chunk_length: int,
    lr: float,
    seed: int,
) -> dict:
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_dataset(zarr_root, chunk_length=chunk_length)

    sample = dataset[0]
    data_schematic = DataSchematic(SCHEMATIC, norm_mode="quantile")
    data_schematic.infer_shapes_from_batch(sample)
    data_schematic.infer_norm_from_dataset(
        dataset,
        "so100_singlearm",
        sample_frac=1.0,
        batch_size=max(1, min(batch_size, len(dataset))),
        num_workers=0,
    )

    model = build_hpt(data_schematic, chunk_length=chunk_length, device=device)
    model.nets.train()
    optimizer = torch.optim.AdamW(model.nets.parameters(), lr=lr, weight_decay=1e-4)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    losses = []
    iterator = iter(loader)
    for _ in range(int(steps)):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        wrapped_batch = {"so100_singlearm": batch}
        processed = model.process_batch_for_training(wrapped_batch)
        predictions = model.forward_training(processed)
        loss_dict = model.compute_losses(predictions, processed)
        loss = loss_dict["action_loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    return {
        "zarr_root": str(zarr_root),
        "dataset_len": int(len(dataset)),
        "batch_size": int(batch_size),
        "steps": int(steps),
        "chunk_length": int(chunk_length),
        "device": str(device),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "losses": losses,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny SO100 HPT overfit smoke training.")
    parser.add_argument(
        "--zarr_root",
        default="/Users/zxwang/repos/egoverse_so100_hpt_zarr_tiny",
    )
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--chunk_length", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_tiny_training(
        zarr_root=Path(args.zarr_root),
        steps=args.steps,
        batch_size=args.batch_size,
        chunk_length=args.chunk_length,
        lr=args.lr,
        seed=args.seed,
    )
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
