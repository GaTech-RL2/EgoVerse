"""
Validation smoke for packed H-Net training: runs one validation batch through
``PackedAlgoBase.process_batch_for_training`` + ``PackedAlgoBase.forward_eval`` (per-episode AR
rollout) + the new ``HNetEvalVideo.compute_metrics_and_viz``. Verifies that
predicted-vs-GT MSE comes back finite, viz images are produced, and the
graph doesn't crash on packed obs.

Per-episode AR rollout uses ``policy.generate(..., T=T_ep)`` for each
sub-sequence in the pack. With B=2 short-ish episodes this should finish in
under a minute on a single A40 even with the pure-Python EMA fallback.

Run:
    python scripts/smoke_packed_validation.py
"""

from __future__ import annotations

import time

import torch
from torch.utils.data import DataLoader

from egomimic.algo.packed_base import PackedAlgoBase as HNetAlgo
from egomimic.eval.eval_hnet import HNetEvalVideo
from egomimic.models.stems.cond_encoders import CondEncoderModule
from egomimic.models.hnet.hnet import HNet as HNetCore
from egomimic.models.stems.image_encoders import SimpleConv
from egomimic.models.hnet.stages import (
    ChunkerStage,
    ComputeStage,
    EncoderDecoderStage,
)
from egomimic.rldb.embodiment.pushshapes import get_keymap, viz_gt_preds
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset
from egomimic.rldb.zarr.zarr_dataset_packed import (
    ZarrEpisodePackedDataset,
    pack_collate,
)

FOLDER = "/coc/cedarp-dxu345-0/Tsim_datasets2/circle"
BATCH_SIZE = 2  # AR rollout per episode is slow → keep tiny for smoke
ACTION_HORIZON = 1024


def build_algo(norm_stats):
    img_enc = SimpleConv(
        in_channels=3,
        channels=[32, 64, 128, 256],
        kernel_size=3,
        stride=2,
        embed_dim=128,
        norm_groups=8,
    )
    cond_encoder = CondEncoderModule(
        d_cond=128,
        obs_specs={"state_agent_obj": dict(input_dim=5, embed_dim=64, widths=[128])},
        img_encoders={"front_img_1": img_enc},
        cond_proj_widths=[128, 128],
        output_key="fused_cond",
    )
    hnet = HNetCore(
        [
            EncoderDecoderStage(
                input_hidden_dim=128,
                output_hidden_dim=128,
                encoder=dict(
                    arch_layout="T2",
                    d_model=128,
                    d_intermediate=256,
                    num_heads=4,
                    cond=True,
                ),
                decoder=dict(
                    arch_layout="T2",
                    d_model=128,
                    d_intermediate=256,
                    num_heads=4,
                    cond=True,
                ),
                cond_key="fused_cond",
                d_cond=128,
                causal=True,
            ),
            ChunkerStage(
                input_hidden_dim=128,
                output_hidden_dim=128,
                target_compression_ratio=8.0,
                ratio_loss_weight=0.03,
                cond_key="fused_cond",
            ),
            ComputeStage(
                input_hidden_dim=128,
                output_hidden_dim=128,
                main_network=dict(
                    arch_layout="T2",
                    d_model=128,
                    d_intermediate=256,
                    num_heads=4,
                    cond=False,
                ),
                cond_key="fused_cond",
                d_cond=0,
                causal=True,
            ),
        ]
    )
    return HNetAlgo(
        action_dim=2,
        action_horizon=ACTION_HORIZON,
        d_model=128,
        d_cond=128,
        cond_encoder=cond_encoder,
        hnet=hnet,
        norm_stats=norm_stats,
        domains=["pushshapes_sim"],
        ac_keys={"pushshapes_sim": "actions"},
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda"

    dataset = ZarrEpisodePackedDataset.from_local_folder(
        folder_path=FOLDER,
        key_map=get_keymap(action_horizon=ACTION_HORIZON),
        max_seq_len=None,
        chunking="none",
        min_seq_len=64,
        transform_list=None,
    )
    print(f"[data] {len(dataset)} entries")

    norm_stats = MultiDataset(state={}, norm_mode="quantile")
    norm_stats.populate_from_datasets({"pushshapes_sim": dataset})
    norm_stats.infer_shapes_from_batch(dataset[0])
    norm_stats.infer_norm_from_dataset(
        dataset,
        "pushshapes_sim",
        sample_frac=0.02,
        num_workers=0,
        batch_size=4,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=pack_collate,
        num_workers=0,
    )

    algo = build_algo(norm_stats)
    algo.nets.eval()
    print(f"[algo] {sum(p.numel() for p in algo.nets.parameters())/1e6:.2f}M params")

    raw_batch = next(iter(loader))
    seq_lens = raw_batch["seq_lens"].tolist()
    print(
        f"[batch] B={BATCH_SIZE}  seq_lens={seq_lens}  T_total={int(raw_batch['cu_seqlens'][-1])}"
    )

    batch = {"pushshapes_sim": raw_batch}
    processed = algo.process_batch_for_training(batch)

    # Build the evaluator (minimal — no Lightning trainer). Plug in the
    # pushshapes viz function manually since we're not going through Hydra.
    eval_video = HNetEvalVideo.__new__(HNetEvalVideo)
    eval_video.trainer = None
    eval_video.model = algo
    eval_video.viz_func = {
        "pushshapes_sim": (lambda preds, batch: viz_gt_preds(preds, batch)),
    }
    eval_video.val_image_buffer = {}
    eval_video.val_counter = {}

    t0 = time.perf_counter()
    metrics, images_dict = eval_video.compute_metrics_and_viz(processed)
    elapsed = time.perf_counter() - t0
    print(f"[eval] {elapsed:.1f}s")

    print("[metrics]")
    for k, v in metrics.items():
        val = v.item() if torch.is_tensor(v) else v
        print(f"  {k:60s} {val:.4f}")

    for emb_id, imgs in images_dict.items():
        print(f"[viz] emb={emb_id} images shape={imgs.shape} dtype={imgs.dtype}")
        # Sanity: should be (B, H, W, 3) uint8.
        assert imgs.ndim == 4 and imgs.shape[-1] == 3
        assert imgs.dtype.kind == "u" and imgs.dtype.itemsize == 1

    # Sanity: predictions and GT should both be finite.
    assert all(
        torch.isfinite(v).all() if torch.is_tensor(v) else True
        for v in metrics.values()
    )
    print("\n=== VALIDATION SMOKE PASSED ===")


if __name__ == "__main__":
    main()
