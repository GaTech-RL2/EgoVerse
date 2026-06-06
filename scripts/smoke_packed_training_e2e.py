"""
End-to-end packed training smoke through the actual algo path.

Unlike ``smoke_packed_training.py`` (which calls ``HNetPolicy.forward_packed``
directly), this script wires the full Lightning-style flow:

    pack_collate -> HNet.process_batch_for_training (incl. normalize via
    norm_stats) -> HNet.forward_training -> HNet.compute_losses -> backward.

This is the assertion that the legacy ``data_schematic`` removal didn't break
anything: ``process_batch_for_training`` now resolves keys via
``norm_stats.zarr_key_to_keyname`` and normalization runs via
``norm_stats.normalize`` (per-feature, packed-shape-safe).

Run:
    python scripts/smoke_packed_training_e2e.py
"""

from __future__ import annotations

import sys
import time

import torch
from torch.utils.data import DataLoader

from egomimic.algo.packed_base import HNet as HNetAlgo
from egomimic.models.stems.cond_encoders import CondEncoderModule
from egomimic.models.hnet.hnet import (
    HNet as HNetCore,
)
from egomimic.models.stems.image_encoders import SimpleConv
from egomimic.models.hnet.stages import (
    ChunkerStage,
    ComputeStage,
    EncoderDecoderStage,
)
from egomimic.rldb.embodiment.pushshapes import get_keymap
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset
from egomimic.rldb.zarr.zarr_dataset_packed import (
    ZarrEpisodePackedDataset,
    pack_collate,
)

FOLDER = "/coc/cedarp-dxu345-0/Tsim_datasets2/circle"
BATCH_SIZE = 4  # smaller than smoke_packed_training to keep this snappy
ACTION_HORIZON = 1024


def _build_hnet_core(d_model: int, d_cond: int) -> HNetCore:
    return HNetCore(
        [
            EncoderDecoderStage(
                input_hidden_dim=d_model,
                output_hidden_dim=d_model,
                encoder=dict(
                    arch_layout="T2",
                    d_model=d_model,
                    d_intermediate=256,
                    num_heads=4,
                    cond=True,
                ),
                decoder=dict(
                    arch_layout="T2",
                    d_model=d_model,
                    d_intermediate=256,
                    num_heads=4,
                    cond=True,
                ),
                cond_key="fused_cond",
                d_cond=d_cond,
                causal=True,
            ),
            ChunkerStage(
                input_hidden_dim=d_model,
                output_hidden_dim=d_model,
                target_compression_ratio=8.0,
                ratio_loss_weight=0.03,
                cond_key="fused_cond",
            ),
            ComputeStage(
                input_hidden_dim=d_model,
                output_hidden_dim=d_model,
                main_network=dict(
                    arch_layout="T2",
                    d_model=d_model,
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "Packed training smoke needs CUDA."

    # 1. Build the packed dataset + dataloader (what tsimulation.yaml resolves to).
    t0 = time.perf_counter()
    dataset = ZarrEpisodePackedDataset.from_local_folder(
        folder_path=FOLDER,
        key_map=get_keymap(action_horizon=ACTION_HORIZON),
        max_seq_len=None,
        chunking="none",
        min_seq_len=64,
        transform_list=None,
    )
    print(f"[data] dataset entries = {len(dataset)}  ({time.perf_counter()-t0:.1f}s)")

    # 2. Build the norm_stats (mirrors trainHydra: populate -> infer_shapes ->
    #    infer_norm). With our fixes, this works on packed datasets.
    t0 = time.perf_counter()
    norm_stats = MultiDataset(state={}, norm_mode="quantile")
    norm_stats.populate_from_datasets({"pushshapes_sim": dataset})
    norm_stats.infer_shapes_from_batch(dataset[0])
    norm_stats.infer_norm_from_dataset(
        dataset,
        "pushshapes_sim",
        sample_frac=0.05,
        num_workers=0,
        batch_size=4,
    )
    # Crucial: trainHydra also calls set_norm_stats_from so the training
    # MultiDataset sees the stats. ZarrEpisodePackedDataset doesn't have that
    # method (normalization runs on the algo side, not at dataset __getitem__),
    # so we just hand norm_stats to the algo directly.
    print(f"[norm_stats] populated  ({time.perf_counter()-t0:.1f}s)")
    for emb_id, by_key in norm_stats.norm_stats.items():
        if by_key:
            for k, stats in by_key.items():
                print(f"  emb{emb_id} {k:25s} mean={list(stats['mean'].round(2))}")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=pack_collate,
        num_workers=0,
    )

    # 3. Build the algo (HNet), passing norm_stats — mirrors pl_model.
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
    algo = HNetAlgo(
        action_dim=2,
        action_horizon=ACTION_HORIZON,
        d_model=128,
        d_cond=128,
        cond_encoder=cond_encoder,
        hnet=_build_hnet_core(128, 128),
        norm_stats=norm_stats,
        domains=["pushshapes_sim"],
        ac_keys={"pushshapes_sim": "actions"},
        device=device,
    )
    algo.nets.train()
    n_params = sum(p.numel() for p in algo.nets.parameters())
    print(f"[algo] params = {n_params/1e6:.2f}M, device = {device}")

    # 4. One training step through the full path.
    raw_batch = next(iter(loader))
    print(
        f"[batch] B={BATCH_SIZE}  seq_lens={raw_batch['seq_lens'].tolist()}  "
        f"T_total={int(raw_batch['cu_seqlens'][-1])}"
    )

    # process_batch_for_training expects the dict keyed by embodiment name.
    batch = {"pushshapes_sim": raw_batch}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    processed = algo.process_batch_for_training(batch)
    t_proc = time.perf_counter() - t0

    # Sanity: processed should be keyed by embodiment_id, with packed keys.
    emb_id = next(iter(processed.keys()))
    pkeys = set(processed[emb_id].keys())
    print(
        f"[process_batch] {t_proc*1000:.0f} ms  "
        f"emb={emb_id}  keys={sorted(k for k in pkeys if not k.startswith('_'))}"
    )
    assert "actions" in pkeys, pkeys
    assert "state_agent_obj" in pkeys, pkeys
    assert "front_img_1" in pkeys, pkeys
    assert "cu_seqlens" in pkeys, pkeys
    assert processed[emb_id]["_packed"] is True

    # Verify normalization actually happened on actions: mean should be ~0.
    a_mean = processed[emb_id]["actions"].mean().item()
    a_std = processed[emb_id]["actions"].std().item()
    print(
        f"[normalize] normalized actions  mean={a_mean:.3f}  std={a_std:.3f}  "
        f"(expect mean ~0, std ~1)"
    )
    assert abs(a_mean) < 1.0, f"normalize did nothing? mean={a_mean}"

    t0 = time.perf_counter()
    predictions = algo.forward_training(processed)
    torch.cuda.synchronize()
    t_fwd = time.perf_counter() - t0
    fwd_peak = torch.cuda.max_memory_allocated()

    avg_len = predictions[f"{emb_id}_avg_chunk_len"].item()
    boundary_rate = predictions[f"{emb_id}_boundary_rate"].item()
    print(
        f"[forward] {t_fwd*1000:.0f} ms  peak={fwd_peak/1024**3:.2f} GiB  "
        f"avg_chunk_len={avg_len:.1f}  boundary_rate={boundary_rate:.3f}"
    )

    losses = algo.compute_losses(predictions, processed)
    print(f"[losses] keys: {[k for k in losses.keys() if not k.startswith('_')][:6]}")
    total = losses["action_loss"]
    print(
        f"  action_loss={total.item():.4f}  "
        f"(includes ratio_loss + mse averaged across embodiments)"
    )

    t0 = time.perf_counter()
    total.backward()
    torch.cuda.synchronize()
    t_bwd = time.perf_counter() - t0
    bwd_peak = torch.cuda.max_memory_allocated()
    n_with_grad = sum(1 for p in algo.nets.parameters() if p.grad is not None)
    print(
        f"[backward] {t_bwd*1000:.0f} ms  peak={bwd_peak/1024**3:.2f} GiB  "
        f"{n_with_grad} params with grad"
    )

    for n, p in algo.nets.named_parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            print(f"[bad-grad] {n}")
            sys.exit(1)

    print("\n=== END-TO-END SMOKE PASSED ===")


if __name__ == "__main__":
    main()
