"""
End-to-end smoke test for packed training of the H-Net pushshapes policy.

Builds the packed dataset + dataloader directly (skipping Lightning and the
data_schematic normalization layer for simplicity), then runs one batch
through ``HNetPolicy.forward_packed`` + MSE + ratio loss + backward, and
reports memory + boundary stats.

This is the assertion that the new packed plumbing works end-to-end:
    [packed dataset] -> pack_collate -> forward_packed -> backward.

Run:
    python scripts/smoke_packed_training.py
"""

from __future__ import annotations

import time

import torch
from torch.utils.data import DataLoader

from egomimic.algo.packed_base import HNetPolicy
from egomimic.models.hnet.cond_encoders import CondEncoderModule
from egomimic.models.hnet.hnet import (
    HNet as HNetCore,
)
from egomimic.models.hnet.hnet import (
    chunk_stats_from_aux,
    ratio_loss_from_aux,
)
from egomimic.models.hnet.image_encoders import SimpleConv
from egomimic.models.hnet.stages import (
    ChunkerStage,
    ComputeStage,
    EncoderDecoderStage,
)
from egomimic.rldb.embodiment.pushshapes import get_keymap
from egomimic.rldb.zarr.zarr_dataset_packed import (
    ZarrEpisodePackedDataset,
    pack_collate,
)

FOLDER = "/coc/cedarp-dxu345-0/Tsim_datasets2/circle"
BATCH_SIZE = 8
ACTION_HORIZON = 1024  # >= longest episode (958 frames in this dataset)
D_MODEL = 128
D_COND = 128


def build_policy() -> HNetPolicy:
    img_enc = SimpleConv(
        in_channels=3,
        channels=[32, 64, 128, 256],
        kernel_size=3,
        stride=2,
        embed_dim=128,
        norm_groups=8,
    )
    cond_encoder = CondEncoderModule(
        d_cond=D_COND,
        obs_specs={"state_agent_obj": dict(input_dim=5, embed_dim=64, widths=[128])},
        img_encoders={"front_img_1": img_enc},
        cond_proj_widths=[128, 128],
        output_key="fused_cond",
    )
    hnet = HNetCore(
        [
            EncoderDecoderStage(
                input_hidden_dim=D_MODEL,
                output_hidden_dim=D_MODEL,
                encoder=dict(
                    arch_layout="T4",
                    d_model=D_MODEL,
                    d_intermediate=512,
                    num_heads=4,
                    cond=True,
                ),
                decoder=dict(
                    arch_layout="T4",
                    d_model=D_MODEL,
                    d_intermediate=512,
                    num_heads=4,
                    cond=True,
                ),
                cond_key="fused_cond",
                d_cond=D_COND,
                causal=True,
            ),
            ChunkerStage(
                input_hidden_dim=D_MODEL,
                output_hidden_dim=192,
                target_compression_ratio=8.0,
                ratio_loss_weight=0.03,
                cond_key="fused_cond",
            ),
            ComputeStage(
                input_hidden_dim=192,
                output_hidden_dim=192,
                main_network=dict(
                    arch_layout="T4",
                    d_model=192,
                    d_intermediate=768,
                    num_heads=4,
                    cond=False,
                ),
                cond_key="fused_cond",
                d_cond=0,
                causal=True,
            ),
        ]
    )
    return HNetPolicy(
        action_dim=2,
        action_horizon=ACTION_HORIZON,
        d_model=D_MODEL,
        cond_encoder=cond_encoder,
        hnet=hnet,
    )


def fmt_mem(n: int) -> str:
    return f"{n / 1024**3:.2f} GiB"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "Packed training smoke needs CUDA."

    dataset = ZarrEpisodePackedDataset.from_local_folder(
        folder_path=FOLDER,
        key_map=get_keymap(action_horizon=ACTION_HORIZON),
        max_seq_len=None,
        chunking="none",
        min_seq_len=64,
        transform_list=None,
    )
    print(f"[data] dataset entries = {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=pack_collate,
        num_workers=0,
    )
    batch = next(iter(loader))
    cu = batch["cu_seqlens"]
    seq_lens = batch["seq_lens"]
    max_seqlen = int(batch["max_seq_len"])
    print(
        f"[batch] B={BATCH_SIZE}  seq_lens={seq_lens.tolist()}  "
        f"max_seqlen={max_seqlen}  T_total={int(cu[-1])}"
    )

    actions_packed = batch["actions"].to(device)  # (T_total, 2)
    obs_packed = {
        "state_agent_obj": batch["state_agent_obj"].float().to(device),  # (T_total, 5)
        "front_img_1": batch["front_img_1"].float().to(device),  # (T_total, 3, H, W)
    }
    cu = cu.to(device)

    policy = build_policy().to(device).train()
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"[model] params = {n_params/1e6:.2f}M, action_horizon = {ACTION_HORIZON}")
    print(
        f"[device] {torch.cuda.get_device_name(0)} "
        f"({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB)"
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    pred, aux = policy.forward_packed(actions_packed, obs_packed, cu, max_seqlen)
    torch.cuda.synchronize()
    fwd_time = time.perf_counter() - t0
    fwd_peak = torch.cuda.max_memory_allocated()
    print(
        f"[forward] pred {tuple(pred.shape)}  time {fwd_time*1000:.0f} ms  "
        f"peak {fmt_mem(fwd_peak)}"
    )

    stats = chunk_stats_from_aux(aux)
    print(
        f"[chunker] boundary_rate={stats['boundary_rate']:.3f}  "
        f"avg_chunk_len={stats['avg_chunk_len']:.1f}  "
        f"(target avg_chunk_len = 8.0)"
    )

    mse = torch.nn.functional.mse_loss(pred, actions_packed)
    rloss = ratio_loss_from_aux(aux, device=device)
    loss = mse + rloss
    print(
        f"[loss] mse={mse.item():.3e}  ratio={rloss.item():.4f}  total={loss.item():.3e}"
    )

    t0 = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize()
    bwd_time = time.perf_counter() - t0
    bwd_peak = torch.cuda.max_memory_allocated()
    print(f"[backward] time {bwd_time*1000:.0f} ms  peak {fmt_mem(bwd_peak)}")

    # Sanity: every parameter that received gradient should be finite.
    for n, p in policy.named_parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            print(f"[bad-grad] {n}")
            return
    print(
        f"[grad] all finite (covered {sum(1 for _, p in policy.named_parameters() if p.grad is not None)} params)"
    )
    print("\n=== SMOKE PASSED ===")
    print(
        f"B={BATCH_SIZE}, T_total={int(cu[-1])}, peak={fmt_mem(bwd_peak)} "
        f"(A40 {torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f} GiB)"
    )


if __name__ == "__main__":
    main()
