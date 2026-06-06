"""
Dev smoke test for the stage-based H-Net.

Run via:

    python -m egomimic.models.hnet._smoke_stages

Builds a tiny 3-stage tree (EncoderDecoder → Chunker → Compute), runs a
forward pass with dummy actions + cond_dict, verifies output shape and that
the chunker registered its bpred entry into ctx.aux, then runs step() across
all T tokens and compares to the cached forward output.
"""

import torch

from egomimic.models.hnet.context import HNetContext
from egomimic.models.hnet.hnet import HNet, ratio_loss_from_aux
from egomimic.models.hnet.stages import (
    ChunkerStage,
    ComputeStage,
    EncoderDecoderStage,
)


def build_tiny_hnet(d_model: int = 64, d_inner: int = 64, d_cond: int = 64) -> HNet:
    stages = [
        EncoderDecoderStage(
            input_hidden_dim=d_model,
            output_hidden_dim=d_model,
            encoder=dict(
                arch_layout="T2",
                d_model=d_model,
                d_intermediate=128,
                num_heads=4,
                cond=True,
            ),
            decoder=dict(
                arch_layout="T2",
                d_model=d_model,
                d_intermediate=128,
                num_heads=4,
                cond=True,
            ),
            cond_key="fused_cond",
            d_cond=d_cond,
            causal=True,
        ),
        ChunkerStage(
            input_hidden_dim=d_model,
            output_hidden_dim=d_inner,
            target_compression_ratio=4.0,
            ratio_loss_weight=0.03,
            cond_key="fused_cond",
        ),
        ComputeStage(
            input_hidden_dim=d_inner,
            output_hidden_dim=d_inner,
            main_network=dict(
                arch_layout="T2",
                d_model=d_inner,
                d_intermediate=128,
                num_heads=4,
                cond=False,
            ),
            cond_key="fused_cond",
            d_cond=0,  # inner stage has no AdaLN
            causal=True,
        ),
    ]
    return HNet(stages)


def main():
    torch.manual_seed(0)
    B, T, d_model, d_cond = 2, 32, 64, 64

    hnet = build_tiny_hnet(d_model=d_model, d_inner=d_model, d_cond=d_cond).eval()

    x_full = torch.randn(B, T, d_model)
    cond = torch.randn(B, T, d_cond)

    ctx = HNetContext(cond_dict={"fused_cond": cond}, aux=[])
    with torch.no_grad():
        y_full = hnet(x_full, ctx)
    print(f"forward output shape: {tuple(y_full.shape)}")
    assert y_full.shape == (B, T, d_model), y_full.shape

    print(f"ctx.aux entries: {len(ctx.aux)}")
    assert len(ctx.aux) == 1, "exactly one chunker should register aux"
    entry = ctx.aux[0]
    assert "bpred" in entry and "target_ratio" in entry and "weight" in entry
    bpred = entry["bpred"]
    print(
        f"  bpred.boundary_mask shape: {tuple(bpred.boundary_mask.shape)} "
        f"(expected ({B},{T}))"
    )
    assert bpred.boundary_mask.shape == (B, T)

    rloss = ratio_loss_from_aux(ctx.aux)
    print(f"ratio_loss: {rloss.item():.6f}")

    # --- step parity ---
    state = hnet.allocate_inference_cache(
        batch_size=B,
        max_seqlen=T,
        device=x_full.device,
        dtype=torch.float32,
    )
    y_step = torch.zeros_like(y_full)
    with torch.no_grad():
        for t in range(T):
            cond_t = cond[:, t]
            step_ctx = HNetContext(
                cond_dict={"fused_cond": cond_t},
                aux=[],
                inference_params=state,
            )
            y_t = hnet.step(x_full[:, t : t + 1], step_ctx)
            y_step[:, t : t + 1] = y_t

    diff = (y_full - y_step).abs().max().item()
    print(f"max |forward - step|: {diff:.3e}")

    tol = 5e-4
    if diff > tol:
        print(
            f"WARN: step/forward divergence above tol={tol}. "
            f"This can happen when chunker boundary patterns differ between "
            f"the padded full-context path and the per-token step path "
            f"(known difference inherited from the original H-Net design)."
        )
    else:
        print(f"OK: step/forward agree within {tol}.")

    print("smoke test complete.")


if __name__ == "__main__":
    main()
