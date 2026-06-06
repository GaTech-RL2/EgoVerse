"""
A40 memory probe: packed-mode forward+backward on a single full pushT episode
through the current H-Net pushshapes arch. No episode chunking
(``chunking='none'``); the full episode flows as one packed sub-sequence with
``cu_seqlens=[0, T]``, exercising the new ``ctx.packed`` plumbing through the
entire stage tree.

What this script does NOT do: go through ``HNetPolicy.forward`` (which is
still padded-only). The action tokenizer / BOS-shift / pos_emb steps are
inlined here so the model side of the call sees true packed shapes. This is
intentional for the probe; the algo-level packed forward is a separate piece
of plumbing.

Run:
    python scripts/debug_full_episode_mem.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import zarr

from egomimic.models.stems.cond_encoders import CondEncoderModule
from egomimic.models.hnet.context import HNetContext
from egomimic.models.hnet.hnet import HNet as HNetCore
from egomimic.models.hnet.hnet import ratio_loss_from_aux
from egomimic.models.stems.image_encoders import SimpleConv
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

DEFAULT_FOLDER = "/coc/cedarp-dxu345-0/Tsim_datasets2/circle"


def find_longest_episode_len(folder: Path) -> int:
    best = 0
    for p in sorted(folder.iterdir()):
        if not p.name.endswith(".zarr"):
            continue
        try:
            g = zarr.open_group(str(p), mode="r")
            best = max(best, int(g["actions"].shape[0]))
        except Exception:
            continue
    return best


def build_model(d_model: int, d_cond: int, action_dim: int, max_pos: int):
    """Construct the pushshapes arch: action tokenizer + cond_encoder + HNet.

    Returns (action_in, action_out, bos, pos_emb, cond_encoder, hnet). Kept as
    individual modules (not a HNetPolicy) so we can drive them in packed mode
    without going through HNetPolicy.forward (which is padded-only today).
    """
    action_in = nn.Linear(action_dim, d_model)
    action_out = nn.Linear(d_model, action_dim)
    bos = nn.Parameter(torch.zeros(1, 1, d_model))
    nn.init.normal_(bos, std=0.02)
    pos_emb = nn.Parameter(torch.zeros(1, max_pos, d_model))
    nn.init.normal_(pos_emb, std=0.02)

    img_enc = SimpleConv(
        in_channels=3,
        channels=[32, 64, 128, 256],
        kernel_size=3,
        stride=2,
        embed_dim=128,
        norm_groups=8,
    )
    cond_encoder = CondEncoderModule(
        d_cond=d_cond,
        obs_specs={"state_agent_obj": dict(input_dim=5, embed_dim=64, widths=[128])},
        img_encoders={"front_img_1": img_enc},
        cond_proj_widths=[128, 128],
        output_key="fused_cond",
    )
    hnet = HNetCore(
        [
            EncoderDecoderStage(
                input_hidden_dim=d_model,
                output_hidden_dim=d_model,
                encoder=dict(
                    arch_layout="T4",
                    d_model=d_model,
                    d_intermediate=512,
                    num_heads=4,
                    cond=True,
                ),
                decoder=dict(
                    arch_layout="T4",
                    d_model=d_model,
                    d_intermediate=512,
                    num_heads=4,
                    cond=True,
                ),
                cond_key="fused_cond",
                d_cond=d_cond,
                causal=True,
            ),
            ChunkerStage(
                input_hidden_dim=d_model,
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

    bundle = nn.ModuleDict(
        {
            "action_in": action_in,
            "action_out": action_out,
            "cond_encoder": cond_encoder,
            "hnet": hnet,
        }
    )
    bundle.bos = bos
    bundle.pos_emb = pos_emb
    return bundle


def fmt_mem(n_bytes: int) -> str:
    return f"{n_bytes / (1024 ** 3):.2f} GiB"


def _patch_router_to_all_boundaries(router):
    """Force every token to be a boundary so the inner ComputeStage sees the
    full T. This is the worst-case "no compression at all" memory footprint.

    Saves the original forward and returns a function that restores it.
    """

    from egomimic.models.hnet.routing import RoutingModuleOutput

    original_forward = router.forward

    def forced_forward(
        hidden_states, cu_seqlens=None, mask=None, inference_params=None
    ):
        # Run the real cos-sim so boundary_prob still has gradient flow.
        out = original_forward(
            hidden_states,
            cu_seqlens=cu_seqlens,
            mask=mask,
            inference_params=inference_params,
        )
        # Override boundary_mask: every position is a boundary.
        bm = torch.ones_like(out.boundary_mask, dtype=torch.bool)
        # selected_probs is gathered from boundary_prob along the chosen branch;
        # forcing all-boundary means we want boundary_prob[..., 1] everywhere.
        if out.selected_probs.dim() == 3:
            sp = out.boundary_prob[..., 1:2]
        else:
            sp = out.boundary_prob[..., 1:2]
        return RoutingModuleOutput(
            boundary_prob=out.boundary_prob,
            boundary_mask=bm,
            selected_probs=sp,
        )

    router.forward = forced_forward
    return lambda: setattr(router, "forward", original_forward)


def _run_one(model, build_inputs, actions_packed, cu, max_seqlen, device, label):
    """build_inputs() returns (x_packed, cond_packed). Called fresh per run so
    the input subgraph is rebuilt (otherwise the second .backward() would try
    to traverse freed tensors from action_in / cond_encoder)."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None

    x_packed, cond_packed = build_inputs()
    ctx = HNetContext(
        cond_dict=cond_packed,
        aux=[],
        inference_params=None,
        cu_seqlens=cu,
        max_seqlen=max_seqlen,
    )
    t0 = time.perf_counter()
    h = model["hnet"](x_packed, ctx)
    pred = model["action_out"](h)
    torch.cuda.synchronize()
    fwd_time = time.perf_counter() - t0
    fwd_peak = torch.cuda.max_memory_allocated()

    bm = ctx.aux[0]["bpred"].boundary_mask
    boundary_rate = bm.float().mean().item()

    loss = torch.nn.functional.mse_loss(pred, actions_packed) + ratio_loss_from_aux(
        ctx.aux, device=device
    )

    t0 = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize()
    bwd_time = time.perf_counter() - t0
    bwd_peak = torch.cuda.max_memory_allocated()

    print(
        f"[{label}] boundary_rate={boundary_rate:.3f}  "
        f"fwd {fwd_time*1000:.0f}ms ({fmt_mem(fwd_peak)})  "
        f"bwd {bwd_time*1000:.0f}ms ({fmt_mem(bwd_peak)})  "
        f"loss={loss.item():.3e}"
    )
    return fwd_peak, bwd_peak, boundary_rate


def main(folder: str = DEFAULT_FOLDER) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "This probe only makes sense on CUDA."
    folder_path = Path(folder)

    max_len = find_longest_episode_len(folder_path)
    print(f"[data] folder = {folder_path}")
    print(f"[data] longest episode = {max_len} frames")

    dataset = ZarrEpisodePackedDataset.from_local_folder(
        folder_path=str(folder_path),
        key_map=get_keymap(action_horizon=max_len),
        max_seq_len=None,
        chunking="none",
        transform_list=None,
    )
    print(f"[data] dataset entries = {len(dataset)}")

    # Pick the longest entry actually present in the packed-dataset index. The
    # raw zarr action array can be slightly longer than what the dataset
    # exposes (the index uses min-length across all key types), so we trust
    # the dataset's view.
    lengths = [(end - start, i) for i, (_k, start, end) in enumerate(dataset.index)]
    lengths.sort(reverse=True)
    longest_idx = lengths[0][1]
    print(
        f"[data] dataset.index longest: {lengths[0][0]} frames " f"(idx={longest_idx})"
    )

    sample = dataset[longest_idx]
    batch = pack_collate([sample])
    cu = batch["cu_seqlens"].to(device)
    max_seqlen = int(batch["max_seq_len"])
    T = int(batch["seq_lens"][0].item())
    print(
        f"[data] episode '{sample.get('episode_key', '?')}' "
        f"T={T}, cu_seqlens={cu.tolist()}"
    )

    # Packed tensors: (T_total, ...). For B=1 single episode T_total == T.
    actions_packed = batch["actions"].to(device)  # (T, 2)
    state_packed = batch["state_agent_obj"].float().to(device)  # (T, 5)
    img_packed = batch["front_img_1"].float().to(device)  # (T, 3, H, W)
    print(
        f"[data] actions {tuple(actions_packed.shape)}, "
        f"state {tuple(state_packed.shape)}, "
        f"img {tuple(img_packed.shape)}"
    )

    model = build_model(d_model=128, d_cond=128, action_dim=2, max_pos=T)
    model = model.to(device).train()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params = {n_params/1e6:.2f}M")
    print(
        f"[device] {torch.cuda.get_device_name(0)} "
        f"({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB total)"
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated()
    print(f"[mem] baseline after model+inputs on device: {fmt_mem(base_mem)}")

    def build_inputs():
        """Per-run input subgraph: action tokenizer + BOS-shift + pos_emb +
        per-token cond. Called fresh each run so backward doesn't trip on
        freed saved tensors from a prior call."""
        a_emb = model["action_in"](actions_packed)  # (T, d_model)
        bos = model.bos.squeeze(0)  # (1, d_model)
        x_shifted = torch.cat([bos, a_emb[:-1]], dim=0)  # (T, d_model)
        pos = model.pos_emb.squeeze(0)[:T]  # (T, d_model)
        x = x_shifted + pos  # (T, d_model)

        obs_padded = {
            "state_agent_obj": state_packed.unsqueeze(0),
            "front_img_1": img_packed.unsqueeze(0),
        }
        cond_padded = model["cond_encoder"].encode(obs_padded, T_action=T)
        cond = {k: v.squeeze(0) for k, v in cond_padded.items()}  # (T, d_cond)
        return x, cond

    # --- Run 1: default chunker (boundary firing rate depends on init) ---
    print(f"\n=== T={T}, packed cu_seqlens=[0,{T}] ===")
    fwd1, bwd1, br1 = _run_one(
        model,
        build_inputs,
        actions_packed,
        cu,
        max_seqlen,
        device,
        label="default chunker",
    )

    # --- Run 2: force all-boundaries (worst case: zero compression) ---
    router = model["hnet"].stages[1].routing_module
    restore = _patch_router_to_all_boundaries(router)
    try:
        fwd2, bwd2, br2 = _run_one(
            model,
            build_inputs,
            actions_packed,
            cu,
            max_seqlen,
            device,
            label="ALL boundaries (no compression)",
        )
    finally:
        restore()

    print("\n=== SUMMARY (packed, single full episode, no chunking) ===")
    print(f"T (episode len)       : {T}")
    print(f"cu_seqlens            : {cu.tolist()}")
    print(f"params                : {n_params/1e6:.2f}M")
    print(
        f"A40 total memory      : "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB"
    )
    print("")
    print("default chunker:")
    print(f"  boundary rate     : {br1:.3f}  (inner stage sees ~{int(br1*T)} tokens)")
    print(f"  peak after fwd    : {fmt_mem(fwd1)}")
    print(f"  peak after bwd    : {fmt_mem(bwd1)}")
    print("")
    print("all-boundary worst case (no compression):")
    print(f"  boundary rate     : {br2:.3f}  (inner stage sees {T} tokens)")
    print(f"  peak after fwd    : {fmt_mem(fwd2)}")
    print(f"  peak after bwd    : {fmt_mem(bwd2)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default=DEFAULT_FOLDER)
    args = parser.parse_args()
    main(folder=args.folder)
