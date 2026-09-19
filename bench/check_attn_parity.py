"""Does need_weights=False change the numbers?

The flow head's attentions used to call nn.MultiheadAttention with the default
need_weights=True, which forces the materialized-softmax path; they now pass
False and dispatch to scaled_dot_product_attention. That is the same function
with a different reduction order, so the loss and gradients should agree to
floating-point tolerance and nothing more. This forces the old path back on by
wrapping MultiheadAttention.forward and compares a full train step.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.bench_hpt import EMB_HUMAN, synth_batch  # noqa: E402
from bench.check_compile_parity import build, make_deterministic, run  # noqa: E402

_MHA_FORWARD = nn.MultiheadAttention.forward


def force_need_weights(enabled: bool):
    """Restore the pre-change call: need_weights defaulted to True."""
    if not enabled:
        nn.MultiheadAttention.forward = _MHA_FORWARD
        return

    def forward(self, *a, **kw):
        kw["need_weights"] = True
        kw.pop("average_attn_weights", None)
        return _MHA_FORWARD(self, *a, **kw)

    nn.MultiheadAttention.forward = forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--rtol", type=float, default=2e-2)
    ap.add_argument("--atol", type=float, default=2e-3)
    ap.add_argument("--min-cos", type=float, default=0.999)
    ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args()

    keys = {
        "camera_keys": ["observations.images.front_img_1"],
        "proprio_keys": ["observations.state.ee_pose"],
        "action_keys": ["actions_cartesian"],
        "lang_keys": [],
    }
    torch.manual_seed(1234)
    batch = synth_batch(
        keys,
        EMB_HUMAN,
        args.batch_size,
        torch.device("cuda"),
        18,
        20,
        100,
        (224, 224),
        1,
        "fold the towel",
    )
    noise = torch.randn(args.batch_size, 100, 18, device="cuda")
    time = torch.rand(args.batch_size, device="cuda") * 0.9 + 0.05

    results = {}
    for label, old_path in (
        ("sdpa (need_weights=False)", False),
        ("materialized (need_weights=True)", True),
    ):
        force_need_weights(old_path)
        algo = build(keys, compile_it=False, mode=None)
        make_deterministic(algo, noise, time)
        results[label] = run(algo, batch, seed=7, bf16=not args.fp32)
        del algo
        torch.cuda.empty_cache()
    force_need_weights(False)

    (new_label, old_label) = list(results)
    new_loss, new_grads = results[new_label]
    old_loss, old_grads = results[old_label]
    print(
        f"loss  {old_label}={old_loss:.6f}  {new_label}={new_loss:.6f}  "
        f"delta={abs(new_loss - old_loss):.3e}"
    )
    assert abs(new_loss - old_loss) <= args.atol + args.rtol * abs(
        old_loss
    ), "loss differs beyond floating-point tolerance"

    assert set(new_grads) == set(old_grads), "gradient key sets differ"
    rows, dead = [], 0
    for name, g in old_grads.items():
        h = new_grads[name]
        if g.count_nonzero() == 0 and h.count_nonzero() == 0:
            dead += 1
            continue
        scale = g.abs().max().clamp(min=1e-8)
        rel = ((g - h).abs().max() / scale).item()
        cos = torch.nn.functional.cosine_similarity(
            g.flatten().float(), h.flatten().float(), dim=0
        ).item()
        rows.append((rel, cos, name))
    rows.sort(reverse=True)
    print(f"{len(rows)} gradient tensors compared ({dead} zero in both)")
    print("worst by relative max-abs delta:")
    for rel, cos, name in rows[:5]:
        print(f"  rel={rel:.2e}  cos={cos:.6f}  {name}")
    min_cos = min(c for _, c, _ in rows)
    print(f"min cosine similarity: {min_cos:.6f}")
    assert min_cos >= args.min_cos, "a gradient points somewhere else"
    print("ATTENTION PARITY OK")


if __name__ == "__main__":
    main()
