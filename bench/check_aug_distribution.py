"""Does BatchedColorJitter draw from the same distribution as the per-image loop?

The two cannot be compared pointwise: they consume different RNG streams, so
every sample differs by construction. What must match is the DISTRIBUTION of
augmented images. This draws the same images through both paths many times and
compares the empirical distributions of a set of per-sample statistics with a
two-sample Kolmogorov-Smirnov test.

The one known deviation is op ORDER: the loop drew a permutation per image, the
batched version draws one per call. A single sample's marginal is unaffected
(the order is still uniform), so the KS test should pass; what it cannot see is
the within-batch correlation that introduces. --order-check measures that
separately.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from torchvision import transforms as T

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from egomimic.models.image_augs import PerSampleAugs  # noqa: E402

JITTER = dict(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)


def ks_2samp(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    """Two-sample KS statistic and its asymptotic p-value."""
    a, b = torch.sort(a)[0], torch.sort(b)[0]
    n, m = a.numel(), b.numel()
    allv = torch.cat([a, b])
    cdf_a = torch.searchsorted(a, allv, right=True).double() / n
    cdf_b = torch.searchsorted(b, allv, right=True).double() / m
    d = float((cdf_a - cdf_b).abs().max())
    en = (n * m / (n + m)) ** 0.5
    lam = (en + 0.12 + 0.11 / en) * d
    p = 2.0 * sum(
        (-1) ** (j - 1) * torch.exp(torch.tensor(-2.0 * j * j * lam * lam)).item()
        for j in range(1, 101)
    )
    return d, min(max(p, 0.0), 1.0)


def stats_of(img: torch.Tensor) -> torch.Tensor:
    """Per-sample summary: per-channel mean and std, then overall min and max."""
    per_ch_mean = img.mean(dim=(-2, -1))
    per_ch_std = img.std(dim=(-2, -1))
    lo = img.amin(dim=(-3, -2, -1), keepdim=False).unsqueeze(-1)
    hi = img.amax(dim=(-3, -2, -1), keepdim=False).unsqueeze(-1)
    return torch.cat([per_ch_mean, per_ch_std, lo, hi], dim=-1)


STAT_NAMES = ["mean_R", "mean_G", "mean_B", "std_R", "std_G", "std_B", "min", "max"]


def collect(module, images, draws, device, seed):
    torch.manual_seed(seed)
    out = []
    for _ in range(draws):
        out.append(stats_of(module(images.clone())).cpu())
    return torch.cat(out, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    augs = T.Compose([T.ColorJitter(**JITTER)])

    # A fixed, non-degenerate image set: flat colour would make several of the
    # adjustments indistinguishable from each other.
    torch.manual_seed(0)
    images = torch.rand(args.batch_size, 3, args.size, args.size, device=device)

    batched = PerSampleAugs(augs).to(device)
    assert batched.vectorized is not None, "the shipped aug list did not vectorize"

    loop = PerSampleAugs(augs).to(device)
    loop.vectorized = None  # the pre-change path

    n = args.draws * args.batch_size
    print(f"{n} augmented samples per path ({args.draws} draws of {args.batch_size})")
    a = collect(loop, images, args.draws, device, seed=1)
    b = collect(batched, images, args.draws, device, seed=2)

    print(
        f"{'statistic':>10} {'loop mean':>12} {'batched mean':>13} "
        f"{'loop sd':>10} {'batched sd':>11} {'KS D':>8} {'p':>8}"
    )
    worst_p = 1.0
    for i, name in enumerate(STAT_NAMES):
        x, y = a[:, i].double(), b[:, i].double()
        d, p = ks_2samp(x, y)
        worst_p = min(worst_p, p)
        print(
            f"{name:>10} {x.mean():12.6f} {y.mean():13.6f} "
            f"{x.std():10.6f} {y.std():11.6f} {d:8.4f} {p:8.4f}"
        )

    print(
        f"\nworst p = {worst_p:.4f} over {len(STAT_NAMES)} statistics "
        f"(alpha = {args.alpha})"
    )
    if worst_p < args.alpha:
        print("DISTRIBUTIONS DIFFER")
        return 1
    print("DISTRIBUTION OK: no detectable difference in the per-sample marginal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
