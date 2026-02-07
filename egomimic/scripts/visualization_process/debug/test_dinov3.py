"""
Quick smoke test for DinoV3 from `egomimic/models/hpt_nets.py`.

Runs a random forward pass and prints output shapes

Checking if DinoV3 is working as expected.
"""

import torch

from egomimic.models.hpt_nets import DinoV3


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # `DinoV3.forward` expects images shaped [B, T, N, 3, H, W]
    B, T, N, C, H, W = 2, 3, 1, 3, 224, 224
    x = torch.randn(B, T, N, C, H, W, device=device)

    model = DinoV3(
        output_dim=256,
        # default is "facebook/dinov3-vits16plus-pretrain-lvd1689m"
        freeze_backbone=True,
    ).to(device)

    model.eval()
    with torch.no_grad():
        y = model(x)

    print("input:  {}".format(tuple(x.shape)))
    # typically [(B*T*N), tokens, output_dim]
    print("output: {}".format(tuple(y.shape)))


if __name__ == "__main__":
    main()
