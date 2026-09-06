"""Small torch building blocks.

Lives here rather than in vendored/robomimic_tensor_utils.py: that file is a
verbatim copy of robomimic's tensor_utils and must stay that way. These two
are ours (they came from the old egomimicUtils).
"""

import einops
import math
import torch
import torch.nn as nn


def get_sinusoid_encoding_table(position_start, position_end, d_hid):
    """Sinusoid position encoding table"""

    # Create position tensor
    positions = torch.arange(position_start, position_end, dtype=torch.float32)

    # Create division term for angles
    div_term = torch.exp(
        torch.arange(0, d_hid, 2).float() * (-math.log(10000.0) / d_hid)
    )

    # Create empty table
    sinusoid_table = torch.zeros((position_end - position_start, d_hid))

    # Fill even indices with sin and odd indices with cos
    sinusoid_table[:, 0::2] = torch.sin(positions.unsqueeze(1) * div_term)
    sinusoid_table[:, 1::2] = torch.cos(positions.unsqueeze(1) * div_term[: d_hid // 2])

    return sinusoid_table.unsqueeze(0)


class EinOpsRearrange(nn.Module):
    def __init__(self, rearrange_expr: str, **kwargs) -> None:
        super().__init__()
        self.rearrange_expr = rearrange_expr
        self.kwargs = kwargs

    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        return einops.rearrange(x, self.rearrange_expr, **self.kwargs)
