"""DFoT forward/reverse DIFFUSION processes (DESIGN.md §2
``models/diffusion/diffusion/``).

The noise processes + schedules for the diffusion stack, relocated here from
``egomimic/algo/dfot`` in DESIGN.md step 7 (``git mv``, no behaviour change):

  * :class:`ContinuousDiffusion` — continuous-time (cosine-schedule) process.
  * :class:`DiscreteDiffusion`   — discrete-timestep (beta-schedule) process.
  * ``noise_schedule``           — the beta / cosine schedule helpers both use.
"""

from egomimic.models.diffusion.diffusion.continuous_diffusion import (
    ContinuousDiffusion,
)
from egomimic.models.diffusion.diffusion.discrete_diffusion import (
    DiscreteDiffusion,
)

__all__ = ["ContinuousDiffusion", "DiscreteDiffusion"]
