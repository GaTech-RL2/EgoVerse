"""Embodiment-agnostic DIFFUSION model pieces (DESIGN.md §2 ``models/diffusion/``).

Role home for the *model* half of the DFoT (Diffusion Forcing Transformer)
stack — the trunk backbones, the forward/reverse diffusion processes, the
timestep/Fourier embeddings, and the sampling schedules. Relocated here from
``egomimic/algo/dfot`` in DESIGN.md step 7 via ``git mv`` (no behaviour
change); the *algo* half (``DFoT`` + the per-variant outer-stages + the VAE
pre-train algo) lives at ``egomimic.algo.diffusion``.

  * backbones/  — :class:`DFoTBackbone`, :class:`DFoTDiT3DBackbone`,
                  :class:`DFoTSpatialBackbone` (interchangeable trunk cores,
                  ``_target_``-swappable like the LSTM/TX/HNet cores).
  * diffusion/  — :class:`ContinuousDiffusion`, :class:`DiscreteDiffusion`
                  forward/reverse processes + the noise schedules.
  * embeddings  — timestep / Fourier / stochastic-time conditioning embeddings.
  * sampling    — DDIM/DDPM sampling + the causal/staircase AR schedules.

The legacy ``egomimic.algo.dfot.<piece>`` import paths (incl. the yaml
``_target_``s such as ``egomimic.algo.dfot.DFoTBackbone`` /
``egomimic.algo.dfot.discrete_diffusion.DiscreteDiffusion``) stay ALIVE through
the ``egomimic.algo.dfot`` facade shim until the final flip (DESIGN.md step 13).
"""

from egomimic.models.diffusion.backbones.backbone import DFoTBackbone
from egomimic.models.diffusion.backbones.dit3d_backbone import DFoTDiT3DBackbone
from egomimic.models.diffusion.backbones.spatial_backbone import (
    DFoTSpatialBackbone,
)
from egomimic.models.diffusion.diffusion.continuous_diffusion import (
    ContinuousDiffusion,
)
from egomimic.models.diffusion.diffusion.discrete_diffusion import (
    DiscreteDiffusion,
)
from egomimic.models.diffusion.sampling import (
    causal_ar_schedule,
    ddim_sample,
    ddpm_sample,
    sample,
    sample_step,
    staircase_ar_schedule,
    vanilla_schedule,
)

__all__ = [
    "DFoTBackbone",
    "DFoTDiT3DBackbone",
    "DFoTSpatialBackbone",
    "ContinuousDiffusion",
    "DiscreteDiffusion",
    "sample_step",
    "sample",
    "vanilla_schedule",
    "causal_ar_schedule",
    "staircase_ar_schedule",
    "ddim_sample",
    "ddpm_sample",
]
