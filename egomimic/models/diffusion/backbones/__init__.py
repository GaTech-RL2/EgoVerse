"""DFoT trunk BACKBONES (DESIGN.md §2 ``models/diffusion/backbones/``).

The interchangeable AR-trunk cores for the diffusion stack, relocated here from
``egomimic/algo/dfot`` in DESIGN.md step 7 (``git mv``, no behaviour change):

  * :class:`DFoTBackbone`        — isotropic-transformer DFoT backbone.
  * :class:`DFoTDiT3DBackbone`   — 3D DiT (temporal + 2D spatial) backbone.
  * :class:`DFoTSpatialBackbone` — patch-embed spatial DFoT backbone.
"""

from egomimic.models.diffusion.backbones.backbone import DFoTBackbone
from egomimic.models.diffusion.backbones.dit3d_backbone import DFoTDiT3DBackbone
from egomimic.models.diffusion.backbones.spatial_backbone import (
    DFoTSpatialBackbone,
)

__all__ = ["DFoTBackbone", "DFoTDiT3DBackbone", "DFoTSpatialBackbone"]
