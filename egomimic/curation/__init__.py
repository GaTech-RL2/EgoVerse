"""
DemInf data curation module for EgoVerse.

Implements mutual information-based trajectory scoring following
Hejna et al. 2025 (Google DeepMind + Stanford).

Pipeline:
    1. Load episodes from local Zarr stores (or via EgoVerse config)
    2. Apply preprocessing filters (pauses, clipping, min length)
    3. Embed (state, action) pairs into a shared latent space
    4. Estimate pointwise MI using the KSG k-NN estimator
    5. Score each trajectory by its mean MI contribution
    6. Export filtered episode hashes for downstream SQL filtering
"""

from egomimic.curation.curator import CurationResult, DemInfCurator
from egomimic.curation.embedders import (
    ActionEmbedder,
    ImageStateEmbedder,
    StateEmbedder,
)
from egomimic.curation.filters import ActionClipFilter, MinLengthFilter, PauseFilter
from egomimic.curation.ksg import ksg_mi, ksg_mi_averaged
from egomimic.curation.scorer import TrajectoryScorer
from egomimic.curation.utils import (
    Episode,
    load_episodes_from_config,
    load_episodes_from_dir,
)

__all__ = [
    "ActionClipFilter",
    "ActionEmbedder",
    "CurationResult",
    "DemInfCurator",
    "Episode",
    "ImageStateEmbedder",
    "MinLengthFilter",
    "PauseFilter",
    "StateEmbedder",
    "TrajectoryScorer",
    "ksg_mi",
    "ksg_mi_averaged",
    "load_episodes_from_config",
    "load_episodes_from_dir",
]
