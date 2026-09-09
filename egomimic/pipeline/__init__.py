"""Dependency-aware pipelines."""

from egomimic.pipeline.core import (
    Pipeline,
    Stage,
    resolve_homogeneous_scalar,
    sum_losses,
)
from egomimic.pipeline.stages_sampler import KeyedFeatureProjection

__all__ = [
    "KeyedFeatureProjection",
    "Pipeline",
    "Stage",
    "resolve_homogeneous_scalar",
    "sum_losses",
]
