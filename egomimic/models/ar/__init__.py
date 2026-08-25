"""Autoregressive / causal state-action models.

Four variants over one causal backbone, differing only in WHAT is predicted,
WHAT is fed back, and HOW actions are decoded. See `variants.py`.
"""

from egomimic.models.ar.variants import (  # noqa: F401
    ARVariantConfig,
    ActionAR,
    JointTeacherForced,
    StateIDM,
    StateActionAR,
    build_variant,
    VARIANTS,
)
