#!/usr/bin/env python3
"""Print the baked inference contract (metadata) for an aria_egoposer checkpoint.

This is the authoritative source of what the receiver must send (camera_keys /
proprio_keys) and what it gets back (action_dim / action_horizon).

Usage:
    python inspect_aria_egoposer.py [path/to/last.ckpt]
"""
import sys

from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.serving.egoverse_policy import EgoVersePolicy

ckpt = sys.argv[1] if len(sys.argv) > 1 else (
    "checkpoints/aria_egoposer/hier/checkpoints/last.ckpt"
)

m = ModelWrapper.load_from_checkpoint(ckpt, weights_only=False)
meta = EgoVersePolicy(m).metadata
print(f"checkpoint: {ckpt}")
for k, v in meta.items():
    print(f"  {k}: {v}")
