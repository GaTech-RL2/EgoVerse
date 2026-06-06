"""Backward-compat FACADE — ``egomimic.algo.vae`` moved to
``egomimic.algo.diffusion.vae_algo`` (DESIGN.md step 7).

The ConvVAE pre-train algo (``VAE`` + ``load_pretrained_vae``) now lives with
the rest of the diffusion stack at ``egomimic.algo.diffusion.vae_algo`` (the
frozen VAE checkpoint it produces is consumed by the image-DFoT outer-stages).
This package is kept ALIVE as a thin FACADE so the legacy import paths + yaml
``_target_``s keep working unchanged until the final flip (DESIGN.md step 13):

  * ``from egomimic.algo.vae import VAE``                  (name)
  * ``from egomimic.algo.vae.algo import VAE, load_pretrained_vae``  (submodule)
  * ``_target_: egomimic.algo.vae.VAE``                   (yaml)

Mechanism: the real ``egomimic.algo.diffusion.vae_algo`` module is registered in
``sys.modules`` under the legacy ``egomimic.algo.vae.algo`` key, so the
submodule import path resolves to the SAME module object. DO NOT add logic
here — pure forwarder; deleted at DESIGN.md step 13.
"""

import importlib as _importlib
import sys as _sys

_mod = _importlib.import_module("egomimic.algo.diffusion.vae_algo")
# Keep ``egomimic.algo.vae.algo`` resolving to the relocated module.
_sys.modules[f"{__name__}.algo"] = _mod

from egomimic.algo.diffusion.vae_algo import VAE, load_pretrained_vae  # noqa: E402

__all__ = ["VAE", "load_pretrained_vae"]

del _sys, _importlib, _mod
