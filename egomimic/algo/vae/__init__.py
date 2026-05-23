"""ConvVAE algo for pre-training an image encoder/decoder on the project's
image stream, separate from any downstream policy training. The frozen
checkpoint is later loaded by ObsActionImageDFoTOuterStage so the
diffusion model can predict latents instead of raw pixels.
"""

from egomimic.algo.vae.algo import VAE

__all__ = ["VAE"]
