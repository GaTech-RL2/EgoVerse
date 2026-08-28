# Unified latent ImageNet objective

`unified_latent` replaces the diagnostic `endpoint_latent` objective. It keeps
the 16×16 Gaussian latent grid and shared time-conditioned field, but fixes the
two failure modes observed in the first run:

- The shared field tokenizes a fully observed image through patch-token
  cross-attention and denoises class-conditioned latent noise through the same
  weights. The learned tokenizer target is detached before the denoising loss,
  preventing the generative objective from shrinking the latent distribution.
- A jointly trained multiscale convolutional decoder replaces the independent
  per-token patch MLP. No pretrained image encoder, decoder, or discriminator is
  used.

Each optimizer step combines multiscale image reconstruction with clean-latent
denoising. Sampling converts the clean-latent prediction to an ODE velocity and
uses the same Euler sampler as the matched experiment. The generator has
131,399,427 trainable parameters versus 131,123,712 for JiT-B/16.

The design follows the single-stage, weight-shared principle in
[UNITE](https://arxiv.org/abs/2603.22283) and the stop-gradient/clean-latent
stability analysis in
[Diffusion as Self-Distillation](https://arxiv.org/abs/2511.14716), without
copying their implementations or pretrained perceptual losses.
