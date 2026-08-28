# JiT backbone on the endpoint algorithm

This is a diagnostic control, not the canonical JiT objective. It removes
JiT's RGB patch input stem and never encodes the target image.

```text
Gaussian tokens (B,256,768)
  -> 12 JiT-B/16 blocks
  -> JiT terminal head interpreted as latent velocity
  -> differentiable endpoint Euler curriculum
  -> learned Linear(768,768) RGB patch decoder
  -> unpatchify to (B,3,256,256)
  -> terminal pixel MSE
```

The optimizer follows the action run: AdamW at `3e-5`, 3,000-step linear
warmup from `3e-6`, then cosine decay to `3e-6` at step 240,000. No
effective-batch LR scaling is applied. Validation and W&B use optimizer steps.

The control isolates the backbone while preserving the target-blind endpoint
algorithm. Its loss must not be normalized against JiT's different denoising
loss; compare matched samples, noise/class effects, diversity, and FID.
