# Flow Transfer config status

## Supported latent recipes

The active full-data latent recipes are:

- ChainGripper Medium: latent width 96, denoiser width 384, 16 blocks;
  153,976,934 sampler parameters and 165,174,022 including observations.
- ChainGripper Large: latent width 128, denoiser width 512, 16 blocks;
  272,346,630 sampler parameters and 283,543,718 including observations.
- U-Socket + ChainGripper cotrain at both capacities.

All four consume immutable native simulator actions. ChainGripper training
composes native4-to-points6 FK and optional anchored-Phi arc tokenization in the
dataset transform list. Rollout detokenizes and applies the shared constrained
IK revert transform before the existing native4 simulator controller. U-Socket
uses its existing rotvec/SE(2)-arc transforms and returns native theta only at
the simulator boundary.

The cotrain recipes currently contain 2,999 clean U-Socket episodes and 3,000
ChainGripper episodes. The U-Socket directory name says `3000`, but episode
index 270 is absent, so documentation and provenance use the observed count.
No verified 960-episode ChainGripper obstacle dataset is active. The obstacle
domain is therefore not represented by a path, placeholder, or misleading
experiment name.

## Standard action-space Diffusion Policy

The genuine baseline is implemented by
`egomimic.pipeline.stages_diffusion.MultiDomainDiffusionPolicyStage` and the
`pipeline_diffusion_usocket_chain_h16` recipe. It delegates to the tracked
repository implementations:

- `egomimic.models.diffusion_policy.DiffusionPolicy`
- `egomimic.models.denoising_nets.ConditionalUnet1D`
- `egomimic.models.ddim_scheduler.DDIMScheduler`

This is not a latent-width imitation. ChainGripper owns a direct 6D points
policy and U-Socket owns a direct 4D rotvec policy; the observation encoder is
shared. Each policy adds Gaussian noise to normalized action targets, predicts
epsilon at a sampled diffusion timestep, and runs the complete DDIM reverse
chain during inference. There is no action encoder and no latent decoder.

The mode-aware graph contract is explicit:

- train reads `condition`, `target`, and `embodiment`, and writes the epsilon
  loss plus diagnostics;
- rollout reads `condition` and `embodiment`, and writes `pred_action` plus
  diagnostics.

Chain rollout uses the consolidated points6-to-native4 constrained IK adapter;
U-Socket rollout converts rotvec back to native theta only at the simulator
boundary. Exact source blobs and objective details are recorded in
`egomimic/pipeline/DIFFUSION_POLICY_PROVENANCE.md`.

## Validation state

- Hydra composition, dataset/action-width contracts, native transformed-loader
  samples, BF16 rollout conversion, FK/IK replay, and relevant CPU tests pass.
- The exact two-step BF16 optimizer + scheduled-validation smokes for Chain
  Medium, U-Socket+Chain Medium, and standard DP are submitted as Slurm job
  `3717405` against immutable commit `034c303f`.
- No full ChainGripper or Flow Transfer cotrain job should launch until that
  smoke produces finite validation metrics and checkpoints for all three arms.
- Obstacle evaluation and obstacle-code consolidation remain blocked pending
  review of the separate session's fix and explicit user confirmation.
