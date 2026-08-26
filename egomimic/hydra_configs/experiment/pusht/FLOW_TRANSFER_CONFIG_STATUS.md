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

The cotrain recipes currently contain 3,000 U-Socket and 3,000 ChainGripper
episodes. No verified 960-episode ChainGripper obstacle dataset exists, so the
obstacle domain is not represented by a path, environment placeholder, or
misleading experiment name.

## Standard DP is not yet supported by PipelineAlgo

Do not create a fake DP baseline by setting the latent width equal to the
action width. `MultiJActionSampler` still integrates in a latent space and then
uses a learned decoder, so that topology is latent denoising rather than
standard action-space Diffusion Policy.

Reusable code already in this repository:

- `egomimic.models.diffusion_policy.DiffusionPolicy`
- `egomimic.models.denoising_nets.ConditionalUnet1D`

Those modules implement direct action-space DDPM training/inference, but expose
the older policy-head API rather than the flat `Stage` read/write contract.
The old HPT wrapper can host them, but adding an HPT transformer trunk would not
be a standard DP baseline against the DP-style observation encoder used here.
The separately snapshotted stock-DP experiment is immutable and is not an
importable component of this worktree.

A clean Pipeline port needs one direct-action diffusion stage (or a noise stage
plus sampler stage) that:

1. reads `condition`, `target`, and `embodiment` during training;
2. adds DDPM noise directly to the 4D or 6D normalized action target;
3. owns one conventional ConditionalUnet1D action head per action width/domain
   (no action encoder and no latent decoder);
4. predicts epsilon with the configured scheduler and emits `pred_action` at
   inference;
5. preserves the existing per-domain rollout-adapter lookup;
6. is gated by Hydra composition, a real transformed-loader batch, bf16 rollout,
   strict target/prediction shape tests, and a short optimization+validation
   smoke before any full launch.

Until that stage exists and passes those gates, there is deliberately no
Flow Transfer DP experiment YAML in this directory.
