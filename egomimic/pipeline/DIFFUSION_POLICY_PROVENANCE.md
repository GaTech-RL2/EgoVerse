# Pipeline Diffusion Policy provenance

The Flow Transfer DP baseline reuses the repository's tracked implementation;
it does not copy code from a dirty worktree or an immutable experiment source.
The clean integration base is:

- commit: `bf632fd182375df0920232aaa42298673bede29b`
- source worktree at integration time:
  `/coc/flash7/paphiwetsa3/worktrees/pipeline-dp-baseline-20260826`

Reused source blobs at that commit:

| Role | Path | Git blob |
|---|---|---|
| epsilon policy/objective | `egomimic/models/diffusion_policy.py` | `8f809b3b9a7bcbd3f3033339a12196f888eb6d26` |
| denoising policy contract | `egomimic/models/denoising_policy.py` | `645a8c44e49059f2d0409a9c12102ea60c0ca1c3` |
| ConditionalUnet1D | `egomimic/models/denoising_nets.py` | `fb47d3a5646fa503195ee76d413928f89e59a67e` |
| DDIM scheduler | `egomimic/models/ddim_scheduler.py` | `b072c7f8857634472dc67fdd7544f48ef9b06d01` |

The original repository DP/UNet recipe was introduced in commit
`4ee39c34f78c294cddc5e68765985dc81af9351d`. Its historical configuration
`egomimic/hydra_configs/model/hpt_cotrain_diffusion_unet.yaml` has Git blob
`a7436e776c96dfd6af04310cb814a5f41d95bdb4`. The Pipeline recipe preserves its
essential contract:

- `ConditionalUnet1D` predicts additive Gaussian noise, not clean actions.
- Training samples a random discrete diffusion timestep.
- The target is the exact noise added by the scheduler and the objective is
  MSE between predicted and sampled epsilon.
- Rollout starts from Gaussian action noise and executes the complete reverse
  scheduler chain.
- `squaredcos_cap_v2`, clipping, and `prediction_type: epsilon` are explicit.

The only new policy code is
`egomimic/pipeline/stages_diffusion.py`, a thin multi-domain Stage adapter. It
uses separate width-specific policies because ChainGripper points6 and
U-Socket rotvec4 cannot share the input/output channels of one convolutional
