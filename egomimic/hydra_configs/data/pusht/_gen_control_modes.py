"""Generate the control-mode study's data, model and evaluator configs.

Experiment: ONE embodiment (`gripper`), six controller modes. Train on four,
hold out two (`ideal` and `jittery`) on opposite sides of the training noise
range. Four arms at two capacities.

Two structural decisions worth stating, because they are not obvious from the
generated YAML:

1. ONE domain, not six. Every control mode maps to the same embodiment key
   `pushshapes_sim_gripper` and the launcher stages every seen mode into a
   single directory. Giving each mode its own domain would hand the model a
   per-mode embedding — i.e. it would be TOLD which controller it is driving,
   which is exactly the inference the study asks whether it can make. It also
   avoids registering four embodiment IDs for what is one embodiment.

2. The held-out modes are measured by ROLLOUT, not by validation loss. A
   held-out validation set would need its own domain key, whose domain
   embedding would never be trained — so the number would report an untrained
   embedding rather than distribution shift. SR under a gap needs no episodes
   from that mode at all, only the gap. A held-out BC loss is still worth
   having; it is a post-hoc pass over the trained checkpoint, not an in-run
   metric.

Run:  python egomimic/hydra_configs/data/pusht/_gen_control_modes.py
"""

import pathlib

HERE = pathlib.Path(__file__).resolve()
REPO = HERE.parents[4]
DATA_DIR = REPO / "egomimic/hydra_configs/data/pusht"
MODEL_DIR = REPO / "egomimic/hydra_configs/model/bf"
EVAL_DIR = REPO / "egomimic/hydra_configs/evaluator"

DOMAIN = "pushshapes_sim_gripper"

# Controller modes ordered by sensing-noise floor, which is the axis the
# held-out question is actually asked along:
#
#   ideal 0.0 | tight 0.3 | laggy 0.4 | loose 0.8 | jittery 2.5
#
# TRAINING spans 0.3-0.8. Both held-out modes sit OUTSIDE that range, on
# opposite sides, which brackets the extrapolation instead of testing only one
# direction:
#   ideal   (0.0) — below the training range: can it stop compensating for a
#                   noise floor that is no longer there?
#   jittery (2.5) — 3x above it: can it cope with an irreducible floor it has
#                   never seen? This is the headline test.
#
# `ideal` began as a training mode and became a holdout because 714/1000 of its
# generated episodes were unreadable (zarr 3.1.0 wrote corrupt numeric arrays
# while reporting success). Training on its 286 survivors would have put a 3.5x
# data imbalance under the comparison, which is worse than one fewer training
# mode. When a clean ideal cell lands it can move back; the 4-mode and 5-mode
# results are worth reporting together, since the difference measures what
# controller diversity buys.
SEEN = ["tight", "loose", "laggy", "sticky"]
HELD_OUT = ["ideal", "jittery"]
PRIMARY_HELD_OUT = "jittery"
NOISE_STD = {"ideal": 0.0, "tight": 0.3, "laggy": 0.4, "loose": 0.8,
             "sticky": 0.0, "jittery": 2.5}
# Explicit, because sorting by NOISE_STD alone puts `sticky` next to `ideal`:
# sticky's gap is a 4.0 deadband and 0.88 gain, not sensing noise, so ordering
# it by noise would imply a similarity to `ideal` that does not exist.
EVAL_ORDER = ["ideal", "tight", "laggy", "loose", "sticky", "jittery"]

# eval_sim.py:375 REQUIRES len(init_seeds) == limit_val_batches under
# init_mode="seeds" and raises otherwise:
#   ValueError: explicit seed count must equal requested rollout count (20 != 2)
# The shipped eval_sim_pushshapes.yaml pairs 20 seeds with limit_val_batches 2,
# which never fired only because its `init_mode: seed` was rejected at
# construction first — fixing that typo exposes this. Keep them equal.
#
# 10 rather than 20: every DDP rank runs the full rollout set unguarded (there
# is no rank-0 guard in the eval path), so a validation pass costs
# ranks x evaluators x N episodes of CPU-bound pymunk. At 8 ranks and 6
# evaluators, 20 seeds is 960 rollouts per validation and starves the SR curve
# of points. 10 seeds halves that; SR granularity becomes 10% per episode,
# which is coarse but reported across many validations.
N_ROLLOUTS = 10
SEED_LIST = list(range(N_ROLLOUTS))

# Arc token: D=10, M=16, rotation_radius=0, velocity_layout=append.
# M=16 matches the dense h16 baseline's width so the arc and dense arms have
# the same scalar output budget. append -> (M+1, 5), hence horizon 17.
D, M, ROTATION_RADIUS, LAYOUT = 10.0, 16, 0.0, "append"
HORIZON = M + 1
ACTION_DIM = 5
COND_DIM = 67  # 3 low-dim state channels + 64 VisualCore features
LATENT_DIM = 96
ROOT = "${oc.env:PUSHSHAPES_ROOT,/workspace/pushshapes}"

# Measured, not derived from 12*L*d^2 — that rule ignores heads, per-domain
# projections and ff_mult, and is ~2x off here. Totals include the shared
# 11.20M observation encoder.
#   arms 2-4 large 313.63M   arm 1 large 309.12M  (1.44%)
#   arms 2-4 small  49.09M   arm 1 small  50.01M  (1.88%)
# Both inside the 5% budget; without this arm 1 sits at 165M and the
# comparison measures capacity instead of attention.
CAPACITIES = {
    "large": dict(d_model=1024, n_layers=24, n_heads=16, nblocks=31),
    "small": dict(d_model=512, n_layers=12, n_heads=8, nblocks=4),
}
DENOISER_HIDDEN = 384

ARMS = {
    "arm1_dp_flow": None,  # flow matching — the established baseline
    "arm2_causal_bidir": "causal_bidir",
    "arm3_state_action_ar": "state_action_ar",
    "arm4_state_idm": "state_idm",
}


def data_config() -> str:
    return f"""_target_: egomimic.pl_utils.pl_data_utils.MultiDataModuleWrapper

# CONTROL-MODE STUDY — gripper only. Seen: {", ".join(SEEN)}.
# Held out: {", ".join(HELD_OUT)}.
#
# The seen modes are staged by the launcher into ONE
# directory under a single embodiment key. The model is deliberately not told
# which controller produced an episode: inferring and compensating for the
# controller is the capability under test. See _gen_control_modes.py.
#
# {", ".join(HELD_OUT)} are absent from training entirely and are measured by
# rollout SR under their control gaps (evaluator/eval_sim_control_modes.yaml).
# They sit on OPPOSITE sides of the training noise range (0.3-0.8): ideal at
# 0.0 and jittery at 2.5.
#
# Arc token: D={D} M={M} rotation_radius={ROTATION_RADIUS} velocity_layout={LAYOUT}
# -> action_horizon {HORIZON}.

train_datasets:
  {DOMAIN}:
    _target_: egomimic.rldb.zarr.zarr_dataset_multi.MultiDataset._from_resolver
    resolver:
      _target_: egomimic.rldb.zarr.zarr_dataset_multi.LocalEpisodeResolverWithEmbodimentOverride
      folder_path: {ROOT}/train/gripper/T
      embodiment_override: {DOMAIN}
      key_map:
        _target_: egomimic.rldb.embodiment.pushshapes.get_keymap_hpt
        action_horizon: 200
      transform_list:
        _target_: egomimic.rldb.embodiment.pushshapes.get_planar_arc_length_transform_list
        min_distance_unit: {D}
        resampled_vector_length: {M}
        dt: 0.03333333333333333
        rotation_radius: {ROTATION_RADIUS}
        velocity_mode: mean_scalar
        velocity_layout: {LAYOUT}
    mode: train
    valid_ratio: 0.02
    bounds_check: false

valid_datasets:
  {DOMAIN}:
    _target_: egomimic.rldb.zarr.zarr_dataset_multi.MultiDataset._from_resolver
    resolver:
      _target_: egomimic.rldb.zarr.zarr_dataset_multi.LocalEpisodeResolverWithEmbodimentOverride
      folder_path: {ROOT}/train/gripper/T
      embodiment_override: {DOMAIN}
      key_map:
        _target_: egomimic.rldb.embodiment.pushshapes.get_keymap_hpt
        action_horizon: 200
      transform_list:
        _target_: egomimic.rldb.embodiment.pushshapes.get_planar_arc_length_transform_list
        min_distance_unit: {D}
        resampled_vector_length: {M}
        dt: 0.03333333333333333
        rotation_radius: {ROTATION_RADIUS}
        velocity_mode: mean_scalar
        velocity_layout: {LAYOUT}
    mode: valid
    valid_ratio: 0.02
    bounds_check: false

train_dataloader_params:
  {DOMAIN}:
    batch_size: 16
    num_workers: 6
    pin_memory: true
    persistent_workers: true
    prefetch_factor: 3

valid_dataloader_params:
  {DOMAIN}:
    batch_size: 16
    num_workers: 4
    pin_memory: true
    persistent_workers: true
    prefetch_factor: 3
"""


ENCODER_STAGE = f"""  - _target_: egomimic.pipeline.stages_sampler.FusedObsEncoder
    n_obs_steps: 1
    encoder:
      _target_: egomimic.pipeline.stages_sampler.DPStyleObsEncoder
      obs_specs:
        state_agent_obj:
          input_dim: 3
          input_slice:
          - 0
          - 3
      img_encoders:
        front_img_1:
          _target_: egomimic.models.stems.visual_core.VisualCore
          in_channels: 3
          image_size: 96
          num_kp: 32
          feature_dimension: 64
          pretrained: false
          crop_aug: true
          crop_height: 84
          crop_width: 84
          crop_eval_mode: center
          crop_sample_mode: v02
          crop_scope: frame
          norm_layer: group
          pool_type: spatial_softmax
"""


def flow_stages(nblocks: int) -> str:
    """Arm 1: unchanged flow-matching head, scaled to the shared budget.

    act_seq MUST equal action_horizon. CrossTransformer adds a
    (1, act_seq, D) positional table to the token sequence, so the shipped
    arc configs (act_seq 16, horizon 17) die on the first forward pass with a
    broadcast error — after the dataset pull and norm-stats phase.
    """
    return f"""  - _target_: egomimic.pipeline.stages_sampler.GaussianLatentNoise
    action_horizon: {HORIZON}
    latent_dim: {LATENT_DIM}
  - _target_: egomimic.pipeline.stages_sampler.MultiJActionSampler
    condition_input_dim: {COND_DIM}
    condition_dim: {DENOISER_HIDDEN}
    gradient_accumulation_steps: 1
    schedule_anchor_domain: {DOMAIN}
    action_horizon: {HORIZON}
    action_dims:
      {DOMAIN}: {ACTION_DIM}
    latent_dim: {LATENT_DIM}
    decoder_hidden_dim: 512
    denoiser_hidden_dim: {DENOISER_HIDDEN}
    num_inference_steps: 16
    sampling_schedule:
      1:
        1: 0.5
        2: 0.5
      2001:
        2: 0.8
        4: 0.15
        8: 0.05
    gradient_checkpointing: true
    denoising_module:
      _target_: egomimic.models.denoising_nets.CrossTransformer
      nblocks: {nblocks}
      cond_dim: {DENOISER_HIDDEN}
      hidden_dim: {DENOISER_HIDDEN}
      act_dim: {LATENT_DIM}
      act_seq: {HORIZON}
      n_heads: 8
      dropout: 0.1
      mlp_layers: 4
      mlp_ratio: 4
      time_conditioning: additive
"""


def ar_stages(variant: str, d_model: int, n_layers: int, n_heads: int) -> str:
    return f"""  - _target_: egomimic.pipeline.stages_ar.ARActionDecoder
    variant: {variant}
    condition_input_dim: {COND_DIM}
    action_horizon: {HORIZON}
    action_dims:
      {DOMAIN}: {ACTION_DIM}
    d_model: {d_model}
    n_layers: {n_layers}
    n_heads: {n_heads}
    ff_mult: 4
    dropout: 0.1
    n_waypoints: {M}
    pose_channels: 4
    idm_hidden: 256
    gradient_checkpointing: true
"""


HEADERS = {
    "arm1_dp_flow": (
        "ARM 1 — established baseline: bidirectional, flow matching.\n"
        "# The external reference point. Arms 2-4 use regression+MSE, so arm 1\n"
        "# differs from them in BOTH objective and attention; arm 2 is what\n"
        "# isolates attention on its own."
    ),
    "arm2_causal_bidir": (
        "ARM 2 — THE CONTROL. Same backbone, same head, same loss as arms 3/4;\n"
        "# BIDIRECTIONAL attention and learned queries instead of shifted rows.\n"
        "# Do not drop it: it is the only arm that separates 'causal generation\n"
        "# helps' from 'this backbone + MSE head helps'. Arm 1 cannot, because\n"
        "# it also changes the objective."
    ),
    "arm3_state_action_ar": (
        "ARM 3 — the causal model. Row m of the arc token is predicted from\n"
        "# rows < m, teacher-forced in training and fed back at rollout."
    ),
    "arm4_state_idm": (
        "ARM 4 — causal path + inverse dynamics. Predicts the pose path\n"
        "# causally; the grip channel is recovered from consecutive predicted\n"
        "# poses. Action channels are never fed back, so the arm cannot lean on\n"
        "# action history as a shortcut."
    ),
}


def model_config(arm: str, cap_name: str, cap: dict, params_m: str) -> str:
    variant = ARMS[arm]
    stages = (
        flow_stages(cap["nblocks"])
        if variant is None
        else ar_stages(variant, cap["d_model"], cap["n_layers"], cap["n_heads"])
    )
    detail = (
        f"CrossTransformer nblocks={cap['nblocks']} hidden={DENOISER_HIDDEN}"
        if variant is None
        else f"d_model={cap['d_model']} n_layers={cap['n_layers']} "
             f"n_heads={cap['n_heads']}"
    )
    return f"""# {HEADERS[arm]}
#
# Capacity: {cap_name} — {detail} -> ~{params_m} total params
# (including the shared 11.20M observation encoder). All four arms are matched
# to within 5% at each capacity; see _gen_control_modes.py.
#
# Control-mode study: gripper only, trained on {", ".join(SEEN)},
# held out {", ".join(HELD_OUT)}. Arc token D={D} M={M} r={ROTATION_RADIUS}
# layout={LAYOUT}.
_target_: egomimic.pl_utils.pl_model.ModelWrapper
robomimic_model:
  _target_: egomimic.pipeline.algo.PipelineAlgo
  action_horizon: {HORIZON}
  domains:
  - {DOMAIN}
  ac_keys:
    {DOMAIN}: actions
  stages:
{ENCODER_STAGE}{stages}  - _target_: egomimic.pipeline.stages_sampler.NativeActionMSELoss
  rollout_adapters:
    {DOMAIN}:
      _target_: egomimic.pipeline.pushshapes.PlanarArcRolloutAdapter
      embodiment: {DOMAIN}
      velocity_layout: {LAYOUT}
enable_grad_norm: false
optimizer:
  _target_: torch.optim.AdamW
  _partial_: true
  lr: 0.0001
  weight_decay: 0.0001
scheduler:
  _target_: egomimic.utils.schedulers.warmup_cosine_scheduler
  _partial_: true
  max_steps: 240000
  warmup_steps: 3000
  warmup_start_factor: 0.1
  eta_min: 1.0e-05
"""


def evaluator_config() -> str:
    """One SimRolloutEval per control mode, all on gripper.

    `init_mode` is "seeds". The shipped eval_sim_pushshapes.yaml says "seed",
    which SimRolloutEval rejects outright — that config cannot instantiate.
    """
    blocks = []
    for mode in EVAL_ORDER:
        tag = "unseen" if mode in HELD_OUT else "seen"
        blocks.append(f"""  - _target_: egomimic.eval.core.eval_sim.SimRolloutEval
    embodiment_name: {DOMAIN}
    control_gap: {mode}
    metric_tag: {tag}_{mode}
    env_kwargs:
      object_shape: T
      pusher_shape: gripper
      obstacle_level: 0
      image_size: 96
    init_mode: seeds
    init_seeds: {SEED_LIST}
    max_steps: 600
    coverage_threshold: 0.95
    limit_val_batches: {N_ROLLOUTS}
    max_videos: 1""")
    return f"""# Control-mode rollout eval: {len(SEEN)} SEEN controller modes plus {len(HELD_OUT)} held out.
#
# Ordered by sensing-noise floor, which is the axis the held-out question is
# asked along. TRAINING spans 0.3-0.8; both holdouts sit outside it, on
# opposite sides, so extrapolation is bracketed rather than tested in one
# direction only:
#   ideal   0.0  UNSEEN — below the training range
#   tight   0.3  seen
#   laggy   0.4  seen
#   loose   0.8  seen
#   sticky  0.0  seen   (deadband/gain bias, no sensing noise)
#   jittery 2.5  UNSEEN — 3x above the range; the headline test
#
# Every instance is the same embodiment (gripper) and differs ONLY in the
# control gap the policy is evaluated under, which is the study's independent
# variable. `control_gap` cannot travel through env_kwargs — it is a class
# attribute on PushShapesEnv, so passing it there raises "unexpected
# PushShapesEnv option(s)". SimRolloutEval applies it to the agent instead.
#
# init_mode is "seeds", not "seed": SimRolloutEval validates against
# {{replay, random, seeds}} and rejects anything else at construction.
#
# metric_tag is REQUIRED here and is not decoration. Metrics default to
# `Valid/emb<id>_sim_success_rate`, keyed by embodiment — but all six of these
# share ONE embodiment and differ only in the control gap. Composed evaluators
# merge their metric dicts, so without distinct tags the six collapse to
# whichever ran last: one plausible number instead of six, and nothing raised.
#
# coverage_threshold 0.95 matches the simulator's own SUCCESS_THRESHOLD; a
# lower bar reports successes the environment does not count.
#
# This must be passed as `evaluator=eval_sim_control_modes` in the TRAINING
# phase. Passing ~evaluator there produces a run that reports loss only,
# finishes cleanly, and cannot answer the question it was launched for.
_target_: egomimic.eval.core.eval_composite.EvalVideoList
evals:
{chr(10).join(blocks)}
"""


def main() -> None:
    written = []

    path = DATA_DIR / "control_modes_gripper_arc_D10_M16_append_r0.yaml"
    path.write_text(data_config())
    written.append(path)

    totals = {"large": "313M", "small": "49M"}
    for arm in ARMS:
        for cap_name, cap in CAPACITIES.items():
            path = MODEL_DIR / f"bf_ctrlmode_{arm}_{cap_name}.yaml"
            path.write_text(model_config(arm, cap_name, cap, totals[cap_name]))
            written.append(path)

    path = EVAL_DIR / "eval_sim_control_modes.yaml"
    path.write_text(evaluator_config())
    written.append(path)

    for p in written:
        print(f"wrote {p.relative_to(REPO)}")


if __name__ == "__main__":
    main()
