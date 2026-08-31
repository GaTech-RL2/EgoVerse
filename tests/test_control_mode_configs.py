"""Preflight for the control-mode study's configs.

Every failure this catches is otherwise paid for on the training node AFTER the
dataset pull and the norm-stats phase — hours in, with the GPUs already
allocated. Two real examples this file would have caught on a laptop in
seconds:

  * act_seq hardcoded to 16 against action_horizon 17. CrossTransformer adds a
    (1, act_seq, D) positional table to the token sequence, so the add fails
    with a broadcast error on the first forward pass.
  * `init_mode: seed` in an evaluator config, which SimRolloutEval rejects
    ("must be replay/random/seeds") — the evaluator never constructs and the
    run silently reports loss only.

So this instantiates the real modules and RUNS them, rather than checking that
YAML parses.
"""

from __future__ import annotations

import pathlib

import hydra
import pytest
import torch
from omegaconf import OmegaConf

REPO = pathlib.Path(__file__).resolve().parents[1]
MODEL_DIR = REPO / "egomimic/hydra_configs/model/bf"
DATA_CFG = (REPO / "egomimic/hydra_configs/data/pusht"
            / "control_modes_gripper_arc_D10_M16_append_r0.yaml")
EVAL_CFG = REPO / "egomimic/hydra_configs/evaluator/eval_sim_control_modes.yaml"

DOMAIN = "pushshapes_sim_gripper"
HORIZON, ACTION_DIM, COND_DIM = 17, 5, 67
CONFIGS = sorted(p.name for p in MODEL_DIR.glob("bf_ctrlmode_*.yaml"))
CAPACITY_BUDGET = 0.05  # handover TODO-2: match arms to within ~5%


def _stages(name: str):
    cfg = OmegaConf.load(MODEL_DIR / name)
    return [hydra.utils.instantiate(s) for s in cfg.robomimic_model.stages]


def _head_stages(name: str):
    """Everything after the shared observation encoder.

    The encoder needs real images; the head is what the arms actually differ
    in, and it can be driven from a synthetic `condition`.
    """
    return _stages(name)[1:]


def test_there_are_eight_configs_four_arms_at_two_capacities():
    assert len(CONFIGS) == 8, CONFIGS
    for arm in ("arm1_dp_flow", "arm2_causal_bidir", "arm3_state_action_ar",
                "arm4_state_idm"):
        for cap in ("large", "small"):
            assert f"bf_ctrlmode_{arm}_{cap}.yaml" in CONFIGS


@pytest.mark.parametrize("name", CONFIGS)
def test_config_instantiates_and_runs_a_training_forward(name):
    batch = {
        "condition": torch.randn(2, COND_DIM),
        "embodiment": DOMAIN,
        "target": torch.randn(2, HORIZON, ACTION_DIM),
    }
    for stage in _head_stages(name):
        stage.eval()
        with torch.no_grad():
            batch = stage(batch)
    assert batch["pred_action"].shape == (2, HORIZON, ACTION_DIM)
    assert torch.isfinite(batch["loss/native_action"])


@pytest.mark.parametrize("name", CONFIGS)
def test_config_runs_a_rollout_forward_without_a_target(name):
    """At rollout there is no action target. A stage that needs one would
    fail here rather than mid-evaluation on the node."""
    batch = {"condition": torch.randn(1, COND_DIM), "embodiment": DOMAIN,
             "rollout_t": 0}
    for stage in _head_stages(name):
        if getattr(stage, "train_only", False):
            continue
        stage.eval()
        with torch.no_grad():
            batch = stage(batch)
    assert batch["pred_action"].shape == (1, HORIZON, ACTION_DIM)


@pytest.mark.parametrize("capacity", ["large", "small"])
def test_arms_are_parameter_matched(capacity):
    """Otherwise the study measures capacity, not attention."""
    totals = {}
    for name in CONFIGS:
        if not name.endswith(f"_{capacity}.yaml"):
            continue
        n = sum(
            sum(p.numel() for p in stage.parameters())
            for stage in _stages(name)
        )
        totals[name] = n
    assert len(totals) == 4, totals
    lo, hi = min(totals.values()), max(totals.values())
    spread = (hi - lo) / lo
    assert spread <= CAPACITY_BUDGET, (
        f"{capacity}: parameter spread {spread*100:.1f}% exceeds "
        f"{CAPACITY_BUDGET*100:.0f}% — "
        + ", ".join(f"{k}={v/1e6:.1f}M" for k, v in sorted(totals.items()))
    )


@pytest.mark.parametrize("name", CONFIGS)
def test_act_seq_matches_action_horizon(name):
    """The exact bug that kills 9 of the shipped arc configs."""
    cfg = OmegaConf.load(MODEL_DIR / name)
    horizon = int(cfg.robomimic_model.action_horizon)
    for stage in cfg.robomimic_model.stages:
        denoiser = stage.get("denoising_module")
        if denoiser is not None and "act_seq" in denoiser:
            assert int(denoiser.act_seq) == horizon, (
                f"{name}: act_seq={denoiser.act_seq} != "
                f"action_horizon={horizon}; CrossTransformer adds a "
                f"(1, act_seq, D) table and will fail to broadcast"
            )


@pytest.mark.parametrize("name", CONFIGS)
def test_rollout_adapter_layout_matches_the_data_config(name):
    """`append` carries a trailing velocity ROW that is not a waypoint.

    A mismatch here is silent: a mis-shaped command is not rejected, just
    misinterpreted.
    """
    data = OmegaConf.load(DATA_CFG)
    layout = (data.train_datasets[DOMAIN].resolver.transform_list
              .velocity_layout)
    model = OmegaConf.load(MODEL_DIR / name)
    adapter = model.robomimic_model.rollout_adapters[DOMAIN]
    assert adapter.velocity_layout == layout
    m = int(data.train_datasets[DOMAIN].resolver.transform_list
            .resampled_vector_length)
    expected = m + 1 if layout == "append" else m
    assert int(model.robomimic_model.action_horizon) == expected


def test_data_config_transform_instantiates_exactly_as_hydra_will():
    """A config key with no matching factory parameter raises TypeError only
    once the job is on the node, after the pull. That cost 8 of 9 sweep runs."""
    data = OmegaConf.load(DATA_CFG)
    for split in ("train_datasets", "valid_datasets"):
        transforms = hydra.utils.instantiate(
            data[split][DOMAIN].resolver.transform_list
        )
        assert transforms, f"{split}: empty transform list"


def test_evaluator_covers_every_control_mode_exactly_once():
    """All six modes, in the documented order, no duplicates or omissions."""
    cfg = OmegaConf.load(EVAL_CFG)
    gaps = [str(e.control_gap) for e in cfg.evals]
    assert gaps == ["ideal", "tight", "laggy", "loose", "sticky", "jittery"]
    for e in cfg.evals:
        assert e.init_mode == "seeds", "SimRolloutEval rejects 'seed'"
        assert e.embodiment_name == DOMAIN
        assert e.env_kwargs.pusher_shape == "gripper"
        # Must match the simulator's own SUCCESS_THRESHOLD.
        assert float(e.coverage_threshold) == 0.95


def test_training_data_excludes_the_held_out_mode():
    """`jittery` must not reach training through the staged directory."""
    data = OmegaConf.load(DATA_CFG)
    for split in ("train_datasets", "valid_datasets"):
        path = str(data[split][DOMAIN].resolver.folder_path)
        assert "jittery" not in path, path


def test_every_evaluator_has_a_distinct_metric_tag():
    """Six evaluators share ONE embodiment, so emb-keyed metrics collide.

    SimRolloutEval keys metrics as `Valid/emb<id>_sim_success_rate` by default.
    Every instance here is `pushshapes_sim_gripper`, and composed evaluators
    merge their metric dicts — so without distinct tags the six control modes
    collapse to whichever ran last. That failure reports ONE plausible success
    rate instead of six and raises nothing, which would invalidate the study
    silently.
    """
    cfg = OmegaConf.load(EVAL_CFG)
    tags = [e.get("metric_tag") for e in cfg.evals]
    assert all(t is not None for t in tags), f"missing metric_tag: {tags}"
    assert len(set(tags)) == len(tags), f"duplicate metric_tag: {tags}"
    # The tag must name the gap it measures, or the dashboard column is a lie.
    for e in cfg.evals:
        assert str(e.metric_tag).endswith(str(e.control_gap)), (
            f"metric_tag {e.metric_tag!r} does not name gap {e.control_gap!r}")
    assert sorted(tags) == sorted([
        "seen_tight", "seen_loose", "seen_laggy", "seen_sticky",
        "unseen_ideal", "unseen_jittery",
    ])


def test_seen_and_unseen_tags_match_the_training_set():
    """A mode tagged `seen_` must be trainable and `unseen_` must not be.

    Mislabelling here would put a held-out number in a seen column, which is
    the one mistake that inverts the study's conclusion.
    """
    cfg = OmegaConf.load(EVAL_CFG)
    seen = {str(e.control_gap) for e in cfg.evals
            if str(e.metric_tag).startswith("seen_")}
    unseen = {str(e.control_gap) for e in cfg.evals
              if str(e.metric_tag).startswith("unseen_")}
    assert seen == {"tight", "loose", "laggy", "sticky"}
    assert unseen == {"ideal", "jittery"}
    assert not (seen & unseen)


def test_evaluator_config_instantiates_through_hydra():
    """Build the real composite the way trainHydra does.

    The YAML checks above are structural; this is the one that would have
    caught `init_mode: seed`, because SimRolloutEval only rejects it at
    construction. On the node this happens after the dataset pull, so a
    failure here costs hours.
    """
    pytest.importorskip("pymunk")
    ev = hydra.utils.instantiate(OmegaConf.load(EVAL_CFG))
    evals = list(ev.evals)
    assert len(evals) == 6
    tags = [e.metric_tag for e in evals]
    assert len(set(tags)) == 6, tags
    for e in evals:
        assert e.control_gap is not None
        assert e.embodiment_name == DOMAIN
        assert e.init_mode == "seeds"


def test_each_evaluator_env_really_carries_its_own_gap():
    """Construct every env and read the gap back off the agent.

    Six evaluators that all silently ran under `ideal` would produce six
    plausible, nearly identical numbers and no error — the failure mode that
    makes the whole study meaningless.
    """
    pytest.importorskip("pymunk")
    from Tsimulation.pushshapes.agents import CONTROL_GAPS

    ev = hydra.utils.instantiate(OmegaConf.load(EVAL_CFG))
    seen_gaps = {}
    for e in ev.evals:
        env = e._get_env()
        seen_gaps[e.metric_tag] = env.agent.control_gap.as_dict()
        assert seen_gaps[e.metric_tag] == CONTROL_GAPS[e.control_gap].as_dict()
    # Distinct gaps must remain distinct after construction.
    distinct = {tuple(sorted(v.items())) for v in seen_gaps.values()}
    assert len(distinct) == len(seen_gaps), (
        f"control gaps collapsed: {seen_gaps}")
