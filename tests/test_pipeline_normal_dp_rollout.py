import torch

from egomimic.eval.inference_graph import ActionCacheState
from egomimic.pipeline.algo import PipelineAlgo
from egomimic.pipeline.core import Pipeline, Stage
from egomimic.pipeline.stages_io import NormalObsCollapse, NormalObsExpand
from egomimic.pipeline.stages_seq import ObsStack


class _Encode(Stage):
    reads = ["obs/value"]
    writes = ["a_top", "s"]

    def forward(self, batch):
        batch["a_top"] = batch["obs/value"].float()
        batch["s"] = batch["obs/value"].float()
        return batch


class _Head(Stage):
    reads = ["a_top", "s", "embodiment"]
    writes = ["pred_action"]

    def forward(self, batch):
        # One action row whose two values expose [previous, current].
        batch["pred_action"] = batch["a_top"].unsqueeze(1).repeat(1, 2, 1)
        return batch


class _NormalDP(PipelineAlgo):
    def __init__(self):
        # Minimal rollout-only surface; avoid constructing the training runner.
        self.policy = Pipeline([
            NormalObsExpand(n_obs_steps=2),
            _Encode(),
            ObsStack(
                in_keys=["a_top", "s"], out_keys=["a_top", "s"],
                n_obs_steps=2,
            ),
            NormalObsCollapse(keys=["a_top", "s"]),
            _Head(),
        ])

    def activate_rollout_apex_attention(self):
        return None


def test_normal_dp_rollout_reuses_training_adapters_without_target():
    policy = _NormalDP()
    state = policy.init_step_state(1, 20, torch.device("cpu"), torch.float32)

    first = policy.step(
        state, {"value": torch.tensor([[1.0]])}, 0, "eva_bimanual")
    second = policy.step(
        state, {"value": torch.tensor([[2.0]])}, 1, "eva_bimanual")

    torch.testing.assert_close(first[0], torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(second[0], torch.tensor([1.0, 2.0]))
    assert [type(stage).__name__ for stage in state["plan"]] == [
        "NormalObsExpand", "_Encode", "ObsStack", "NormalObsCollapse", "_Head"
    ]


def test_normal_obs_expand_still_requires_actions_outside_rollout():
    stage = NormalObsExpand(n_obs_steps=2)
    try:
        stage({"obs/value": torch.zeros(1, 2, 1)})
    except ValueError as error:
        assert "outside rollout" in str(error)
    else:
        raise AssertionError("target-free training batch was accepted")


def test_normal_obs_training_target_contract_is_unchanged():
    expand = NormalObsExpand(n_obs_steps=2)
    collapse = NormalObsCollapse(keys=["a_top", "s"])
    actions = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2)
    batch = expand({
        "obs/value": torch.tensor([[[1.0], [2.0]]]),
        "actions": actions.clone(),
    })
    batch["a_top"] = batch["obs/value"]
    batch["s"] = batch["obs/value"]
    batch = collapse(batch)

    torch.testing.assert_close(batch["target"], actions)
    torch.testing.assert_close(batch["a_top"], torch.tensor([[2.0]]))


class _IdentityNorm:
    def normalize(self, obs, emb_id):
        return obs

    def unnormalize(self, values, emb_id):
        return values


class _GraphNormalDP(_NormalDP):
    def __init__(self):
        super().__init__()
        self.nets = torch.nn.ModuleDict({"anchor": torch.nn.Linear(1, 1)})
        self.norm_stats = _IdentityNorm()
        self.action_horizon = 20
        self.replan_stride = 2
        self.domain_by_id = {15: "eva_bimanual"}
        self.ac_keys = {"eva_bimanual": "actions"}
        self.inference_stages = None
        self._sim_action_cache = ActionCacheState()
        self._inference_graph = self._build_inference_graph()


def test_dp_cache_keeps_immediately_previous_environment_frame():
    policy = _GraphNormalDP()
    policy._reset_inference_graph()
    policy._sim_emb_id = 15
    policy._sim_ac_key = "actions"

    outputs = []
    for t, value in enumerate((1.0, 2.0, 3.0)):
        policy._sim_t = t
        outputs.append(policy._inference_graph(
            obs={"value": torch.tensor([[value]])}))

    # t1 is a cache hit. At t2 the two-frame DP input must be [t1,t2], not
    # [previous model query t0, current t2].
    torch.testing.assert_close(torch.from_numpy(outputs[0]),
                               torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(torch.from_numpy(outputs[1]),
                               torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(torch.from_numpy(outputs[2]),
                               torch.tensor([2.0, 3.0]))
