import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from astra_reversal import codex_relay
from astra_reversal.action_adapter import ActionSpec
from astra_reversal.demo_skill_agent import png_wire
from astra_reversal.reasoning_learning import online_grounding as grounding
from astra_reversal.reasoning_learning import visual_grounding as visual
from astra_reversal.reasoning_learning.rollout import LearningRollout
from astra_reversal.records import digest


def fixture():
    rng = np.random.default_rng(10)
    steps = grounding.FIT_STEPS + grounding.VALIDATION_STEPS
    xyz = {s: rng.uniform(-0.1, 0.1, 3) for s in steps}
    jacobian = np.asarray([[100, -150, 10], [15, 45, -200]])
    pixels = {s: jacobian @ xyz[s] + 112 for s in steps}
    history = {
        s: {
            "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/state": np.r_[xyz[s], np.zeros(5)],
            "prompt": "place",
        }
        for s in steps
    }
    calls = []

    def ask(frames):
        assert all(set(row) == {"step", "image"} for row in frames)
        calls.append([r["step"] for r in frames])
        return {
            "frames": [
                {
                    "step": r["step"],
                    "visible": True,
                    "confidence": 0.8,
                    "uv": (pixels[r["step"]] / 223).tolist(),
                }
                for r in frames
            ]
        }

    return history, ask, calls, jacobian


def test_projection_requires_separate_validation_and_records_extrapolation():
    history, ask, calls, jacobian = fixture()
    receipt = grounding.estimate(history, ask)
    assert receipt["accepted"]
    assert calls == [list(grounding.FIT_STEPS), list(grounding.VALIDATION_STEPS)]
    assert not set(calls[0]) & set(calls[1])
    assert receipt["validation_rms_pixels"] < 1e-10
    np.testing.assert_allclose(
        receipt["projection"]["jacobian_pixels_per_meter"], jacobian
    )
    current = grounding.context(receipt, [0.5] * 8)
    assert min(current["extrapolation_outside_box_meters"]) > 0.3
    assert receipt["environment_actions_added"] == 0
    assert receipt["source_observation_sha256"][0] == digest(history[0])


@pytest.mark.parametrize("failure", ["few_fit_labels", "bad_validation"])
def test_failed_geometry_supplies_no_numeric_hint(failure):
    history, ask, calls, _ = fixture()

    def altered(frames):
        proposal = ask(frames)
        if failure == "few_fit_labels":
            for row in proposal["frames"][5:]:
                row.update(visible=False, confidence=0, uv=[0, 0])
        elif frames[0]["step"] == 5:
            for row in proposal["frames"]:
                row["uv"][0] += 0.1
        return proposal

    receipt = grounding.estimate(history, altered)
    assert not receipt["accepted"]
    assert grounding.context(receipt, [0] * 8) is None
    assert len(calls) == (1 if failure == "few_fit_labels" else 2)


def test_online_pixel_job_is_episode_bound_and_cannot_contain_future_frames():
    frames = [
        {"step": i, "image": png_wire(np.zeros((8, 8, 3), np.uint8))} for i in range(6)
    ]
    request = visual.build_request(
        frames,
        "place",
        identity={
            "episode_id": "rollout1",
            "request_index": 4,
            "observation_step": 10,
        },
    )
    point = {
        "visible": True,
        "confidence": 0.8,
        "uv": [0.5, 0.5],
        "evidence": "Fingers visible",
    }
    proposal = {
        "frames": [{"step": i, **point} for i in range(6)],
        "destination": point,
    }
    bound = visual.parse_proposal(proposal, request)
    module = codex_relay._module("visual_grounding")
    assert (
        codex_relay._proposal(bound, request, module, "visual_grounding")["episode_id"]
        == "rollout1"
    )
    wrong = copy.deepcopy(bound)
    wrong["episode_id"] = "rollout2"
    with pytest.raises(codex_relay.ClientError, match="identity"):
        codex_relay._proposal(wrong, request, module, "visual_grounding")
    with pytest.raises(ValueError, match="earlier prefix"):
        visual.build_request(
            frames,
            "place",
            identity={
                "episode_id": "rollout1",
                "request_index": 4,
                "observation_step": 5,
            },
        )


def test_rollout_projection_is_once_per_episode_and_never_assessment_input(tmp_path):
    history, ask, calls, _ = fixture()

    def label(request):
        proposal = ask(request["frames"])
        for row in proposal["frames"]:
            row["evidence"] = "Synthetic visible gripper for this unit test"
        proposal["destination"] = {
            "visible": False,
            "confidence": 0,
            "uv": [0, 0],
            "evidence": "Unresolved",
        }
        return visual.parse_proposal(proposal, request)

    seen = []
    teacher = SimpleNamespace(propose=lambda request: seen.append(request) or {})
    policy = SimpleNamespace(
        policy=torch.nn.Linear(1, 1).requires_grad_(False),
        input_transform=lambda x: x,
        output_transform=lambda x: x,
    )
    spec = ActionSpec("test", 10, 32, 0.05, (-1,) * 7, (1,) * 7, {})
    kwargs = dict(
        episode_id="this",
        instruction="place",
        policy_version=0,
        seed=1,
        visual_grounding_client=SimpleNamespace(propose=label),
    )
    with pytest.raises(ValueError, match="Autonomous"):
        LearningRollout(policy, None, spec, tmp_path / "invalid", **kwargs)
    loop = LearningRollout(policy, teacher, spec, tmp_path / "valid", **kwargs)
    loop.observations = {digest(raw): raw for raw in history.values()}
    loop.steps = [
        {
            "episode_id": "this",
            "observation_id": digest(history[s]) if s in history else "unused",
        }
        for s in range(120)
    ]
    loop._maybe_projection(119)
    assert calls == []
    loop._maybe_projection(120)
    loop._maybe_projection(125)
    assert len(calls) == loop.request_index == 2
    assert loop.projection_receipt["accepted"]
    assert (loop.directory / "visual_projection.json").exists()
    snapshots = [{"step": 125, "observation": history[0]}]
    loop._ask("diagnose", 125, snapshots, {"native": np.zeros((10, 7)).tolist()})
    loop._ask("assess", 125, snapshots, {})
    assert "measured_local_image_projection" in seen[0]["context"]
    assert "measured_local_image_projection" not in seen[1]["context"]
    assert loop.request_index == 4
