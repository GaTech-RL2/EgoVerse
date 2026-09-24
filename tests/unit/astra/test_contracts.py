import copy
import json

import numpy as np
import pytest

from astra_reversal.action_adapter import ActionAdapter
from astra_reversal.agent import ReplayBackend, build_request, parse_proposal
from astra_reversal.augmentation import augment

from .conftest import SyntheticBackend, SyntheticEnvironment, SyntheticPolicy


def request(spec, stage=1):
    return build_request(
        episode_id="e1",
        step=0,
        observation=SyntheticEnvironment().observe(),
        instruction="move cup",
        history=[],
        active=None,
        spec=spec,
        stage=stage,
        completion_types=("eef_position", "gripper_width"),
    )


def parse(raw, req):
    return parse_proposal(
        raw,
        req,
        model_version="synthetic-test",
        request_time=0,
        response_time=1,
        max_timeout=60,
        completion_types=("eef_position", "gripper_width"),
    )


def test_transforms_normalize_then_pad_and_preserve_internal_channels(spec):
    policy = SyntheticPolicy()
    adapter = ActionAdapter(spec, policy.input_transform, policy.output_transform)
    observation = SyntheticEnvironment().observe()
    actions = np.full((10, 7), 0.25)
    model = adapter.encode(actions, observation)
    assert model.shape == (1, 10, 32)
    np.testing.assert_allclose(model[..., :7], 0.1)
    np.testing.assert_array_equal(model[..., 7:], 0)
    # Internal channels remain present until the output transform drops them.
    model[..., 7:] = 20
    decoded, clipped = adapter.decode(
        model, policy.input_transform(observation)["state"][None]
    )
    np.testing.assert_allclose(decoded, actions)
    assert clipped["count"] == 0
    np.testing.assert_array_equal(actions, np.full((10, 7), 0.25))


@pytest.mark.parametrize(
    "change",
    [
        lambda x: x.update(observation_step=1),
        lambda x: x.update(episode_id="different"),
        lambda x: x.update(action_spec_id="wrong"),
        lambda x: x.update(action_chunk=[[0] * 7] * 9),
        lambda x: x.update(action_chunk=[[0] * 6] * 10),
        lambda x: x["action_chunk"][0].__setitem__(0, float("nan")),
        lambda x: x["action_chunk"][0].__setitem__(0, 2),
        lambda x: x["action_chunk"][0].__setitem__(0, True),
        lambda x: x["action_chunk"][0].__setitem__(0, "0.2"),
        lambda x: x.update(timeout_env_steps=100),
        lambda x: x.update(completion={"type": "sim_success", "parameters": {}}),
        lambda x: x.update(extra_motion="fill this in"),
    ],
)
def test_invalid_proposals_are_rejected_without_repairs(spec, change):
    req = request(spec)
    data = json.loads(SyntheticBackend().generate(req))
    change(data)
    with pytest.raises((ValueError, TypeError)):
        parse(json.dumps(data), req)


def test_replay_is_bound_to_pixels_not_only_episode_step(spec, tmp_path):
    req = request(spec)
    path = tmp_path / "replay.jsonl"
    raw = SyntheticBackend().generate(req)
    path.write_text(
        json.dumps(
            {
                "request_fingerprint": req["request_fingerprint"],
                "raw_response": raw,
                "model_version": "recorded-model",
            }
        )
        + "\n"
    )
    assert ReplayBackend(path).generate(req) == raw
    changed = copy.deepcopy(req)
    changed["observation"]["observation/image"][0, 0] = 2
    from astra_reversal.records import digest

    changed.pop("request_fingerprint")
    changed["request_fingerprint"] = digest(changed)
    with pytest.raises(ValueError, match="differs"):
        ReplayBackend(path).generate(changed)


def test_stage2_annotations_expire_per_camera_and_keep_raw_state(spec):
    req = request(spec, stage=2)
    proposal = parse(SyntheticBackend().generate(req), req)
    original = req["observation"]
    rendered, events = augment(original, proposal, original)
    assert events[0]["event"] == "annotation_applied"
    assert np.any(rendered["observation/image"] != original["observation/image"])
    np.testing.assert_array_equal(
        rendered["observation/state"], original["observation/state"]
    )
    assert np.all(original["observation/image"] == 0)
    changed = copy.deepcopy(original)
    changed["observation/image"][10, 10] = 1
    expired, events = augment(changed, proposal, original)
    assert events[0]["event"] == "annotation_expired"
    np.testing.assert_array_equal(
        expired["observation/image"], changed["observation/image"]
    )


def test_stage2_rejects_unavailable_camera_and_out_of_bounds_pixels(spec):
    req = request(spec, stage=2)
    data = json.loads(SyntheticBackend().generate(req))
    data["annotations"][0]["camera"] = "extra_camera"
    with pytest.raises(ValueError):
        parse(json.dumps(data), req)
    data["annotations"][0]["camera"] = "observation/image"
    data["annotations"][0]["coordinates"] = [0, 0, 999, 999]
    with pytest.raises(ValueError):
        parse(json.dumps(data), req)


def test_policy_output_bounds_are_explicit_and_logged(spec):
    policy = SyntheticPolicy()
    adapter = ActionAdapter(spec, policy.input_transform, policy.output_transform)
    model = np.full((1, 10, 32), 2, np.float32)
    decoded, clipping = adapter.decode(model, np.zeros((1, 32)), bounds="clip")
    assert clipping["count"] == 70
    np.testing.assert_array_equal(decoded, 1)
    with pytest.raises(ValueError, match="bounds"):
        adapter.decode(model, np.zeros((1, 32)), bounds="reject")
