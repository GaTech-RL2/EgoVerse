"""Freeze the requested benchmark coverage and independently identify new runs."""

import json
from pathlib import Path

import pytest
import yaml

from astra_reversal.frs_experiment import load_protocol as frs_protocol
from astra_reversal.representation_search import load_protocol as vision_protocol

ROOT = Path(__file__).resolve().parents[3]
CONFIGS = ROOT / "astra_reversal/configs"


def test_full_frs_has_200_cases_per_method_and_no_direct_or_learning_arm():
    protocol = frs_protocol(CONFIGS / "frs_frozen_evaluation_v1.json")
    assert len(protocol["suites"]) * protocol["tasks_per_suite"] == 20
    assert protocol["evaluation_states"] == list(range(1, 11))
    assert len(protocol["evaluation_methods"]) == 3
    assert protocol["rounds"] == 0 and not protocol["adaptation_methods"]
    assert frs_protocol()["rounds"] == 3


def test_vision_screen_keeps_native_and_random_controls_with_220_rollout_cap():
    protocol = vision_protocol(CONFIGS / "vision_representation_screen_v1.json")
    assert protocol["evaluation_cases"] == [[i, 0] for i in range(10)]
    assert protocol["arms"] == [
        "native_retry",
        "random_vei",
        "random_vli",
        "astra_vei",
        "astra_vli",
    ]
    assert 20 * (1 + len(protocol["arms"]) * protocol["revisions"]) == 220
    assert len(vision_protocol()["arms"]) == 10


@pytest.mark.parametrize(
    "loader,filename",
    [
        (frs_protocol, "frs_frozen_evaluation_v1.json"),
        (vision_protocol, "vision_representation_screen_v1.json"),
    ],
)
def test_new_campaigns_cannot_disable_provider_stop(loader, filename, tmp_path):
    value = loader(CONFIGS / filename)
    value["astra"]["stop_on_provider_unavailable"] = False
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        loader(path)


@pytest.mark.parametrize(
    "stem,filename",
    [
        ("frs_frozen_evaluation", "frs_frozen_evaluation_v1.json"),
        ("vision_representation_screen", "vision_representation_screen_v1.json"),
    ],
)
def test_workers_bind_the_new_protocol_and_request_l40s(stem, filename):
    workflow = yaml.safe_load(
        (ROOT / f"astra_reversal/osmo/{stem}_l40s.yaml").read_text()
    )["workflow"]
    assert workflow["resources"]["default"]["gpu"] == 1
    assert workflow["resources"]["default"]["platform"] == "ovx-l40s"
    assert len(workflow["tasks"]) == 8
    for i, worker in enumerate(workflow["tasks"]):
        assert worker["environment"]["ASTRA_WORKER_INDEX"] == str(i)
        assert (
            worker["environment"]["ASTRA_PROTOCOL_PATH"]
            == f"astra_reversal/configs/{filename}"
        )
