"""Prevent implicit provider changes and cross-harness result attribution."""

import copy
import json
from pathlib import Path

import pytest
import yaml

from astra_reversal.frs_experiment import load_protocol as frs_protocol
from astra_reversal.provider_stop import ProviderUnavailable, require_provider_available
from astra_reversal.reasoner_backend import make_client
from astra_reversal.representation_search import load_protocol as vision_protocol

CONFIGS = Path(__file__).resolve().parents[3] / "astra_reversal/configs"


@pytest.mark.parametrize(
    "loader,filename",
    [
        (frs_protocol, "frs_codex_frozen_evaluation_v1.json"),
        (vision_protocol, "vision_codex_representation_screen_v1.json"),
    ],
)
def test_codex_has_an_explicit_separate_study_and_cap_limit(loader, filename, tmp_path):
    protocol = loader(CONFIGS / filename)
    assert protocol["astra"]["backend"] == "codex_relay"
    assert protocol["astra"]["model"] == "gpt-6-astra"
    assert protocol["astra"]["max_completion_tokens"] is None
    for changes in (
        {"model": "gpt-6-sol"},
        {"backend": "nvidia_http"},
        {"stop_on_provider_unavailable": False},
        {"max_completion_tokens": 8192},
    ):
        modified = copy.deepcopy(protocol)
        modified["astra"].update(changes)
        path = tmp_path / "invalid.json"
        path.write_text(json.dumps(modified))
        with pytest.raises(ValueError):
            loader(path)


def test_existing_http_protocol_cannot_silently_become_codex(tmp_path):
    protocol = frs_protocol(CONFIGS / "frs_frozen_evaluation_v1.json")
    protocol["astra"]["backend"] = "codex_relay"
    path = tmp_path / "mixed.json"
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError):
        frs_protocol(path)


def test_codex_transport_failure_stops_even_after_job_started():
    with pytest.raises(ProviderUnavailable, match="reasoner_backend_unavailable"):
        require_provider_available(
            {"provider_call": True, "provider_unavailable": True}
        )
    require_provider_available(
        {"provider_call": True, "provider_unavailable": False, "accepted": False}
    )


def test_factory_cannot_fall_back_to_http_for_unknown_backend(tmp_path):
    settings = frs_protocol(CONFIGS / "frs_frozen_evaluation_v1.json")["astra"]
    settings["backend"] = "typo"
    with pytest.raises(ValueError, match="Unknown reasoner backend"):
        make_client("frs", settings, tmp_path / "ledger.jsonl")


@pytest.mark.parametrize(
    "name",
    ("frs_codex_frozen_evaluation_l40s", "vision_codex_representation_screen_l40s"),
)
def test_codex_workers_require_no_external_inference_key(name):
    path = CONFIGS.parent / "osmo" / f"{name}.yaml"
    workflow = yaml.safe_load(path.read_text())["workflow"]
    assert workflow["resources"]["default"]["gpu"] == 1
    assert len(workflow["tasks"]) == 8
    for task in workflow["tasks"]:
        assert "astra-reversal-inference-20260924" not in task["credentials"]
        assert task["environment"]["ASTRA_RETAIN_FAILED_WORKER"] == "0"
        assert task["environment"]["ASTRA_CODEX_RELAY_HOST"] == "127.0.0.1"
        assert "ASTRA_CODEX_RELAY_TOKEN" not in task["environment"]
