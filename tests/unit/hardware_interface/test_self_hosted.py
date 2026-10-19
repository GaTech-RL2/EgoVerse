"""Self-hosting must preserve model identity, images and private credential boundaries."""

import copy
import io
import json
from pathlib import Path

import pytest
from test_contract import proxy

from astra_reversal.hardware_interface.protocol import load_manifest, validate
from astra_reversal.hardware_interface.provider import (
    HTTP,
    LoopbackHTTP,
    Meter,
    Session,
    model_http,
)
from astra_reversal.hardware_interface.replication import (
    DEEPSEEK_MODEL,
    DEEPSEEK_REVISION,
    VLLM_IMAGE,
    self_hosted_design,
)
from experiments.libero_hardware_interface.tools.self_hosted_campaign import (
    serving_host,
)
from experiments.libero_hardware_interface.tools.self_hosted_server import command

STUDY = Path(__file__).resolve().parents[3] / "experiments/libero_hardware_interface"


@pytest.mark.parametrize("suffix", ["", "-adaptive"])
def test_self_hosted_manifest_is_explicit_and_retains_the_matched_design(suffix):
    original = load_manifest(STUDY / ("preregistration" + suffix + ".yaml"))
    new = self_hosted_design(original)
    saved = load_manifest(
        STUDY / ("preregistration-deepseek-selfhost" + suffix + ".yaml")
    )
    assert saved == new
    assert new["model_replication"]["requested_hf_revision"] == DEEPSEEK_REVISION
    assert new["model_replication"]["serving_image"] == VLLM_IMAGE
    assert new["observer_model"] == new["model"]["identifier"] == DEEPSEEK_MODEL
    for field in (
        "conditions",
        "limits",
        "environment_seed",
        "pilot_indices",
        "task_ids",
        "randomization_seed",
        "container_digest",
    ):
        assert new[field] == original[field]
    for field, wrong in (
        ("identifier", "deepseek-flash"),
        ("numeric_reasoning_effort", 50),
        ("snapshot_pinned", False),
        ("base_url", "http://10.0.0.1/v1"),
    ):
        bad = copy.deepcopy(new)
        bad["model"][field] = wrong
        if field == "identifier":
            bad["observer_model"] = wrong
        with pytest.raises(ValueError, match="unrecognized_model_replication"):
            validate(bad)


def test_loopback_transport_never_reads_or_sends_hosted_credentials(monkeypatch):
    monkeypatch.setenv("HARDWARE_API_KEY_FILE", "/must/not/be/read")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-forwarded")
    client = LoopbackHTTP("http://127.0.0.1:18080/v1")
    requests = []

    class Opener:
        def open(self, request, timeout):
            requests.append(request)
            return io.BytesIO(b'{"ok":true}')

    client.opener = Opener()
    assert client("responses", {}, 180) == {"ok": True}
    assert requests[0].get_header("Authorization") is None
    for url in (
        "http://127.0.0.1:18080/v1?redirect=evil",
        "http://example.com/v1",
        "https://example.com/v1",
    ):
        with pytest.raises(ValueError):
            LoopbackHTTP(url)
    with pytest.raises(ValueError, match="https"):
        HTTP("http://example.com/v1", key="fixture")
    manifest = load_manifest(STUDY / "preregistration-deepseek-selfhost.yaml")
    assert isinstance(model_http(manifest["model"]), LoopbackHTTP)


def test_vllm_requests_record_numeric_effort_and_preserve_images_and_reasoning(
    tmp_path,
):
    manifest = load_manifest(STUDY / "preregistration-deepseek-selfhost.yaml")
    p = proxy(tmp_path, limits=validate(manifest))
    captured = []
    reasoning = {
        "type": "reasoning",
        "content": [{"type": "reasoning_text", "text": "fixture only"}],
    }

    def post(path, body, timeout):
        captured.append(copy.deepcopy(body))
        return {
            "model": DEEPSEEK_MODEL,
            "status": "completed",
            "usage": {"input_tokens": 20, "output_tokens": 5},
            "output": [
                reasoning,
                {
                    "type": "function_call",
                    "name": "finish",
                    "arguments": "{}",
                    "call_id": "c",
                },
            ],
        }

    session = Session(manifest["model"], p.limits, Meter(None), p.events, post=post)
    session.request("same interface prompt", [{"type": "function", "name": "finish"}])
    session.tool_result("c", p.observe(p.camera_keys))
    session.request("same interface prompt", [{"type": "function", "name": "finish"}])
    assert captured[1]["input"][0] == reasoning
    assert (
        sum(
            item["type"] == "input_image"
            for item in captured[1]["input"][-1]["content"]
        )
        == 2
    )
    for request in captured:
        assert "reasoning" not in request
        assert request["chat_template_kwargs"] == {
            "thinking": True,
            "reasoning_effort": 75,
            "drop_thinking": False,
        }
        assert request["temperature"] == request["top_p"] == 1
        assert request["seed"] == 42 and "max_output_tokens" not in request
        assert request["model"] == DEEPSEEK_MODEL


def test_serving_group_targets_private_address_and_native_mixed_precision():
    assert serving_host("10.244.1.2") == "10.244.1.2"
    for value in ("127.0.0.1", "8.8.8.8", "0.0.0.0", "example.com"):
        with pytest.raises(ValueError):
            serving_host(value)
    args = command(Path("/pinned/snapshot"))
    assert args[args.index("--tensor-parallel-size") + 1] == "8"
    assert args[args.index("--served-model-name") + 1] == DEEPSEEK_MODEL
    assert json.loads(args[args.index("--engram-config") + 1]) == {"cpu_offload": True}
    assert (
        json.loads(args[args.index("--speculative-config") + 1])["method"] == "dspark"
    )
    assert "--language-model-only" not in args
