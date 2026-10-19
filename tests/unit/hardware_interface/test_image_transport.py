"""Generated debug summaries must not turn into invalid provider image inputs."""

import base64
import copy
import io
import json

import pytest
from PIL import Image

from astra_reversal.hardware_interface.provider import observation_content


def frame():
    buffer = io.BytesIO()
    Image.new("RGB", (7, 5), (13, 27, 51)).save(buffer, format="PNG")
    return {
        "encoding": "image/png",
        "base64": base64.b64encode(buffer.getvalue()).decode(),
        "shape": [5, 7, 3],
    }


@pytest.mark.parametrize(
    "change",
    [
        {"shape": "[5, 7, 3]"},
        {"base64": "not base64"},
        {"shape": [6, 7, 3]},
        {"shape": [5, 7, 4]},
        {"shape": [True, 7, 3]},
        {"shape": None},
        {"base64": None},
    ],
)
def test_image_shaped_debug_values_stay_exact_text(change):
    value = {"result": {"preview": {**frame(), **change}}}
    original = copy.deepcopy(value)
    text, images = observation_content(value)
    assert json.loads(text) == original and images == []
    assert value == original


@pytest.mark.parametrize("length", [80, -16])
def test_truncated_camera_payload_is_not_attached_even_with_valid_shape(length):
    value = frame()
    value["base64"] = value["base64"][:length]
    text, images = observation_content(value)
    assert json.loads(text) == value and images == []


def test_exact_live_debug_preview_does_not_hide_real_camera_pixels():
    camera = frame()
    # The live CAP agent summarized each field using str(value)[:80]. The old
    # adapter interpreted that nested summary as an actual PNG and got HTTP 400.
    preview = {key: str(value)[:80] for key, value in camera.items()}
    value = {"result": {"val_repr": preview}, "camera": {"value": camera}}
    original = copy.deepcopy(value)
    text, images = observation_content(value)
    converted = json.loads(text)
    assert converted["result"]["val_repr"] == preview
    assert converted["camera"]["value"]["image_attachment"] == "result/camera/value"
    assert len(images) == 2 and images[1] == {
        "type": "input_image",
        "image_url": "data:image/png;base64," + camera["base64"],
    }
    assert value == original
