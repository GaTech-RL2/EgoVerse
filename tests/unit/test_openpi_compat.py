"""openpi pins transformers 4.53 and ships a patch written against it; this repo
runs transformers 5. The shims in egomimic/models/openpi_compat.py bridge the
gap and must disappear on their own once the fork's patch is ported forward."""

import pytest
import torch

pytest.importorskip("openpi.models_pytorch.gemma_pytorch")

from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel  # noqa: E402

from egomimic.models import openpi_compat  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_embed_image():
    original = PaliGemmaWithExpertModel.embed_image
    applied = openpi_compat._APPLIED
    yield
    PaliGemmaWithExpertModel.embed_image = original
    openpi_compat._APPLIED = applied


class _Outputs:
    """What transformers 5's ``get_image_features`` hands back: the vision
    output object with the projected features on ``pooler_output``."""

    def __init__(self, features):
        self.last_hidden_state = torch.zeros_like(features)
        self.pooler_output = features


class _Prefix:
    def __init__(self, result):
        self._result = result

    class _Inner:
        def __init__(self, result):
            self._result = result

        def get_image_features(self, image):
            return self._result

    @property
    def model(self):
        return self._Inner(self._result)


def _embed(result):
    fake = object.__new__(PaliGemmaWithExpertModel)
    fake.paligemma = _Prefix(result)
    return PaliGemmaWithExpertModel.embed_image(fake, torch.zeros(1, 3, 224, 224))


def test_a_wrapped_output_is_unwrapped_to_its_features():
    # Without this, openpi's embed_prefix reads `.shape` off the output object.
    features = torch.randn(2, 256, 2048)
    openpi_compat._patch_embed_image()
    assert torch.equal(_embed(_Outputs(features)), features)


def test_a_plain_tensor_passes_straight_through():
    # openpi's own patched transformers already returns the tensor; the shim
    # must not change that behaviour.
    features = torch.randn(2, 256, 2048)
    openpi_compat._patch_embed_image()
    assert torch.equal(_embed(features), features)


def test_apply_is_a_no_op_the_second_time(caplog):
    openpi_compat._APPLIED = False
    openpi_compat.apply()
    first = PaliGemmaWithExpertModel.embed_image
    openpi_compat.apply()
    assert PaliGemmaWithExpertModel.embed_image is first


def test_the_shim_stands_down_on_the_transformers_openpi_pins(monkeypatch):
    import transformers

    original = PaliGemmaWithExpertModel.embed_image
    monkeypatch.setattr(transformers, "__version__", "4.53.2")
    openpi_compat._APPLIED = False
    openpi_compat.apply()
    assert PaliGemmaWithExpertModel.embed_image is original
