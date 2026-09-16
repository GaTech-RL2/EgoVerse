"""A fake Qwen 3.5 (multi-image chat layout) for the QwenVLA unit tests.

Same shape of stub as ``tests/unit/test_qwen35_vlm.py`` uses for the HPT stem
pair, generalised to N images per sample: every ``<img>`` in the template
expands to one token per (2x2) patch, ``hidden_states`` are returned for
every layer, and the image-token hidden states depend on the pixels so a
wrong image/text split shows up.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import torch
import torch.nn as nn

STUB_HIDDEN = 16
STUB_LAYERS = 2
STUB_IMAGE_SIZE = (4, 8)  # (H, W) -> (4/2) * (8/2) = 8 tokens per frame
STUB_TOKENS_PER_FRAME = 8
IMAGE_TOKEN_ID = 900
STRIDE = 2


class FakeTokenizer:
    """Whitespace tokenizer with a growing vocabulary (ids start at 1)."""

    def __init__(self) -> None:
        self.padding_side = "left"
        self._ids: dict[str, int] = {}
        self._words: dict[int, str] = {}

    def _id(self, word: str) -> int:
        if word not in self._ids:
            index = len(self._ids) + 1
            self._ids[word] = index
            self._words[index] = word
        return self._ids[word]

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [self._id(w) for w in str(text).split()]}

    def decode(self, ids):
        return " ".join(self._words[i] for i in ids)


class FakeProcessor:
    """Chat layout ``<img> <img> ... <prompt>``, right padded."""

    IMAGE = "<img>"

    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False
    ):
        content = messages[0]["content"]
        images = [c for c in content if c["type"] == "image"]
        texts = [c["text"] for c in content if c["type"] == "text"]
        assert content[: len(images)] == images, "images must come first"
        return " ".join([self.IMAGE] * len(images) + texts).strip()

    def __call__(
        self,
        text,
        images,
        padding=True,
        return_tensors="pt",
        do_rescale=None,
        do_resize=None,
    ):
        assert do_rescale is False and do_resize is False
        images = list(images)  # flat: all frames of sample 0, then sample 1, ...
        n_vis = (images[0].shape[-2] // STRIDE) * (images[0].shape[-1] // STRIDE)
        rows = []
        for item in text:
            ids: list[int] = []
            for word in item.split():
                if word == self.IMAGE:
                    ids.extend([IMAGE_TOKEN_ID] * n_vis)
                else:
                    ids.append(self.tokenizer._id(word))
            rows.append(ids)
        width = max(len(r) for r in rows)
        input_ids = torch.zeros(len(rows), width, dtype=torch.long)
        attention = torch.zeros(len(rows), width, dtype=torch.long)
        for i, row in enumerate(rows):
            input_ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
            attention[i, : len(row)] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention,
            "mm_token_type_ids": (input_ids == IMAGE_TOKEN_ID).long(),
            "pixel_values": torch.stack(images).flatten(1),
        }


class FakeInner(nn.Module):
    """``model.model``: the VLM without the lm_head."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(1024, STUB_HIDDEN)
        self.layers = nn.ModuleList(
            [nn.Linear(STUB_HIDDEN, STUB_HIDDEN) for _ in range(STUB_LAYERS)]
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        output_hidden_states=False,
        **kwargs,
    ):
        hidden = self.embed(input_ids)
        if pixel_values is not None:
            batch = input_ids.shape[0]
            # one scalar per sample from ALL of its frames, added on its image tokens
            per_sample = pixel_values.reshape(batch, -1).mean(dim=-1)
            image = (input_ids == IMAGE_TOKEN_ID).unsqueeze(-1).float()
            hidden = hidden + image * per_sample[:, None, None]
        states = [hidden]
        for layer in self.layers:
            hidden = torch.tanh(layer(hidden))
            states.append(hidden)
        out = {"last_hidden_state": hidden}
        if output_hidden_states:
            out["hidden_states"] = tuple(states)
        return SimpleNamespace(**out)


class FakeVLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            image_token_id=IMAGE_TOKEN_ID,
            text_config=SimpleNamespace(
                hidden_size=STUB_HIDDEN, num_hidden_layers=STUB_LAYERS
            ),
            vision_config=SimpleNamespace(patch_size=STRIDE, spatial_merge_size=1),
        )
        self.model = FakeInner()
        self.lm_head = nn.Linear(STUB_HIDDEN, 8, bias=False)
        self.lm_head.weight = nn.Parameter(torch.randn(8, STUB_HIDDEN))

    def gradient_checkpointing_enable(self, **kwargs) -> None:
        self.gradient_checkpointing = True


def _write_stub_snapshot(tmp_path) -> str:
    from safetensors.torch import save_file

    torch.manual_seed(0)
    reference = FakeVLM()
    snapshot = tmp_path / "qwen35_stub"
    snapshot.mkdir()
    state = {k: v.contiguous() for k, v in reference.state_dict().items()}
    state.pop("lm_head.weight")  # tied in the real checkpoint too
    state["mtp.fc.weight"] = torch.zeros(2, 2)
    shard = "model.safetensors-00001-of-00001.safetensors"
    save_file(state, str(snapshot / shard))
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: shard for k in state}}), encoding="utf-8"
    )
    return str(snapshot)


def install_fake_qwen35(tmp_path, monkeypatch) -> str:
    """Write a stub snapshot and patch the two ``transformers`` entry points
    ``Qwen35VLMEncoder`` (hence ``QwenVLBackbone``) loads through."""
    import transformers
    from safetensors.torch import load_file

    path = _write_stub_snapshot(tmp_path)

    def _load_model(load_path, dtype=None, **kwargs):
        model = FakeVLM()
        state = load_file(
            os.path.join(load_path, "model.safetensors-00001-of-00001.safetensors")
        )
        state.pop("mtp.fc.weight", None)
        model.load_state_dict(state, strict=False)
        model.lm_head.weight = nn.Parameter(model.model.embed.weight[:8].clone())
        return model if dtype is None else model.to(dtype)

    monkeypatch.setattr(
        transformers.AutoProcessor, "from_pretrained", lambda *a, **k: FakeProcessor()
    )
    monkeypatch.setattr(
        transformers.AutoModelForImageTextToText,
        "from_pretrained",
        staticmethod(_load_model),
    )
    return path
