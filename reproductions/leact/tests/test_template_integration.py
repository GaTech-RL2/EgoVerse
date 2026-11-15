"""Exercise the installed LLaMA-Factory template, without model weights."""
from copy import deepcopy
import json

import pytest
import yaml

from reproductions.leact.data import file_sha256
from reproductions.leact.training import conversation, write_recipe


class CharacterTokenizer:
    """Reversible tokenizer for checking template boundaries and loss targets."""
    bos_token_id = None
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return list(map(ord, text))

    def decode(self, tokens):
        return "".join(map(chr, tokens))


@pytest.mark.parametrize("method,thought", [("bc", ""), ("leact", "A weak hand faces pressure.")])
def test_actual_template_masks_empty_bc_thought_but_preserves_reasoning(tmp_path, method, thought):
    template_module = pytest.importorskip("llamafactory.data.template")
    answer = f'<think>\n{thought}\n</think>\n\nAction: fold\nPolicy: {{"fold": 1.0}}'
    data = [conversation("Choose a strategy.", answer)]
    (tmp_path / "data.json").write_text(json.dumps(data))
    (tmp_path / "manifest.json").write_text(json.dumps({
        "method": method, "context_length_checked": True, "cutoff_len": 2048,
        "data_sha256": file_sha256(tmp_path / "data.json"),
    }))
    recipe = yaml.safe_load(write_recipe(tmp_path, "unused-model").read_text())
    messages = [{"role": "user", "content": "Choose a strategy."},
                {"role": "assistant", "content": answer}]
    tokenizer = CharacterTokenizer()
    original = deepcopy(template_module.TEMPLATES["qwen3"])
    original.enable_thinking = True
    old_prompt, old_target = original.encode_oneturn(tokenizer, messages)
    configured = deepcopy(original)
    configured.enable_thinking = recipe["enable_thinking"]
    prompt, target = configured.encode_oneturn(tokenizer, messages)
    assert prompt + target == old_prompt + old_target
    decoded = tokenizer.decode(target)
    if method == "bc":
        assert decoded.startswith("Action: fold")
        assert "<think>" not in decoded and "</think>" not in decoded
        assert tokenizer.decode(prompt).endswith("<think>\n\n</think>\n\n")
    else:
        assert decoded.startswith("<think>") and thought in decoded
