import numpy as np
import torch

from egomimic.algo.pi import PI

# NOTE: the former ``test_visualize_preds_*`` tests were removed: visualization
# was moved off the ``PI`` algo into the external evaluators (commit
# "Refactor Eval: Metrics are computed in external eval class"), so the
# ``PI.visualize_preds`` method and ``pi.draw_actions`` import they exercised no
# longer exist.


# ---------------------------------------------------------------------------
# Short-term memory: "Prev:" prompt block built from recent EE-pose history.
# These exercise prompt assembly only (no tokenizer / model / data).
# ---------------------------------------------------------------------------


def _make_prompt_pi(**overrides):
    """Bare PI with just the attributes the prompt-assembly methods read."""
    pi = object.__new__(PI)
    pi.annotation_key = None
    pi.default_prompt = "do task"
    pi.sampling_mode = "first"
    pi.proprio_in_prompt = True
    pi.embodiment_label = False
    pi.control_mode = None
    pi.prev_state_in_prompt = False
    pi.history_len = 1
    pi.proprio_keys_for_prompt = ["observations.state.ee_pose"]
    pi.state_num_bins = 256
    pi.prev_state_num_bins = 64
    pi._state_bin_edges = np.linspace(-1.0, 1.0, 256 + 1)[:-1]
    pi._prev_bin_edges = np.linspace(-1.0, 1.0, 64 + 1)[:-1]
    for key, val in overrides.items():
        setattr(pi, key, val)
    return pi


def _hist_batch(prev=0.4, cur=0.5, dim=4):
    # [B=1, K=2, D]: row 0 is the previous frame, row 1 (last) is the current.
    return {
        "observations.state.ee_pose": torch.tensor(
            [[[prev] * dim, [cur] * dim]], dtype=torch.float32
        )
    }


def test_prev_block_absent_when_disabled():
    pi = _make_prompt_pi(prev_state_in_prompt=False, history_len=2)
    out = pi._build_prompts(_hist_batch(), "eva_bimanual", 1)
    assert len(out) == 1
    assert "Prev1:" not in out[0]
    assert "State:" in out[0]
    assert out[0].endswith(";\nAction: ")


def test_prev_block_present_and_ordered_when_enabled():
    pi = _make_prompt_pi(prev_state_in_prompt=True, history_len=2)
    out = pi._build_prompts(_hist_batch(), "eva_bimanual", 1)[0]
    assert "State:" in out and "Prev1:" in out
    assert out.index("Prev1:") < out.index("State:")  # recent motion then current
    assert out.endswith(";\nAction: ")


def test_prev_delta_bins_zero_motion_is_center_bin():
    pi = _make_prompt_pi(prev_state_in_prompt=True, history_len=2)
    # K=2 window -> one past frame -> a single-element list.
    s = pi._discretize_prev_states_for_sample(_hist_batch(prev=0.3, cur=0.3, dim=3), 0)
    center = int(np.digitize(0.0, np.linspace(-1.0, 1.0, 65)[:-1]) - 1)
    assert s == [" ".join([str(center)] * 3)]


def test_prev_delta_bins_known_value():
    pi = _make_prompt_pi(prev_state_in_prompt=True, history_len=2)
    s = pi._discretize_prev_states_for_sample(_hist_batch(prev=0.4, cur=0.5, dim=2), 0)
    edges = np.linspace(-1.0, 1.0, 65)[:-1]
    expected = int(np.digitize(0.4 - 0.5, edges) - 1)  # delta = prev - cur
    assert s == [" ".join([str(expected)] * 2)]


def test_prev_returns_empty_without_history():
    pi = _make_prompt_pi(prev_state_in_prompt=True, history_len=2)
    # Single-frame proprio [B, D] -> per-sample [D] -> no history window.
    batch = {
        "observations.state.ee_pose": torch.tensor(
            [[0.5, 0.5, 0.5]], dtype=torch.float32
        )
    }
    assert pi._discretize_prev_states_for_sample(batch, 0) == []
    assert "Prev1:" not in pi._build_prompts(batch, "eva_bimanual", 1)[0]


def test_prev_caps_number_of_past_frames():
    pi = _make_prompt_pi(prev_state_in_prompt=True, history_len=5)
    # K=5 window -> 4 past frames, but only _PREV_MAX_FRAMES are encoded, each
    # as its own Prev<j> bin-string of width `dim`.
    dim = 2
    window = torch.arange(5 * dim, dtype=torch.float32).reshape(1, 5, dim) / 100.0
    s = pi._discretize_prev_states_for_sample({"observations.state.ee_pose": window}, 0)
    assert len(s) == PI._PREV_MAX_FRAMES
    assert all(len(frame.split()) == dim for frame in s)


def test_tokenize_truncation_warns_once():
    pi = object.__new__(PI)
    pi.tokenizer_max_length = 4
    pi._prompt_trunc_warned = False

    def fake_tokenizer(prompts, padding, truncation, max_length, return_tensors):
        n = len(prompts)
        # No padding column -> the window is full -> truncation suspected.
        return {
            "input_ids": torch.ones(n, 4, dtype=torch.long),
            "attention_mask": torch.ones(n, 4, dtype=torch.long),
        }

    pi.tokenizer = fake_tokenizer
    out = pi._tokenize_prompts(["a"])
    assert pi._prompt_trunc_warned is True
    assert out["tokenized_prompt"].shape == (1, 4)
    # token_loss_mask masks out the final (anchor) position.
    assert out["token_loss_mask"][0, -1].item() is False
    pi._tokenize_prompts(["b"])  # stays warned, no crash
    assert pi._prompt_trunc_warned is True
