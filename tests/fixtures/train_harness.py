"""Shared pieces of the train-step tiers: config composition, environment
hygiene, the batch-key spy and the Pi stub network."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from hydra import compose, initialize_config_module
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

import egomimic.trainHydra as train_hydra
from fixtures.synthetic_episodes import write_episode

STATE_DIM = {"eva_bimanual": 14, "human_bimanual": 12}
PI_TOKENIZER = (
    "google/paligemma-3b-mix-224"  # tokenizer_model_name in model/pi0.5_base.yaml
)


def compose_recipe(recipe, overrides: list[str], out_dir: Path):
    """Compose the recipe's real top-level/data/model configs with overrides and
    install the HydraConfig singleton so ${hydra:...} resolves. Tests reset the
    singleton afterwards (see the autouse fixture in tests/conftest.py)."""
    with initialize_config_module(
        config_module="egomimic.hydra_configs", version_base=None
    ):
        cfg = compose(
            config_name=recipe.top,
            overrides=[
                f"data={recipe.data}",
                f"model={recipe.model}",
                *recipe.extra,
                *overrides,
            ],
            return_hydra_config=True,
        )
    with open_dict(cfg):
        cfg.hydra.runtime.output_dir = str(out_dir)
    HydraConfig.instance().set_config(cfg)
    return cfg


def reset_hydra_config() -> None:
    HydraConfig.instance().cfg = None


def hermetic_env(monkeypatch) -> None:
    """No network, no wandb, no ~/.egoverse_env, and no Lightning Slurm plugin
    inside an srun step."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    monkeypatch.setenv("SLURM_JOB_NAME", "bash")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(train_hydra, "load_env", lambda *a, **k: None)


def write_fixtures(tmp_path: Path, vendor: str, n: int = 3) -> tuple[Path, Path]:
    data, out = tmp_path / "data", tmp_path / "out"
    data.mkdir()
    out.mkdir()
    for i in range(n):
        write_episode(data, vendor, seed=i)
    return data, out


class BatchKeySpy:
    """Records the raw batch keys the policy's process_batch_for_training saw
    (ModelWrapper.training_step hands it the CombinedLoader dict keyed by
    dataset name)."""

    def __init__(self, monkeypatch, cls):
        self.keys: dict[str, set[str]] = {}
        original = cls.process_batch_for_training

        def wrapped(model_self, batch):
            for emb_name, sub in batch.items():
                self.keys.setdefault(emb_name, set()).update(sub.keys())
            return original(model_self, batch)

        monkeypatch.setattr(cls, "process_batch_for_training", wrapped)


def pi_unavailable() -> str | None:
    """Reason to skip Pi tests, or None."""
    try:
        import openpi.models_pytorch.pi0_pytorch  # noqa: F401
    except ImportError:
        # Also the case in CI: openpi is not in uv.lock (pyproject explains).
        return "openpi not importable in this venv (see pi05.md); Pi cases run by hand"
    from huggingface_hub import try_to_load_from_cache

    if try_to_load_from_cache(PI_TOKENIZER, "tokenizer_config.json") is None:
        return (
            f"{PI_TOKENIZER} tokenizer not in the HF cache; run once online: "
            f"AutoTokenizer.from_pretrained({PI_TOKENIZER!r})"
        )
    return None


class StubPI0(nn.Module):
    """Stand-in for openpi.models_pytorch.pi0_pytorch.PI0Pytorch: same
    constructor and forward signature, trivial compute. It asserts on every
    observation field the EgoVerse Pi wrapper must fill, so a wiring gap fails
    here instead of being trained on. Set ``expect`` per test."""

    PI_CAMS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    expect: dict = {}  # {"state_dim": int, "wrist_present": bool}

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.action_in_proj = nn.Linear(32, 64)
        self.action_out_proj = nn.Linear(64, 32)

    def forward(self, observation, actions):
        B = actions.shape[0]
        state_dim = self.expect["state_dim"]
        wrist_present = self.expect["wrist_present"]
        assert observation.state.shape == (B, state_dim), observation.state.shape
        prompt = observation.tokenized_prompt
        assert prompt.shape[0] == B and prompt.ndim == 2 and prompt.shape[1] > 0
        assert tuple(observation.images) == self.PI_CAMS, list(observation.images)
        for k, v in observation.images.items():
            assert v.shape[0] == B and v.ndim == 4 and tuple(v.shape[-2:]) == (224, 224)
            mask = observation.image_masks[k]
            assert mask.shape == (B,), (k, mask.shape)
            expected_mask = True if k == "base_0_rgb" else wrist_present
            assert (
                bool(mask.all()) == expected_mask and bool(mask.any()) == expected_mask
            ), (
                k,
                mask,
            )
        assert actions.shape == (B, self.config.action_horizon, 32), actions.shape
        pred = self.action_out_proj(torch.tanh(self.action_in_proj(actions)))
        return ((pred - actions) ** 2).mean(dim=(-1, -2))

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10):
        B = observation.state.shape[0]
        return torch.zeros(B, self.config.action_horizon, 32, device=device)
