"""CPU tier: every (vendor, algo) recipe builds and takes 2 optimizer steps on
synthetic data through the real trainHydra.train() path. Pi's openpi network
is a stub; the EgoVerse Pi wrapper (keymaps, converters, prompts, batch
assembly) is real."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from fixtures.recipes import (
    RECIPES,
    common_overrides,
    cpu_trainer_overrides,
    hpt_small_overrides,
    pi_cpu_overrides,
)
from fixtures.synthetic_episodes import write_episode
from hydra import compose, initialize_config_module
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

import egomimic.trainHydra as train_hydra
from egomimic.algo.hpt import HPT
from egomimic.rldb.embodiment.eva import Eva
from egomimic.rldb.embodiment.human import Human

VENDOR_NAMES = ["eva", "aria", "mecka", "scale"]
STEPS = 2


def compose_recipe(recipe, overrides: list[str], out_dir: Path):
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


class BatchKeySpy:
    """Records the raw batch keys the policy's process_batch_for_training saw."""

    def __init__(self, monkeypatch, cls):
        self.keys: dict[str, set[str]] = {}
        original = cls.process_batch_for_training

        def wrapped(model_self, batch):
            for emb_name, sub in batch.items():
                self.keys.setdefault(emb_name, set()).update(sub.keys())
            return original(model_self, batch)

        monkeypatch.setattr(cls, "process_batch_for_training", wrapped)


def hermetic_env(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    monkeypatch.setenv("SLURM_JOB_NAME", "bash")  # Lightning: srun step is interactive
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)


def _write_fixtures(tmp_path: Path, vendor: str) -> tuple[Path, Path]:
    data, out = tmp_path / "data", tmp_path / "out"
    data.mkdir()
    out.mkdir()
    for i in range(3):
        write_episode(data, vendor, seed=i)
    return data, out


def _run(cfg) -> float:
    metrics, objects = train_hydra.train(cfg)
    loss = float(metrics["Train/action_loss"])
    assert math.isfinite(loss), f"non-finite loss {loss}"
    assert objects["trainer"].global_step == STEPS
    return loss


@pytest.mark.parametrize("vendor", VENDOR_NAMES)
def test_hpt_train_step(tmp_path, monkeypatch, vendor):
    hermetic_env(monkeypatch)
    recipe = RECIPES[(vendor, "hpt")]
    data, out = _write_fixtures(tmp_path, vendor)
    cfg = compose_recipe(
        recipe,
        common_overrides(recipe.embodiment, data, out, batch_size=2, num_workers=0)
        + cpu_trainer_overrides(STEPS)
        + hpt_small_overrides(recipe.embodiment),
        out,
    )
    spy = BatchKeySpy(monkeypatch, HPT)
    _run(cfg)
    seen = spy.keys[recipe.embodiment]
    is_eva = recipe.embodiment == "eva_bimanual"
    front = Eva.VIZ_IMAGE_KEY if is_eva else Human.VIZ_IMAGE_KEY
    expected = {"actions_cartesian", "observations.state.ee_pose", front}
    if is_eva:
        expected |= {
            "observations.images.right_wrist_img",
            "observations.images.left_wrist_img",
        }
    assert (
        expected <= seen
    ), f"{vendor}/hpt missing {expected - seen}; saw {sorted(seen)}"


# ---------------------------------------------------------------- Pi (stub net)


def _pi_unavailable() -> str | None:
    try:
        import openpi.models_pytorch.pi0_pytorch  # noqa: F401
    except ImportError:
        return "openpi not importable in this venv (see pi05.md)"
    from huggingface_hub import try_to_load_from_cache

    cached = try_to_load_from_cache(
        "google/paligemma-3b-mix-224", "tokenizer_config.json"
    )
    if cached is None:
        return (
            "PaliGemma tokenizer not in the HF cache; run once online: "
            "AutoTokenizer.from_pretrained('google/paligemma-3b-mix-224')"
        )
    return None


class StubPI0(nn.Module):
    """Stand-in for openpi.models_pytorch.pi0_pytorch.PI0Pytorch with the same
    call surface. It asserts on every observation field the EgoVerse wrapper
    is supposed to fill, so a wiring gap fails here instead of training on it."""

    PI_CAMS = {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05
        self.action_in_proj = nn.Linear(32, 64)
        self.action_out_proj = nn.Linear(64, 32)
        self.gradient_checkpointing_enabled = False

    def forward(self, observation, actions):
        assert observation.state.ndim == 2, observation.state.shape
        prompt = observation.tokenized_prompt
        assert prompt.ndim == 2 and prompt.shape[1] > 0, prompt.shape
        assert set(observation.images) == self.PI_CAMS, sorted(observation.images)
        for k, v in observation.images.items():
            assert v.ndim == 4 and tuple(v.shape[-2:]) == (224, 224), (k, v.shape)
            assert observation.image_masks[k].shape[0] == v.shape[0], k
        assert actions.shape[-1] == 32, actions.shape
        pred = self.action_out_proj(torch.tanh(self.action_in_proj(actions)))
        return ((pred - actions) ** 2).mean(dim=(-1, -2))

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10):
        B = observation.state.shape[0]
        return torch.zeros(B, self.config.action_horizon, 32, device=device)

    def gradient_checkpointing_enable(self):
        pass

    def gradient_checkpointing_disable(self):
        pass


@pytest.mark.parametrize("vendor", VENDOR_NAMES)
def test_pi_train_step(tmp_path, monkeypatch, vendor):
    reason = _pi_unavailable()
    if reason:
        pytest.skip(reason)
    import openpi.models_pytorch.pi0_pytorch as pi0_pytorch

    from egomimic.algo.pi import PI

    hermetic_env(monkeypatch)
    monkeypatch.setattr(pi0_pytorch, "PI0Pytorch", StubPI0)
    recipe = RECIPES[(vendor, "pi")]
    data, out = _write_fixtures(tmp_path, vendor)
    cfg = compose_recipe(
        recipe,
        common_overrides(recipe.embodiment, data, out, batch_size=2, num_workers=0)
        + cpu_trainer_overrides(STEPS)
        + pi_cpu_overrides(),
        out,
    )
    spy = BatchKeySpy(monkeypatch, PI)
    _run(cfg)
    seen = spy.keys[recipe.embodiment]
    expected = {
        "actions_cartesian",
        "observations.state.ee_pose",
        "base_0_rgb",
        "annotations",
    }
    if recipe.embodiment == "eva_bimanual":
        expected |= {"left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert (
        expected <= seen
    ), f"{vendor}/pi missing {expected - seen}; saw {sorted(seen)}"
