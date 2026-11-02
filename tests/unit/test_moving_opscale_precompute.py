"""The real normalization CLI must compose a budget-directory primary config."""

import json
import sys
from pathlib import Path

import pytest

import egomimic
from egomimic.scripts import precompute_norm_stats as precompute


def test_precompute_external_primary_config_and_reference_seed(monkeypatch, tmp_path):
    root = Path(egomimic.__file__).parent.parent
    private = tmp_path / "private"
    (private / "splits").mkdir(parents=True)
    split = {
        "train": ["synthetic_train"],
        "operators": ["synthetic_operator"],
        "seen_matched": ["synthetic_matched"],
        "seen_perlevel": ["synthetic_perlevel"],
        "unseen": ["synthetic_unseen"],
        "video_seen_matched": ["synthetic_matched"],
        "video_seen_perlevel": ["synthetic_perlevel"],
        "video_unseen": ["synthetic_unseen"],
    }
    (private / "splits/fold_moving_b24_op4.json").write_text(json.dumps(split))
    monkeypatch.setenv("EGOVERSE_MECKA_PRIVATE_DIR", str(private))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "precompute_norm_stats.py",
            "--config-name",
            "op_4",
            "--config-dir",
            str(root / "experiments/fold_clothes_moving_opscale/budget_24"),
            "--data",
            "mecka_fold_moving_budget",
            "--model",
            "rdt_bc_mecka_kp_dinov3_vitb",
            "--sample-frac",
            "1.0",
            "--max-samples",
            "200000",
            "--num-workers",
            "4",
            "--out",
            str(tmp_path / "stats"),
            f"hydra.searchpath=[file://{root}/egomimic/hydra_configs]",
            "seed=42026",
        ],
    )
    captured = {}
    real_compose = precompute.compose

    class ConfigCaptured(Exception):
        pass

    def compose_only(*args, **kwargs):
        captured["config"] = real_compose(*args, **kwargs)
        raise ConfigCaptured

    monkeypatch.setattr(precompute, "compose", compose_only)
    with pytest.raises(ConfigCaptured):
        precompute.main()
    cfg = captured["config"]
    assert cfg.seed == 42026
    assert cfg.norm_stats.sample_frac == 1.0
    assert cfg.norm_stats.max_samples == 200000
    assert cfg.model.robomimic_model.width == 1280
    assert cfg.data.train_dataloader_params.human_bimanual.batch_size == 64
    assert list(cfg.data.train_datasets.human_bimanual.filters.episode_hashes) == [
        "synthetic_train"
    ]
    assert cfg.trainer.limit_val_batches == 800
