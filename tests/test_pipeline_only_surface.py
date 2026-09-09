import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_legacy_first_party_policy_surface_is_removed():
    removed = [
        "egomimic/models/act_nets.py",
        "egomimic/models/hpt_nets.py",
        "egomimic/models/preprocess_pi_obs.py",
        "egomimic/models/denoising_policy.py",
        "egomimic/models/fm_policy.py",
        "egomimic/eval/eval_act.py",
        "egomimic/eval/eval_hpt.py",
        "egomimic/eval/eval_pi.py",
        "egomimic/eval/eval_latent.py",
        "egomimic/eval/eval_video.py",
        "egomimic/eval/latent_dataset.py",
        "egomimic/robot/rollout.py",
        "egomimic/utils/hydra_resolvers.py",
        "egomimic/utils/scheduler_utils.py",
        "egomimic/utils/tensor_utils.py",
        "external/openpi",
    ]
    leftovers = [relative for relative in removed if (ROOT / relative).exists()]
    assert leftovers == []
    assert list((ROOT / "egomimic/algo").glob("*.py")) == []


def test_surviving_model_configs_are_pipeline_only():
    config_dir = ROOT / "egomimic/hydra_configs/model"
    configs = sorted(config_dir.rglob("*.yaml"))
    for config in configs:
        text = config.read_text()
        assert "egomimic.algo." not in text, config
        assert "robomimic_model" not in text, config


def test_distribution_discovers_only_the_egomimic_package():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    package_find = project["tool"]["setuptools"]["packages"]["find"]
    assert package_find["include"] == ["egomimic*"]
    assert not (ROOT / "setup.py").exists()


def test_training_entrypoint_has_one_strict_evaluation_path():
    entrypoint = (ROOT / "egomimic/trainHydra.py").read_text()
    assert 'hasattr(eval_obj, "run")' not in entrypoint
    assert 'hasattr(eval_obj, "bind_data_context")' not in entrypoint
    assert "strict=False" not in entrypoint
    assert "register_new_resolver" not in entrypoint
