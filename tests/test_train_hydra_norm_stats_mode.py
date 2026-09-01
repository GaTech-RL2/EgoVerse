from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

import egomimic.trainHydra as train_hydra


class _FakeDataset:
    def __len__(self):
        return 7

    def __getitem__(self, index):
        assert index == 0
        return {"actions": [0.0]}

    def set_norm_stats_from(self, norm_stats):
        raise AssertionError("stats-only mode must return before dataset wiring")


class _FakeNormStats:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.shape_inferences = []
        self.inferred = []
        self.cache_calls = []
        self.__class__.instances.append(self)

    def populate_from_datasets(self, datasets):
        self.datasets = datasets

    def infer_shapes_from_batch(self, batch, dataset_name):
        assert batch == {"actions": [0.0]}
        self.shape_inferences.append(dataset_name)

    def infer_norm_from_dataset(self, dataset, dataset_name, **kwargs):
        self.inferred.append((dataset_name, kwargs))

    def cache_stats(self, save_cache_dir):
        assert len(self.inferred) == 2
        self.cache_calls.append(save_cache_dir)


def test_norm_stats_mode_caches_once_then_skips_model_and_trainer(
    monkeypatch, tmp_path
):
    _FakeNormStats.instances.clear()
    dataset = _FakeDataset()
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "mode": "norm_stats",
            "paths": {"output_dir": str(tmp_path)},
            "data": {
                "_target_": "example.MultiDataModuleWrapper",
                "train_datasets": {
                    "pushshapes_sim_u_socket": {
                        "resolver": {"key_map": {"actions": {}}}
                    },
                    "pushshapes_sim_chain_gripper": {
                        "resolver": {"key_map": {"actions": {}}}
                    },
                },
                "valid_datasets": {},
            },
            "norm_stats": {
                "norm_mode": "minmax",
                "reduce_all_but_last": True,
                "sample_frac": 1.0,
                "num_workers": 4,
                "save_cache_dir": str(tmp_path / "artifact"),
                "precomputed_norm_path": None,
            },
        }
    )

    def fake_instantiate(config, **kwargs):
        if "train_datasets" in kwargs:
            return SimpleNamespace(
                train_datasets=kwargs["train_datasets"],
                valid_datasets=kwargs["valid_datasets"],
            )
        return dataset

    monkeypatch.setattr(train_hydra.L, "seed_everything", lambda *a, **k: None)
    monkeypatch.setattr(train_hydra, "set_global_seed", lambda *a, **k: None)
    monkeypatch.setattr(train_hydra, "load_env", lambda: None)
    monkeypatch.setattr(train_hydra.hydra.utils, "instantiate", fake_instantiate)
    monkeypatch.setattr(train_hydra, "MultiDataset", _FakeNormStats)
    monkeypatch.setattr(
        train_hydra,
        "ModelWrapper",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("ModelWrapper must not be instantiated")
        ),
    )
    monkeypatch.setattr(
        train_hydra,
        "instantiate_callbacks",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("callbacks must not be instantiated")
        ),
    )
    monkeypatch.setattr(
        train_hydra,
        "instantiate_loggers",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("loggers must not be instantiated")
        ),
    )

    metrics, objects = train_hydra.train(cfg)

    norm_stats = _FakeNormStats.instances[-1]
    assert metrics == {}
    assert objects["norm_stats"] is norm_stats
    assert [name for name, _ in norm_stats.inferred] == [
        "pushshapes_sim_u_socket",
        "pushshapes_sim_chain_gripper",
    ]
    assert norm_stats.shape_inferences == [
        "pushshapes_sim_u_socket",
        "pushshapes_sim_chain_gripper",
    ]
    assert norm_stats.cache_calls == [str(tmp_path / "artifact")]


def test_model_wrapper_receives_configured_runtime_flags(monkeypatch):
    captured = {}
    cfg = OmegaConf.create(
        {
            "model": {
                "robomimic_model": {"_target_": "example.Policy"},
                "scheduler_interval": "step",
                "enable_grad_norm": False,
                "train_metrics_on_step": True,
                "train_metrics_on_epoch": False,
                "unite_flow_updates_per_reconstruction": 14,
                "unite_gradient_telemetry_every_n_steps": 100,
            }
        }
    )
    norm_stats = SimpleNamespace(to_state=lambda: {"stats": "sentinel"})

    monkeypatch.setattr(
        train_hydra,
        "ModelWrapper",
        lambda **kwargs: captured.update(kwargs) or SimpleNamespace(),
    )
    train_hydra._instantiate_model_wrapper(cfg, norm_stats)

    assert captured["enable_grad_norm"] is False
    assert captured["train_metrics_on_step"] is True
    assert captured["train_metrics_on_epoch"] is False
    assert captured["unite_flow_updates_per_reconstruction"] == 14
    assert captured["unite_gradient_telemetry_every_n_steps"] == 100
    assert captured["norm_stats_state"] == {"stats": "sentinel"}


def test_model_wrapper_emits_step_metrics_and_separate_epoch_aggregates():
    calls = []
    wrapper = SimpleNamespace(
        train_metrics_on_step=True,
        train_metrics_on_epoch=True,
        log=lambda name, value, **kwargs: calls.append((name, value, kwargs)),
    )

    train_hydra.ModelWrapper._log_train_metric(
        wrapper, "Train/action_loss", 1.25
    )

    assert calls == [
        (
            "Train/action_loss",
            1.25,
            {"on_step": True, "on_epoch": False, "sync_dist": True},
        ),
        (
            "Train/action_loss_epoch",
            1.25,
            {"on_step": False, "on_epoch": True, "sync_dist": True},
        ),
    ]


def test_model_wrapper_emits_lr_each_optimizer_step_with_grad_norm_disabled():
    calls = []
    wrapper = SimpleNamespace(
        train_metrics_on_step=True,
        enable_grad_norm=False,
        log=lambda name, value, **kwargs: calls.append((name, value, kwargs)),
    )
    optimizer = SimpleNamespace(param_groups=[{"lr": 3.0e-5}])

    train_hydra.ModelWrapper.on_before_optimizer_step(wrapper, optimizer)

    assert calls == [
        (
            "Optimizer/param_group_0_lr",
            3.0e-5,
            {"on_step": True, "on_epoch": False, "sync_dist": True},
        )
    ]


def test_norm_stats_mode_rejects_precomputed_input_before_instantiation(
    monkeypatch, tmp_path
):
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "mode": "norm_stats",
            "paths": {"output_dir": str(tmp_path)},
            "norm_stats": {
                "save_cache_dir": str(tmp_path / "artifact"),
                "precomputed_norm_path": str(tmp_path / "old.json"),
            },
        }
    )
    monkeypatch.setattr(train_hydra.L, "seed_everything", lambda *a, **k: None)
    monkeypatch.setattr(train_hydra, "set_global_seed", lambda *a, **k: None)
    monkeypatch.setattr(train_hydra, "load_env", lambda: None)
    monkeypatch.setattr(
        train_hydra.hydra.utils,
        "instantiate",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("stats-only validation must run before instantiation")
        ),
    )

    with pytest.raises(ValueError, match="must compute rather than reload"):
        train_hydra.train(cfg)
