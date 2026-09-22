"""`trainHydra` passes `dataset_name=` to every dataset target, so the latent
data configs' factory (`egomimic.eval.latent_dataset.build_dataset`) must accept
it and forward it to `MultiDataset._from_resolver` -- otherwise
`data=cotrain_pi_latent` dies at launch with a TypeError."""

from __future__ import annotations

import hydra
import pytest
from fixtures.synthetic_episodes import write_episode
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf, open_dict

from egomimic.eval import latent_dataset
from egomimic.eval.latent_dataset import build_dataset

LOCAL_RESOLVER = "egomimic.rldb.zarr.zarr_dataset_multi.LocalEpisodeResolver"


@pytest.mark.parametrize("mode", ["random", "custom"])
def test_build_dataset_forwards_dataset_name(monkeypatch, mode) -> None:
    seen: dict = {}

    def fake_from_resolver(resolver, **kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(
        latent_dataset.MultiDataset, "_from_resolver", fake_from_resolver
    )
    monkeypatch.setattr(latent_dataset, "EvenStrideDataset", lambda base, **kw: base)

    build_dataset(
        mode,
        task="pick_place",
        embodiment="eva_bimanual",
        resolver=object(),
        dataset_name="eva_bimanual",
        hashes=["eva_00"],
    )
    assert seen["dataset_name"] == "eva_bimanual"


def _latent_nodes():
    with initialize_config_module(
        config_module="egomimic.hydra_configs", version_base=None
    ):
        cfg = compose(
            config_name="train_zarr_cartesian_pi",
            overrides=["data=cotrain_pi_latent", "model=pi0.5_bc_eva"],
        )
    return cfg


def test_latent_config_nodes_instantiate_without_typeerror(tmp_path) -> None:
    """Every node of `data=cotrain_pi_latent` must survive
    `instantiate(node, dataset_name=<key>)`. The synthetic episodes carry
    task_name='synthetic', so the 'random' mode filter legitimately matches
    nothing -- a ValueError about episodes is fine; a TypeError is the bug."""
    data = tmp_path / "data"
    data.mkdir()
    write_episode(data, "eva", seed=0)
    write_episode(data, "aria", seed=0)

    cfg = _latent_nodes()
    with open_dict(cfg):
        cfg.paths.dataset_dir = str(data)

    checked = 0
    for group in ("train_datasets", "valid_datasets"):
        for key in cfg.data[group]:
            node = OmegaConf.create(
                OmegaConf.to_container(cfg.data[group][key], resolve=True)
            )
            with open_dict(node):
                node.resolver._target_ = LOCAL_RESOLVER
                # LocalEpisodeResolver has no `require_annotations` kwarg.
                del node.resolver["require_annotations"]
            try:
                hydra.utils.instantiate(node, dataset_name=key)
            except Exception as e:  # noqa: BLE001 - we only care about the type
                chain = []
                cur: BaseException | None = e
                while cur is not None:
                    chain.append(cur)
                    cur = cur.__cause__
                assert not any(isinstance(c, TypeError) for c in chain), (
                    f"{group}.{key} raised TypeError: "
                    + "; ".join(f"{type(c).__name__}: {c}" for c in chain)
                )
            checked += 1
    assert checked == 4
