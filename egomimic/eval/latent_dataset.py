"""Latent-eval-specific dataset assembly.

Three things live here:

  - ``build_dataset``:   hydra factory dispatching by `mode`:
                           random  -> MultiDataset (whole-task filter)
                           pairs   -> EvenStrideDataset on a fixed
                                      (eva, aria) hash list
                           custom  -> EvenStrideDataset on user-supplied
                                      hash lists

  - ``build_MultiDataModuleWrapper``: thin wrapper that strips the
                           latent-config-specific top-level keys before
                           constructing the real PL data module.

  - ``from_resolver``:   backward-compat alias for older yamls that
                           wrapped ``MultiDataset._from_resolver`` plus
                           optional EvenStrideDataset subsampling.

The reusable building blocks (``SafeS3EpisodeResolver``,
``EvenStrideDataset``) live in ``egomimic.rldb.zarr.zarr_dataset_multi``
next to their parents — this module only provides the latent-eval
orchestration.

Hydra usage (in the data yaml):

    eva_bimanual:
      _target_: egomimic.eval.latent_dataset.build_dataset
      mode: ${data.latent_mode}
      task: pick_place
      embodiment: eva_bimanual
      hashes: ${oc.select:data.${data.latent_mode}_hashes.eva, []}
      frames_per_episode: ${data.frames_per_episode}
      stride: ${data.stride}
      resolver:
        _target_: egomimic.rldb.zarr.zarr_dataset_multi.SafeS3EpisodeResolver
        require_annotations: true
        ...
"""

from __future__ import annotations

import logging

from egomimic.rldb.filters import DatasetFilter
from egomimic.rldb.zarr.zarr_dataset_multi import EvenStrideDataset, MultiDataset

logger = logging.getLogger(__name__)


# Top-level keys this module's yaml templates use that the underlying
# MultiDataModuleWrapper does NOT accept. Must stay in sync with the
# `data:` block in cotrain_pi_latent.yaml.
_LATENT_CONFIG_KEYS = frozenset(
    {
        "latent_mode",
        "pair_hashes",
        "custom_hashes",
        "frames_per_episode",
        "stride",
        "_shuffle_random",
        "_shuffle_pairs",
        "_shuffle_custom",
    }
)


def build_dataset(
    mode: str,
    *,
    task: str,
    embodiment: str,
    resolver,
    hashes: list[str] | None = None,
    frames_per_episode: int | None = 128,
    stride: int | None = None,
    valid_ratio: float = 0.05,
    filters=None,  # base yaml may pass filters=null; ignored — we build our own
):
    """Build a dataset object based on `mode`:
      - 'random' -> MultiDataset filtered by (task, robot_name)
      - 'pairs' / 'custom' -> EvenStrideDataset on the supplied hash list,
        subsampled by `frames_per_episode` (default) or `stride` (if set).

    Unknown `mode` values raise. Extra kwargs are NOT accepted (no silent
    swallowing) — typos in yaml fail loudly.
    """
    if mode == "random":
        lam = (
            "lambda row: row['task'] == "
            f"{task!r} and row['robot_name'] == {embodiment!r}"
        )
        filters = DatasetFilter(filter_lambdas=[lam])
        logger.info("[build_dataset] %s | random mode | filter=%s", embodiment, lam)
        return MultiDataset._from_resolver(
            resolver,
            filters=filters,
            mode="total",
            valid_ratio=valid_ratio,
        )

    if mode in ("pairs", "custom"):
        if not hashes:
            raise ValueError(
                f"latent_mode={mode!r} requires a non-empty `hashes` list "
                f"for embodiment={embodiment!r} (got {hashes!r})."
            )
        hash_tuple = "(" + ",".join(repr(str(h)) for h in hashes) + ",)"
        lam = f"lambda row: row['episode_hash'] in {hash_tuple}"
        filters = DatasetFilter(filter_lambdas=[lam])
        if stride is not None and stride > 0:
            sub_kwargs = {"stride": stride}
            rule = f"stride={stride}"
        else:
            sub_kwargs = {"frames_per_episode": frames_per_episode}
            rule = f"frames_per_episode={frames_per_episode}"
        logger.info(
            "[build_dataset] %s | %s mode | %d hashes | %s",
            embodiment,
            mode,
            len(hashes),
            rule,
        )
        base = MultiDataset._from_resolver(
            resolver,
            filters=filters,
            mode="total",
            valid_ratio=valid_ratio,
        )
        return EvenStrideDataset(base, **sub_kwargs)

    raise ValueError(
        f"Unknown latent_mode: {mode!r}. Must be one of: 'random', 'pairs', 'custom'."
    )


def build_MultiDataModuleWrapper(**kwargs):
    """Thin wrapper around MultiDataModuleWrapper that strips the latent-
    config-specific top-level keys (see `_LATENT_CONFIG_KEYS`) so they
    don't reach the real constructor as unexpected kwargs. The keys are
    still resolvable via OmegaConf interpolation from inside the data
    yaml's per-dataset blocks. Also drops any private keys that start
    with underscore (yaml lookup tables we use only for
    ``${oc.select:_shuffle_${...}}`` dispatch)."""
    filtered = {
        k: v
        for k, v in kwargs.items()
        if k not in _LATENT_CONFIG_KEYS and not k.startswith("_")
    }
    from egomimic.pl_utils.pl_data_utils import MultiDataModuleWrapper

    return MultiDataModuleWrapper(**filtered)


def from_resolver(
    resolver,
    frames_per_episode: int | None = None,
    stride: int | None = None,
    filters=None,
    mode: str = "total",
    valid_ratio: float = 0.05,
    **kwargs,
):
    """Backward-compat alias: ``MultiDataset._from_resolver`` plus optional
    ``EvenStrideDataset`` subsampling. Prefer ``build_dataset`` for new
    yamls; this exists only for older configs that haven't migrated yet."""
    base = MultiDataset._from_resolver(
        resolver,
        filters=filters,
        mode=mode,
        valid_ratio=valid_ratio,
        **kwargs,
    )
    if frames_per_episode is None and stride is None:
        return base
    return EvenStrideDataset(base, frames_per_episode=frames_per_episode, stride=stride)
