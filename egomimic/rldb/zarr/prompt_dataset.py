"""
Whole-demonstration prompt sampling for the BPP algo (see
docs/2026-09-02_bpp_episode_prompting.md).

``EpisodePromptMultiDataset`` wraps the usual per-timestep ``MultiDataset``
and attaches, to every sample, a prompt built from a *different* episode of
the same group. A group is ``(task, operator)`` by default, so a prompt never
crosses operators. The prompt covers the whole prompt episode at one chunk per
``chunk_n_actions * prompt_stride`` raw frames (one chunk per second on 30 fps
data with chunk 30, stride 1):

    prompt:
      obs:
        <camera batch key>: (P, 3, H, W) float in [0, 1], resized, unnormalized
        <state batch key>:  (P, D_state)          normalized like the rollout
      action:               (P, chunk_n, D_act)   normalized like the rollout
      length:               P

Variable ``P`` is padded to the batch maximum by ``prompt_collate`` in
``egomimic/pl_utils/pl_data_utils.py``, which also builds ``metadata.mask``.

Prompts are only attached once normalization stats have been set on the
dataset (``set_norm_stats_from``). Before that (shape probing and norm-stat
inference in ``trainHydra``) samples are plain rollout samples, so the default
collate keeps working.

Own-episode history (``prompt.history.max_chunks > 0``, see
docs/plan/2026-09-08_bpp_rollout_history.md): every sample also carries
``history``, the same payload shape as ``prompt`` but holding the last
``max_chunks`` chunks of the sample's *own* episode that end before its frame
(chunk grid anchored at the episode start, so it is a slice of the cached
whole-episode chunks). ``length`` may be 0 near the episode start.
"""

from __future__ import annotations

import logging
import random
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import simplejpeg
import torch
import torchvision.transforms.functional as TF

from egomimic.rldb.zarr.zarr_dataset_multi import SEED, MultiDataset, ZarrDataset

logger = logging.getLogger(__name__)

DEFAULT_GROUP_KEY = "lambda row: (row['task'], row['operator'])"


# ---------------------------------------------------------------------------
# Reading one episode as prompt chunks
# ---------------------------------------------------------------------------


def read_prompt_chunks(
    leaf: ZarrDataset,
    chunk_n_actions: int,
    prompt_stride: int,
    transform_list: list,
    image_size: tuple[int, int] | None,
    action_key: str,
    state_key: str,
    chunk_starts: np.ndarray | None = None,
    action_steps: int | None = None,
) -> dict:
    """Build prompt chunks for a whole episode.

    Chunk ``p`` starts at raw frame ``s_p = p * chunk_n_actions * prompt_stride``
    and is assembled exactly like a rollout sample at frame ``s_p`` would be by
    ``ZarrDataset.__getitem__``: every windowed key_map entry (the action keys)
    reads ``chunk_n_actions`` frames at ``prompt_stride`` from ``s_p`` (repeating
    the last frame past the end, like ``_pad_sequences``), every other entry
    reads frame ``s_p``, then ``transform_list`` runs. The transform list must
    therefore emit chunks of ``chunk_n_actions`` steps (for the human cartesian
    pipeline: ``Human.get_transform_list(mode, stride=1, chunk_length=chunk_n_actions)``).

    ``action_steps`` is the number of action steps the transform list emits
    per chunk (default ``chunk_n_actions``; the rollout convention test passes
    100 with a 30-frame read window).

    Returns unnormalized numpy/torch data:
        {"obs": {<camera key>: uint8 (P, 3, h, w), state_key: float32 (P, D)},
         "action": float32 (P, chunk_n_actions, D_act), "length": P}
    """
    T = int(leaf.total_frames)
    if action_steps is None:
        action_steps = chunk_n_actions
    window = chunk_n_actions * prompt_stride
    if chunk_starts is None:
        chunk_starts = np.arange(0, T, window)
    image_keys = set(leaf._image_keys)

    numeric_zarr_keys = {
        spec["zarr_key"]
        for spec in leaf.key_map.values()
        if spec.get("key_type") != "annotation_keys"
        and spec["zarr_key"] not in image_keys
    }
    full = leaf.episode_reader.read({zk: (0, T) for zk in numeric_zarr_keys})

    camera_names = [
        name
        for name, spec in leaf.key_map.items()
        if spec.get("key_type") != "annotation_keys" and spec["zarr_key"] in image_keys
    ]

    chunks = [
        assemble_chunk(
            _read_raw_chunk(leaf, int(s), full, chunk_n_actions, prompt_stride),
            transform_list=transform_list,
            camera_names=camera_names,
            state_key=state_key,
            action_key=action_key,
            image_size=image_size,
            action_steps=action_steps,
        )
        for s in chunk_starts
    ]
    obs = {name: torch.stack([c["obs"][name] for c in chunks]) for name in camera_names}
    obs[state_key] = torch.stack([c["obs"][state_key] for c in chunks])
    return {
        "obs": obs,
        "action": torch.stack([c["action"] for c in chunks]),
        "length": int(len(chunk_starts)),
    }


def _read_raw_chunk(
    leaf: ZarrDataset, s: int, full: dict, chunk_n_actions: int, prompt_stride: int
) -> dict:
    """Raw per-key data for the chunk starting at frame ``s``, keyed by
    key_map name (annotation keys skipped): images decoded to ``(3, H, W)``
    float in [0, 1] at ``s``; windowed keys (``horizon`` set) as
    ``(chunk_n_actions, ...)`` over the chunk, index-clamped past the episode
    end like ``_pad_sequences``; every other key at frame ``s``. ``full`` is
    the whole-episode read of the numeric zarr keys."""
    T = int(leaf.total_frames)
    image_keys = set(leaf._image_keys)
    data = {}
    for name, spec in leaf.key_map.items():
        zk = spec["zarr_key"]
        if spec.get("key_type") == "annotation_keys":
            continue
        if zk in image_keys:
            raw = leaf.episode_reader.read({zk: (s, None)})[zk]
            decoded = simplejpeg.decode_jpeg(raw, colorspace="RGB")
            data[name] = np.transpose(decoded, (2, 0, 1)) / 255.0
        elif spec.get("horizon") is not None:
            idx = np.minimum(s + prompt_stride * np.arange(chunk_n_actions), T - 1)
            data[name] = np.asarray(full[zk])[idx]
        else:
            data[name] = np.asarray(full[zk])[s]
    return data


def assemble_chunk(
    raw: dict,
    *,
    transform_list: list,
    camera_names: list,
    state_key: str,
    action_key: str,
    image_size: tuple[int, int] | None,
    action_steps: int,
) -> dict:
    """Run ``transform_list`` on one chunk's raw data (``_read_raw_chunk``
    layout) and pack it as one prompt/history chunk:
    ``{"obs": {<camera>: uint8 (3, h, w), state_key: float32 (D,)},
    "action": float32 (action_steps, D_act)}``."""
    data = dict(raw)
    for transform in transform_list or []:
        data = transform.transform(data)
    obs = {}
    for name in camera_names:
        img = torch.from_numpy(np.ascontiguousarray(data[name])).float()
        if image_size is not None:
            img = TF.resize(img, list(image_size), antialias=True)
        obs[name] = (img.clamp(0, 1) * 255.0).round().to(torch.uint8)
    obs[state_key] = torch.from_numpy(np.asarray(data[state_key], dtype=np.float32))
    act = np.asarray(data[action_key], dtype=np.float32)
    if act.shape[0] != action_steps:
        raise ValueError(
            f"prompt transform emitted {act.shape[0]} action steps per chunk, "
            f"expected {action_steps}; build the prompt transform list with "
            f"chunk_length={action_steps}."
        )
    return {"obs": obs, "action": torch.from_numpy(act)}


def build_history_chunk(
    raw_chunk: dict,
    *,
    key_map: dict,
    transform_list: list,
    image_size: tuple[int, int] | None = (224, 224),
    action_key: str = "actions_cartesian",
    state_key: str = "observations.state.ee_pose",
    action_steps: int | None = None,
    chunk_n_actions: int | None = None,
) -> dict:
    """Deployment helper: build one *unnormalized* history chunk from raw data
    a rollout buffered for one chunk window, in the ``_read_raw_chunk`` layout
    keyed by key_map name (camera keys ``(3, H, W)`` float in [0, 1] at the
    chunk start, windowed keys ``(chunk_n_actions, ...)`` over the window,
    others at the chunk start). Same transforms and packing as the training
    prompt/history chunks. Returns ``{"obs": {k: (1, ...)}, "action":
    (1, action_steps, D_act), "length": 1}``; normalize it like a prompt
    (``EpisodePromptMultiDataset._normalize_prompt``) before
    ``BPP.push_history_chunk``.
    """
    if action_steps is None:
        if chunk_n_actions is None:
            raise ValueError(
                "build_history_chunk needs action_steps or chunk_n_actions"
            )
        action_steps = int(chunk_n_actions)
    camera_names = [
        name for name, spec in key_map.items() if spec.get("key_type") == "camera_keys"
    ]
    chunk = assemble_chunk(
        raw_chunk,
        transform_list=transform_list,
        camera_names=camera_names,
        state_key=state_key,
        action_key=action_key,
        image_size=image_size,
        action_steps=action_steps,
    )
    return {
        "obs": {k: v.unsqueeze(0) for k, v in chunk["obs"].items()},
        "action": chunk["action"].unsqueeze(0),
        "length": 1,
    }


def reduce_stats_to_last_dim(stats: dict, ndim_target: int = 1) -> dict:
    """Collapse per-timestep norm stats (e.g. ``(100, 12)`` for a 100-step
    action horizon) to per-dimension stats ``(12,)`` so a prompt chunk of any
    length can be normalized with the rollout's action statistics. Low
    quantiles/min take the min over the leading axes, high quantiles/max the
    max, mean the mean, std the pooled std; median the median."""
    out = {}
    means = None
    for name, arr in stats.items():
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim <= ndim_target:
            out[name] = arr
            continue
        axes = tuple(range(arr.ndim - ndim_target))
        if name == "mean":
            means = arr
            out[name] = arr.mean(axis=axes)
        elif name == "min":
            out[name] = arr.min(axis=axes)
        elif name == "max":
            out[name] = arr.max(axis=axes)
        elif name == "median":
            out[name] = np.median(arr, axis=axes)
        elif name.startswith("quantile_"):
            q = float(name[len("quantile_") :].replace("_", "."))
            out[name] = arr.min(axis=axes) if q < 50 else arr.max(axis=axes)
        elif name == "std":
            out[name] = arr  # resolved below once mean is known
        else:
            out[name] = arr.mean(axis=axes)
    if "std" in out and out["std"].ndim > ndim_target:
        std = np.asarray(stats["std"], dtype=np.float32)
        if means is None:
            means = np.asarray(stats["mean"], dtype=np.float32)
        axes = tuple(range(std.ndim - ndim_target))
        pooled_mean = means.mean(axis=axes)
        out["std"] = np.sqrt(
            np.maximum((std**2 + means**2).mean(axis=axes) - pooled_mean**2, 0.0)
        )
    return out


def _nbytes(prompt: dict) -> int:
    n = prompt["action"].numel() * prompt["action"].element_size()
    for v in prompt["obs"].values():
        n += v.numel() * v.element_size()
    return n


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class EpisodePromptMultiDataset(MultiDataset):
    """``MultiDataset`` whose samples carry a whole-episode prompt from the
    same ``(task, operator)`` group. Leaves must be ``ZarrDataset``s with
    ``metadata_row`` set by the resolver.

    ``prompt`` config keys (all optional except ``chunk_n_actions`` and
    ``max_sequence_length``):

    - ``chunk_n_actions``: action steps per prompt chunk (30 = 1 Hz at 30 fps).
    - ``max_sequence_length``: cap on episode length in raw frames; longer
      episodes raise at construction (matches the policy's positional table).
    - ``prompt_stride``: raw frames between prompt action steps (default 1).
    - ``group_key``: lambda source evaluated on ``metadata_row``
      (default ``(task, operator)``).
    - ``min_episodes_per_group``: groups with fewer episodes are dropped (2).
    - ``heldout_groups``: explicit list of operator ids (or full groups) held
      out of training entirely. In the split modes their episodes all land in
      the ``valid`` split tagged ``operator_seen=0``; seen groups' validation
      episodes are tagged ``operator_seen=1``. In ``total`` mode nothing is
      moved or dropped (total keeps every episode); the list only decides the
      ``operator_seen`` tag. Default: none.
    - ``ignore``: when true no prompt is ever attached (samples still carry
      ``group_idx`` and ``operator_seen``), so an unprompted baseline sharing
      this data config skips the prompt IO. Default: false.
    A sample is never prompted with its own episode, so every group needs at
    least two episodes in each split it appears in (train, valid, held-out).
    - ``balance_by``: ``none`` | ``group`` | ``task`` for ``sample_weights()``.
    - ``image_size``: prompt frames are resized to this at read time ([224, 224]).
    - ``cache_bytes``: per-process LRU cap for decoded prompt episodes (2e9).
    - ``action_key`` / ``state_key``: post-transform keys ("actions_cartesian",
      "observations.state.ee_pose").
    - ``seed``: RNG seed for the split and prompt draws.
    - ``history``: optional ``{max_chunks, gap_frames}``. ``max_chunks > 0``
      attaches ``history`` to every sample: the last ``max_chunks`` chunks of
      the sample's own episode (same grid and reader as the prompt, so a
      slice of the cached whole-episode chunks) whose window ends at least
      ``gap_frames`` (default 0) before the sample's served frame. Empty
      (``length`` 0) near the episode start. Default: off.
    """

    def __init__(
        self,
        datasets: dict | None = None,
        mode: str = "train",
        prompt: dict | None = None,
        prompt_transform_list: list | None = None,
        valid_ratio: float = 0.2,
        state: dict | None = None,
        **kwargs,
    ):
        if state is not None:
            super().__init__(datasets=None, state=state, **kwargs)
            self._ignore_prompt = True
            self.use_history = False
            return
        if datasets is None:
            raise ValueError("EpisodePromptMultiDataset requires `datasets`.")

        cfg = dict(prompt or {})
        try:
            self.chunk_n_actions = int(cfg["chunk_n_actions"])
            self.max_sequence_length = int(cfg["max_sequence_length"])
        except KeyError as e:
            raise ValueError(
                "prompt config needs `chunk_n_actions` and `max_sequence_length`"
            ) from e
        self.prompt_stride = int(cfg.get("prompt_stride", 1))
        if "exclude_self" in cfg and not bool(cfg["exclude_self"]):
            raise ValueError(
                "exclude_self=False is not supported: a sample is never prompted "
                "with its own episode."
            )
        self.min_episodes_per_group = int(cfg.get("min_episodes_per_group", 2))
        heldout_spec = cfg.get("heldout_groups") or []
        # hydra hands lists over as omegaconf ListConfig: accept any
        # non-string sequence.
        if isinstance(heldout_spec, (str, bytes)) or not isinstance(
            heldout_spec, Sequence
        ):
            raise ValueError(
                "heldout_groups must be a list of operator ids / groups, "
                f"got {heldout_spec!r}"
            )
        self._heldout_spec = {str(x) for x in heldout_spec}
        self.balance_by = str(cfg.get("balance_by", "none"))
        if self.balance_by not in ("none", "group", "task"):
            raise ValueError(
                f"balance_by must be none|group|task, got {self.balance_by!r}"
            )
        image_size = cfg.get("image_size", [224, 224])
        self.image_size = (
            None if image_size is None else tuple(int(x) for x in image_size)
        )
        self.cache_bytes = float(cfg.get("cache_bytes", 2e9))
        self.action_key = str(cfg.get("action_key", "actions_cartesian"))
        self.state_key = str(cfg.get("state_key", "observations.state.ee_pose"))
        self.seed = int(cfg.get("seed", SEED))
        group_key_src = str(cfg.get("group_key", DEFAULT_GROUP_KEY))
        self._group_fn = eval(group_key_src)
        if not callable(self._group_fn):
            raise ValueError(f"group_key must evaluate to a callable: {group_key_src}")
        if prompt_transform_list is None:
            raise ValueError(
                "EpisodePromptMultiDataset requires `prompt_transform_list` "
                "(e.g. Human.get_transform_list(mode, stride=1, chunk_length=chunk_n_actions))."
            )
        self.prompt_transform_list = list(prompt_transform_list)
        self._ignore_prompt = bool(cfg.get("ignore", False))
        history_cfg = dict(cfg.get("history") or {})
        self.history_max_chunks = int(history_cfg.get("max_chunks", 0))
        self.history_gap_frames = int(history_cfg.get("gap_frames", 0))
        if self.history_max_chunks < 0 or self.history_gap_frames < 0:
            raise ValueError(
                "prompt.history.max_chunks and gap_frames must be >= 0, got "
                f"{self.history_max_chunks} / {self.history_gap_frames}"
            )
        self.use_history = self.history_max_chunks > 0
        self._rng: random.Random | None = None
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._cache_size = 0

        # ---- group every episode by (task, operator) ----
        group_of: dict[str, object] = {}
        task_of: dict[str, object] = {}
        for name, ds in datasets.items():
            if not isinstance(ds, ZarrDataset):
                raise TypeError(
                    "EpisodePromptMultiDataset leaves must be ZarrDataset "
                    f"(got {type(ds).__name__} for {name!r})"
                )
            row = ds.metadata_row
            if row is None:
                raise ValueError(
                    f"episode {name!r} has no metadata_row; use a resolver that "
                    "attaches SQL/zarr metadata (S3EpisodeResolver, LocalEpisodeResolver)."
                )
            group_of[name] = self._group_fn(row)
            task_of[name] = row.get("task")
        by_group: dict[object, list[str]] = {}
        for name in sorted(datasets):
            by_group.setdefault(group_of[name], []).append(name)

        # ---- drop small groups, then split per group ----
        # Self-prompting is never allowed, so every split a group appears in
        # needs >= 2 of its episodes: train and valid both (split modes), or
        # the whole pool (total mode / held-out groups).
        train_min = 2
        valid_min = 2 if (valid_ratio > 0 and mode != "total") else 0
        need = max(self.min_episodes_per_group, train_min + valid_min)
        if mode == "total":
            need = max(self.min_episodes_per_group, train_min)
        heldout_need = max(self.min_episodes_per_group, 2)

        def _is_heldout(g):
            if not self._heldout_spec:
                return False
            parts = {str(g)} | ({str(x) for x in g} if isinstance(g, tuple) else set())
            return bool(parts & self._heldout_spec)

        dropped = {
            g: n
            for g, n in by_group.items()
            if len(n) < (heldout_need if _is_heldout(g) else need)
        }
        for g in dropped:
            del by_group[g]
        if dropped:
            logger.warning(
                "EpisodePromptMultiDataset: dropped %d group(s) with too few "
                "episodes (need %d, held-out %d): %s",
                len(dropped),
                need,
                heldout_need,
                {str(g): len(n) for g, n in dropped.items()},
            )
        if not by_group:
            raise ValueError(
                f"No group has >= {need} episodes (min_episodes_per_group="
                f"{self.min_episodes_per_group}; self-prompting is never allowed)."
            )

        # ---- held-out groups (unseen operators), explicit list only ----
        # Resolved in every mode so ``operator_seen`` is tagged; only the
        # split modes move the held-out episodes (total keeps everything).
        all_groups = sorted(by_group, key=str)
        heldout: set = set()
        if self._heldout_spec:
            for g in all_groups:
                parts = {str(g)} | (
                    {str(x) for x in g} if isinstance(g, tuple) else set()
                )
                if parts & self._heldout_spec:
                    heldout.add(g)
            matched = set()
            for g in heldout:
                matched |= {str(g)} | (
                    {str(x) for x in g} if isinstance(g, tuple) else set()
                )
            missing = self._heldout_spec - matched
            if missing:
                logger.warning(
                    "heldout_groups entries not found in data: %s", sorted(missing)
                )
            if mode != "total" and heldout and len(heldout) >= len(all_groups):
                raise ValueError("heldout_groups would hold out every group.")

        train_names: list[str] = []
        valid_names: list[str] = []
        heldout_names: list[str] = []
        rng = random.Random(self.seed)
        for g in all_groups:
            names = list(by_group[g])
            rng.shuffle(names)
            if g in heldout and mode != "total":
                heldout_names.extend(names)
                continue
            if valid_ratio <= 0 or mode == "total":
                n_val = 0
            else:
                n_val = max(valid_min, int(round(valid_ratio * len(names))))
                n_val = min(n_val, len(names) - train_min)
            valid_names.extend(names[:n_val])
            train_names.extend(names[n_val:])

        if mode == "train":
            chosen = train_names
        elif mode == "valid":
            chosen = valid_names + heldout_names
        elif mode == "total":
            chosen = train_names + valid_names
        else:
            raise ValueError(
                f"EpisodePromptMultiDataset supports mode train|valid|total, got {mode!r}"
            )
        if not chosen:
            raise ValueError(f"No episodes left for mode={mode!r}")

        super().__init__(
            datasets={n: datasets[n] for n in chosen},
            mode="total",
            valid_ratio=valid_ratio,
            **kwargs,
        )
        self.mode = mode
        self.train_collections = set(train_names)
        self.valid_collections = set(valid_names) | set(heldout_names)
        self.heldout_groups_resolved = sorted(heldout, key=str)
        # 1 = the sample's operator has training episodes; 0 = held out.
        self._operator_seen = {n: (group_of[n] not in heldout) for n in chosen}

        # ---- tables ----
        self._group_of_episode = {n: group_of[n] for n in chosen}
        self._task_of_episode = {n: task_of[n] for n in chosen}
        self.group_names = sorted({group_of[n] for n in chosen}, key=str)
        self._group_index = {g: i for i, g in enumerate(self.group_names)}
        self._episodes_by_group: dict[object, list[str]] = {}
        for n in sorted(chosen):
            self._episodes_by_group.setdefault(group_of[n], []).append(n)
        self.episode_names = sorted(chosen)
        self._episode_index = {n: i for i, n in enumerate(self.episode_names)}

        too_long = {
            n: int(self.datasets[n].total_frames)
            for n in chosen
            if int(self.datasets[n].total_frames) > self.max_sequence_length
        }
        if too_long:
            raise ValueError(
                f"{len(too_long)} episode(s) exceed max_sequence_length="
                f"{self.max_sequence_length} raw frames: {too_long}. Raise "
                "shape_meta.max_sequence_length or filter the task."
            )

        logger.info(
            "EpisodePromptMultiDataset[%s]: %d episodes in %d groups "
            "(chunk_n=%d, stride=%d, max_len=%d, balance_by=%s, "
            "history_max_chunks=%d, history_gap_frames=%d)",
            mode,
            len(chosen),
            len(self.group_names),
            self.chunk_n_actions,
            self.prompt_stride,
            self.max_sequence_length,
            self.balance_by,
            self.history_max_chunks,
            self.history_gap_frames,
        )
        for i, g in enumerate(self.group_names):
            logger.info(
                "  group %d = %s: %d episodes%s",
                i,
                g,
                len(self._episodes_by_group[g]),
                " [held-out operator]" if g in heldout else "",
            )

    # ------------------------------------------------------------------
    # Public knobs
    # ------------------------------------------------------------------

    def set_ignore_prompt(self, ignore: bool) -> None:
        self._ignore_prompt = bool(ignore)

    def get_ignore_prompt(self) -> bool:
        return self._ignore_prompt

    def group_of_index(self, idx: int):
        dataset_name, _ = self.index_map[idx]
        return self._group_of_episode[dataset_name]

    def sample_weights(self) -> torch.Tensor | None:
        """Per-index weights for ``WeightedRandomSampler`` (``None`` when
        ``balance_by == "none"``). ``group``: every group gets equal total
        weight; ``task``: every task does."""
        if self.balance_by == "none":
            return None
        if self.balance_by == "group":
            key_of = self._group_of_episode
        else:
            key_of = self._task_of_episode
        frames_per_key: dict[object, int] = {}
        for name, ds in self.datasets.items():
            frames_per_key[key_of[name]] = frames_per_key.get(key_of[name], 0) + len(ds)
        weights = torch.empty(len(self.index_map), dtype=torch.double)
        for i, (name, _) in enumerate(self.index_map):
            weights[i] = 1.0 / frames_per_key[key_of[name]]
        return weights

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _rng_for_worker(self) -> random.Random:
        if self._rng is None:
            # torch.initial_seed() differs per DataLoader worker and per epoch.
            self._rng = random.Random(self.seed + int(torch.initial_seed()) % (2**31))
        return self._rng

    def _raw_prompt(self, episode_name: str) -> dict:
        cached = self._cache.get(episode_name)
        if cached is not None:
            self._cache.move_to_end(episode_name)
            return cached
        prompt = read_prompt_chunks(
            self.datasets[episode_name],
            chunk_n_actions=self.chunk_n_actions,
            prompt_stride=self.prompt_stride,
            transform_list=self.prompt_transform_list,
            image_size=self.image_size,
            action_key=self.action_key,
            state_key=self.state_key,
        )
        size = _nbytes(prompt)
        if size <= self.cache_bytes:
            self._cache[episode_name] = prompt
            self._cache_size += size
            while self._cache_size > self.cache_bytes and len(self._cache) > 1:
                _, evicted = self._cache.popitem(last=False)
                self._cache_size -= _nbytes(evicted)
        return prompt

    def _prompt_stats(self, embodiment_id, key):
        """Per-dimension stats for ``key`` (rollout action stats are per
        timestep and get collapsed; see ``reduce_stats_to_last_dim``)."""
        stats = self.norm_stats.get(embodiment_id, {}) if self.norm_stats else {}
        raw = stats.get(key)
        if raw is None:
            return None
        cache = getattr(self, "_prompt_stats_cache", None)
        if cache is None:
            cache = self._prompt_stats_cache = {}
        ck = (embodiment_id, key, id(raw))
        if ck not in cache:
            cache[ck] = reduce_stats_to_last_dim(raw)
        return cache[ck]

    def _normalize_prompt(self, prompt: dict, embodiment_id) -> dict:
        obs = {}
        for key, value in prompt["obs"].items():
            if value.dtype == torch.uint8:
                obs[key] = value.float() / 255.0
            else:
                obs[key] = value.clone()
        action = prompt["action"].clone()
        state_stats = self._prompt_stats(embodiment_id, self.state_key)
        if self.state_key in obs and state_stats is not None:
            obs[self.state_key] = self._apply_norm_one(obs[self.state_key], state_stats)
        action_stats = self._prompt_stats(embodiment_id, self.action_key)
        if action_stats is not None:
            action = self._apply_norm_one(action, action_stats)
        return {"obs": obs, "action": action, "length": int(prompt["length"])}

    def _embodiment_id_of(self, episode_name: str):
        embodiment_id = self.datasets[episode_name].embodiment
        if isinstance(embodiment_id, str):
            from egomimic.rldb.embodiment.embodiment import get_embodiment_id

            embodiment_id = get_embodiment_id(embodiment_id)
        return embodiment_id

    def build_prompt_for_episode(self, episode_name: str, embodiment_id=None) -> dict:
        """Normalized prompt for one episode (deployment helper; same code path
        as training)."""
        prompt = self._raw_prompt(episode_name)
        if embodiment_id is None:
            embodiment_id = self._embodiment_id_of(episode_name)
        return self._normalize_prompt(prompt, embodiment_id)

    # ------------------------------------------------------------------
    # Own-episode history
    # ------------------------------------------------------------------

    def _history_for(self, episode_name: str, t: int, embodiment_id) -> dict:
        """Normalized history for a sample at served frame ``t`` of
        ``episode_name``: the last ``history_max_chunks`` chunks of the own
        episode (same grid as the prompt, ``p * chunk_n_actions *
        prompt_stride``) whose window ends at or before ``t -
        history_gap_frames``. A slice of the cached whole-episode chunks, so
        no extra IO once the episode is in the LRU."""
        own = self._raw_prompt(episode_name)
        window = self.chunk_n_actions * self.prompt_stride
        n_done = max(0, (int(t) - self.history_gap_frames) // window)
        n_done = min(n_done, int(own["length"]))
        # max(0, ...): a negative start would wrap around and silently return
        # nothing for the first ``max_chunks`` chunks of every episode.
        start = max(0, n_done - self.history_max_chunks)
        sliced = {
            "obs": {k: v[start:n_done] for k, v in own["obs"].items()},
            "action": own["action"][start:n_done],
            "length": n_done - start,
        }
        return self._normalize_prompt(sliced, embodiment_id)

    def build_history_for_index(self, idx: int) -> dict:
        """History a sample at (requested, unsubstituted) global index ``idx``
        would carry (test / deployment helper)."""
        episode_name, t = self.index_map[idx]
        return self._history_for(
            episode_name, int(t), self._embodiment_id_of(episode_name)
        )

    def __getitem__(self, idx, _attempts: int | None = None):
        data, served = self._getitem_with_index(idx, _attempts=_attempts)
        # The served index may differ from ``idx`` after a bounds/NaN
        # substitution, but never leaves the episode; the served frame is the
        # leaf's ``frame_idx`` (set by ZarrDataset.__getitem__).
        dataset_name, served_local = self.index_map[served]
        frame_idx = data.get("frame_idx")
        if torch.is_tensor(frame_idx):
            frame_idx = int(frame_idx.item())
        t = int(served_local if frame_idx is None else frame_idx)
        group = self._group_of_episode[dataset_name]
        data["group_idx"] = int(self._group_index[group])
        data["operator_seen"] = int(self._operator_seen[dataset_name])
        # Own episode (index into ``episode_names``); the evaluator groups
        # viz frames by it. ``prompt_episode_idx`` below is the prompt's.
        data["episode_idx"] = int(self._episode_index[dataset_name])
        if self._ignore_prompt or not self.norm_stats:
            return data

        pool = [n for n in self._episodes_by_group[group] if n != dataset_name]
        if not pool:
            raise RuntimeError(
                f"episode {dataset_name!r} is alone in group {group!r}; cannot "
                "prompt without self-prompting (should have been dropped at build)."
            )
        prompt_name = self._rng_for_worker().choice(pool)
        emb = data.get("embodiment")
        if torch.is_tensor(emb):
            emb = int(emb.item())
        data["prompt"] = self._normalize_prompt(self._raw_prompt(prompt_name), emb)
        data["prompt_episode_idx"] = int(self._episode_index[prompt_name])
        if self.use_history:
            data["history"] = self._history_for(dataset_name, t, emb)
        return data


def build_episode_prompt(
    episode_path: str | Path,
    key_map: dict,
    prompt_transform_list: list,
    norm_stats: MultiDataset,
    chunk_n_actions: int,
    prompt_stride: int = 1,
    image_size: tuple[int, int] | None = (224, 224),
    action_key: str = "actions_cartesian",
    state_key: str = "observations.state.ee_pose",
) -> dict:
    """Deployment helper: build a batch-size-1 normalized prompt for one local
    episode using the same reader, transforms and stats as training. Returns
    ``{"obs": {key: (1, P, ...)}, "action": (1, P, chunk_n, D),
    "metadata": {"mask": (1, P) all False}}`` with batch-key names; the BPP
    adapter renames keys to shape_meta names in
    ``BPP.episode_prompt_to_policy``.
    """
    leaf = ZarrDataset(Path(episode_path), key_map=key_map, transform_list=None)
    raw = read_prompt_chunks(
        leaf,
        chunk_n_actions=chunk_n_actions,
        prompt_stride=prompt_stride,
        transform_list=prompt_transform_list,
        image_size=image_size,
        action_key=action_key,
        state_key=state_key,
    )
    from egomimic.rldb.embodiment.embodiment import get_embodiment_id

    emb = leaf.embodiment
    emb_id = get_embodiment_id(emb) if isinstance(emb, str) else int(emb)
    helper = EpisodePromptMultiDataset.__new__(EpisodePromptMultiDataset)
    helper.norm_stats = norm_stats.norm_stats
    helper.norm_mode = norm_stats.norm_mode
    helper.state_key = state_key
    helper.action_key = action_key
    helper._prompt_stats_cache = {}
    normalized = helper._normalize_prompt(raw, emb_id)
    P = normalized["length"]
    return {
        "obs": {k: v.unsqueeze(0) for k, v in normalized["obs"].items()},
        "action": normalized["action"].unsqueeze(0),
        "metadata": {"mask": torch.zeros(1, P, dtype=torch.bool)},
    }
