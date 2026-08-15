"""Offline evaluation for the DreamZero WAM model.

Loads a hydra config (defaults to ``train_zarr_human_wam_wan22_5b``), forces an
OOD train/valid split on the data yaml, keeps exactly one window per selected
episode (each window's cam_horizon already spans a full ~6s episode), loads a
checkpoint, and runs a single Lightning ``trainer.validate`` pass through
``WAMEvalVideo`` — emitting one ``predicted_video_*.mp4`` + one
``validation_video_*.mp4`` per full episode.

Two important differences vs ``trainHydra.py``:

  1. **Purely autoregressive causal rolling**: ``algo.val_rollout`` is
     monkey-patched to run an AR rolling variant that
     (a) uses ``clean_x = concat(history, video[n_hist:])`` sliding K latents
     per step (matches training's causal ``clean_x`` contract and avoids the
     ``clean_x = latents0`` broadcast that ``WAMModel.sample()`` does), AND
     (b) **never** teacher-forces GT into the sliding history the way
     ``WAMModel.sample_rolling`` does — history slides only on the model's
     own K new predictions each step. Only the initial anchor
     (``gt_latents[:, :, :1]``) is GT. Teacher-forcing GT into history mid-
     rollout produces "video jumps back to a point in the GT" seams at chunk
     boundaries whenever the model's predictions have drifted from GT.

  2. **Per-episode limit**: ``num_val_episodes`` is the only tunable; the
     trainer's ``limit_val_batches`` is auto-set equal to it because we keep
     exactly one full-episode window per episode.

Run (from the repo root):

    python -m egomimic.eval.eval_dreamzero \
        --config-name=train_zarr_human_wam_wan22_5b \
        ckpt_path=/path/to/checkpoints/last.ckpt \
        num_val_episodes=3 \
        <training-time overrides...>
"""

from __future__ import annotations

import copy
import os
from collections import OrderedDict

import hydra
import lightning as L
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig, OmegaConf, open_dict

import egomimic.utils.hydra_resolvers  # noqa: F401  -- registers OmegaConf resolvers
from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.rldb.embodiment.embodiment import get_embodiment
from egomimic.rldb.zarr.utils import set_global_seed
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset
from egomimic.utils.aws.aws_data_utils import load_env
from egomimic.utils.pylogger import RankedLogger

OmegaConf.register_new_resolver("eval", eval, replace=True)

log = RankedLogger(__name__, rank_zero_only=True)


def _restrict_to_first_n_episodes(mds: MultiDataset, n: int) -> list[str]:
    """Keep only the first ``n`` (sorted) episodes on a MultiDataset and expose
    each as exactly one sample window (local_idx=0). The default cam_horizon
    (33 downsampled frames = ~6.6s of source) already spans a full aria
    fold_clothes episode, so one window per episode is the "full pass".

    Rebuilds the flat sample index in place. Returns the kept episode names."""
    all_names = sorted(mds.datasets.keys())
    if len(all_names) < n:
        log.warning(
            f"[eval_dreamzero] valid split has {len(all_names)} episodes < "
            f"requested {n}; using all of them."
        )
    kept = all_names[:n]
    mds.datasets = {name: mds.datasets[name] for name in kept}
    mds.index_map = []
    mds._global_indices_by_dataset = {name: [] for name in kept}
    for name in kept:
        # index 0 = first cam_horizon-length window = full-episode span at aria's 6s length.
        mds.index_map.append((name, 0))
        mds._global_indices_by_dataset[name].append(0)
    log.info(
        f"[eval_dreamzero] restricted valid dataset to {len(kept)} episodes "
        f"(1 window each, full-episode span): {kept}"
    )
    return kept


def _rolling_denoise_step(
    model,
    sched,
    video,
    action,
    history,
    n_hist,
    F_total,
    seq_len,
    context,
    state,
    emb,
):
    """One rolling step's denoising loop — shared by AR + TF variants. Runs the
    flow-match scheduler over all timesteps, updating ``video[:, :, n_hist:]``
    (the K noisy positions) while pinning ``video[:, :, :n_hist]`` to
    ``history`` (clean past) each step. Returns (new_latents, new_actions).
    """
    for t in sched.timesteps:
        ts_v = torch.zeros(video.shape[0], F_total, device=video.device)
        ts_v[:, n_hist:] = t
        ts_a = t.to(video.device).expand(video.shape[0], action.shape[1])
        clean_x = torch.cat([history, video[:, :, n_hist:]], dim=2)
        v_vel, a_vel = model.dit(
            video,
            ts_v,
            ts_a,
            context,
            seq_len,
            action=action,
            state=state,
            embodiment_id=emb,
            clean_x=clean_x,
        )
        video = sched.step(v_vel, t, video)
        video[:, :, :n_hist] = history
        action = sched.step(a_vel, t, action)
    new_latents = video[:, :, n_hist:]  # (B, Cz, K, h, w)
    return new_latents, action


def _rolling_setup(model, step_data, num_steps):
    """Common setup for both rolling variants — encodes the GT clip, builds
    scheduler, primes history/state/context, and returns everything the
    per-step loop needs plus (K, F_gt, num_steps, num_action_per_block)."""
    from egomimic.models.wam_nets import FlowMatchScheduler

    sched = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
    sched.set_timesteps(model.num_inference_steps, training=False)

    K = getattr(model.dit, "num_frame_per_block", 1)
    F_total = model.num_video_frames
    n_hist = F_total - K

    gt_latents = model._encode(step_data["video"])  # (B, Cz, F_gt, h, w)
    B, Cz, F_gt, h, w = gt_latents.shape
    device, dtype = gt_latents.device, gt_latents.dtype

    max_steps_by_gt = max(1, (F_gt - 1) // K)
    num_steps = (
        max_steps_by_gt if num_steps is None else min(num_steps, max_steps_by_gt)
    )

    seq_len = F_total * (h // 2) * (w // 2)
    history = gt_latents[:, :, :1].repeat(1, 1, n_hist, 1, 1)

    state, _ = model._prep_state_action(step_data)
    context = model._zero_context(B, device, dtype)
    # Cross-embodiment conditioning: the cotrained WAM DiT has a per-embodiment
    # embedding, so at eval we MUST pass the batch's real embodiment id (3 for
    # HUMAN_BIMANUAL, 6 for EVA_BIMANUAL) rather than 0. Zero was the historical
    # default when only aria was trained and the embedding table only had one
    # index — but for cotrain it maps to an untrained slot, producing degenerate
    # predictions (especially for eva, the minority split).
    eid = int(step_data.get("embodiment_id", 0))
    emb = torch.full((B,), eid, dtype=torch.long, device=device)

    num_action_per_block = getattr(
        model.dit,
        "num_action_per_block",
        model.action_horizon // ((F_total - 1) // K),
    )

    return {
        "sched": sched,
        "K": K,
        "F_total": F_total,
        "F_gt": F_gt,
        "n_hist": n_hist,
        "seq_len": seq_len,
        "history": history,
        "gt_latents": gt_latents,
        "state": state,
        "context": context,
        "emb": emb,
        "num_steps": num_steps,
        "num_action_per_block": num_action_per_block,
        "B": B,
        "Cz": Cz,
        "h": h,
        "w": w,
        "device": device,
        "dtype": dtype,
    }


@torch.no_grad()
def _sample_rolling_ar(model, step_data, num_steps=None):
    """Fully-autoregressive rolling: history slides ONLY on the model's own K
    new latents each step. Only the initial anchor (``gt_latents[:, :, :1]``)
    is GT — every subsequent frame is generated. Right for measuring what the
    model can produce open-loop; drift accumulates step-to-step.
    """
    ctx = _rolling_setup(model, step_data, num_steps)
    sched, K, F_total, n_hist = ctx["sched"], ctx["K"], ctx["F_total"], ctx["n_hist"]
    seq_len, history, gt_latents = ctx["seq_len"], ctx["history"], ctx["gt_latents"]
    state, context, emb = ctx["state"], ctx["context"], ctx["emb"]
    num_steps, num_action_per_block = ctx["num_steps"], ctx["num_action_per_block"]
    B, Cz, h, w, device, dtype = (
        ctx["B"],
        ctx["Cz"],
        ctx["h"],
        ctx["w"],
        ctx["device"],
        ctx["dtype"],
    )

    pred_latents = []
    pred_actions_chunks = []
    for _step in range(num_steps):
        video = torch.randn(B, Cz, F_total, h, w, device=device, dtype=dtype)
        video[:, :, :n_hist] = history
        action = torch.randn(
            B, model.action_horizon, model.action_dim, device=device, dtype=dtype
        )

        new_latents, new_action = _rolling_denoise_step(
            model,
            sched,
            video,
            action,
            history,
            n_hist,
            F_total,
            seq_len,
            context,
            state,
            emb,
        )
        pred_latents.append(new_latents)
        pred_actions_chunks.append(new_action[:, -num_action_per_block:])

        # AR: history slides on the model's own K new latents (no GT peek).
        history = torch.cat([history[:, :, K:], new_latents], dim=2)

    pred_actions = torch.cat(pred_actions_chunks, dim=1)
    all_latents = torch.cat([gt_latents[:, :, :1]] + pred_latents, dim=2)
    frames = model.vae.decode(all_latents)
    return pred_actions, frames


@torch.no_grad()
def _sample_rolling_tf(model, step_data, num_steps=None):
    """GT teacher-forced rolling (dreamzero Fig-14a): history slides with
    ``gt_latents[:, :, step*K+1 : step*K+K+1]`` each step, so chunk k
    conditions on GT for the previous chunk's K positions instead of the
    model's own predictions. Right for short-horizon / one-chunk-ahead
    forecasting quality since drift can't accumulate. If we run off the end
    of the encoded clip we fall back to the model's own predictions (matches
    ``WAMModel.sample_rolling`` semantics).
    """
    ctx = _rolling_setup(model, step_data, num_steps)
    sched, K, F_total, F_gt, n_hist = (
        ctx["sched"],
        ctx["K"],
        ctx["F_total"],
        ctx["F_gt"],
        ctx["n_hist"],
    )
    seq_len, history, gt_latents = ctx["seq_len"], ctx["history"], ctx["gt_latents"]
    state, context, emb = ctx["state"], ctx["context"], ctx["emb"]
    num_steps, num_action_per_block = ctx["num_steps"], ctx["num_action_per_block"]
    B, Cz, h, w, device, dtype = (
        ctx["B"],
        ctx["Cz"],
        ctx["h"],
        ctx["w"],
        ctx["device"],
        ctx["dtype"],
    )

    pred_latents = []
    pred_actions_chunks = []
    for step in range(num_steps):
        video = torch.randn(B, Cz, F_total, h, w, device=device, dtype=dtype)
        video[:, :, :n_hist] = history
        action = torch.randn(
            B, model.action_horizon, model.action_dim, device=device, dtype=dtype
        )

        new_latents, new_action = _rolling_denoise_step(
            model,
            sched,
            video,
            action,
            history,
            n_hist,
            F_total,
            seq_len,
            context,
            state,
            emb,
        )
        pred_latents.append(new_latents)
        pred_actions_chunks.append(new_action[:, -num_action_per_block:])

        # TF: history slides with GT latents from the input clip; if we've
        # exhausted GT, tail with the model's own predictions.
        next_gt_start = step * K + 1
        next_gt_end = next_gt_start + K
        if next_gt_end <= F_gt:
            newest = gt_latents[:, :, next_gt_start:next_gt_end]
        elif next_gt_start < F_gt:
            newest_gt = gt_latents[:, :, next_gt_start:F_gt]
            fill = K - (F_gt - next_gt_start)
            newest = torch.cat([newest_gt, new_latents[:, :, -fill:]], dim=2)
        else:
            newest = new_latents
        history = torch.cat([history[:, :, K:], newest], dim=2)

    pred_actions = torch.cat(pred_actions_chunks, dim=1)
    all_latents = torch.cat([gt_latents[:, :, :1]] + pred_latents, dim=2)
    frames = model.vae.decode(all_latents)
    return pred_actions, frames


def _patch_algo_use_sample_rolling(algo, teacher_force: bool = False) -> None:
    """Replace ``WAM.val_rollout`` (and ``forward_eval``) on the loaded algo
    with EITHER ``_sample_rolling_ar`` (fully autoregressive, no GT
    reconditioning after the anchor) OR ``_sample_rolling_tf`` (dreamzero
    Fig-14a GT teacher-forced, recondition every K latents).

    Which of the two is used depends on ``teacher_force``, which is
    typically set by the offline eval script from the evaluator yaml
    (``evaluator=eval_dreamzero_ar`` vs ``eval_dreamzero_tf``).

    Why not modify ``WAMModel.sample_rolling`` directly: we keep ``wam.py``
    untouched to preserve training-time val behavior. Rebinding bound methods
    on the loaded algo instead of subclassing means trainer / evaluator see
    the patched behavior without any Hydra-side plumbing changes.
    """
    _rolling_fn = _sample_rolling_tf if teacher_force else _sample_rolling_ar

    def val_rollout(eid, batch):
        model = algo.nets["policy"]

        video_clip = None
        for key in algo.camera_keys[eid]:
            if key in batch:
                video_clip = batch[key]
                break

        states = []
        for key in algo.proprio_keys[eid]:
            if key in batch:
                s = batch[key]
                states.append(s.unsqueeze(1) if s.dim() == 2 else s)
        full_state = torch.cat(states, dim=-1) if states else None

        step_data = {
            "video": video_clip,
            "state": full_state,
            "action": batch[algo.ac_keys[eid]],
            "embodiment_id": eid,  # per-batch conditioning for the DiT emb table
        }
        pred_actions, pred_frames = _rolling_fn(model, step_data)
        viz_video = pred_frames[:, :, 1:]  # drop the VAE recon of the anchor
        return pred_actions, viz_video

    def forward_eval(batch):
        # Same structure as WAM.forward_eval but calls our sample_rolling
        # rollout above. Preserves the {name}_loss, {name}_action_loss,
        # {name}_world_loss, {name}_{ac_key} keys the evaluator reads.
        unnorm_preds = {}
        algo._eval_frames = {}
        for eid, _batch in batch.items():
            name = get_embodiment(eid).lower()
            ac_key = algo.ac_keys[eid]
            data = algo._to_wam_data(eid, _batch)
            loss, parts = algo.nets["policy"].compute_loss(
                {"domain": name, "data": data}
            )
            unnorm_preds[f"{name}_loss"] = loss
            for pk, pv in parts.items():
                unnorm_preds[f"{name}_{pk}"] = pv
            pred_actions, viz_video = val_rollout(eid, _batch)
            algo._eval_frames[eid] = viz_video
            ref = _batch[ac_key]
            B, T, D = ref.shape
            # Keep the FULL rolled prediction so viz can draw arrows past GT's
            # dataset horizon (eva longeva: rolling produces N_steps*48 actions
            # which can be 3× longer than dataset action_horizon=192). Old code
            # truncated to ``[:, :T, :D]`` for MSE-compat, but that also killed
            # the overlay past 6 s. MSE now slices inside compute_metrics_and_viz.
            preds = OrderedDict({ac_key: pred_actions[:, :, :D]})
            for key, val in algo.norm_stats.unnormalize(preds, eid).items():
                unnorm_preds[f"{name}_{key}"] = val
        return unnorm_preds

    algo.val_rollout = val_rollout
    algo.forward_eval = forward_eval
    mode_str = (
        "GT teacher-forced (recondition every K latents)"
        if teacher_force
        else "fully-autoregressive (no GT reconditioning after anchor)"
    )
    log.info(
        f"[eval_dreamzero] algo.val_rollout / forward_eval patched — "
        f"rolling mode: {mode_str}"
    )


def _force_ood_split(
    cfg: DictConfig, valid_ratio: float, valid_mode: str = "valid"
) -> None:
    """Rewrite ``cfg.data.train_datasets`` / ``cfg.data.valid_datasets`` so the
    valid loop iterates the requested slice of the seed-split.

    The reference training config (``aria_wam.yaml``) sets ``mode: total`` on
    train and points valid at the SAME instance via interpolation
    (``valid_datasets.human_bimanual: ${data.train_datasets.human_bimanual}``),
    so no split is applied — the model was trained on every episode. We resolve
    that interpolation, then rewrite each train dataset to ``mode: train`` and
    each valid dataset to a fresh instance of the same target with
    ``mode: valid_mode`` (default ``"valid"`` → OOD; pass ``"train"`` to eval on
    the training side of the split; pass ``"total"`` to eval on the exact set
    the training used).
    """
    with open_dict(cfg):
        for name, ds_cfg in cfg.data.train_datasets.items():
            ds_cfg.mode = "train"
            ds_cfg.valid_ratio = valid_ratio

        new_valid: dict = {}
        for name in cfg.data.valid_datasets:
            train_ds = cfg.data.train_datasets[name]
            valid_cfg = copy.deepcopy(train_ds)
            valid_cfg.mode = valid_mode
            valid_cfg.valid_ratio = valid_ratio
            new_valid[name] = valid_cfg
        cfg.data.valid_datasets = OmegaConf.create(new_valid)


def _apply_eval_trainer_overrides(cfg: DictConfig, limit_val_batches: int) -> None:
    """Force the trainer into a single-GPU one-epoch validate-only run.

    We keep the ``ddp_find_unused_parameters_true`` strategy the training run
    used (with devices=1, num_nodes=1) so the process group is initialised —
    otherwise ``ModelWrapper.on_validation_end`` crashes on
    ``torch.distributed.barrier()``.
    """
    with open_dict(cfg):
        cfg.trainer.strategy = "ddp_find_unused_parameters_true"
        cfg.trainer.devices = 1
        cfg.trainer.num_nodes = 1
        cfg.trainer.limit_train_batches = 0
        cfg.trainer.limit_val_batches = limit_val_batches
        cfg.trainer.check_val_every_n_epoch = 1
        cfg.trainer.max_epochs = 1
        cfg.trainer.min_epochs = 1
        cfg.trainer.num_sanity_val_steps = 0
        cfg.trainer.sync_batchnorm = False
        cfg.logger = None


def _build_model_config_tree(cfg: DictConfig) -> DictConfig:
    """Mirror trainHydra: strip norm_stats out of the model tree that gets
    stored as a hyperparameter (it's rebuilt from ``norm_stats_state`` inside
    ModelWrapper)."""
    model_cfg = copy.deepcopy(cfg.model)
    if (
        "robomimic_model" in model_cfg
        and isinstance(model_cfg.robomimic_model, DictConfig)
        and "norm_stats" in model_cfg.robomimic_model
    ):
        model_cfg.robomimic_model.norm_stats = None
    return OmegaConf.create({"model": model_cfg})


@hydra.main(
    version_base="1.3",
    config_path="../hydra_configs",
    config_name="train_zarr_human_wam_wan22_5b",
)
def main(cfg: DictConfig) -> None:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
        set_global_seed(cfg.seed)
    else:
        raise ValueError("Seed must be provided in cfg for reproducibility!")

    load_env()

    num_val_episodes: int = int(cfg.get("num_val_episodes", 3))
    valid_ratio: float = float(cfg.get("valid_ratio", 0.2))
    # Which slice of the seed-split to iterate at eval time. Default "valid"
    # (OOD side). Set to "train" to eval on the training side, or "total" to
    # replicate the training run's ``mode: total`` (whole dataset).
    valid_mode: str = str(cfg.get("valid_mode", "valid"))
    ckpt_path: str | None = cfg.get("ckpt_path")
    if not ckpt_path:
        raise ValueError(
            "ckpt_path must be provided (pass ckpt_path=/path/to/checkpoint.ckpt)."
        )

    # ---- config surgery ----------------------------------------------------
    _force_ood_split(cfg, valid_ratio=valid_ratio, valid_mode=valid_mode)

    # Instantiate train datasets first — needed to bootstrap norm_stats.
    train_datasets = {
        name: hydra.utils.instantiate(cfg.data.train_datasets[name])
        for name in cfg.data.train_datasets
    }
    valid_datasets = {
        name: hydra.utils.instantiate(cfg.data.valid_datasets[name])
        for name in cfg.data.valid_datasets
    }

    # Sanity: with valid_mode="valid" the two sides must be disjoint (OOD split).
    # For "train" / "total" we intentionally overlap — the eval loop is meant to
    # look at episodes the checkpoint was trained on — so we skip that check.
    for name in valid_datasets:
        tr_names = set(getattr(train_datasets[name], "datasets", {}).keys())
        va_names = set(getattr(valid_datasets[name], "datasets", {}).keys())
        overlap = tr_names & va_names
        if valid_mode == "valid":
            assert not overlap, (
                f"OOD split violated for dataset {name!r}: {len(overlap)} episodes "
                f"appear in both train and valid ({sorted(overlap)[:3]}...)"
            )
        assert va_names, f"Valid split for {name!r} is empty."
        log.info(
            f"[eval_dreamzero] {name}: {len(tr_names)} train / {len(va_names)} "
            f"valid episodes (valid_mode={valid_mode}, overlap={len(overlap)})"
        )

    # Restrict valid to first N episodes (sorted for determinism), one full-
    # episode window each. Trainer's ``limit_val_batches`` (set below) equals
    # the resulting length so the val loop is exactly one pass per episode.
    total_windows = 0
    for name, mds in valid_datasets.items():
        _restrict_to_first_n_episodes(mds, num_val_episodes)
        total_windows += len(mds)
    log.info(
        f"[eval_dreamzero] Sweeping {total_windows} total val windows "
        f"(one full-episode window × {num_val_episodes} episodes per dataset)."
    )

    # Assemble datamodule (WAM uses MultiDataModuleWrapper).
    assert (
        "MultiDataModuleWrapper" in cfg.data._target_
    ), "cfg.data._target_ must be 'MultiDataModuleWrapper'"
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.data, train_datasets=train_datasets, valid_datasets=valid_datasets
    )

    # ---- norm_stats: same recipe as trainHydra, reusing precomputed path ---
    norm_stats = MultiDataset(
        state={},
        norm_mode=OmegaConf.select(cfg, "norm_stats.norm_mode", default="quantile"),
    )
    norm_stats.populate_from_datasets(datamodule.train_datasets)

    for dataset_name, dataset in datamodule.train_datasets.items():
        log.info(f"[eval_dreamzero] Inferring shapes for dataset <{dataset_name}>")
        norm_stats.infer_shapes_from_batch(dataset[0])
        instantiate_copy = copy.deepcopy(cfg.data.train_datasets[dataset_name])
        keymap_cfg = instantiate_copy.resolver.key_map
        km = OmegaConf.to_container(keymap_cfg, resolve=False)
        km["norm_mode"] = True
        instantiate_copy.resolver.key_map = km
        norm_dataset = hydra.utils.instantiate(instantiate_copy)
        norm_stats.infer_norm_from_dataset(
            norm_dataset,
            dataset_name,
            sample_frac=OmegaConf.select(cfg, "norm_stats.sample_frac", default=1.0),
            num_workers=OmegaConf.select(cfg, "norm_stats.num_workers", default=4),
            precomputed_norm_path=OmegaConf.select(
                cfg, "norm_stats.precomputed_norm_path", default=None
            ),
        )

    for ds in datamodule.train_datasets.values():
        ds.set_norm_stats_from(norm_stats)
    for ds in datamodule.valid_datasets.values():
        ds.set_norm_stats_from(norm_stats)

    # Cache the computed stats under the hydra run dir so subsequent runs on
    # the same data can reuse them via ``norm_stats.precomputed_norm_path``
    # (cross-embodiment norm inference on ~250k samples takes ~20 min).
    save_cache_dir = OmegaConf.select(cfg, "norm_stats.save_cache_dir", default=None)
    if save_cache_dir:
        norm_stats.cache_stats(save_cache_dir=save_cache_dir)

    # ---- model wrap --------------------------------------------------------
    log.info(f"[eval_dreamzero] Instantiating model <{cfg.model._target_}>")
    model: LightningModule = ModelWrapper(
        config_tree=_build_model_config_tree(cfg),
        norm_stats_state=norm_stats.to_state(),
        scheduler_interval=cfg.model.get("scheduler_interval", "step"),
    )

    # ---- eval / trainer overrides ------------------------------------------
    _apply_eval_trainer_overrides(cfg, limit_val_batches=total_windows or 1)

    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=None, logger=None)

    # Videos will land under <output_dir>/videos/epoch_0/HUMAN_BIMANUAL/*.mp4
    os.makedirs(os.path.join(trainer.default_root_dir, "videos"), exist_ok=True)

    # ---- evaluator + checkpoint --------------------------------------------
    eval_obj = hydra.utils.instantiate(cfg.evaluator)
    eval_obj.trainer = trainer
    eval_obj.model = model.model
    model.evaluator = eval_obj

    log.info(f"[eval_dreamzero] Loading checkpoint {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
    log.info(
        f"[eval_dreamzero] load_state_dict: {len(missing)} missing / "
        f"{len(unexpected)} unexpected keys"
    )
    # Free the second copy of the weights before Lightning moves the module
    # to GPU (each 5B ckpt is ~70 GB in bf16 on disk).
    del checkpoint
    import gc

    gc.collect()

    # Route eval through sample_rolling — must happen AFTER load_state_dict
    # since we rebind bound methods on the loaded ``model.model`` instance.
    # ``teacher_force_rolling`` toggles fully-AR vs. dreamzero Fig-14a
    # teacher-forced rolling; both modes share the ``clean_x = concat(history,
    # video[n_hist:])`` prefix layout that matches training. Read the flag off
    # the instantiated evaluator (WAMEvalVideo declares it as a kwarg) so the
    # eval yaml controls which mode we're in.
    teacher_force_rolling = bool(getattr(eval_obj, "teacher_force_rolling", False))
    _patch_algo_use_sample_rolling(model.model, teacher_force=teacher_force_rolling)

    log.info("[eval_dreamzero] Starting evaluation!")
    trainer.validate(model=model, datamodule=datamodule)

    videos_dir = os.path.join(trainer.default_root_dir, "videos")
    log.info(f"[eval_dreamzero] Done. Videos under: {videos_dir}")


if __name__ == "__main__":
    main()
