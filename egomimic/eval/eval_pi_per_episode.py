import csv
import json
import logging
import os

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from egomimic.eval.eval_pi import PIEvalVideo
from egomimic.pl_utils.pl_data_utils import annotation_collate
from egomimic.rldb.embodiment.embodiment import get_embodiment
from egomimic.rldb.zarr.utils import set_global_seed

logger = logging.getLogger(__name__)


class PIEvalVideoPerEpisode(PIEvalVideo):
    """
    Per-episode validation for Pi/Pi0.5 models.

    Where the parent ``PIEvalVideo`` produces a single dataset-level aggregate
    pushed to wandb, this evaluator iterates the validation ``MultiDataset`` one
    episode at a time and writes **one record per episode** to disk. It reuses
    the parent's ``compute_metrics_and_viz`` unchanged: when a DataLoader yields
    only one episode's frames, the batch-reduced metrics ARE that episode's
    metrics (accumulated frame-weighted across the episode's batches).

    Invoked via the ``mode=eval`` ``run()`` hook in ``trainHydra.py`` — it does
    NOT call ``trainer.validate``, so Lightning's ``on_validation_*`` hooks (and
    the unconditional ``torch.distributed.barrier()`` in
    ``ModelWrapper.on_validation_end``) never fire. Eval mode forces
    ``devices=1``, so this is strictly single-process and file writes need no
    rank guard.

    Designed for an OLD euler checkpoint: pair with ``transform_lists: {}`` (so
    the 6D cam-frame revert branch in ``compute_metrics_and_viz`` is skipped) and
    a ``norm_stats.precomputed_norm_path`` pointing at the checkpoint's own
    euler norm stats.
    """

    def __init__(
        self,
        *args,
        dataset_name: str = "mecka_bimanual",
        output_subdir: str = "per_episode_stats",
        seed: int = 42,
        max_frames_per_episode: int | None = None,
        num_shards: int = 1,
        shard_index: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Which entry of ``datamodule.valid_datasets`` to iterate. Also the
        # embodiment-name key expected by ``process_batch_for_training``.
        self.dataset_name = dataset_name
        self.output_subdir = output_subdir
        self.seed = seed
        # Episode sharding for multi-GPU runs: each process scores
        # ``sorted(episodes)[shard_index::num_shards]`` and writes its own
        # ``episodes.shard{i}of{n}.jsonl``. Merge the shard files afterwards.
        # num_shards=1 -> single ``episodes.jsonl`` + summary (no merge needed).
        if not (0 <= shard_index < num_shards):
            raise ValueError(
                f"shard_index {shard_index} out of range for num_shards {num_shards}"
            )
        self.num_shards = num_shards
        self.shard_index = shard_index
        # Cap frames scored per episode (evenly spaced across the episode) to
        # bound runtime. None = score every frame. The per-episode metric is a
        # mean over frames, and consecutive frames are highly autocorrelated, so
        # an evenly-spaced subset of a few hundred frames is an ample estimate
        # for ranking/filtering episodes.
        self.max_frames_per_episode = max_frames_per_episode

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_float(v):
        if torch.is_tensor(v):
            return float(v.detach().cpu().item())
        return float(v)

    def _episode_dir(self):
        return os.path.join(self.root_dir(), self.output_subdir)

    @staticmethod
    def _cfg_get(cfg, path):
        try:
            v = OmegaConf.select(cfg, path)
            return None if v is None else str(v)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # entrypoint (called by trainHydra.py mode=eval)
    # ------------------------------------------------------------------
    def run(self, trainer, model, datamodule, cfg):
        # ``model`` is the LightningModule (ModelWrapper); ``model.model`` is the
        # algo. trainHydra already sets self.trainer / self.model, but be explicit.
        self.trainer = trainer
        self.model = model.model
        algo = self.model

        # --- load checkpoint weights into the wrapper (mirrors eval_latent.run) ---
        ckpt_path = cfg.get("ckpt_path")
        if ckpt_path:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["state_dict"], strict=False)
            logger.info("Loaded weights from %s", ckpt_path)
        else:
            logger.warning(
                "No ckpt_path provided — evaluating current (untrained) weights."
            )

        # --- device setup (on_validation_start won't run, so do it by hand) ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        # The algo reads ``self.device`` when building pad masks / sampling.
        algo.device = device

        # Disable the max-autotune torch.compile on sample_actions (set in
        # PI0Pytorch.__init__). Compiling it costs >20 min on first call and
        # would re-trigger on every new batch size (each episode's short final
        # batch differs), so eager is both faster end-to-end and stable here.
        # Mirrors egomimic/robot/rollout.py's rollout-inference unwrap.
        pi0 = algo.nets["policy"]
        if "sample_actions" in vars(pi0):
            del pi0.sample_actions
            logger.info("Disabled torch.compile on sample_actions (eager eval).")

        set_global_seed(self.seed)

        valid_md = datamodule.valid_datasets[self.dataset_name]
        gibd = valid_md._global_indices_by_dataset  # {episode_hash: [global idx]}

        dl_params = dict(datamodule.valid_dataloader_params.get(self.dataset_name, {}))
        batch_size = int(dl_params.get("batch_size", 32))
        num_workers = int(dl_params.get("num_workers", 0))

        out_dir = self._episode_dir()
        os.makedirs(out_dir, exist_ok=True)
        if self.num_shards > 1:
            jsonl_path = os.path.join(
                out_dir, f"episodes.shard{self.shard_index}of{self.num_shards}.jsonl"
            )
        else:
            jsonl_path = os.path.join(out_dir, "episodes.jsonl")

        provenance = {
            "ckpt_path": str(ckpt_path),
            "transform_mode": self._cfg_get(
                cfg,
                f"data.valid_datasets.{self.dataset_name}.resolver.transform_list.mode",
            ),
            "precomputed_norm_path": self._cfg_get(
                cfg, "norm_stats.precomputed_norm_path"
            ),
            "seed": self.seed,
            "dataset_name": self.dataset_name,
        }

        records = []
        all_episodes = sorted(gibd.items())
        episodes = all_episodes[self.shard_index :: self.num_shards]
        n_eps = len(episodes)
        logger.info(
            "Per-episode eval: %d episodes (shard %d/%d of %d total, "
            "dataset=%s, batch_size=%d, num_workers=%d)",
            n_eps,
            self.shard_index,
            self.num_shards,
            len(all_episodes),
            self.dataset_name,
            batch_size,
            num_workers,
        )

        with open(jsonl_path, "w") as jf:
            for ep_i, (episode_hash, global_idxs) in enumerate(episodes):
                # Re-seed per episode so numbers don't depend on episode ordering.
                set_global_seed(self.seed + ep_i)

                idxs = list(global_idxs)
                n_frames_total = len(idxs)
                if (
                    self.max_frames_per_episode
                    and n_frames_total > self.max_frames_per_episode
                ):
                    sel = np.linspace(
                        0, n_frames_total - 1, self.max_frames_per_episode
                    )
                    sel = sorted(set(int(round(x)) for x in sel))
                    idxs = [idxs[i] for i in sel]

                subset = Subset(valid_md, idxs)
                loader = DataLoader(
                    subset,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=annotation_collate,
                    num_workers=num_workers,
                )

                agg = {}  # metric_key -> frame-weighted running sum
                total_frames = 0
                n_batches = 0
                n_resampled = 0
                task_label = None
                emb_id = None
                action_converter = None

                for collated in loader:
                    batch = algo.process_batch_for_training(
                        {self.dataset_name: collated}
                    )
                    metrics, _ = self.compute_metrics_and_viz(batch, do_viz=False)

                    emb_id = next(iter(batch.keys()))
                    hashes = batch[emb_id].get("episode_hash")
                    if hashes is None:
                        hashes = collated.get("episode_hash", [])
                    b = len(hashes)
                    n_resampled += sum(1 for h in hashes if str(h) != str(episode_hash))

                    if task_label is None:
                        tlist = batch[emb_id].get("task") or collated.get("task")
                        if tlist is not None and len(tlist) > 0:
                            task_label = str(tlist[0])
                    if action_converter is None:
                        try:
                            ac_key = algo.ac_keys[emb_id]
                            action_converter = type(
                                algo.action_registry.get(emb_id, ac_key)
                            ).__name__
                        except Exception:
                            action_converter = None

                    for k, v in metrics.items():
                        agg[k] = agg.get(k, 0.0) + self._to_float(v) * b
                    total_frames += b
                    n_batches += 1

                if total_frames == 0:
                    logger.warning(
                        "Episode %s yielded 0 frames; skipping.", episode_hash
                    )
                    continue

                ep_metrics = {k: (s / total_frames) for k, s in agg.items()}
                if n_resampled:
                    logger.warning(
                        "Episode %s: %d/%d frames resampled to a different episode "
                        "(MultiDataset bounds fallback); metrics include them.",
                        episode_hash,
                        n_resampled,
                        total_frames,
                    )

                rec = {
                    "episode_hash": str(episode_hash),
                    "task": task_label,
                    "embodiment_id": int(emb_id) if emb_id is not None else None,
                    "embodiment": (
                        get_embodiment(emb_id) if emb_id is not None else None
                    ),
                    "n_frames": int(total_frames),
                    "n_frames_total": int(n_frames_total),
                    "n_batches": int(n_batches),
                    "n_resampled_frames": int(n_resampled),
                    "action_converter": action_converter,
                    **provenance,
                    "metrics": ep_metrics,
                }
                jf.write(json.dumps(rec) + "\n")
                jf.flush()
                records.append(rec)
                logger.info(
                    "[%d/%d] %s  task=%s  frames=%d  action_loss=%.5f",
                    ep_i + 1,
                    n_eps,
                    episode_hash,
                    task_label,
                    total_frames,
                    ep_metrics.get("Valid/action_loss", float("nan")),
                )

        if self.num_shards == 1:
            self._write_summary(out_dir, records)
        else:
            logger.info(
                "Shard %d/%d done — wrote %s. Merge shard jsonls + build summary "
                "after all shards finish.",
                self.shard_index,
                self.num_shards,
                jsonl_path,
            )
        logger.info(
            "Per-episode eval complete: %d episodes (shard %d/%d) -> %s",
            len(records),
            self.shard_index,
            self.num_shards,
            out_dir,
        )

    # ------------------------------------------------------------------
    # roll-up
    # ------------------------------------------------------------------
    def _write_summary(self, out_dir, records):
        if not records:
            logger.warning("No episode records to summarize.")
            return

        metric_keys = sorted({k for r in records for k in r["metrics"]})

        # Per-episode CSV (one row per episode, flattened metrics).
        base_cols = [
            "episode_hash",
            "task",
            "embodiment",
            "n_frames",
            "n_batches",
            "n_resampled_frames",
        ]
        csv_path = os.path.join(out_dir, "summary.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(base_cols + metric_keys)
            for r in records:
                w.writerow(
                    [r.get(c) for c in base_cols]
                    + [r["metrics"].get(k) for k in metric_keys]
                )

        # Aggregate over episodes (unweighted stats + frame-weighted mean) so the
        # frame-weighted mean is directly comparable to the old wandb aggregate.
        agg = {}
        for k in metric_keys:
            vals = np.array(
                [r["metrics"][k] for r in records if k in r["metrics"]], dtype=float
            )
            fw = np.array(
                [r["n_frames"] for r in records if k in r["metrics"]], dtype=float
            )
            agg[k] = {
                "episode_mean": float(vals.mean()),
                "episode_median": float(np.median(vals)),
                "episode_std": float(vals.std()),
                "frame_weighted_mean": (
                    float((vals * fw).sum() / fw.sum()) if fw.sum() else None
                ),
            }

        summary = {
            "n_episodes": len(records),
            "total_frames": int(sum(r["n_frames"] for r in records)),
            "metrics": agg,
        }
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
