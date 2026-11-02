"""Episode coverage and the reference open-loop videos for moving opscale.

The renderer is the archived reference implementation, imported under isolated
module names. Metrics remain owned by the three ordinary validation loaders.
This callback supports the reference's single-GPU/global-batch-64 execution.
"""

from __future__ import annotations

import functools
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch
from lightning.pytorch import Callback
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from egomimic.eval.opscale_reference_open_loop import OpenLoopSimEval
from egomimic.pl_utils.pl_data_utils import annotation_collate
from egomimic.rldb.embodiment.human import (
    Human,
    _build_human_keypoints_revert_6d_wristframe_transform_list,
)
from egomimic.rldb.zarr.zarr_dataset_multi import (
    EvenStrideDataset,
    pinned_episode_subset,
)

COHORT_HEADS = {
    "seen_matched": "valid",
    "seen_perlevel": "extra_valid",
    "unseen": "unseen_op_valid",
}


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def validate_selections(split: dict, episodes: dict) -> dict:
    """Fail before evaluation if counts, pins, or operator identities disagree."""
    n = len(split["operators"])
    expected_pins = {"seen_matched": 4, "seen_perlevel": min(5, n), "unseen": 5}
    seen = set()
    pins = {}
    for cohort in COHORT_HEADS:
        ids = split[cohort]
        if len(ids) != len(set(ids)) or not ids:
            raise ValueError(f"{cohort}: empty or duplicate validation episodes")
        if seen.intersection(ids) or set(split["train"]).intersection(ids):
            raise ValueError(f"{cohort}: validation or training overlap")
        seen.update(ids)
        selected = split[f"video_{cohort}"]
        if len(selected) != expected_pins[cohort] or len(set(selected)) != len(
            selected
        ):
            raise ValueError(f"{cohort}: wrong number of visualization pins")
        if not set(selected).issubset(ids):
            raise ValueError(f"{cohort}: visualization pins outside validation")
        operators = [episodes[eid]["user_id"] for eid in selected]
        if len(set(operators)) != len(operators):
            raise ValueError(f"{cohort}: repeated visualization operator")
        pins[cohort] = list(selected)
    if len(split["seen_matched"]) != 40 or len(split["seen_perlevel"]) != 3 * n:
        raise ValueError("Wrong seen validation episode counts")
    train_ops = set(split["operators"])
    for cohort in ("seen_matched", "seen_perlevel"):
        if not {episodes[e]["user_id"] for e in split[cohort]} <= train_ops:
            raise ValueError(f"{cohort}: operator absent from training")
    if {episodes[e]["user_id"] for e in split["unseen"]} & train_ops:
        raise ValueError("Unseen operator appears in training")
    return pins


class _NamedReferenceRenderer(OpenLoopSimEval):
    def __init__(
        self, *, leaves: dict, cohort: str, operator: str, episode: str, **kwargs
    ):
        super().__init__(**kwargs)
        self.leaves = leaves
        self.cohort = cohort
        self.operator = operator
        self.episode = episode
        self.uploads = []

    def _episode_leaf(self, episode_hash):
        return self.leaves[episode_hash]

    def _upload_open_loop_video(self, group, path, index):
        key = f"videos_open_loop/{self.cohort}/{self.operator}/{self.episode}"
        uploaded = False
        for logger in self.trainer.loggers:
            if isinstance(logger, WandbLogger):
                logger.log_video(
                    key,
                    [path],
                    step=self.trainer.global_step,
                    caption=[
                        f"epoch {self.trainer.current_epoch}; {self.cohort}; "
                        f"operator {self.operator}; episode {self.episode}"
                    ],
                    format=["mp4"],
                )
                uploaded = True
        if not uploaded:
            raise RuntimeError("The requested video logging requires a W&B logger")
        self.uploads.append(
            {
                "cohort": self.cohort,
                "operator": self.operator,
                "episode": self.episode,
                "path": path,
                "wandb_key": key,
            }
        )


class MovingOpscaleEvaluation(Callback):
    def __init__(
        self, private_dir: str, split_name: str, viz_every_n_epochs: int = 200
    ):
        self.private_dir = Path(private_dir)
        self.split_name = split_name
        self.viz_every_n_epochs = int(viz_every_n_epochs)
        if self.viz_every_n_epochs < 1:
            raise ValueError("viz_every_n_epochs must be positive")
        self.split = None
        self.episodes = None
        self.pins = None
        self.observed = {}
        self.final_evidence = None

    def _load(self):
        if self.split is None:
            self.split = json.loads(
                (self.private_dir / "splits" / f"{self.split_name}.json").read_text()
            )
            design = json.loads(
                (self.private_dir / "moving_opscale_design.json").read_text()
            )
            self.episodes = design["episodes"]
            self.pins = validate_selections(self.split, self.episodes)

    def on_validation_start(self, trainer, pl_module):
        self._load()
        if trainer.world_size != 1:
            raise ValueError("Moving opscale preserves the single-GPU reference recipe")
        self.observed = {cohort: set() for cohort in COHORT_HEADS}

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        names = pl_module.val_loader_names or []
        if dataloader_idx >= len(names):
            raise ValueError("Missing validation loader identity")
        cohort = next(
            (c for c, head in COHORT_HEADS.items() if head == names[dataloader_idx]),
            None,
        )
        if cohort is None:
            return
        if isinstance(batch, tuple) and len(batch) == 3 and isinstance(batch[0], dict):
            batch = batch[0]
        for samples in batch.values():
            hashes = samples.get("episode_hash")
            if hashes is None:
                raise ValueError("Validation batches must preserve episode_hash")
            self.observed[cohort].update(
                [hashes] if isinstance(hashes, str) else map(str, hashes)
            )

    def _render(self, trainer, pl_module):
        uploads = []
        model = pl_module.model
        model.device = pl_module.device
        for cohort, head in COHORT_HEADS.items():
            sources = getattr(trainer.datamodule, f"{head}_datasets")
            for episode in self.pins[cohort]:
                progress = {
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "epoch": trainer.current_epoch,
                    "global_step": trainer.global_step,
                    "cohort": cohort,
                    "episode": episode,
                    "completed_videos": len(uploads),
                    "expected_videos": sum(len(ids) for ids in self.pins.values()),
                    "state": "rendering",
                }
                print(
                    f"[MOVING_OPSCALE_VIDEO] {cohort}/{episode} "
                    f"{len(uploads) + 1}/{progress['expected_videos']}",
                    flush=True,
                )
                _atomic_json(
                    Path(trainer.default_root_dir) / "evaluation_progress.json",
                    progress,
                )
                matches = [
                    (name, ds) for name, ds in sources.items() if episode in ds.datasets
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"Pinned episode {episode} must occur in exactly one {head} dataset"
                    )
                name, source = matches[0]
                subset = pinned_episode_subset(source, [episode], dataset_name=name)
                walk = EvenStrideDataset(
                    subset, stride=30, preserve_indices=True, replace_invalid=False
                )
                renderer = _NamedReferenceRenderer(
                    leaves=subset.datasets,
                    cohort=cohort,
                    operator=self.episodes[episode]["user_id"],
                    episode=episode,
                    execute_steps=30,
                    seen_episodes=[episode],
                    video_group=cohort,
                    wandb_videos=1,
                    log_metrics=False,
                    viz_every_n_epochs=self.viz_every_n_epochs,
                    viz_mode="overlay",
                    viz_func={
                        "human_bimanual": functools.partial(
                            Human.viz_gt_preds,
                            image_key="observations.images.front_img_1",
                            action_key="actions_keypoints",
                            mode="keypoints",
                        )
                    },
                    transform_lists={
                        "human_bimanual": _build_human_keypoints_revert_6d_wristframe_transform_list()
                    },
                )
                renderer.trainer = trainer
                renderer.model = model
                renderer.on_validation_start()
                reset = getattr(model, "reset_eval_pass_counter", None)
                if callable(reset):
                    reset()
                # Rendering uses the same source frames and chunk timing as the
                # reference. No workers means no additional CUDA fork lifetime.
                loader = DataLoader(
                    walk,
                    batch_size=16,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=annotation_collate,
                )
                with torch.inference_mode(), trainer.precision_plugin.forward_context():
                    for index, samples in enumerate(loader):
                        processed = model.process_batch_for_training({name: samples})
                        renderer.on_validation_step(processed, index)
                renderer.on_validation_end()
                if len(renderer.uploads) != 1:
                    raise RuntimeError(f"Expected one W&B video for {cohort}/{episode}")
                record = renderer.uploads[0]
                if (
                    not Path(record["path"]).is_file()
                    or Path(record["path"]).stat().st_size == 0
                ):
                    raise RuntimeError(f"Missing rendered video: {record['path']}")
                uploads.append(record)
                progress.update(
                    {
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "completed_videos": len(uploads),
                        "state": "video_logged",
                    }
                )
                _atomic_json(
                    Path(trainer.default_root_dir) / "evaluation_progress.json",
                    progress,
                )
        return uploads

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        coverage = {}
        for cohort in COHORT_HEADS:
            expected = set(self.split[cohort])
            actual = self.observed[cohort]
            if actual != expected:
                raise RuntimeError(
                    f"Incomplete {cohort} evaluation: {len(actual)}/{len(expected)} episodes; "
                    f"missing={len(expected - actual)}, extra={len(actual - expected)}"
                )
            coverage[cohort] = {"expected": len(expected), "observed": len(actual)}
        final = trainer.current_epoch + 1 >= trainer.max_epochs
        render_now = final or (trainer.current_epoch + 1) % self.viz_every_n_epochs == 0
        videos = self._render(trainer, pl_module) if render_now else []
        metrics = {}
        for cohort in COHORT_HEADS:
            prefix = f"val/{cohort}/"
            selected = {
                k: float(v)
                for k, v in trainer.callback_metrics.items()
                if k.startswith(prefix)
            }
            if not selected or not all(math.isfinite(v) for v in selected.values()):
                raise RuntimeError(f"Missing or nonfinite metrics for {cohort}")
            metrics.update(selected)
        evidence = {
            "completed": False,
            "split_name": self.split_name,
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            "coverage": coverage,
            "metrics": metrics,
            "videos": videos,
        }
        root = Path(trainer.default_root_dir)
        _atomic_json(root / "evaluation_status.json", evidence)
        if final:
            self.final_evidence = evidence

    def on_fit_end(self, trainer, pl_module):
        # Completion is published after Lightning has logged epoch metrics and
        # returned from final train/checkpoint hooks, never during a smoke run.
        evidence = self.final_evidence
        if evidence is None or trainer.global_step < 400000:
            return
        if evidence["global_step"] != trainer.global_step:
            raise RuntimeError(
                "The final evaluation does not match the final training step"
            )
        loggers = [
            logger for logger in trainer.loggers if isinstance(logger, WandbLogger)
        ]
        if len(loggers) != 1:
            raise RuntimeError(
                "Exactly one W&B logger is required for completion evidence"
            )
        evidence.update(
            {
                "completed": True,
                "status": "passed",
                "step": trainer.global_step,
                "wandb_id": loggers[0].version,
                "sets": {
                    cohort: {
                        "episodes": evidence["coverage"][cohort]["observed"],
                        "videos": sum(
                            v["cohort"] == cohort for v in evidence["videos"]
                        ),
                        "metrics_logged": True,
                    }
                    for cohort in COHORT_HEADS
                },
            }
        )
        _atomic_json(
            Path(trainer.default_root_dir) / "evaluation_complete.json", evidence
        )
