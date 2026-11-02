"""Generate the 36 Hydra recipes from the verified W&B reference settings.

Episode IDs remain in EGOVERSE_MECKA_PRIVATE_DIR/splits. Generation does not
assert dataset readiness; validate_setup.py and the launcher enforce that gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BUDGETS = (24, 32, 48, 64, 72, 96)
SCALES = (4, 8, 16, 32, 64, 128)


def emit(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# @package _global_\n" + yaml.safe_dump(value, sort_keys=False))


def main():
    reference = json.loads((HERE / "reference_recipe.json").read_text())
    data = {
        "_target_": "egomimic.pl_utils.pl_data_utils.MultiDataModuleWrapper",
        "valid_prefix": "val/seen_matched",
        "extra_valid_prefix": "val/seen_perlevel",
        "unseen_op_valid_prefix": "val/unseen",
        "exact_val_reads": True,
        "replace_invalid_val_frames": False,
    }
    resolver = {
        "_target_": "egomimic.rldb.zarr.zarr_dataset_multi.LocalEpisodeResolver",
        "folder_path": "${oc.env:EGOVERSE_MECKA_PRIVATE_DIR}/canonical_zarr",
        "key_map": reference["data_key_map"],
        "transform_list": reference["data_transform_list"],
    }
    for head in ("train", "valid", "extra_valid", "unseen_op_valid"):
        dataset = {
            "_target_": "egomimic.rldb.zarr.zarr_dataset_multi.MultiDataset._from_resolver",
            "resolver": resolver
            if head == "train"
            else "${data.train_datasets.human_bimanual.resolver}",
            "filters": {
                "_target_": "egomimic.rldb.filters.DatasetFilter",
                "episode_hashes": "???",
            },
            "pose_glitch_mask": reference["data_pose_glitch_mask"],
            "mode": "total",
        }
        if head != "train":
            dataset["bounds_rel_slack"] = 10.0
        data[f"{head}_datasets"] = {"human_bimanual": dataset}
        data[f"{head}_dataloader_params"] = {
            "human_bimanual": {
                "batch_size": 64 if head == "train" else 32,
                "num_workers": 8,
                **({"shuffle": False} if head != "train" else {}),
            }
        }
    data["metric_frames_per_episode"] = {
        h: "auto" for h in ("valid", "extra_valid", "unseen_op_valid")
    }
    data["video_episodes"] = {
        h: "???" for h in ("valid", "extra_valid", "unseen_op_valid")
    }
    data_path = REPO / "egomimic/hydra_configs/data/mecka_fold_moving_budget.yaml"
    data_path.write_text(yaml.safe_dump(data, sort_keys=False))
    common = {
        "defaults": [
            "/train_zarr_mecka_flagship_fold_kp_rdt",
            "/_scheduler_reset",
            {"override /data": "mecka_fold_moving_budget"},
            {"override /hydra/launcher": "submitit_pace"},
            "_self_",
        ],
        "name": "fold_clothes_moving_opscale",
        "description": "unconfigured",
        "seed": reference["seed"],
        "ckpt_path": None,
        "mode": "train",
        "train_viz": False,
        "launch_params": {"gpus_per_node": 1, "nodes": 1},
        "model": reference["model"],
        "trainer": {**reference["trainer"], "limit_val_batches": 800},
        "data": {
            "train_dataloader_params": {
                "human_bimanual": {"batch_size": 64, "num_workers": 8}
            },
            "train_datasets": {
                "human_bimanual": {
                    "resolver": {
                        "key_map": reference["data_key_map"],
                        "transform_list": reference["data_transform_list"],
                    }
                }
            },
        },
        "norm_stats": {
            **reference["norm_stats"],
            "save_cache_dir": "${paths.output_dir}",
            "cache_dir": "${paths.cache_dir}/norm_stats",
            "precomputed_norm_path": None,
        },
        "logger": {
            "wandb": {
                "project": "egoverse",
                "entity": "rl2-group",
                "group": "fold_clothes_moving_opscale_v1",
                "job_type": "moving_budget_opscale",
                "tags": ["fold_clothes_moving_opscale", "v1"],
            }
        },
        "callbacks": {
            "moving_opscale_evaluation": {
                "_target_": "egomimic.eval.moving_opscale_eval.MovingOpscaleEvaluation",
                "private_dir": "${oc.env:EGOVERSE_MECKA_PRIVATE_DIR}",
                "split_name": "???",
                "viz_every_n_epochs": 200,
            },
            "model_checkpoint": {
                "every_n_epochs": 50,
                "save_last": True,
                "save_top_k": 2,
                "monitor": "epoch",
                "mode": "max",
            },
        },
        # Metrics use the existing evaluator. The recovered reference open-loop
        # callback will supply explicitly pinned episode videos; launch is gated
        # until evaluation integration has passed a real smoke check.
        "evaluator": {"viz_every_n_epochs": 0, "viz_max_batches": 10},
    }
    emit(REPO / "egomimic/hydra_configs/fold_moving_common.yaml", common)
    for budget in BUDGETS:
        for count in SCALES:
            split = f"fold_moving_b{budget}_op{count}"

            def ref(key):
                return "${mecka_private:" + split + "," + key + "}"

            cfg = {
                "defaults": ["/fold_moving_common", "_self_"],
                "description": f"b{budget}_op{count}_v1",
                "callbacks": {"moving_opscale_evaluation": {"split_name": split}},
                "logger": {
                    "wandb": {
                        "name": f"fold_clothes_b{budget}_op{count}_rdt300m_v1",
                        "tags": [
                            "fold_clothes_moving_opscale",
                            "v1",
                            f"budget_hours={budget}",
                            f"operators={count}",
                        ],
                    }
                },
                "data": {
                    "train_operators": ref("operators"),
                    "video_episodes": {
                        "valid": ref("video_seen_matched"),
                        "extra_valid": ref("video_seen_perlevel"),
                        "unseen_op_valid": ref("video_unseen"),
                    },
                },
            }
            for head, key in [
                ("train", "train"),
                ("valid", "seen_matched"),
                ("extra_valid", "seen_perlevel"),
                ("unseen_op_valid", "unseen"),
            ]:
                cfg["data"][f"{head}_datasets"] = {
                    "human_bimanual": {"filters": {"episode_hashes": ref(key)}}
                }
            emit(HERE / f"budget_{budget}" / f"op_{count}.yaml", cfg)
    print(
        "Generated 36 Hydra recipes; private split resolution and launch gates remain mandatory."
    )


if __name__ == "__main__":
    main()
