"""Validate all private splits and actual Hydra composition; require real smoke evidence for launch."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import build_splits as splits
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def compare(expected, actual, prefix):
    for key, value in expected.items():
        full = prefix + "." + key
        if isinstance(value, dict):
            compare(value, actual[key], full)
        elif OmegaConf.is_config(actual[key]):
            assert OmegaConf.to_container(actual[key], resolve=True) == value, full
        else:
            assert actual[key] == value, full


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--private-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--evaluation-smoke", type=Path)
    a = p.parse_args()
    root = a.private_dir.resolve()
    os.environ["EGOVERSE_MECKA_PRIVATE_DIR"] = str(root)
    import egomimic.utils.hydra_resolvers  # noqa: F401

    design = json.loads((root / "moving_opscale_design.json").read_text())
    all_splits = {
        p.stem: json.loads(p.read_text())
        for p in (root / "splits").glob("fold_moving_b*_op*.json")
    }
    inputs = root / "inputs"
    validation = splits.validate(
        design,
        all_splits,
        splits.read_ids(inputs / "banned_episodes.txt"),
        splits.read_ids(inputs / "banned_users.txt"),
    )
    reference = json.loads((HERE / "reference_recipe.json").read_text())
    entries = []
    for budget in splits.BUDGETS:
        for count in splits.SCALES:
            folder = HERE / f"budget_{budget}"
            path = folder / f"op_{count}.yaml"
            name = f"fold_moving_b{budget}_op{count}"
            s = all_splits[name]
            with initialize_config_dir(config_dir=str(folder), version_base=None):
                cfg = compose(
                    config_name=f"op_{count}",
                    overrides=[
                        f"hydra.searchpath=[file://{REPO}/egomimic/hydra_configs]"
                    ],
                )
                compare(reference["model"], cfg.model, "model")
                compare(
                    {**reference["trainer"], "limit_val_batches": 800},
                    cfg.trainer,
                    "trainer",
                )
                compare(
                    reference["data_key_map"],
                    cfg.data.train_datasets.human_bimanual.resolver.key_map,
                    "key_map",
                )
                compare(
                    reference["data_transform_list"],
                    cfg.data.train_datasets.human_bimanual.resolver.transform_list,
                    "transform",
                )
                assert cfg.data.train_dataloader_params.human_bimanual.batch_size == 64
                assert (
                    cfg.launch_params.gpus_per_node == 1
                    and cfg.trainer.accumulate_grad_batches == 1
                )
                assert cfg.seed == 42026 and cfg.model.ema.enabled is False
                assert (
                    cfg.data.exact_val_reads and not cfg.data.replace_invalid_val_frames
                )
                assert cfg.callbacks.moving_opscale_evaluation.split_name == name
                for head, key, prefix in [
                    ("valid", "seen_matched", "val/seen_matched"),
                    ("extra_valid", "seen_perlevel", "val/seen_perlevel"),
                    ("unseen_op_valid", "unseen", "val/unseen"),
                ]:
                    actual = list(
                        cfg.data[
                            f"{head}_datasets"
                        ].human_bimanual.filters.episode_hashes
                    )
                    assert actual == s[key]
                    assert cfg.data.metric_frames_per_episode[head] == "auto"
                    batch = cfg.data[
                        f"{head}_dataloader_params"
                    ].human_bimanual.batch_size
                    assert (
                        batch == 32
                        and len(actual) <= cfg.trainer.limit_val_batches * batch
                    )
                    assert (
                        cfg.data[
                            {
                                "valid": "valid_prefix",
                                "extra_valid": "extra_valid_prefix",
                                "unseen_op_valid": "unseen_op_valid_prefix",
                            }[head]
                        ]
                        == prefix
                    )
                    assert list(cfg.data.video_episodes[head]) == s["video_" + key]
                assert (
                    list(cfg.data.train_datasets.human_bimanual.filters.episode_hashes)
                    == s["train"]
                )
                assert list(cfg.data.train_operators) == s["operators"]
                assert cfg.norm_stats.precomputed_norm_path is None
            entries.append(
                {
                    "budget_hours": budget,
                    "operator_count": count,
                    "config_sha256": sha(path),
                    "split_sha256": sha(root / "splits" / f"{name}.json"),
                    "unseen_episode_count": len(s["unseen"]),
                }
            )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    smoke = json.loads(a.evaluation_smoke.read_text()) if a.evaluation_smoke else {}
    smoke_ok = (
        smoke.get("status") == "passed"
        and smoke.get("commit") == commit
        and smoke.get("design_sha256") == sha(root / "moving_opscale_design.json")
        and smoke.get("checks")
        == {
            "finite_loss": True,
            "checkpoint_resume": True,
            "all_validation_episodes": True,
            "wandb_metrics": True,
            "video_counts": True,
        }
    )
    report = {
        "schema_version": 1,
        "status": "passed" if smoke_ok else "blocked",
        "commit": commit,
        "private_root": str(root),
        "design_sha256": sha(root / "moving_opscale_design.json"),
        "checks": {
            "dataset_integrity": True,
            "operator_consistency": True,
            "budget_consistency": True,
            "configuration_integrity": True,
            "reference_recipe": True,
            "evaluation_smoke": smoke_ok,
        },
        "configurations": entries,
        "actual_hours": validation["actual_hours"],
        "blockers": []
        if smoke_ok
        else [
            "Real GPU evaluation/logging/checkpoint-resume smoke evidence is missing."
        ],
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "configurations": len(entries),
                "data_and_config_checks": "passed",
                "launch_status": report["status"],
                "blockers": report["blockers"],
            }
        )
    )


if __name__ == "__main__":
    main()
