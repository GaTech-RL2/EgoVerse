#!/usr/bin/env python3
"""Fail closed unless a resolved config is the approved fresh clean DP campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf

DOMAINS = {"pushshapes_sim_u_socket", "pushshapes_sim_chain_gripper"}
STAGES = [
    "FusedObsEncoder",
    "ActionTargetBuilder",
    "DiffusionNoisingStage",
    "DiffusionDenoiserStage",
    "DiffusionEpsilonLossStage",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-audit", type=Path, required=True)
    parser.add_argument("--norm", type=Path, required=True)
    parser.add_argument("--parameter-count", type=int, required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--validation-interval", type=int, required=True)
    parser.add_argument("--checkpoint-interval", type=int, required=True)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    split = json.loads(args.split_audit.read_text())
    assert cfg.name == "planar_v2_cotrain_clean_dp_standard_h16"
    assert cfg.mode == "train" and cfg.ckpt_path is None
    assert set(cfg.data.train_datasets) == DOMAINS
    assert set(cfg.data.valid_datasets) == DOMAINS
    assert set(cfg.evaluator.native_decoders) == DOMAINS
    assert cfg.run_provenance.obstacle_data is False
    assert set(cfg.run_provenance.domains) == DOMAINS
    assert int(cfg.trainer.devices) == 1 and int(cfg.trainer.num_nodes) == 1
    assert str(cfg.trainer.precision) == "bf16"
    assert int(cfg.trainer.max_steps) == args.expected_steps
    assert int(cfg.trainer.val_check_interval) == args.validation_interval
    assert int(cfg.callbacks.model_checkpoint.every_n_train_steps) == args.checkpoint_interval
    assert cfg.callbacks.model_checkpoint.save_top_k == -1
    assert Path(cfg.norm_stats.precomputed_norm_path).resolve() == args.norm.resolve()
    assert float(cfg.norm_stats.sample_frac) == 1.0
    stage_names = [stage._target_.rsplit(".", 1)[-1] for stage in cfg.model.pipeline.stages]
    assert stage_names == STAGES
    assert split["status"] == "PASS"
    assert split["cross_domain_train_valid_resolved_path_overlap_count"] == 0
    counts = {(d["total_count"], d["train_count"], d["valid_count"]) for d in split["domains"].values()}
    assert counts == {(2999, 2970, 29), (3000, 2970, 30)}
    model = instantiate(cfg.model.pipeline)
    assert sum(p.numel() for p in model.nets.parameters()) == args.parameter_count
    print("[contract] PASS fresh clean two-domain standard DP")


if __name__ == "__main__":
    main()
