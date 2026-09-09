"""Compose-only check of the queued eval configs: does hydra resolve, and does
the evaluator instantiate with the new arcmatch_gt_span_m knob? No model, no
dataset, so it is safe on a login node."""
import os, sys
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

WT = sys.argv[1]
DATA = os.path.expanduser("~/scratch/data/e1_fold")
ok = True
for exp, variant in (("fold_time", "time"), ("fold_arcmean", "arcmean"), ("fold_arcvel", "arcvel"),
                     ("abc_arcdur", "arcdur"), ("abc_time", "time"), ("abc_arclogdur", "arclogdur")):
    d = DATA if exp.startswith("fold") else os.path.expanduser("~/scratch/data/e1_abc")
    try:
        with initialize_config_dir(config_dir=f"{WT}/egomimic/hydra_configs", version_base=None):
            cfg = compose(config_name="train_zarr_cartesian", overrides=[
                f"+experiment=e1/{exp}", "mode=eval", "e1.spread=full",
                f"e1.train_root={d}/test_mid", f"e1.valid_root={d}/test_mid",
                "e1.h_match_frames=33", "e1.limit_val_batches=1.0", "seed=42",
                "evaluator.results_path=/tmp/x.json"])
        ev = OmegaConf.to_container(cfg.evaluator, resolve=True)
        gs = ev.get("arcmatch_gt_span_m")
        e = instantiate(cfg.evaluator, _recursive_=False)
        print(f"  {exp:18} variant={variant:10} D={cfg.e1.D:<5} gt_span_m={gs}  "
              f"-> evaluator.gt_span_m={e.gt_span_m}  velocity_norm={cfg.e1.velocity_norm}")
        assert e.variant == variant, f"variant mismatch {e.variant} != {variant}"
    except Exception as exc:
        ok = False
        print(f"  {exp:18} FAILED: {type(exc).__name__}: {exc}")
print("COMPOSE_OK" if ok else "COMPOSE_FAILED")
sys.exit(0 if ok else 1)
