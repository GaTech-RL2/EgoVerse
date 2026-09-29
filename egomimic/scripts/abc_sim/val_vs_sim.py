"""Does the offline validation metric predict closed-loop sim success?

Input: a JSON manifest of checkpoints, each with the W&B run its validation
metrics live in, its epoch, its data hours (the ladder rung) and the
sim_client summary.json of its fine-tune's rollout:

    [{"name": "k4_ep400", "wandb": "rl2-group/egoverse/<run id>", "epoch": 400,
      "hours": 1.096, "summary": ".../summary.json"}, ...]

    python -m egomimic.scripts.abc_sim.val_vs_sim manifest.json --metric \
        "seen_op_valid/Valid/human_bimanual_actions_cartesian_paired_mse_avg" --out report/

Writes report.json + report.md: per-checkpoint rows, Spearman and Pearson
between the val metric and success_rate / mean_max_progress with 2000-sample
bootstrap CIs, and the log-linear fit of the val metric against data hours
(one point per rung at its final epoch) with its slope and R^2.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

# constant inputs inside a bootstrap resample give nan by design
warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)


def wandb_metric(run_path: str, metric: str, epoch: int) -> float:
    """The metric's value at (the last logged step of) ``epoch``."""
    import wandb

    run = wandb.Api().run(run_path)
    rows = [r for r in run.scan_history(keys=[metric, "epoch"]) if r.get(metric) is not None]
    at = [r for r in rows if r.get("epoch") is not None and int(r["epoch"]) <= epoch]
    if not at:
        raise ValueError(f"{run_path}: no {metric} at epoch <= {epoch}")
    return float(at[-1][metric])


def corr(x, y, kind: str) -> float:
    return float((stats.spearmanr if kind == "spearman" else stats.pearsonr)(x, y)[0])


def bootstrap(x, y, kind: str, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile interval of ``corr`` over resampled checkpoints."""
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    vals = [corr(x[i], y[i], kind) for i in (rng.integers(0, len(x), len(x)) for _ in range(n))]
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def loglinear(hours, metric) -> dict:
    """metric = intercept + slope * log10(hours)."""
    f = stats.linregress(np.log10(hours), metric)
    return {"slope": float(f.slope), "intercept": float(f.intercept), "r2": float(f.rvalue**2)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("manifest", type=Path)
    p.add_argument("--metric", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--no-wandb", action="store_true", help="manifest rows already carry 'val'")
    a = p.parse_args(argv)
    rows = json.loads(a.manifest.read_text())
    for r in rows:
        if "val" not in r or not a.no_wandb:
            r["val"] = wandb_metric(r["wandb"], a.metric, int(r["epoch"]))
        s = json.loads(Path(r["summary"]).read_text())
        r["success_rate"], r["mean_max_progress"], r["num_worlds"] = s["success_rate"], s.get("mean_max_progress"), s["num_worlds"]
    val = [r["val"] for r in rows]
    out = {"metric": a.metric, "rows": rows, "n": len(rows), "correlations": {}}
    for target in ("success_rate", "mean_max_progress"):
        y = [r[target] for r in rows]
        if any(v is None for v in y):
            continue
        for kind in ("spearman", "pearson"):
            lo, hi = bootstrap(val, y, kind)
            out["correlations"][f"{kind}_{target}"] = {"r": corr(val, y, kind), "ci95": [lo, hi]}
    # log-linear in data: the final epoch of each rung
    final = {}
    for r in rows:
        if "hours" in r and (r["hours"] not in final or r["epoch"] > final[r["hours"]]["epoch"]):
            final[r["hours"]] = r
    if len(final) >= 2:
        hs = sorted(final)
        out["loglinear_val_vs_hours"] = {"hours": hs, "val": [final[h]["val"] for h in hs], **loglinear(hs, [final[h]["val"] for h in hs])}
        out["loglinear_success_vs_hours"] = {"hours": hs, "success": [final[h]["success_rate"] for h in hs], **loglinear(hs, [final[h]["success_rate"] for h in hs])}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "report.json").write_text(json.dumps(out, indent=2))
    md = [f"# val metric vs sim success ({len(rows)} checkpoints)", "", f"metric: `{a.metric}`", "",
          "| name | hours | epoch | val | success | max progress | worlds |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        prog = "" if r["mean_max_progress"] is None else f"{r['mean_max_progress']:.3f}"
        md.append(f"| {r['name']} | {r.get('hours', '')} | {r['epoch']} | {r['val']:.4g} | {r['success_rate']:.3f} | {prog} | {r['num_worlds']} |")
    md += ["", "| correlation | r | 95% CI |", "|---|---|---|"]
    md += [f"| {k} | {v['r']:.3f} | [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] |" for k, v in out["correlations"].items()]
    for key in ("loglinear_val_vs_hours", "loglinear_success_vs_hours"):
        if key in out:
            f = out[key]
            md += ["", f"{key}: slope {f['slope']:.4g} per decade, R^2 {f['r2']:.3f} over hours {f['hours']}"]
    (a.out / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
