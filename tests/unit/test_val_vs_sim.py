"""val_vs_sim: correlations and the log-linear fit from a manifest with
pre-filled val metrics (no W&B)."""

import json

from egomimic.scripts.abc_sim import val_vs_sim as vs


def test_report_from_manifest(tmp_path):
    rows = []
    for i, (h, ep, val, succ) in enumerate([(0.237, 1000, 0.9, 0.1), (0.559, 1000, 0.6, 0.4), (1.096, 1000, 0.3, 0.7), (1.096, 500, 0.5, 0.5)]):
        s = tmp_path / f"s{i}.json"
        s.write_text(json.dumps({"success_rate": succ, "mean_max_progress": succ + 0.1, "num_worlds": 50}))
        rows.append({"name": f"r{i}", "epoch": ep, "hours": h, "val": val, "summary": str(s)})
    m = tmp_path / "m.json"
    m.write_text(json.dumps(rows))
    assert vs.main([str(m), "--metric", "x", "--out", str(tmp_path / "out"), "--no-wandb"]) == 0
    rep = json.loads((tmp_path / "out" / "report.json").read_text())
    assert rep["correlations"]["spearman_success_rate"]["r"] == -1.0  # lower val mse, higher success
    assert rep["loglinear_val_vs_hours"]["r2"] > 0.95 and rep["loglinear_val_vs_hours"]["slope"] < 0
    assert (tmp_path / "out" / "report.md").read_text().count("| r") >= 4
