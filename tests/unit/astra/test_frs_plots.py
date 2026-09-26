"""Scientific exports bind complete synthetic data, never collect outcomes."""

import copy
import hashlib
import json

import pytest

from astra_reversal.frs_plots import export_plots, plotted_values
from astra_reversal.frs_report import build_report
from astra_reversal.records import file_sha256

from .test_frs_report import fixture_tree, write


def test_checkpoint_curve_is_not_cumulative_and_controls_have_one_measurement(tmp_path):
    workers, receipts = fixture_tree(tmp_path, phase="evaluation")
    report = build_report(workers, phase="evaluation", audit_receipts=receipts)
    values = plotted_values(report)
    assert (values["phase"], values["seed"], values["tasks"]) == ("evaluation", 43, 20)
    assert values["cohorts"]["adaptation"]["episodes"] == 20
    assert values["cohorts"]["evaluation"]["episodes"] == 200
    methods = values["cohorts"]["evaluation"]["methods"]
    assert [point["successes"] for point in methods["learned_noise"]["points"]] == [
        0,
        200,
        0,
    ]
    assert [
        point["unsuccessful_episodes"] for point in methods["learned_noise"]["points"]
    ] == [200, 0, 200]
    for name, method in methods.items():
        if name != "learned_noise":
            assert [point["round"] for point in method["points"]] == [3]
    assert values["physical_cost"]["unique_rollouts"] == 1740
    assert len(values["source_tasks"]) == 20


def test_partial_or_unaudited_data_cannot_export_scientific_results(tmp_path):
    workers, receipts = fixture_tree(tmp_path / "inputs")
    report = build_report(workers, phase="development", audit_receipts=receipts[:2])
    write(tmp_path / "report.json", report)
    with pytest.raises(ValueError, match="complete coverage and passed task audits"):
        export_plots(tmp_path / "report.json", tmp_path / "figures")
    assert not (tmp_path / "figures").exists()
    complete = build_report(workers, phase="development", audit_receipts=receipts)
    complete["producer_evidence"]["tasks"][0]["audited"] = False
    with pytest.raises(ValueError, match="complete coverage and passed task audits"):
        plotted_values(complete)


def test_role_partition_cannot_drop_failed_call_cost(tmp_path):
    workers, receipts = fixture_tree(tmp_path)
    report = build_report(workers, phase="development", audit_receipts=receipts)
    altered = copy.deepcopy(report)
    altered["producer_evidence"]["tasks"][0]["provider_by_role"]["critique"]["tokens"][
        "reasoning_tokens"
    ]["sum"] -= 1
    with pytest.raises(
        ValueError, match="conserve the unique physical provider ledger"
    ):
        plotted_values(altered)


def test_missing_usage_exports_lower_bounds_and_binds_every_plotted_source(tmp_path):
    workers, receipts = fixture_tree(tmp_path / "inputs", missing_usage=True)
    report = build_report(workers, phase="development", audit_receipts=receipts)
    source = tmp_path / "report.json"
    write(source, report)
    raw = source.read_bytes()
    output = tmp_path / "figures"
    result = export_plots(source, output)
    values = json.loads((output / "plotted_values.json").read_text())
    assert (values["phase"], values["seed"], values["tasks"]) == ("development", 19, 3)
    assert values["report_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["plotted_values_sha256"] == file_sha256(
        output / "plotted_values.json"
    )
    assert result["files"] == {
        path.name: {"sha256": file_sha256(path), "bytes": path.stat().st_size}
        for path in output.iterdir()
        if path.name != "manifest.json"
    }
    usage = values["physical_cost"]["provider_usage"]
    assert usage["calls"] == 63 and usage["failed_calls"] == 3
    assert usage["tokens"]["total_tokens"] == {"sum": 930, "missing_calls": 1}
    assert (
        values["provider_by_role"]["critique"]["tokens"]["input_tokens"][
            "missing_calls"
        ]
        == 1
    )
    for method in values["cohorts"]["adaptation"]["methods"].values():
        assert (
            method["rescue"]["rescued"] + method["rescue"]["censored"]
            == method["rescue"]["baseline_failed"]
        )
    token_svg = (output / "provider_token_cost.svg").read_text()
    assert "lower bound" in token_svg and "1 calls missing usage" in token_svg
    assert "seed 19" in token_svg and "development" in token_svg
    assert (
        (output / "provider_token_cost.png")
        .read_bytes()
        .startswith(b"\x89PNG\r\n\x1a\n")
    )
    assert (output / "success_and_learning.pdf").read_bytes().startswith(b"%PDF-")
    assert source.read_bytes() == raw
    with pytest.raises(ValueError, match="overwrite"):
        export_plots(source, output)
