"""Distributed archives must remain remote unless a one-byte receipt is requested."""

import json
import sys
from unittest.mock import MagicMock

import pytest

from astra_reversal.osmo import download_reports


@pytest.mark.parametrize("key", ["artifacts.tar.gz", "worker_3/artifacts.tar.gz"])
def test_large_archive_is_skipped_without_network(tmp_path, monkeypatch, capsys, key):
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({key: "https://example.invalid/private"}))
    destination = tmp_path / "reports"
    monkeypatch.setattr(
        sys, "argv", ["download_reports", str(catalog), str(destination), key]
    )
    open_url = MagicMock(side_effect=AssertionError("Archive must not be fetched"))
    monkeypatch.setattr(download_reports.urllib.request, "urlopen", open_url)

    download_reports.main()

    open_url.assert_not_called()
    assert not (destination / key).exists()
    assert json.loads(capsys.readouterr().out)["status"] == "skipped_large_archive"


def test_distributed_receipts_are_separate_and_read_one_byte(tmp_path, monkeypatch):
    keys = [f"worker_{index}/artifacts.tar.gz" for index in (1, 3)]
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps({key: f"https://example.invalid/{key}" for key in keys})
    )
    destination = tmp_path / "reports"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_reports",
            str(catalog),
            str(destination),
            *keys,
            "--archive-receipt",
        ],
    )
    responses = []

    def open_url(request, timeout):
        assert request.get_header("Range") == "bytes=0-0"
        assert timeout == 30
        response = MagicMock()
        response.__enter__.return_value = response
        response.status = 206
        response.headers = {
            "Content-Range": "bytes 0-0/1000000000",
            "ETag": request.full_url.rsplit("/", 2)[1],
        }
        responses.append(response)
        return response

    monkeypatch.setattr(download_reports.urllib.request, "urlopen", open_url)

    download_reports.main()

    assert len(responses) == 2
    for response in responses:
        response.read.assert_called_once_with(1)
    for index in (1, 3):
        receipt = json.loads(
            (destination / f"worker_{index}/archive_receipt.json").read_text()
        )
        assert receipt["status"] == 206
        assert receipt["etag"] == f"worker_{index}"
        assert not (destination / f"worker_{index}/artifacts.tar.gz").exists()
    assert not (destination / "archive_receipt.json").exists()
