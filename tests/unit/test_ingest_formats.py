"""Format detection and dispatch (egomimic/ingest/formats.py)."""

import shutil
import subprocess

import pytest
from fixtures.mcap_episodes import write_mcap_episode

from egomimic.ingest import formats
from egomimic.ingest.formats import (
    AmbiguousFormat,
    by_name,
    detect,
    prepare_delivery,
)


def make(tmp_path, *names):
    d = tmp_path / "delivery"
    for n in names:
        (d / n).parent.mkdir(parents=True, exist_ok=True)
        (d / n).write_bytes(b"x")
    return d


@pytest.mark.parametrize(
    "names, expected",
    [
        (["episode.mcap"], "mcap"),
        (["episode.mcap", "annotation.mcap"], "mcap"),
        (
            ["episode.mcap", "notes.txt"],
            "mcap",
        ),  # recognized; the extra file rejects it
        (["recording.vrs", "recording.vrs.json"], None),
        (
            ["sub/episode.mcap"],
            None,
        ),  # the episode file sits at the top of the delivery
        (["annotation.mcap"], None),
        ([], None),
    ],
)
def test_detect(names, expected):
    f = detect(names)
    assert (f.name if f else None) == expected


def test_mcap_accepts_exactly_its_files():
    mcap = by_name("mcap")
    assert mcap.allowed == {"episode.mcap", "annotation.mcap"}
    assert mcap.unexpected({"episode.mcap", "notes.txt"}) == ["notes.txt"]
    with pytest.raises(KeyError):
        by_name("zarr")


def test_two_formats_claiming_a_delivery_is_an_error(monkeypatch):
    mcap = by_name("mcap")
    clone = formats.Format(**{**mcap.__dict__, "name": "mcap-v2"})
    monkeypatch.setattr(formats, "formats", lambda: [mcap, clone])
    with pytest.raises(AmbiguousFormat):
        detect(["episode.mcap"])


def test_an_unknown_delivery_is_unsupported_not_guessed(tmp_path):
    d = make(tmp_path, "recording.vrs", "mps/slam/closed_loop_trajectory.csv")
    p = prepare_delivery(
        d, tmp_path / "staged", tmp_path / "lance", fingerprint="fp", vendor="acme"
    )
    assert p.status == "unsupported" and p.staged is None
    assert "mps/slam/closed_loop_trajectory.csv" in p.problems[0]
    assert "mcap: episode.mcap" in p.problems[0]


def test_unexpected_files_reject_before_any_conversion(tmp_path):
    d = make(tmp_path, "episode.mcap", "notes.txt")  # not even a valid MCAP: never read
    p = prepare_delivery(
        d, tmp_path / "staged", tmp_path / "lance", fingerprint="fp", vendor="acme"
    )
    assert p.status == "rejected" and "unexpected: ['notes.txt']" in p.problems[0]
    assert not (tmp_path / "staged").exists()


def test_an_mcap_delivery_prepares_through_its_format(tmp_path):
    pytest.importorskip("lance")
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True)
    if not shutil.which("ffmpeg") or "libx264" not in out.stdout.decode():
        pytest.skip("needs ffmpeg with libx264")
    from egomimic.ingest.stage import read_pin

    ep = write_mcap_episode(tmp_path / "eps")
    p = prepare_delivery(
        ep, tmp_path / "staged", tmp_path / "lance", fingerprint="fp", vendor="acme",
        codec="libx264", mano=False,
    )  # fmt: skip
    assert p.status == "staged" and read_pin(p.staged)["format"] == "mcap"


def test_oversized_files_reject_before_any_conversion(tmp_path, monkeypatch):
    small = formats.Format(**{**by_name("mcap").__dict__, "max_file_bytes": 3})
    monkeypatch.setattr(formats, "formats", lambda: [small])
    d = make(tmp_path, "episode.mcap")
    (d / "annotation.mcap").write_bytes(b"four")  # 4 bytes > 3
    p = prepare_delivery(
        d, tmp_path / "staged", tmp_path / "lance", fingerprint="fp", vendor="acme"
    )
    assert p.status == "rejected" and len(p.problems) == 1
    assert p.problems[0].startswith("annotation.mcap is 0.0 GiB, over the mcap limit")
    assert not (tmp_path / "staged").exists()


def test_oversize_works_on_listed_sizes_before_download():
    mcap = by_name("mcap")
    limit = mcap.max_file_bytes
    assert mcap.oversize({"episode.mcap": limit}) == []
    (problem,) = mcap.oversize({"episode.mcap": limit + 1, "annotation.mcap": 10})
    assert problem.startswith(
        "episode.mcap is 200.0 GiB, over the mcap limit of 200 GiB"
    )
