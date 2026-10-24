"""The SharpaDex ID contract and the manifest reader."""

import json
from dataclasses import asdict

import pytest

from egomimic.rldb.zarr.required_metadata import vendor_recording_id_error
from egomimic.scripts.sharpa_process.sharpadex_records import (
    EpisodeRecord,
    ManifestError,
    canonical_id,
    id_problems,
    ids_correspond,
    parse_source_episode_id,
    parse_vendor_recording_id,
    read_manifest,
    source_episode_id,
    vendor_recording_id,
)

TASK = "assemble_gears_on_base"
SEASON = "season_POC22027_2026_04_30_15_04_50_train"


def _record(task=TASK, season=SEASON, index=0, recording_id=None, **changes):
    fields = dict(
        recording_id=recording_id or f"{task}/{season}/{index}",
        task=task,
        season=season,
        episode_index=index,
        rig="POC22027",
        length=10,
        fps=30,
        source_uid=f"uid{index}",
        fingerprint="0" * 16,
        tags={},
    )
    return EpisodeRecord(**{**fields, **changes})


def _manifest(path, records, header=None):
    lines = [json.dumps({"manifest": header or {"release_root": "/r"}})]
    lines += [json.dumps(asdict(r)) for r in records]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_canonical_and_source_ids_round_trip():
    vid = vendor_recording_id(TASK, SEASON, 7)
    assert vid == f"{TASK}__{SEASON}__000007"
    assert vendor_recording_id_error(vid) is None
    assert parse_vendor_recording_id(vid) == (TASK, SEASON, 7)
    sid = source_episode_id(TASK, SEASON, 7)
    assert sid == f"{TASK}/{SEASON}/7"
    assert parse_source_episode_id(sid) == (TASK, SEASON, 7)
    assert canonical_id(sid) == vid
    assert ids_correspond(vid, sid)
    assert not ids_correspond(vid, f"{TASK}/{SEASON}/8")
    assert not ids_correspond(sid, vid)
    # Indices past six digits still round-trip.
    assert parse_vendor_recording_id(vendor_recording_id(TASK, SEASON, 1234567)) == (
        TASK,
        SEASON,
        1234567,
    )


@pytest.mark.parametrize(
    "task, season, index",
    [
        ("a/b", SEASON, 0),  # a "/" would split the S3 path
        ("a__b", SEASON, 0),  # "__" would make the parse ambiguous
        (TASK, "x__y", 0),
        # A trailing "_" runs into the separator: "task___season" splits wrongly.
        ("task_", SEASON, 0),
        (TASK, "season_", 0),
        ("", SEASON, 0),
        (TASK, "has space", 0),
        (TASK, SEASON, -1),
        (TASK, SEASON, True),
    ],
)
def test_parts_that_break_the_contract_are_rejected(task, season, index):
    with pytest.raises(ValueError):
        vendor_recording_id(task, season, index)
    with pytest.raises(ValueError):
        source_episode_id(task, season, index)


@pytest.mark.parametrize(
    "text",
    [
        f"{TASK}__{SEASON}__7",  # not zero-padded
        f"{TASK}__{SEASON}__0000007",  # padded past six digits
        f"{TASK}/{SEASON}/7",
        f"{TASK}__{SEASON}",
        f"a__b__{SEASON}__000001",
        None,
    ],
)
def test_non_canonical_ids_do_not_parse(text):
    with pytest.raises(ValueError):
        parse_vendor_recording_id(text)


@pytest.mark.parametrize(
    "text", [f"{TASK}/{SEASON}/07", f"{TASK}/{SEASON}", f"{TASK}/{SEASON}/x", 3]
)
def test_non_canonical_source_ids_do_not_parse(text):
    with pytest.raises(ValueError):
        parse_source_episode_id(text)


def test_id_problems_finds_mismatches_bad_parts_and_collisions():
    good = [_record(index=0), _record(index=1)]
    assert id_problems(good) == []
    mismatch = _record(index=2, recording_id=f"{TASK}/{SEASON}/3")
    bad_part = _record(task="bad__task", index=0)
    problems = id_problems([*good, mismatch, bad_part])
    assert len(problems) == 2
    assert "does not match" in problems[0] and "bad__task" in problems[1]
    # The map is injective once the parts are valid, so a shared canonical id
    # means the same episode is listed twice.
    twin = _record(index=1, recording_id=f"{TASK}/{SEASON}/1")
    assert any("canonical id of" in p for p in id_problems([*good, twin]))


def test_read_manifest_returns_records_by_source_id(tmp_path):
    path = _manifest(tmp_path / "m.jsonl", [_record(index=0), _record(index=1)])
    header, records = read_manifest(path)
    assert header == {"release_root": "/r"}
    assert list(records) == [f"{TASK}/{SEASON}/0", f"{TASK}/{SEASON}/1"]
    assert records[f"{TASK}/{SEASON}/1"].vendor_recording_id == (
        f"{TASK}__{SEASON}__000001"
    )


def test_read_manifest_rejects_malformed_files(tmp_path):
    def fails(text, match):
        path = tmp_path / "bad.jsonl"
        path.write_text(text)
        with pytest.raises(ManifestError, match=match):
            read_manifest(path)

    record = json.dumps(asdict(_record()))
    fails("", "not a SharpaDex manifest")
    fails('{"not": "a manifest"}\n', "not a SharpaDex manifest")
    fails("not json\n", "not a SharpaDex manifest")
    fails('{"manifest": {}}\n{"recording_id": "x"}\n', "malformed record")
    fails('{"manifest": {}}\n{oops\n', "malformed record")
    fails(f'{{"manifest": {{}}}}\n{record}\n{record}\n', "repeated")
    bad = json.dumps(asdict(_record(task="a__b")))
    fails(f'{{"manifest": {{}}}}\n{bad}\n', "id problem")


def test_every_accepted_id_parses_back_exactly():
    for task, season in ((TASK, SEASON), ("a.b-c", "s-1.x"), ("t1", "season_9")):
        for index in (0, 7, 999_999, 1_000_000):
            vid = vendor_recording_id(task, season, index)
            assert parse_vendor_recording_id(vid) == (task, season, index)


def test_manifest_records_with_a_trailing_underscore_are_refused(tmp_path):
    bad = [_record(task="fold_towel_", index=0), _record(season="season_x_", index=1)]
    problems = id_problems(bad)
    assert len(problems) == 2 and all("'_'" in p for p in problems)
    path = _manifest(tmp_path / "m.jsonl", bad)
    with pytest.raises(ManifestError, match="2 id problem"):
        read_manifest(path)
