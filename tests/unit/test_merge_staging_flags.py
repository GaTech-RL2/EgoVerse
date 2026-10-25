"""Merge guards: fail on a hash conflict, and copy the licence from attributes."""

from __future__ import annotations

import sys

import pytest

from egomimic.scripts.backfill_scripts import merge_staging_to_episodes as merge

STAGING_COLUMNS = [
    "episode_hash",
    "embodiment",
    "task",
    "task_description",
    "num_frames",
    "zarr_processed_path",
    "created_at",
    "source_folder",
    "raw_attrs",
]


class _Result:
    def __init__(self, value, rowcount=None):
        self.value = value
        self.rowcount = rowcount

    def all(self):
        return self.value

    def scalar(self):
        return self.value


class _Block:
    """One ``connect()`` or ``begin()`` block: its SQL and how it ended."""

    def __init__(self, engine, transaction):
        self.engine = engine
        self.transaction = transaction
        self.sql: list[str] = []
        self.error = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        # An exception leaving a begin() block rolls the transaction back.
        self.error = exc_type
        return False

    def execute(self, clause, params=None):
        sql = str(clause)
        self.sql.append(sql)
        self.engine.sql.append(sql)
        if "information_schema" in sql:
            return _Result([(c,) for c in STAGING_COLUMNS])
        if "COUNT(*)" in sql:
            return _Result(self.engine.counts.pop(0))
        if "INSERT INTO" in sql and self.engine.insert_error is not None:
            raise self.engine.insert_error
        return _Result(None, self.engine.rowcount)


class _Engine:
    """Answer the merge's queries: columns, then four counts, then the insert."""

    def __init__(self, counts, rowcount=2, insert_error=None):
        self.counts = list(counts)
        self.rowcount = rowcount
        self.insert_error = insert_error
        self.sql: list[str] = []
        self.blocks: list[_Block] = []

    def connect(self):
        self.blocks.append(_Block(self, transaction=False))
        return self.blocks[-1]

    def begin(self):
        self.blocks.append(_Block(self, transaction=True))
        return self.blocks[-1]

    def dispose(self):
        pass


def _run(monkeypatch, counts, *flags, **engine_kwargs):
    engine = _Engine(counts, **engine_kwargs)
    monkeypatch.setattr(merge, "create_default_engine", lambda: engine)
    monkeypatch.setattr(
        sys, "argv", ["merge", "--staging-table", "app.staging_sharpa_dex", *flags]
    )
    return merge.main(), engine


def test_a_conflict_fails_before_anything_is_written(monkeypatch, capsys):
    status, engine = _run(monkeypatch, [5, 0, 1, 4], "--fail-on-conflict", "--apply")

    assert status == 1
    assert "FAILED: 1 staged hash" in capsys.readouterr().out
    assert not any("INSERT INTO" in sql for sql in engine.sql)


def test_without_the_flag_conflicts_are_skipped_as_before(monkeypatch):
    status, engine = _run(monkeypatch, [5, 0, 1, 4], "--apply")

    assert status == 0
    [insert] = [sql for sql in engine.sql if "INSERT INTO" in sql]
    assert "license" not in insert and "operator" not in insert


@pytest.mark.parametrize("column", ["license", "operator"])
def test_a_column_comes_from_the_staged_attributes(monkeypatch, column):
    status, engine = _run(
        monkeypatch,
        [4, 0, 0, 4, 4],
        "--fail-on-conflict",
        f"--{column}-from-attrs",
        "--apply",
        rowcount=4,
    )

    assert status == 0
    [insert] = [sql for sql in engine.sql if "INSERT INTO" in sql]
    assert column in insert and f"s.raw_attrs->>'{column}'" in insert


@pytest.mark.parametrize("column", ["license", "operator"])
def test_an_attribute_copy_needs_staged_attributes(monkeypatch, column):
    monkeypatch.setattr(sys.modules[__name__], "STAGING_COLUMNS", STAGING_COLUMNS[:-1])
    with pytest.raises(SystemExit, match="raw_attrs"):
        _run(monkeypatch, [4, 0, 0, 4], f"--{column}-from-attrs")


def test_a_strict_merge_selects_and_inserts_in_one_transaction(monkeypatch):
    status, engine = _run(
        monkeypatch, [4, 0, 0, 4, 4], "--fail-on-conflict", "--apply", rowcount=4
    )

    assert status == 0
    [insert] = [b for b in engine.blocks if any("INSERT INTO" in q for q in b.sql)]
    assert insert.transaction and insert.error is None
    # The expected row count is taken in the inserting transaction, and no
    # conflict is suppressed: a hash already in app.episodes aborts the insert.
    assert "COUNT(*)" in insert.sql[0] and "INSERT INTO" in insert.sql[-1]
    assert "ON CONFLICT" not in insert.sql[-1] and "NOT EXISTS" not in insert.sql[-1]


def test_a_strict_merge_rolls_back_a_conflict_inserted_after_the_preview(
    monkeypatch, capsys
):
    from sqlalchemy.exc import IntegrityError

    clash = IntegrityError("INSERT", {}, Exception("duplicate key episode_hash"))
    status, engine = _run(
        monkeypatch,
        [4, 0, 0, 4, 4],
        "--fail-on-conflict",
        "--apply",
        insert_error=clash,
    )

    assert status == 1
    assert engine.blocks[-1].transaction and engine.blocks[-1].error is IntegrityError
    assert "rolled back" in capsys.readouterr().out


def test_a_strict_merge_rolls_back_an_unexpected_row_count(monkeypatch, capsys):
    status, engine = _run(
        monkeypatch, [4, 0, 0, 4, 4], "--fail-on-conflict", "--apply", rowcount=3
    )

    assert status == 1
    assert engine.blocks[-1].transaction and engine.blocks[-1].error is not None
    assert "inserted 3 row(s), expected 4" in capsys.readouterr().out


def test_without_the_flag_the_insert_still_skips_conflicts(monkeypatch):
    status, engine = _run(monkeypatch, [5, 0, 1, 4], "--apply")

    assert status == 0
    [insert] = [q for q in engine.sql if "INSERT INTO" in q]
    assert "ON CONFLICT (episode_hash) DO NOTHING" in insert and "NOT EXISTS" in insert
