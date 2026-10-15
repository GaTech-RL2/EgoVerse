from types import SimpleNamespace

import pytest

lance = pytest.importorskip("lance")
pytest.importorskip("ray")

from egomimic.scripts.aria_process import aria_to_lance_ray as m  # noqa: E402


@pytest.fixture
def chunks(monkeypatch):
    """{mp4 path: chunk bytes} that the fake chunk_one returns."""
    out = {}
    monkeypatch.setattr(m, "chunk_one", SimpleNamespace(remote=lambda mp4, k: out[mp4]))
    monkeypatch.setattr(m.ray, "get", list)
    return out


def _delivery(rid, n):
    return SimpleNamespace(
        recording_id=rid, recording={"num_frames": n}, videos={"front_1": rid}
    )


def _frames(uri):
    t = lance.dataset(str(uri)).to_table(columns=["recording_id", "num_frames"])
    out = {}
    for r, n in zip(*(c.to_pylist() for c in t.columns)):
        out[r] = out.get(r, 0) + n
    return out


def test_reconvert_replaces_chunk_rows_left_by_a_crashed_run(tmp_path, chunks):
    uri = tmp_path / "video_chunks" / "front_1_k2.lance"
    chunks.update(a=[b"a0", b"a1"], b=[b"b0"])
    m.append_chunks(uri, [_delivery("a", 4), _delivery("b", 2)], "front_1", 2)

    chunks["a"] = [b"a0", b"a1", b"a2"]
    m.append_chunks(uri, [_delivery("a", 6)], "front_1", 2)
    assert _frames(uri) == {"a": 4, "b": 2}

    m.append_chunks(uri, [_delivery("a", 6)], "front_1", 2, redo=True)
    assert _frames(uri) == {"a": 6, "b": 2}
    ids = lance.dataset(str(uri)).to_table(columns=["recording_id"]).column(0)
    assert ids.to_pylist() == ["b", "a", "a", "a"]


def test_chunks_must_cover_the_recording(tmp_path, chunks):
    chunks["a"] = [b"a0"]
    with pytest.raises(ValueError, match="num_frames 4"):
        m.append_chunks(tmp_path / "c.lance", [_delivery("a", 4)], "front_1", 2)
