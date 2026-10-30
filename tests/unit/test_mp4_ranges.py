import numpy as np
import pytest

av = pytest.importorskip("av")

from fixtures.videos import write_av1  # noqa: E402

from egomimic.rldb.lance import mp4_ranges  # noqa: E402

N = 12


def _reference(path):
    with av.open(str(path)) as c:
        packets = [(p.pos, p.size, p.is_keyframe) for p in c.demux(video=0) if p.size]
    with av.open(str(path)) as c:
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    return packets, np.stack(frames)


@pytest.fixture(params=[True, False], ids=["faststart", "moov_at_end"])
def mp4(tmp_path, request):
    try:
        write_av1(tmp_path / "v.mp4", N, faststart=request.param)
    except (av.FFmpegError, ValueError) as e:
        pytest.skip(f"no AV1 encoder: {e}")
    return tmp_path / "v.mp4"


def test_index_matches_the_demuxer_and_reads_no_media(mp4):
    data = mp4.read_bytes()
    reads = []
    index = mp4_ranges.read_index(
        lambda a, b: reads.append((a, b)) or data[a:b], first=64
    )
    packets, _ = _reference(mp4)
    assert index.num_frames == N
    assert index.offsets.tolist() == [p[0] for p in packets]
    assert index.sizes.tolist() == [p[1] for p in packets]
    keys = [i for i, p in enumerate(packets) if p[2]]
    assert index.keyframe_of.tolist() == [
        max(k for k in keys if k <= i) for i in range(N)
    ]
    media = (min(index.offsets), max(index.offsets + index.sizes))
    # Past the first read, only box headers and the moov are fetched.
    assert not any(a < media[1] and b > media[0] + 16 for a, b in reads[1:])


def test_frames_decode_from_fetched_ranges_only(mp4):
    data = mp4.read_bytes()
    index = mp4_ranges.read_index(lambda a, b: data[a:b])
    _, want = _reference(mp4)
    frames = [7, 0, 3, 3, N - 1]
    rg = mp4_ranges.ranges(index, frames)
    assert rg == sorted(rg) and all(b <= a2 for (_, b), (a2, _) in zip(rg, rg[1:]))
    fetched = {r: data[r[0] : r[1]] for r in rg}
    got = mp4_ranges.Decoder(index).frames(frames, fetched)
    np.testing.assert_array_equal(got, want[frames])
    with pytest.raises(KeyError):
        mp4_ranges.Decoder(index).frames([5], fetched)


@pytest.mark.parametrize(
    "data",
    [
        b"\x00\x00\x00\x10ftypisom\x00\x00\x00\x00"
        + b"\x00\x00\x00\x00mdat"
        + b"x" * 64,
        b"\x00\x00\x00\x10ftypisom\x00\x00\x00\x00",
    ],
    ids=["mdat_to_eof", "no_moov"],
)
def test_index_of_a_file_without_moov_raises(data):
    with pytest.raises(ValueError, match="moov"):
        mp4_ranges.read_index(lambda a, b: data[a:b], first=16)
