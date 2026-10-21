"""Frames from an MP4 by byte range: read its sample table once, then fetch only the
packets a frame needs (its keyframe through itself) and decode them in memory.

Built for the delivery-spec encodes (one AV1 video track, GOP 2, `+faststart`), where a
frame and its keyframe are adjacent packets, so one range read serves a frame.
"""

import dataclasses
import io
import struct
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

_CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts"}


def _boxes(buf: bytes, start: int = 0, end: int | None = None):
    """(type, payload start, box end) of each box in buf[start:end]."""
    end = len(buf) if end is None else end
    pos = start
    while pos + 8 <= end:
        size, typ = struct.unpack_from(">I4s", buf, pos)
        head = 8
        if size == 1:
            size = struct.unpack_from(">Q", buf, pos + 8)[0]
            head = 16
        elif size == 0:
            size = end - pos
        yield typ, pos + head, pos + size
        pos += size


def _find(buf: bytes, start: int, end: int, path: list[bytes]):
    for typ, a, b in _boxes(buf, start, end):
        if typ == path[0]:
            return (a, b) if len(path) == 1 else _find(buf, a, b, path[1:])
    return None


def _full(buf: bytes, a: int, fmt: str):
    """Fields after a full box's version/flags."""
    return struct.unpack_from(fmt, buf, a + 4)


@dataclass
class Mp4Index:
    offsets: np.ndarray  # (n,) byte offset of each frame's packet
    sizes: np.ndarray  # (n,) packet size
    keyframe_of: np.ndarray  # (n,) the sync sample each frame decodes from
    header: bytes  # the file up to and including moov, enough to build a decoder

    @property
    def num_frames(self) -> int:
        return len(self.offsets)


def _video_stbl(moov: bytes) -> tuple[int, int]:
    for typ, a, b in _boxes(moov):
        if typ != b"trak":
            continue
        hdlr = _find(moov, a, b, [b"mdia", b"hdlr"])
        if hdlr and moov[hdlr[0] + 8 : hdlr[0] + 12] == b"vide":
            return _find(moov, a, b, [b"mdia", b"minf", b"stbl"])
    raise ValueError("no video track")


def parse_index(header: bytes, moov_at: tuple[int, int]) -> Mp4Index:
    """`moov_at` is the moov payload's half-open byte range in `header`."""
    a, b = moov_at
    moov = header[a:b]
    s0, s1 = _video_stbl(moov)

    def box(name):
        return _find(moov, s0, s1, [name])

    stsz = box(b"stsz")
    fixed, n = _full(moov, stsz[0], ">II")
    sizes = (
        np.full(n, fixed, np.int64)
        if fixed
        else np.frombuffer(moov, ">u4", n, stsz[0] + 12).astype(np.int64)
    )
    if (co := box(b"stco")) is not None:
        (k,) = _full(moov, co[0], ">I")
        chunk_off = np.frombuffer(moov, ">u4", k, co[0] + 8).astype(np.int64)
    else:
        co = box(b"co64")
        (k,) = _full(moov, co[0], ">I")
        chunk_off = np.frombuffer(moov, ">u8", k, co[0] + 8).astype(np.int64)
    stsc = box(b"stsc")
    (m,) = _full(moov, stsc[0], ">I")
    runs = np.frombuffer(moov, ">u4", 3 * m, stsc[0] + 8).reshape(m, 3).astype(np.int64)
    per_chunk = np.empty(k, np.int64)
    for j, (first, spc, _) in enumerate(runs):
        last = runs[j + 1, 0] - 1 if j + 1 < m else k
        per_chunk[first - 1 : last] = spc
    chunk_of = np.repeat(np.arange(k), per_chunk)[:n]
    first_in_chunk = np.r_[0, np.cumsum(per_chunk)[:-1]][chunk_of]
    within = np.cumsum(sizes) - sizes
    offsets = chunk_off[chunk_of] + within - within[first_in_chunk]
    keys = np.arange(n)
    if (ss := box(b"stss")) is not None:
        (c,) = _full(moov, ss[0], ">I")
        keys = np.frombuffer(moov, ">u4", c, ss[0] + 8).astype(np.int64) - 1
    keyframe_of = keys[np.searchsorted(keys, np.arange(n), side="right") - 1]
    return Mp4Index(offsets, sizes, keyframe_of, header[:b])


def read_index(read: Callable[[int, int], bytes], first: int = 1 << 16) -> Mp4Index:
    """The index of an MP4 through `read(start, end)` (half-open byte range). A faststart
    file costs one read (two if its moov exceeds `first` bytes); a moov after the media
    costs two more, and beyond the first read the media is never fetched."""
    buf, base = read(0, first), 0  # buf holds file bytes [base, base + len(buf))
    head = b""  # boxes before the media, kept so the header parses as a file
    pos = 0
    while True:
        if pos + 16 > base + len(buf):
            buf, base = read(pos, pos + max(16, first)), pos
        size, typ = struct.unpack_from(">I4s", buf, pos - base)
        hlen = 8
        if size == 1:
            size, hlen = struct.unpack_from(">Q", buf, pos - base + 8)[0], 16
        if typ == b"moov":
            if pos + size > base + len(buf):
                buf += read(base + len(buf), pos + size)
            moov = buf[pos - base : pos - base + size]
            pre = head if base else buf[: pos - base]
            return parse_index(pre + moov, (len(pre) + hlen, len(pre) + size))
        if typ == b"mdat" and not head:
            head = buf[: pos - base] if base == 0 else b""
        pos += size


def ranges(index: Mp4Index, frames, gap: int = 0) -> list[tuple[int, int]]:
    """Merged half-open byte ranges covering `frames` and the packets they decode from;
    ranges closer than `gap` bytes are joined."""
    frames = np.unique(np.asarray(frames, np.int64))
    lo = index.offsets[index.keyframe_of[frames]]
    hi = index.offsets[frames] + index.sizes[frames]
    order = np.argsort(lo)
    out: list[list[int]] = []
    for a, b in zip(lo[order].tolist(), hi[order].tolist()):
        if out and a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _slice(fetched: dict[tuple[int, int], bytes], start: int, size: int) -> bytes:
    for (a, b), data in fetched.items():
        if a <= start and start + size <= b:
            return data[start - a : start - a + size]
    raise KeyError(f"bytes {start}+{size} not fetched")


class Decoder:
    """An AV1 decoder configured from an index's header; decodes frames from fetched bytes."""

    def __init__(self, index: Mp4Index):
        import av
        import av.logging

        # FFmpeg logs "partial file" for a header without its media; expected here.
        with av.logging.Capture(), av.open(io.BytesIO(index.header)) as c:
            self._name = c.streams.video[0].codec_context.name
            self._extradata = c.streams.video[0].codec_context.extradata
        self.index = dataclasses.replace(index, header=b"")

    def _context(self):
        import av

        ctx = av.CodecContext.create(self._name, "r")
        ctx.extradata = self._extradata
        # One context per frame: dav1d's thread pool costs more to start than it saves.
        ctx.thread_count = 1
        return ctx

    def frames(self, frames, fetched: dict[tuple[int, int], bytes]) -> np.ndarray:
        """(len(frames), H, W, 3) uint8 RGB, in the order asked."""
        import av

        idx = self.index
        want = np.asarray(frames, np.int64)
        got: dict[int, np.ndarray] = {}
        for f in np.unique(want).tolist():
            if f in got:
                continue
            k = int(idx.keyframe_of[f])
            ctx = self._context()
            shown = []
            for p in range(k, f + 1):
                pkt = av.Packet(_slice(fetched, int(idx.offsets[p]), int(idx.sizes[p])))
                shown += ctx.decode(pkt)
            shown += ctx.decode(None)
            for j, fr in enumerate(shown):
                got.setdefault(k + j, fr.to_ndarray(format="rgb24"))
        return np.stack([got[f] for f in want.tolist()])
