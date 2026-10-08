"""Bench-only zarr sampler over the stores as written (one shard per array): same samples as
ZarrDataset, fewer and concurrent reads.

zarr-python 3 re-reads `<key>/zarr.json` on every `group[key]` (no consolidated metadata) and the
whole shard index on every partial read (46 KB per JPEG frame), then fetches inner chunks one by
one. Here each worker opens every array once, keeps each shard index after its first read, and
issues every inner-chunk range of a sample, or of a whole DataLoader batch through
`MultiDataset.__getitems__`, in one asyncio.gather. Decoding stays byte-for-byte zarr's: the
inner codecs are checked against the two layouts ZarrWriter produces (bytes|vlen-bytes, then
optional zstd); anything else goes through stock zarr.
"""

import asyncio

import numcodecs
import numpy as np
from zarr.abc.store import RangeByteRequest, SuffixByteRequest
from zarr.codecs import BytesCodec, ShardingCodec, ZstdCodec
from zarr.codecs.vlen_utf8 import VLenBytesCodec
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync

from egomimic.rldb.zarr.zarr_dataset_multi import (
    LocalEpisodeResolver,
    ZarrDataset,
    ZarrEpisode,
)

_EMPTY = 2**64 - 1
_VLEN = numcodecs.VLenBytes()


class _Layout:
    """How to read one array's single shard without zarr's codec pipeline."""

    __slots__ = ("path", "inner", "n_inner", "dtype", "vlen", "zstd", "index")

    def __init__(self, arr):
        meta = arr.metadata
        (shard,) = meta.codecs
        if not isinstance(shard, ShardingCodec) or tuple(
            meta.chunk_grid.chunk_shape
        ) != tuple(arr.shape):
            raise ValueError("not a single-shard array")
        if shard.index_location.value != "end" or [
            type(c).__name__ for c in shard.index_codecs
        ] != [
            "BytesCodec",
            "Crc32cCodec",
        ]:
            raise ValueError("unexpected shard index codecs")
        first, *rest = shard.codecs
        if not isinstance(first, BytesCodec | VLenBytesCodec) or any(
            not isinstance(c, ZstdCodec) for c in rest
        ):
            raise ValueError("unexpected inner codecs")
        self.vlen = isinstance(first, VLenBytesCodec)
        if (
            not self.vlen
            and first.endian is not None
            and first.endian.value != "little"
        ):
            raise ValueError("big-endian")
        self.zstd = bool(rest)
        self.inner = tuple(shard.chunk_shape)
        self.n_inner = int(np.prod([-(-s // c) for s, c in zip(arr.shape, self.inner)]))
        if len(self.inner) > 1 and self.inner[1:] != tuple(arr.shape[1:]):
            raise ValueError("chunked along a non-leading axis")
        self.dtype = None if self.vlen else arr.dtype
        self.path = arr.store_path / meta.encode_chunk_key((0,) * len(arr.shape))
        self.index = None

    def decode(self, raw: bytes) -> np.ndarray:
        if self.zstd:
            raw = numcodecs.Zstd().decode(raw)
        if self.vlen:
            return _VLEN.decode(raw)
        return np.frombuffer(raw, dtype=self.dtype).reshape(self.inner)


async def _get(path, byte_range):
    buf = await path.get(prototype=default_buffer_prototype(), byte_range=byte_range)
    return buf.to_bytes()


class FastZarrEpisode(ZarrEpisode):
    """ZarrEpisode whose `read_intervals` reads only inner-chunk byte ranges, concurrently.

    Caches live in `__dict__` (ZarrEpisode uses __slots__), so the s3patch pickling, which keeps
    only `_path`, gives each worker fresh ones."""

    def _cache(self) -> dict:
        c = self.__dict__.get("_fz")
        if c is None:
            c = self.__dict__["_fz"] = {"layouts": {}, "prefetched": None}
        return c

    def layout(self, key: str):
        layouts = self._cache()["layouts"]
        if key not in layouts:
            try:
                layouts[key] = _Layout(self._store[key])
            except ValueError:
                layouts[key] = None
        return layouts[key]

    def requests(self, intervals: dict) -> tuple[list, list]:
        """(indexes to load, (key, layout, inner chunk ids)) for `intervals`; keys without a
        fast layout are left out and read by stock zarr."""
        need_index, reads = [], []
        for key, ivs in intervals.items():
            lay = self.layout(key)
            if lay is None:
                continue
            ids = set()
            for lo, hi in self.plan_reads(ivs, lay.inner[0]):
                ids.update(range(lo // lay.inner[0], (hi - 1) // lay.inner[0] + 1))
            if lay.index is None:
                need_index.append(lay)
            reads.append((key, lay, sorted(ids)))
        return need_index, reads

    def build(self, intervals: dict, reads: list, chunks: dict) -> dict:
        """`read_intervals`' result from decoded inner chunks `{(id(layout), i): ndarray}`."""
        fast = {key for key, _, _ in reads}
        result = {}
        for key, lay, _ in reads:
            c = lay.inner[0]
            out = []
            for lo, hi in self.plan_reads(intervals[key], c):
                first, last = lo // c, (hi - 1) // c
                rows = np.concatenate(
                    [chunks[(id(lay), i)] for i in range(first, last + 1)]
                )
                out.append((lo, rows[lo - first * c : hi - first * c]))
            result[key] = out
        rest = {k: v for k, v in intervals.items() if k not in fast}
        if rest:
            result.update(super().read_intervals(rest))
        return result

    def read_intervals(self, intervals: dict) -> dict:
        cache = self._cache()
        pre = cache["prefetched"]
        if pre is not None and pre[0] == intervals:
            cache["prefetched"] = None
            return pre[1]
        return fetch_many([(self, intervals)])[0]


def fetch_many(items: list) -> list:
    """`read_intervals` for many (episode, intervals) at once: shard indexes first, then every
    inner chunk, each phase one gather."""
    plans = [(ep, iv, *ep.requests(iv)) for ep, iv in items]
    indexes = {id(lay): lay for _, _, need, _ in plans for lay in need}.values()
    wanted = {}
    for _, _, _, reads in plans:
        for _, lay, ids in reads:
            for i in ids:
                wanted[(id(lay), i)] = (lay, i)

    async def run():
        if indexes:
            raws = await asyncio.gather(
                *(
                    _get(lay.path, SuffixByteRequest(lay.n_inner * 16 + 4))
                    for lay in indexes
                )
            )
            for lay, raw in zip(indexes, raws):
                lay.index = np.frombuffer(raw[:-4], "<u8").reshape(-1, 2)
        keys = list(wanted)
        ranges = []
        for k in keys:
            lay, i = wanted[k]
            off, n = lay.index[i]
            if off == _EMPTY:
                raise ValueError(
                    "empty inner chunk"
                )  # fill value: let stock zarr handle it
            ranges.append(_get(lay.path, RangeByteRequest(int(off), int(off + n))))
        raws = await asyncio.gather(*ranges)
        return {k: wanted[k][0].decode(r) for k, r in zip(keys, raws)}

    try:
        chunks = sync(run())
    except ValueError:
        return [ZarrEpisode.read_intervals(ep, iv) for ep, iv in items]
    return [ep.build(iv, reads, chunks) for ep, iv, _, reads in plans]


class FastZarrDataset(ZarrDataset):
    """ZarrDataset on FastZarrEpisode, batched through `MultiDataset.__getitems__`.

    `assemble_sample` hands the prefetched rows to the stock `__getitem__`, so decoding,
    padding, masks and the JPEG-failure retry (a different index reads afresh) are unchanged."""

    fetch_split = None

    def init_episode(self):
        self.episode_reader = FastZarrEpisode(self.episode_path)
        self.metadata = self.episode_reader.metadata
        self.total_frames = self.metadata["total_frames"]
        self.embodiment = self.metadata["embodiment"]
        self.keys_dict = {k: (0, None) for k in self.episode_reader._collect_keys()}
        self._image_keys = self._detect_image_keys()
        self._json_keys = self._detect_json_keys()

    def intervals(self, idx: int) -> dict:
        """The per-key intervals `__getitem__` reads for frame `idx`."""
        out = {}
        for zarr_key, window in self._read_windows(idx).values():
            if isinstance(window, np.ndarray):
                spans = [(int(i), int(i) + 1) for i in np.unique(window)]
            else:
                start, end = window
                spans = [(start, start + 1 if end is None else end)]
            out.setdefault(zarr_key, []).extend(spans)
        return out

    def plan_sample(self, i: int) -> dict:
        return {"idx": i, "intervals": self.intervals(i)}

    @staticmethod
    def fetch_planned(items: list) -> list:
        return fetch_many(
            [(leaf.episode_reader, plan["intervals"]) for leaf, plan in items]
        )

    def assemble_sample(self, plan: dict, fetched: dict):
        self.episode_reader._cache()["prefetched"] = (plan["intervals"], fetched)
        try:
            return self[plan["idx"]]
        finally:
            self.episode_reader._cache()["prefetched"] = None


class FastLocalEpisodeResolver(LocalEpisodeResolver):
    _dataset_class = FastZarrDataset


def s3_leaves(root: str, key_map: dict, embodiment: str | None) -> dict:
    """`bench_ego.zarr_s3_leaves` with FastZarrDataset leaves."""
    from egomimic.scripts.format_bench import s3patch

    store = s3patch.obstore_s3(root)
    names = sorted(
        p.rstrip("/").split("/")[-1]
        for p in store.list_with_delimiter()["common_prefixes"]
    )
    leaves = {
        n: FastZarrDataset(f"{root.rstrip('/')}/{n}", key_map, None) for n in names
    }
    if embodiment:
        leaves = {n: d for n, d in leaves.items() if d.embodiment.lower() == embodiment}
    return leaves
