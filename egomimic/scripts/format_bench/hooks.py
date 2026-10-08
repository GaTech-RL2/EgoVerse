"""Decoder-cache hit counters, installed inside each DataLoader worker. Module-level functions so
they pickle by reference under forkserver and spawn. FB_DECODER_CACHE=N shrinks Lance's per-worker
LRU (LeRobot reads LEROBOT_VIDEO_DECODER_CACHE_SIZE itself)."""

import os


def lance_decoders(counters, wid):
    from egomimic.rldb.lance import dataset as m

    if os.environ.get("FB_DECODER_CACHE"):
        m._MAX_DECODERS = int(os.environ["FB_DECODER_CACHE"])

    orig = m._decoder

    def counted(path):
        counters[2 * wid + (0 if path in m._DECODERS else 1)] += 1
        return orig(path)

    m._decoder = counted


def lerobot_stream_decoders(counters, wid):
    # StreamingLeRobotDataset builds its own VideoDecoderCache per iterator, so count at the class.
    from lerobot.datasets.video_utils import VideoDecoderCache

    orig = VideoDecoderCache.get_decoder

    def counted(self, path):
        counters[2 * wid + (0 if path in self else 1)] += 1
        return orig(self, path)

    VideoDecoderCache.get_decoder = counted


def lerobot_decoders(counters, wid):
    from lerobot.datasets import video_utils as m

    cache = m._default_decoder_cache
    orig = cache.get_decoder

    def counted(path):
        counters[2 * wid + (0 if path in cache else 1)] += 1
        return orig(path)

    cache.get_decoder = counted
