"""Flask `/thumbnail/<hash>/<frame>` route + in-process LRU.

Decoded JPEG bytes are cached by (video_hash, frame_idx). Backed by
`load_thumbnail_jpeg` which itself uses the zarr-group LRU in images.py.
"""

from __future__ import annotations

import logging
from collections import OrderedDict

from .images import (
    load_thumbnail_jpeg,
)

logger = logging.getLogger(__name__)


class ThumbnailService:
    def __init__(self, zarr_root: str, image_key: str, cache_max: int = 1024):
        self.zarr_root = zarr_root
        self.image_key = image_key
        self.cache_max = cache_max
        self._cache: "OrderedDict[tuple[str, int], bytes]" = OrderedDict()

    def get(self, video_hash: str, frame_idx: int) -> bytes | None:
        key = (video_hash, frame_idx)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        data_bytes = load_thumbnail_jpeg(
            self.zarr_root, video_hash, frame_idx, self.image_key
        )
        if data_bytes is None:
            return None
        self._cache[key] = data_bytes
        while len(self._cache) > self.cache_max:
            self._cache.popitem(last=False)
        return data_bytes

    def register(self, app):
        """Wire `/thumbnail/<video_hash>/<frame_idx>` on `app.server`.

        Uses a string converter (not `<int:>`) so the sentinel `frame_idx=-1`
        — written by eval_latent when the source dataset didn't emit
        `frame_index` — reaches `load_thumbnail_jpeg`'s frame-0 fallback
        instead of 404'ing on the Werkzeug converter."""

        @app.server.route("/thumbnail/<video_hash>/<frame_idx>")
        def _thumbnail_route(video_hash: str, frame_idx: str):
            from flask import Response, abort

            try:
                frame_idx_int = int(frame_idx)
            except ValueError:
                return abort(404)
            data_bytes = self.get(video_hash, frame_idx_int)
            if data_bytes is None:
                return abort(404)
            return Response(
                data_bytes,
                mimetype="image/jpeg",
                headers={"Cache-Control": "public, max-age=86400"},
            )
