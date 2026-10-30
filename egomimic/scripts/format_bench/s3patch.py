"""Points the zarr and Lance loaders at a plain-http S3 endpoint (local MinIO), for the bench only.

Active when FB_S3_ENDPOINT is set. Imported in the parent and preloaded in the forkserver, so
workers inherit the patched modules.

- Lance: `storage_options` gains `allow_http` (plain-http endpoints), reading the same R2_* vars.
- zarr: `ZarrEpisode` opens s3:// paths through obstore, and reopens lazily in each worker rather
  than pickling a live client.
"""

import os
from pathlib import Path

ENDPOINT = os.environ.get("FB_S3_ENDPOINT")


def obstore_s3(uri: str):
    from obstore.store import S3Store

    bucket, prefix = uri[len("s3://") :].split("/", 1)
    return S3Store(
        bucket,
        prefix=prefix.rstrip("/"),
        endpoint=ENDPOINT,
        access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region="us-east-1",
        client_options={"allow_http": True},
    )


def _patch_lance():
    from egomimic.rldb.lance import dataset as m

    orig = m.storage_options

    def storage_options(uri):
        opts = orig(uri)
        if opts is not None:
            opts = {**opts, "allow_http": "true", "region": "us-east-1"}
        return opts

    m.storage_options = storage_options


def _patch_zarr():
    import zarr

    from egomimic.rldb.zarr import zarr_dataset_multi as zdm

    def _init(self, path):
        self._path = path if str(path).startswith("s3://") else Path(path)
        if str(path).startswith("s3://"):
            store = zarr.storage.ObjectStore(obstore_s3(str(path)), read_only=True)
            self._store = zarr.open_group(store=store, mode="r")
        else:
            self._store = zarr.open_group(str(path), mode="r")
        self.metadata = dict(self._store.attrs)
        self.keys = self.metadata["features"]

    def _lazy(self, name):
        # Workers reopen a store on first use, not at unpickle: an eager reopen costs every worker
        # one GET per episode (12 x 10k at startup) for stores the first batch never touches.
        if name in ("_store", "metadata", "keys"):
            _init(self, object.__getattribute__(self, "_path"))
            return object.__getattribute__(self, name)
        raise AttributeError(name)

    zdm.ZarrEpisode.__init__ = _init
    zdm.ZarrEpisode.__getattr__ = _lazy
    zdm.ZarrEpisode.__getstate__ = lambda self: {"_path": self._path}
    zdm.ZarrEpisode.__setstate__ = lambda self, s: setattr(self, "_path", s["_path"])


if ENDPOINT:
    _patch_lance()
    _patch_zarr()

if os.environ.get("FB_NO_GETITEMS"):
    # Ablation: per-sample reads for Lance too, to separate the batched fetch from the format.
    from egomimic.rldb.zarr import zarr_dataset_multi as _zdm

    del _zdm.MultiDataset.__getitems__
