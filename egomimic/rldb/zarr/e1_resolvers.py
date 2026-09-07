"""E1 helpers around the folder resolver.

``LocalFolderEpisodeResolverWithEmbodimentOverride`` — the ABC ("fold and stack
the skirts") zarrs still carry ``attrs.embodiment == "eva_bimanual"`` from before
the 2026-09-01 eva→yam relabel in the episode table, but the lab trains them as
``yam_bimanual`` (their own transforms, no Eva extrinsics). The leaf dataset takes
the embodiment from the zarr attrs, so without this override every sample would
be routed to the eva domain and normalised with the wrong stats.
"""

from __future__ import annotations

from egomimic.rldb.zarr.zarr_dataset_multi import LocalFolderEpisodeResolver, S3EpisodeResolver


class LocalFolderEpisodeResolverWithEmbodimentOverride(LocalFolderEpisodeResolver):
    def __init__(self, *args, embodiment_override: str | None = None, image_hw=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.embodiment_override = embodiment_override
        # The folder resolver drops the base class's image_hw; ABC episodes mix 640x480 with
        # 1280x720 front images and a batch cannot mix sizes, so resize like the lab's S3 resolver.
        self.image_hw = tuple(image_hw) if image_hw else None

    def resolve(self, *args, **kwargs):
        datasets = super().resolve(*args, **kwargs)
        if self.embodiment_override is not None:
            for ds in datasets.values():
                ds.embodiment = self.embodiment_override
        return datasets


class S3EpisodeResolverWithEmbodimentOverride(S3EpisodeResolver):
    """The lab's SQL-driven resolver with the same override. On the Phoenix mirror the ABC
    zarrs still say ``attrs.embodiment == "eva_bimanual"`` (the Skynet copies the lab trains
    from were converted after the eva→yam relabel), so without this the leaves register as
    embodiment 6 while the model's domain is yam_bimanual (7): norm stats land under 6 with no
    keys and ``HPT.ac_keys[7]`` is never set ("Missing key 7" at the first validation batch)."""

    def __init__(self, *args, embodiment_override: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.embodiment_override = embodiment_override

    def resolve(self, *args, **kwargs):
        datasets = super().resolve(*args, **kwargs)
        if self.embodiment_override is not None:
            for ds in datasets.values():
                ds.embodiment = self.embodiment_override
        return datasets
