"""
State and action embedders for DemInf curation.

Both embedders share a fit/embed interface:
    embedder.fit(episodes)          — compute normalisation stats / train
    embedder.embed(data) -> ndarray — map raw data to latent vectors

Embedders must be fit on the *entire* dataset before any MI computation,
because KSG is a global estimator operating on the joint distribution.

Cross-embodiment support
------------------------
When ``cross_embodiment_mode="independent"`` (the default), each embodiment
gets its own normalisation statistics and projection matrix.  The output
dimensionality is always ``latent_dim`` regardless of the input dimension,
so per-embodiment embeddings can be concatenated safely for downstream use,
but MI is scored *within* each embodiment group to avoid comparing
fundamentally different action modalities (e.g., hand keypoints vs. robot
joint positions).

When ``cross_embodiment_mode="shared"`` all episodes are embedded using a
single global statistics set — only makes sense when embodiments have been
retargeted to a common action/state space.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import torch
import torch.nn as nn

from egomimic.curation.utils import Episode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _fit_gaussian_stats(
    arrays: list[np.ndarray],
    min_std: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute global mean and std from a list of (T, D) arrays."""
    stacked = np.concatenate(arrays, axis=0)  # (N_total, D)
    mean = stacked.mean(axis=0)
    std = np.maximum(stacked.std(axis=0), min_std)
    return mean, std


def _build_random_projection(
    in_dim: int,
    out_dim: int,
    seed: int = 42,
) -> np.ndarray | None:
    """
    Build a random orthogonal projection matrix (in_dim → out_dim).

    Returns None when in_dim <= out_dim (identity / no projection needed).
    The matrix is computed via QR decomposition so columns are orthonormal,
    preserving MI up to the Johnson-Lindenstrauss guarantee.
    """
    if in_dim <= out_dim:
        return None
    rng = np.random.default_rng(seed=seed)
    G = rng.standard_normal((in_dim, out_dim))
    Q, _ = np.linalg.qr(G)
    return Q[:, :out_dim].astype(np.float32)


# ---------------------------------------------------------------------------
# StateEmbedder
# ---------------------------------------------------------------------------


class StateEmbedder:
    """
    Embed proprioceptive states or egocentric images to a fixed latent dim.

    Args:
        mode: "proprioceptive" or "image".
        latent_dim: Output dimensionality (default 32).
        device: Torch device for image mode.
        cross_embodiment_mode: "independent" (separate stats per embodiment)
            or "shared" (single global stats).  Only affects proprioceptive mode.
        image_batch_size: Batch size for GPU inference in image mode.
    """

    def __init__(
        self,
        mode: Literal["proprioceptive", "image"] = "proprioceptive",
        latent_dim: int = 32,
        device: str | torch.device = "cpu",
        cross_embodiment_mode: Literal["independent", "shared"] = "independent",
        image_batch_size: int = 256,
    ) -> None:
        self.mode = mode
        self.latent_dim = latent_dim
        self.device = torch.device(device)
        self.cross_embodiment_mode = cross_embodiment_mode
        self.image_batch_size = image_batch_size
        self._fitted = False

        # Proprioceptive — shared stats (always populated, used as fallback)
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._proj: np.ndarray | None = None

        # Per-embodiment stats (populated when mode="independent")
        self._per_embodiment_stats: dict[str, dict] = {}

        # Image
        self._backbone: nn.Module | None = None
        self._proj_layer: nn.Linear | None = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, episodes: list[Episode]) -> None:
        """Compute normalisation stats (and build projection) from all episodes."""
        if self.mode == "proprioceptive":
            self._fit_proprioceptive(episodes)
        else:
            self._fit_image(episodes)
        self._fitted = True

    def _fit_proprioceptive(self, episodes: list[Episode]) -> None:
        if self.cross_embodiment_mode == "independent":
            self._fit_proprioceptive_independent(episodes)
        else:
            self._fit_proprioceptive_shared(episodes)

    def _fit_proprioceptive_shared(self, episodes: list[Episode]) -> None:
        obs_list = [ep.observations for ep in episodes]
        self._mean, self._std = _fit_gaussian_stats(obs_list)
        obs_dim = self._mean.shape[0]
        self._proj = _build_random_projection(obs_dim, self.latent_dim)
        logger.info(
            "StateEmbedder (proprio/shared): obs_dim=%d → latent_dim=%d",
            obs_dim,
            min(obs_dim, self.latent_dim),
        )

    def _fit_proprioceptive_independent(self, episodes: list[Episode]) -> None:
        by_embodiment: dict[str, list[np.ndarray]] = {}
        for ep in episodes:
            by_embodiment.setdefault(ep.embodiment, []).append(ep.observations)

        self._per_embodiment_stats = {}
        for embodiment, obs_list in by_embodiment.items():
            mean, std = _fit_gaussian_stats(obs_list)
            proj = _build_random_projection(mean.shape[0], self.latent_dim)
            self._per_embodiment_stats[embodiment] = {
                "mean": mean,
                "std": std,
                "proj": proj,
            }
            logger.info(
                "StateEmbedder (proprio/independent) '%s': obs_dim=%d → latent_dim=%d",
                embodiment,
                mean.shape[0],
                min(mean.shape[0], self.latent_dim),
            )

        # Shared stats = first embodiment (fallback for unknown embodiments)
        first = next(iter(self._per_embodiment_stats.values()))
        self._mean, self._std, self._proj = (
            first["mean"],
            first["std"],
            first["proj"],
        )

    def _fit_image(self, episodes: list[Episode]) -> None:  # noqa: ARG002
        """Build frozen ResNet-18 + fixed random projection (Xavier init)."""
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except ImportError as exc:
            raise ImportError("torchvision is required for image mode") from exc

        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self._backbone = nn.Sequential(*list(backbone.children())[:-1])
        for p in self._backbone.parameters():
            p.requires_grad_(False)
        self._backbone = self._backbone.to(self.device).eval()

        # Fixed random projection 512 → latent_dim.
        # We use a trainable Linear but initialise with Xavier and freeze it.
        # A fixed random projection is sufficient for MI estimation — we only
        # need a reasonable low-dimensional embedding, not a supervised mapping.
        self._proj_layer = nn.Linear(512, self.latent_dim, bias=True).to(self.device)
        nn.init.xavier_uniform_(self._proj_layer.weight)
        nn.init.zeros_(self._proj_layer.bias)
        for p in self._proj_layer.parameters():
            p.requires_grad_(False)

        logger.info(
            "StateEmbedder (image): ResNet-18 → %d (fixed random proj)", self.latent_dim
        )

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed(self, data: np.ndarray, embodiment: str | None = None) -> np.ndarray:
        """
        Embed a batch of observations.

        Args:
            data: (N, obs_dim) for proprioceptive or (N, C, H, W) for image.
            embodiment: Embodiment name — used in independent mode to select
                per-embodiment normalisation statistics.  Ignored in shared mode.

        Returns:
            (N, latent_dim) float32 numpy array.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before embed()")

        if self.mode == "proprioceptive":
            return self._embed_proprioceptive(data, embodiment)
        return self._embed_image(data)

    def _embed_proprioceptive(
        self, data: np.ndarray, embodiment: str | None
    ) -> np.ndarray:
        if (
            self.cross_embodiment_mode == "independent"
            and embodiment is not None
            and embodiment in self._per_embodiment_stats
        ):
            stats = self._per_embodiment_stats[embodiment]
            mean, std, proj = stats["mean"], stats["std"], stats["proj"]
        else:
            mean, std, proj = self._mean, self._std, self._proj

        data = data.astype(np.float32)
        normalised = (data - mean) / std
        if proj is not None:
            return normalised @ proj
        return normalised

    def _embed_image(self, data: np.ndarray) -> np.ndarray:
        """
        Batched ResNet-18 inference with ImageNet preprocessing.

        Expects uint8 (N, C, H, W) in [0, 255] or float32 in [0, 1].
        Applies center-crop after resize-to-224 to handle non-square Aria images.
        """
        try:
            import torchvision.transforms.functional as TF
        except ImportError as exc:
            raise ImportError("torchvision is required for image mode") from exc

        all_outputs: list[np.ndarray] = []
        N = data.shape[0]

        for start in range(0, N, self.image_batch_size):
            chunk = data[start : start + self.image_batch_size]

            # Convert to float32 [0, 1]
            if chunk.dtype == np.uint8:
                tensor = torch.from_numpy(chunk).float() / 255.0
            else:
                tensor = torch.from_numpy(chunk.astype(np.float32))

            # Resize shortest side to 224, then centre-crop 224×224
            # Using functional API to avoid the Compose/PIL overhead
            tensor = TF.resize(tensor, [224], antialias=True)
            tensor = TF.center_crop(tensor, [224, 224])

            # ImageNet normalisation
            tensor = TF.normalize(
                tensor,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )

            tensor = tensor.to(self.device)
            with torch.no_grad():
                feats = self._backbone(tensor)  # (B, 512, 1, 1)
                feats = feats.flatten(1)  # (B, 512)
                out = self._proj_layer(feats)  # (B, latent_dim)

            all_outputs.append(out.cpu().numpy())

        return np.concatenate(all_outputs, axis=0)

    def embed_batch(self, images: np.ndarray) -> np.ndarray:
        """
        Embed a batch of images efficiently (image mode only).

        Args:
            images: (N, C, H, W) numpy array in [0, 255] uint8 or [0,1] float32.

        Returns:
            (N, latent_dim) float32 numpy array.
        """
        if self.mode != "image":
            raise RuntimeError("embed_batch() is only available in image mode")
        return self._embed_image(images)

    def embed_episodes(self, episodes: list[Episode]) -> tuple[np.ndarray, list[int]]:
        """
        Embed all timesteps from a list of episodes in one pass.

        In independent mode, each episode's observations are normalised using
        its embodiment's statistics, then results are concatenated globally.

        Returns:
            embeddings: (N_total, latent_dim)
            lengths: list of per-episode timestep counts (for grouping back)
        """
        if self.mode == "image":
            # Image mode: concatenate all frames, embed in batches
            all_obs = [ep.observations for ep in episodes]
            lengths = [len(o) for o in all_obs]
            concatenated = np.concatenate(all_obs, axis=0)
            embeddings = self.embed(concatenated)
            return embeddings, lengths

        # Proprioceptive: process per-episode so independent mode can route stats
        all_emb: list[np.ndarray] = []
        lengths: list[int] = []
        for ep in episodes:
            emb = self.embed(ep.observations, embodiment=ep.embodiment)
            all_emb.append(emb)
            lengths.append(len(ep.observations))
        embeddings = np.concatenate(all_emb, axis=0)
        return embeddings, lengths


# ---------------------------------------------------------------------------
# ActionEmbedder
# ---------------------------------------------------------------------------


class ActionEmbedder:
    """
    Embed actions: Gaussian normalisation → random orthogonal linear projection.

    A random orthogonal projection preserves MI (it is a bijective linear map
    when action_dim ≤ latent_dim, and an MI-preserving dimensionality reduction
    via Johnson-Lindenstrauss when action_dim > latent_dim).  This is preferred
    over a random MLP because nonlinear random networks destroy linear
    state-action correlations — the primary signal DemInf relies on.

    Cross-embodiment support
    ~~~~~~~~~~~~~~~~~~~~~~~~
    When ``cross_embodiment_mode="independent"`` (default), each embodiment
    gets its own mean, std, and projection matrix.  This handles heterogeneous
    action spaces (e.g., Aria 42-D hand keypoints vs. ALOHA 14-D joint
    positions) without mixing statistics across fundamentally different
    action modalities.

    When ``cross_embodiment_mode="shared"`` a single global statistics set is
    used — appropriate only when all embodiments share the same action space
    (e.g., retargeted robot data).

    Args:
        latent_dim: Output dimensionality (default 32).
        cross_embodiment_mode: "independent" or "shared".
        device: Unused; retained for API compatibility with image embedders.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 64,  # kept for API compatibility, not used
        device: str | torch.device = "cpu",
        cross_embodiment_mode: Literal["independent", "shared"] = "independent",
    ) -> None:
        self.latent_dim = latent_dim
        self.device = device
        self.cross_embodiment_mode = cross_embodiment_mode
        self._fitted = False

        # Shared / fallback stats
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._proj: np.ndarray | None = None

        # Per-embodiment stats (populated in independent mode)
        self._per_embodiment_stats: dict[str, dict] = {}

    def fit(self, episodes: list[Episode]) -> None:
        """Compute normalisation stats and random orthogonal projection."""
        if self.cross_embodiment_mode == "independent":
            self._fit_independent(episodes)
        else:
            self._fit_shared(episodes)
        self._fitted = True

    def _fit_shared(self, episodes: list[Episode]) -> None:
        act_list = [ep.actions for ep in episodes]
        self._mean, self._std = _fit_gaussian_stats(act_list)
        action_dim = self._mean.shape[0]
        self._proj = _build_random_projection(action_dim, self.latent_dim)
        logger.info(
            "ActionEmbedder (shared): action_dim=%d → latent_dim=%d (proj=%s)",
            action_dim,
            min(action_dim, self.latent_dim),
            self._proj is not None,
        )

    def _fit_independent(self, episodes: list[Episode]) -> None:
        by_embodiment: dict[str, list[np.ndarray]] = {}
        for ep in episodes:
            by_embodiment.setdefault(ep.embodiment, []).append(ep.actions)

        self._per_embodiment_stats = {}
        for embodiment, act_list in by_embodiment.items():
            mean, std = _fit_gaussian_stats(act_list)
            proj = _build_random_projection(mean.shape[0], self.latent_dim)
            self._per_embodiment_stats[embodiment] = {
                "mean": mean,
                "std": std,
                "proj": proj,
            }
            logger.info(
                "ActionEmbedder (independent) '%s': action_dim=%d → latent_dim=%d",
                embodiment,
                mean.shape[0],
                min(mean.shape[0], self.latent_dim),
            )

        # Shared fallback = first embodiment
        first = next(iter(self._per_embodiment_stats.values()))
        self._mean, self._std, self._proj = (
            first["mean"],
            first["std"],
            first["proj"],
        )

    def embed(self, data: np.ndarray, embodiment: str | None = None) -> np.ndarray:
        """
        Embed a batch of actions.

        Args:
            data: (N, action_dim) float32 array.
            embodiment: Embodiment name used to select per-embodiment statistics
                in independent mode.

        Returns:
            (N, latent_dim) float32 numpy array.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before embed()")

        if (
            self.cross_embodiment_mode == "independent"
            and embodiment is not None
            and embodiment in self._per_embodiment_stats
        ):
            stats = self._per_embodiment_stats[embodiment]
            mean, std, proj = stats["mean"], stats["std"], stats["proj"]
        else:
            if self.cross_embodiment_mode == "independent" and embodiment is not None:
                logger.warning(
                    "Unknown embodiment '%s' — falling back to global stats", embodiment
                )
            mean, std, proj = self._mean, self._std, self._proj

        data = data.astype(np.float32)
        normalised = (data - mean) / std
        if proj is not None:
            return normalised @ proj
        return normalised

    def embed_episodes(self, episodes: list[Episode]) -> tuple[np.ndarray, list[int]]:
        """
        Embed all timesteps from a list of episodes.

        In independent mode, each episode is routed through its embodiment's
        statistics before concatenation.

        Returns:
            embeddings: (N_total, latent_dim)
            lengths: list of per-episode timestep counts
        """
        all_emb: list[np.ndarray] = []
        lengths: list[int] = []
        for ep in episodes:
            emb = self.embed(ep.actions, embodiment=ep.embodiment)
            all_emb.append(emb)
            lengths.append(len(ep.actions))
        return np.concatenate(all_emb, axis=0), lengths


# ---------------------------------------------------------------------------
# ImageStateEmbedder (convenience wrapper for pure image pipelines)
# ---------------------------------------------------------------------------


class ImageStateEmbedder:
    """
    Convenience wrapper: frozen ResNet-18 + fixed Xavier-initialised projection.

    Equivalent to ``StateEmbedder(mode="image", ...)``, but exposed as a
    dedicated class so it can be imported/configured explicitly.

    The projection layer (512 → latent_dim) uses Xavier initialisation and is
    *not* trained.  A fixed random projection is sufficient for the KSG
    estimator — we only need a low-dimensional embedding that preserves
    local geometry, not a supervised representation.

    Args:
        latent_dim: Output dimensionality.
        device: Device for GPU inference.
        batch_size: Number of frames per GPU batch.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        device: str = "cuda",
        batch_size: int = 256,
    ) -> None:
        self._embedder = StateEmbedder(
            mode="image",
            latent_dim=latent_dim,
            device=device,
            image_batch_size=batch_size,
        )

    def fit(self, episodes: list[Episode]) -> None:
        self._embedder.fit(episodes)

    @torch.no_grad()
    def embed_batch(self, images: np.ndarray) -> np.ndarray:
        """
        Embed a batch of images.

        Args:
            images: (N, C, H, W) uint8 numpy array in [0, 255].

        Returns:
            (N, latent_dim) float32 numpy array.
        """
        return self._embedder.embed_batch(images)

    def embed_episodes(self, episodes: list[Episode]) -> tuple[np.ndarray, list[int]]:
        return self._embedder.embed_episodes(episodes)
