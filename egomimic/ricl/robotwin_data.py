"""RoboTwin shim data layer for RICL training (joint-space).

Feeds RoboTwin 2.0 bimanual expert demos through EgoVerse's real RICL machinery
(``build_ricl_collate`` -> ``PIRicl`` prefix conditioning -> pi0.5) via the same
"thin shim, not a port" pattern proven on DROID (see ``droid_data.py`` /
``DROID_VERIFICATION.md``). RoboTwin is natively **joint-space** — 16-D qpos
``[left_arm(7), left_gripper(1), right_arm(7), right_gripper(1)]`` — so, exactly
like DROID's 8-D, the raw state/action are quantile-normalized to [-1, 1] and
slot-filled into pi0.5's shared 32-D space (dims 0..15; grippers land at slots
7 and 15). No Zarr/SQL/embodiment port; the Zarr converter
(``scripts/robotwin_to_zarr.py``) is the *separate* reusable-corpus path.

What this module provides (mirrors ``droid_data.py``)
-----------------------------------------------------
- ``RoboTwinCorpus``        : discover task dirs (``<task>/data/episode*.hdf5`` +
  ``<task>/instructions/episode*.json``) and lazily read per-array HDF5 (LRU).
- ``RoboTwinQueryDataset``  : per-frame torch ``Dataset`` yielding the post-transform
  keys the RICL query path consumes (``base_0_rgb`` etc., a 32-D slot-filled state,
  a (horizon,32) action target, ``episode_hash``/``frame_idx``/``annotations``),
  quantile-normalized to [-1, 1].
- ``make_robotwin_bank_provider`` : a ``BankFrameProvider`` returning a neighbor's
  (head image, normalized 16-D state, normalized (horizon,16) action chunk).
- ``build_robotwin_retrieval_cache`` / ``build_random_neighbor_cache`` : within-task
  leave-one-out retrieval cache (and random controls) over a per-frame embedding
  provider, via the same ``RetrievalCache`` the Zarr path uses.
- ``make_dinov2_embedding_provider`` / ``make_fake_embedding_provider`` : per-frame
  ``(T, D)`` embedding providers for cache building (real DINOv2 vs a deterministic
  stub for fast smoke checks / unit tests that don't need the encoder).
- ``RoboTwinNormStats`` / ``Passthrough32`` : the minimal norm-stats surface PI's
  constructor + ``process_batch_for_training`` touch, and a no-op 16->32 converter.

Quantile stats are computed from the corpus (RoboTwin ships none) and may be shared
across a train/eval split via ``quantiles=`` so both sides normalize identically.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from collections import OrderedDict
from functools import lru_cache
from typing import Callable

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from egomimic.ricl.retrieval import RetrievalCache
from egomimic.utils.action_utils import BaseActionConverter, _pad32

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
# RoboTwin's qpos dim is embodiment-dependent: aloha-agilex has 6-DOF arms ->
# 14-D state (6+1+6+1, grippers at 6 and 13); a 7-DOF arm gives 16-D (7/15). The
# corpus DETECTS its own ``state_dim`` + ``gripper_slots`` from the data; these are
# only fallback defaults (and the synthetic fixture's default 6-DOF layout).
RAW_DIM = 14  # default qpos dim (aloha-agilex, the bimanual RoboTwin embodiment)
SHARED_DIM = 32  # pi0.5 shared action/state space
GRIPPER_SLOTS = (6, 13)  # default left/right gripper indices (6-DOF arms)

EMB_ID = 8  # eva_bimanual masquerade (matches DROID's choice; no new embodiment)
EMB_NAME = "eva_bimanual"
AC_KEY = "actions_joint"
STATE_KEY = "observations.state.ee_pose"  # 32-D slot-fill -> model continuous proprio
PROMPT_STATE_KEY = "state_raw"  # normalized raw qpos -> compact text State block
CAM_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
# query cameras map to RoboTwin's three views (head exterior is the retrieval view)
_CAM_TO_ROBOTWIN = {
    "base_0_rgb": "head_camera",
    "left_wrist_0_rgb": "left_camera",
    "right_wrist_0_rgb": "right_camera",
}
IMAGE_HW = (224, 224)  # query/bank images resized to pi0.5's resolution


def slot32(x: np.ndarray) -> np.ndarray:
    """Place a D-dim vector into dims 0..D-1 of a 32-D zero vector (last axis),
    where ``D = x.shape[-1]`` (RoboTwin's per-embodiment qpos dim — 14 for the
    6-DOF aloha-agilex arms, 16 for 7-DOF). ``D`` must be <= 32."""
    x = np.asarray(x, dtype=np.float32)
    d = x.shape[-1]
    if d > SHARED_DIM:
        raise ValueError(f"state/action dim {d} exceeds shared dim {SHARED_DIM}")
    out = np.zeros(x.shape[:-1] + (SHARED_DIM,), dtype=np.float32)
    out[..., :d] = x
    return out


# --------------------------------------------------------------------------- #
# Corpus
# --------------------------------------------------------------------------- #
def _discover_groups(root: str) -> "OrderedDict[str, list[str]]":
    """Map group_id -> sorted episode hdf5 paths.

    A *group* is any dir that contains ``data/episode*.hdf5`` (a RoboTwin
    task/setting), keyed by its path relative to ``root`` (``/`` -> ``__``). This
    handles both a single task dir and a parent dir of many tasks.
    """
    groups: "OrderedDict[str, list[str]]" = OrderedDict()
    for data_dir in sorted(glob.glob(os.path.join(root, "**", "data"), recursive=True)):
        if not os.path.isdir(data_dir):
            continue
        eps = sorted(glob.glob(os.path.join(data_dir, "episode*.hdf5")))
        if not eps:
            continue
        rel = os.path.relpath(os.path.dirname(data_dir), root)
        gid = "root" if rel == "." else rel.replace(os.sep, "__")
        groups[gid] = eps
    # also allow root itself being a task dir (root/data/episode*.hdf5)
    return groups


def _episode_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]  # "episode3"


class RoboTwinCorpus:
    """Discover + lazily load RoboTwin episode HDF5s, grouped by task.

    Episode hash = ``"{group_id}__{episode_name}"`` (filesystem-safe, unique
    across groups). Per-(hash, key) arrays are LRU-cached; images are decoded on
    demand. Quantile stats (q01/q99 over the qpos dims, for state + action) are
    computed once from the corpus unless an explicit ``quantiles`` dict is given
    (pass the train corpus's to an eval corpus so both normalize identically).
    """

    def __init__(
        self,
        root: str,
        groups: list[str] | None = None,
        cache_size: int = 4096,
        image_hw: tuple[int, int] = IMAGE_HW,
        quantiles: dict | None = None,
    ):
        self.root = root
        self.image_hw = image_hw
        self.hash_to_path: "OrderedDict[str, str]" = OrderedDict()
        self.group_to_hashes: "OrderedDict[str, list[str]]" = OrderedDict()
        discovered = _discover_groups(root)
        if groups is not None:
            discovered = OrderedDict(
                (g, hs) for g, hs in discovered.items() if g in set(groups)
            )
        for g, eps in discovered.items():
            hs = []
            for ep in eps:
                h = f"{g}__{_episode_name(ep)}"
                self.hash_to_path[h] = ep
                hs.append(h)
            if hs:
                self.group_to_hashes[g] = hs
        if not self.hash_to_path:
            raise FileNotFoundError(
                f"no <task>/data/episode*.hdf5 found under {root!r}"
            )
        self._arr = lru_cache(maxsize=cache_size)(self._load_arr)
        self._actions_cache = lru_cache(maxsize=cache_size)(self._compute_actions)
        # Detect the embodiment's qpos layout from the first episode: RoboTwin arm
        # DOF varies (aloha-agilex 6-DOF -> 14-D state, grippers at 6/13; 7-DOF ->
        # 16-D, 7/15). vector = [l_arm, l_grip, r_arm, r_grip].
        h0 = next(iter(self.hash_to_path))
        la = int(self._arr(h0, "joint_action/left_arm").shape[-1])
        ra = int(self._arr(h0, "joint_action/right_arm").shape[-1])
        self.state_dim = int(self._arr(h0, "joint_action/vector").shape[-1])
        self.gripper_slots = (la, la + 1 + ra)
        self.quantiles = quantiles or self._compute_quantiles()

    # ---- raw HDF5 access ----
    def _load_arr(self, h: str, key: str) -> np.ndarray:
        """Read one HDF5 dataset (e.g. ``joint_action/vector``) for an episode."""
        with h5py.File(self.hash_to_path[h], "r") as f:
            return f[key][()]

    @property
    def hashes(self) -> list[str]:
        return list(self.hash_to_path.keys())

    def group_of(self, h: str) -> str:
        return h.split("__", 1)[0]

    def num_frames(self, h: str) -> int:
        return int(self._arr(h, "joint_action/vector").shape[0])

    def state(self, h: str, fi: int) -> np.ndarray:
        """Raw 16-D qpos at frame ``fi`` (``joint_action/vector``)."""
        return np.asarray(self._arr(h, "joint_action/vector")[fi], dtype=np.float32)

    def _compute_actions(self, h: str) -> np.ndarray:
        """Action array (T, 16): action[t] = next state (qpos), last repeated —
        RoboTwin's ``action = next state`` convention (process_data.py)."""
        v = np.asarray(self._arr(h, "joint_action/vector"), dtype=np.float32)
        nxt = np.concatenate([v[1:], v[-1:]], axis=0)
        return nxt

    def action_chunk(self, h: str, fi: int, horizon: int) -> np.ndarray:
        """Raw (horizon, 16) action chunk starting at ``fi`` (repeat-pad the tail)."""
        acts = self._actions_cache(h)
        T = acts.shape[0]
        idx = np.clip(np.arange(fi, fi + horizon), 0, T - 1)
        return acts[idx]

    def image(self, h: str, fi: int, cam: str) -> np.ndarray:
        """Decode the JPEG for ``cam`` at frame ``fi`` -> (H, W, 3) RGB uint8.

        RoboTwin stores cv2-encoded JPEG bytes; cv2 encode/decode round-trips the
        array byte-for-byte regardless of channel-order interpretation, so the
        renderer's RGB frame decodes back as RGB — no BGR swap needed. Trailing
        ``\\x00`` padding (fixed-length ``S{maxlen}`` storage) is ignored by the
        JPEG decoder (stops at the EOI marker)."""
        raw = self._arr(h, f"observation/{_CAM_TO_ROBOTWIN[cam]}/rgb")[fi]
        buf = np.frombuffer(bytes(raw), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # (h, w, 3)
        if img is None:
            raise ValueError(f"failed to decode {cam} frame {fi} of {h}")
        if tuple(img.shape[:2]) != tuple(self.image_hw):
            img = cv2.resize(img, (self.image_hw[1], self.image_hw[0]))
        return np.ascontiguousarray(img, dtype=np.uint8)

    def prompt(self, h: str) -> str:
        """First 'seen' instruction for the episode (from ``instructions/*.json``)."""
        ep_path = self.hash_to_path[h]
        task_dir = os.path.dirname(os.path.dirname(ep_path))
        instr_path = os.path.join(
            task_dir, "instructions", f"{_episode_name(ep_path)}.json"
        )
        try:
            with open(instr_path) as f:
                instr = json.load(f)
            seen = instr.get("seen") or instr.get("unseen") or []
            if seen:
                return str(seen[0])
        except FileNotFoundError:
            pass
        return self.group_of(h).replace("__", " ").replace("_", " ")

    # ---- normalization ----
    def _compute_quantiles(self) -> dict:
        """q01/q99 per-dim over all frames, for ``state`` and ``actions``."""
        states, actions = [], []
        for h in self.hashes:
            v = np.asarray(self._arr(h, "joint_action/vector"), dtype=np.float32)
            states.append(v)
            actions.append(self._actions_cache(h))
        S = np.concatenate(states, axis=0)
        A = np.concatenate(actions, axis=0)
        out = {}
        for name, arr in (("state", S), ("actions", A)):
            out[name] = (
                np.quantile(arr, 0.01, axis=0).astype(np.float32),
                np.quantile(arr, 0.99, axis=0).astype(np.float32),
            )
        return out

    def quantile_norm(self, x: np.ndarray, which: str) -> np.ndarray:
        """Map raw 16-D state/actions to [-1, 1] using q01/q99 (mirrors the pi0.5
        State-block range). ``which`` is ``"state"`` or ``"actions"``."""
        q01, q99 = self.quantiles[which]
        x = np.asarray(x, dtype=np.float32)
        return 2.0 * ((x - q01) / (q99 - q01 + 1e-6)) - 1.0

    def save_quantiles(self, path: str) -> None:
        payload = {
            k: {"q01": q01.tolist(), "q99": q99.tolist()}
            for k, (q01, q99) in self.quantiles.items()
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)


def load_quantiles(path: str) -> dict:
    with open(path) as f:
        payload = json.load(f)
    return {
        k: (
            np.asarray(v["q01"], dtype=np.float32),
            np.asarray(v["q99"], dtype=np.float32),
        )
        for k, v in payload.items()
    }


# --------------------------------------------------------------------------- #
# Query dataset
# --------------------------------------------------------------------------- #
class RoboTwinQueryDataset(Dataset):
    """Flat per-frame query dataset over a set of RoboTwin episodes.

    Each item is a flat dict (the RICL collate reads ``episode_hash`` /
    ``frame_idx`` off the top level, then ``annotation_collate`` stacks the rest):
        base_0_rgb / left_wrist_0_rgb / right_wrist_0_rgb : (224,224,3) uint8 HWC
        observations.state.ee_pose : (32,) f32   slot32(quantile_norm(state))
        actions_joint              : (H,32) f32  slot32(quantile_norm(chunk))
        state16                    : (16,) f32   normalized state for the text block
        annotations                : [prompt]
        episode_hash, frame_idx
    """

    def __init__(self, corpus: RoboTwinCorpus, hashes: list[str], action_horizon: int):
        self.corpus = corpus
        self.action_horizon = int(action_horizon)
        self.index: list[tuple[str, int]] = []
        for h in hashes:
            for fi in range(corpus.num_frames(h)):
                self.index.append((h, fi))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        h, fi = self.index[idx]
        c = self.corpus
        state = slot32(c.quantile_norm(c.state(h, fi), "state"))  # (32,)
        chunk = c.action_chunk(h, fi, self.action_horizon)  # (H, 16)
        action = slot32(c.quantile_norm(chunk, "actions"))  # (H, 32)
        sample = {
            STATE_KEY: state.astype(np.float32),
            AC_KEY: action.astype(np.float32),
            PROMPT_STATE_KEY: c.quantile_norm(c.state(h, fi), "state").astype(
                np.float32
            ),
            "annotations": [c.prompt(h)],
            "episode_hash": h,
            "frame_idx": int(fi),
        }
        for cam in CAM_KEYS:
            sample[cam] = c.image(h, fi, cam)  # HWC uint8
        return sample


# --------------------------------------------------------------------------- #
# Bank frame provider
# --------------------------------------------------------------------------- #
def make_robotwin_bank_provider(corpus: RoboTwinCorpus, action_horizon: int = 1):
    """Return a ``BankFrameProvider``: (episode_hash, frame_idx) -> retrieved block.

    Returns the raw head image (HWC uint8; the collate scales to [0,1]) plus the
    quantile-normalized 16-D state and the ``(action_horizon, 16)`` retrieved action
    *chunk* (repeat-padded at the episode tail). Carrying the full chunk — not just
    one step — is what lets a kNN demo's trajectory differ from a random one's (#1).
    The collate keeps these 16-D (text-only prompt encoding); the model's own
    continuous inputs are slot-filled to 32 separately.
    """

    def provider(episode_hash: str, frame_idx: int) -> dict:
        fi = int(np.clip(frame_idx, 0, corpus.num_frames(episode_hash) - 1))
        state = corpus.quantile_norm(corpus.state(episode_hash, fi), "state")  # (16,)
        action = corpus.quantile_norm(
            corpus.action_chunk(episode_hash, fi, action_horizon), "actions"
        )  # (action_horizon, 16)
        return {
            "image": corpus.image(episode_hash, fi, "base_0_rgb"),  # HWC u8
            "state": state.astype(np.float32),
            "action": action.astype(np.float32),
        }

    return provider


# --------------------------------------------------------------------------- #
# Per-frame embedding providers (for the retrieval cache)
# --------------------------------------------------------------------------- #
def make_dinov2_embedding_provider(corpus: RoboTwinCorpus, **kwargs):
    """Per-episode ``(T, EMBED_DIM)`` DINOv2 descriptors of the head view.

    Lazily builds a single ``DinoV2Embedder`` (downloads weights on first use, runs
    on the GPU). Cached per hash.
    """
    from egomimic.ricl.retrieval import DinoV2Embedder

    embedder = DinoV2Embedder(**kwargs)
    cache: dict[str, np.ndarray] = {}

    def provider(episode_hash: str) -> np.ndarray:
        if episode_hash not in cache:
            T = corpus.num_frames(episode_hash)
            frames = np.stack(
                [corpus.image(episode_hash, fi, "base_0_rgb") for fi in range(T)]
            )
            cache[episode_hash] = embedder.embed(frames)
        return cache[episode_hash]

    return provider


def make_fake_embedding_provider(corpus: RoboTwinCorpus, dim: int = 256, seed: int = 0):
    """Deterministic per-frame embeddings for fast smoke checks / unit tests that
    don't need the DINOv2 encoder.

    Frames of one episode cluster around a per-episode centroid (so within-task
    retrieval is sane to assert on); a small per-task component keeps same-task
    episodes nearer each other than cross-task ones."""

    def provider(episode_hash: str) -> np.ndarray:
        T = corpus.num_frames(episode_hash)
        rng = np.random.default_rng(abs(hash(episode_hash)) % (2**32) + seed)
        grp_rng = np.random.default_rng(
            abs(hash(corpus.group_of(episode_hash))) % (2**32)
        )
        task_centroid = grp_rng.standard_normal(dim).astype(np.float32)
        ep_centroid = task_centroid + 0.3 * rng.standard_normal(dim).astype(np.float32)
        return (
            ep_centroid[None, :]
            + 0.02 * rng.standard_normal((T, dim)).astype(np.float32)
        ).astype(np.float32)

    return provider


# --------------------------------------------------------------------------- #
# Retrieval cache (within-group leave-one-out) + random controls
# --------------------------------------------------------------------------- #
def build_robotwin_retrieval_cache(
    corpus: RoboTwinCorpus,
    k: int,
    embeddings_provider: Callable[[str], np.ndarray],
):
    """Within-task leave-one-out ``RetrievalCache`` over all frames.

    For each task group, stack every frame's per-frame embedding, then for each
    query episode compute raw-L2 distances to all group frames, mask the query's
    own frames (leave-one-out), and keep the top-k. Mirrors
    ``build_droid_retrieval_cache`` but takes an explicit per-frame embedding
    provider (RoboTwin ships no precomputed embeddings)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache = RetrievalCache(
        k=k, meta={"source": "robotwin", "k": k, "loeo": True, "device": device}
    )
    emb = {
        h: np.asarray(embeddings_provider(h), dtype=np.float32) for h in corpus.hashes
    }
    for g, hs in corpus.group_to_hashes.items():
        if len(hs) < 2:
            logger.warning("group %s has <2 episodes; skipping (no leave-one-out)", g)
            continue
        vecs, ref_hash, ref_frame = [], [], []
        for h in hs:
            v = emb[h]
            vecs.append(v)
            ref_hash.append(np.full(v.shape[0], h))
            ref_frame.append(np.arange(v.shape[0], dtype=np.int32))
        ref_hash = np.concatenate(ref_hash)
        ref_frame = np.concatenate(ref_frame)
        bank = torch.from_numpy(np.concatenate(vecs, axis=0)).to(device)  # (N, D)

        for qh in hs:
            qv = torch.from_numpy(emb[qh]).to(device)  # (Tq, D)
            d = torch.cdist(qv, bank)  # (Tq, N) raw L2
            self_rows = torch.from_numpy(ref_hash == qh).to(device)
            d[:, self_rows] = float("inf")  # leave-one-out
            kk = min(k, int((ref_hash != qh).sum()))
            dist, top = torch.topk(d, kk, dim=1, largest=False)  # ascending
            top = top.cpu().numpy()
            dist = dist.cpu().numpy().astype(np.float32)
            bh = ref_hash[top]
            bf = ref_frame[top]
            if kk < k:  # pad to fixed width
                pad = k - kk
                bh = np.pad(bh, ((0, 0), (0, pad)), constant_values="")
                bf = np.pad(bf, ((0, 0), (0, pad)), constant_values=-1)
                dist = np.pad(dist, ((0, 0), (0, pad)), constant_values=np.inf)
            cache.add(
                qh,
                bh.astype("U"),
                bf.astype(np.int32),
                dist.astype(np.float32),
                group_id=f"{g}:loeo:{qh}",
            )
    return cache


def build_random_neighbor_cache(
    corpus: RoboTwinCorpus,
    k: int,
    scope: str = "within",
    seed: int = 0,
    embeddings_provider: Callable[[str], np.ndarray] | None = None,
):
    """Random-retrieval control cache (RANDOM neighbors instead of kNN).

      scope="within": random demos from the query's OWN task group (leave-one-out)
                      -> "given the right task bank, does kNN *ranking* matter?"
      scope="bank"  : random demos from ALL groups pooled (leave-one-episode-out)
                      -> "does retrieving the *right task* matter at all?"

    ``dist`` carries the true embedding L2 (so distance-weighted interpolation #2
    barely fires for far random demos) when ``embeddings_provider`` is given,
    else 0 (with a warning). Mirrors ``droid_data.build_random_neighbor_cache``."""
    rng = np.random.default_rng(seed)
    cache = RetrievalCache(k=k, meta={"source": "robotwin", "random": scope, "k": k})
    g2h = corpus.group_to_hashes
    h2g = {h: g for g, hs in g2h.items() for h in hs}
    all_h = [h for hs in g2h.values() for h in hs]
    nf = {h: corpus.num_frames(h) for h in all_h}
    emb = (
        {h: np.asarray(embeddings_provider(h), dtype=np.float32) for h in all_h}
        if embeddings_provider is not None
        else None
    )
    if emb is None:
        logger.warning(
            "build_random_neighbor_cache: no embeddings_provider; dist=0 (random "
            "demos will get full interpolation weight — pass a provider for #2)"
        )
    for h in all_h:
        T, g = nf[h], h2g[h]
        if scope == "within":
            pool = [x for x in g2h[g] if x != h]
        else:
            pool = [x for x in all_h if x != h]
        if not pool:  # singleton group under within-scope: fall back to whole bank
            pool = [x for x in all_h if x != h]
        ep = rng.integers(0, len(pool), size=(T, k))
        bh = np.empty((T, k), dtype=object)
        bf = np.zeros((T, k), dtype=np.int32)
        dist = np.zeros((T, k), dtype=np.float32)
        for t in range(T):
            for j in range(k):
                eh = pool[int(ep[t, j])]
                fj = int(rng.integers(0, nf[eh]))
                bh[t, j] = eh
                bf[t, j] = fj
                if emb is not None:
                    diff = emb[h][t] - emb[eh][fj]
                    dist[t, j] = float(np.sqrt(diff @ diff))
        cache.add(
            h,
            bh.astype("U"),
            bf.astype(np.int32),
            dist,
            group_id=f"{g}:rand-{scope}:{h}",
        )
    return cache


def _verify_cache(cache, corpus: RoboTwinCorpus) -> None:
    """Spot-check: neighbors are in the query's group, never self, dists sorted."""
    n_checked = 0
    for qh in cache.query_hashes:
        g = corpus.group_of(qh)
        group_hashes = set(corpus.group_to_hashes[g])
        bh, bf, dd = cache.neighbors(qh, 0)
        valid = [str(x) for x in bh if str(x)]
        assert qh not in valid, f"{qh}: self in neighbors"
        for h in valid:
            assert h in group_hashes, f"{qh}: neighbor {h} not in group {g}"
        finite = dd[np.isfinite(dd)]
        assert np.all(np.diff(finite) >= -1e-3), f"{qh}: distances not ascending"
        n_checked += 1
    print(f"  verified {n_checked} query episodes (in-group, no self, sorted dists)")


# --------------------------------------------------------------------------- #
# Norm-stats stub + action converter
# --------------------------------------------------------------------------- #
class RoboTwinNormStats:
    """Minimal norm-stats surface PI's constructor + process_batch touch.

    The shim normalizes in the dataset, so only the structural methods are
    exercised in the train + flow-loss-eval path; ``unnormalize`` is a pass-through.
    """

    KEY_TYPES = {
        "base_0_rgb": "camera_keys",
        "left_wrist_0_rgb": "camera_keys",
        "right_wrist_0_rgb": "camera_keys",
        STATE_KEY: "proprio_keys",
        AC_KEY: "action_keys",
    }

    def __init__(self, emb_id: int = EMB_ID):
        self.emb_id = int(emb_id)

    def keys_of_type(self, key_type: str, embodiment_id: int) -> list[str]:
        if embodiment_id != self.emb_id:
            return []
        return [k for k, t in self.KEY_TYPES.items() if t == key_type]

    def is_key_with_embodiment(self, key: str, embodiment_id: int) -> bool:
        return embodiment_id == self.emb_id and key in self.KEY_TYPES

    def zarr_key_to_keyname(self, key: str, embodiment_id: int):
        if embodiment_id == self.emb_id and key in self.KEY_TYPES:
            return key
        return None

    def unnormalize(self, batch, embodiment_id):  # pragma: no cover (sampling-only)
        return batch


class Passthrough32(BaseActionConverter):
    """No-op converter: RoboTwin actions are already slot-filled to 32-D."""

    def to32(self, actions: torch.Tensor) -> torch.Tensor:
        return _pad32(actions)

    def from32(self, actions32: torch.Tensor) -> torch.Tensor:
        return actions32
