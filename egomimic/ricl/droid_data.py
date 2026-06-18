"""DROID shim data layer for the RICL training-pipeline verification.

Feeds the *original* RICL paper's flat ``processed_demo.npz`` demos through
EgoVerse's real RICL machinery (``build_ricl_collate`` -> ``PIRicl`` prefix
conditioning -> pi0.5) without porting anything into Zarr/SQL. This is the
"thin shim, not a port" path from the bring-up plan, scaled up to a real
multi-step train + eval so we can check that retrieval-conditioning actually
*helps* on a known-good retrieval dataset.

What this module provides
-------------------------
- ``DroidCorpus``         : discover + lazily load the npz demos (LRU-cached).
- ``DroidQueryDataset``   : a torch ``Dataset`` yielding the post-transform keys
  the RICL query path consumes (``base_0_rgb`` etc., a 32-D slot-filled state,
  a (horizon,32) action target, ``episode_hash``, ``frame_idx``, ``annotations``),
  already quantile-normalized to [-1, 1] (PI assumes pre-normalized input).
- ``make_droid_bank_provider`` : a ``BankFrameProvider`` returning a neighbor's
  (top image, normalized 8-D state, normalized (1,8) action); the ricl collate's
  ``_fit`` pads state/action to 32, so the bank and query sides share a layout.
- ``build_droid_retrieval_cache`` : within-group leave-one-out retrieval cache
  built directly from the pre-pooled ``top_image_embeddings`` via
  ``egomimic.ricl.retrieval.build_cache`` (same code the Zarr path uses).
- ``DroidNormStats`` / ``Passthrough32`` : the minimal norm-stats surface PI's
  constructor + ``process_batch_for_training`` touch, and a no-op 8->32 action
  converter (DROID actions are slot-filled to 32 already).

DROID layout (verified): per demo ``state (T,8) f64``, ``actions (T,8) f64``
(7 joint vel + gripper), ``top/wrist/right_image (T,224,224,3) u8``,
``*_image_embeddings (T,49152) f32`` (DINOv2 64-patch pooled), ``prompt`` scalar.
Norm stats (q01/q99, 8-D) at
``external/ricl_openpi/assets/pi0_fast_droid_ricl/droid/norm_stats.json``.

Slot-fill note: DROID's gripper is action/state dim 7, so it lands at slot 7 of
the 32-D vector (``DroidRiclEval`` uses ``gripper_indices=(7,)``). The padding
dims 8..31 are zero on both query and bank sides, so they cancel in the
retrieval-vs-floor delta.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from collections import OrderedDict
from functools import lru_cache

import numpy as np
import torch
from torch.utils.data import Dataset

from egomimic.ricl.retrieval import RetrievalCache
from egomimic.utils.action_utils import BaseActionConverter, _pad32

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Paths / constants
# --------------------------------------------------------------------------- #
DROID_ROOT = (
    "/storage/project/r-dxu345-0/rco3/ricl_openpi/preprocessing/"
    "collected_demos_training"
)
# Original RICL eval set: genuinely NEW tasks (idli plate, poke ball, squeegee,
# door, bagel, toaster, ...) the model never trains on. The retrieval bank for an
# eval query is that new task's own demos (within-group leave-one-out) — the
# original repo's `baseline` mode. This is the canonical RICL train/eval split:
# train on `collected_demos_training`, evaluate on these held-out new tasks.
EVAL_ROOT = (
    "/storage/project/r-dxu345-0/rco3/ricl_openpi/preprocessing/" "collected_demos"
)
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
NORM_JSON = os.path.join(
    _REPO_ROOT,
    "external/ricl_openpi/assets/pi0_fast_droid_ricl/droid/norm_stats.json",
)
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(__file__), "outputs", "droid_cache")

EMB_KEY = "top_image_embeddings"  # ricl_openpi retrieves on the top (exterior) view
RAW_DIM = 8  # DROID state/action dim (7 joints/joint-vels + 1 gripper)
SHARED_DIM = 32  # pi0.5 shared action/state space
GRIPPER_SLOT = 7  # DROID gripper index, preserved by slot32

EMB_ID = 8  # eva_bimanual masquerade (get_embodiment_id("eva_bimanual"))
EMB_NAME = "eva_bimanual"
AC_KEY = "actions_joint"
STATE_KEY = "observations.state.ee_pose"  # 32-D slot-fill -> model continuous proprio
PROMPT_STATE_KEY = "state8"  # 8-D normalized state -> compact text State block
CAM_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
# query cameras map to DROID's three views (top -> base_0_rgb is the retrieval view)
_CAM_TO_DROID = {
    "base_0_rgb": "top_image",
    "left_wrist_0_rgb": "wrist_image",
    "right_wrist_0_rgb": "right_image",
}


# --------------------------------------------------------------------------- #
# Normalization helpers (quantile -> [-1, 1], matching the State-block clip)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _norm_quantiles():
    with open(NORM_JSON) as f:
        ns = json.load(f)["norm_stats"]
    out = {}
    for name, key in (("state", "query_state"), ("actions", "query_actions")):
        q01 = np.asarray(ns[key]["q01"], dtype=np.float32)
        q99 = np.asarray(ns[key]["q99"], dtype=np.float32)
        out[name] = (q01, q99)
    return out


def quantile_norm(x: np.ndarray, which: str) -> np.ndarray:
    """Map raw DROID state/actions to [-1, 1] using q01/q99 (mirrors the pi0.5
    State-block range). ``which`` is ``"state"`` or ``"actions"``."""
    q01, q99 = _norm_quantiles()[which]
    x = np.asarray(x, dtype=np.float32)
    return 2.0 * ((x - q01) / (q99 - q01 + 1e-6)) - 1.0


def slot32(x8: np.ndarray) -> np.ndarray:
    """Place an 8-D vector into dims 0..7 of a 32-D zero vector (last axis)."""
    x8 = np.asarray(x8, dtype=np.float32)
    out = np.zeros(x8.shape[:-1] + (SHARED_DIM,), dtype=np.float32)
    out[..., :RAW_DIM] = x8[..., :RAW_DIM]
    return out


# --------------------------------------------------------------------------- #
# Corpus
# --------------------------------------------------------------------------- #
def _demo_hash(group: str, ts: str) -> str:
    # filesystem- and npz-filename-safe (no "/"); used as the cache query key
    return f"{group}__{ts}"


class DroidCorpus:
    """Discover and lazily load DROID ``processed_demo.npz`` demos.

    A *group* is a task-prompt folder (e.g. ``2025-03-14_move_apple_to_the_right``);
    each contains many timestamped demo dirs. Episode hash = ``"{group}__{ts}"``.
    Per-demo arrays are LRU-cached (one demo's 3 cams ~11 MB) so DataLoader
    workers stay memory-bounded.
    """

    def __init__(
        self,
        root: str = DROID_ROOT,
        groups: list[str] | None = None,
        cache_size: int = 2048,
    ):
        self.root = root
        self.hash_to_path: "OrderedDict[str, str]" = OrderedDict()
        self.group_to_hashes: "OrderedDict[str, list[str]]" = OrderedDict()
        all_groups = sorted(
            d
            for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d)) and not d.startswith(".")
        )
        if groups is not None:
            all_groups = [g for g in all_groups if g in set(groups)]
        for g in all_groups:
            hs = []
            for npz in sorted(
                glob.glob(os.path.join(root, g, "*", "processed_demo.npz"))
            ):
                ts = os.path.basename(os.path.dirname(npz))
                h = _demo_hash(g, ts)
                self.hash_to_path[h] = npz
                hs.append(h)
            if hs:
                self.group_to_hashes[g] = hs
        if not self.hash_to_path:
            raise FileNotFoundError(f"no processed_demo.npz found under {root}")
        # Cache *individual* arrays keyed by (hash, npz_key) so loading embeddings
        # for the retrieval build never decompresses the (much larger) image
        # arrays, and per-frame training reads decompress each demo array once.
        self._arr = lru_cache(maxsize=cache_size)(self._load_arr)

    def _load_arr(self, h: str, key: str) -> np.ndarray:
        with np.load(self.hash_to_path[h], allow_pickle=True) as d:
            return d[key]

    # ---- accessors ----
    @property
    def hashes(self) -> list[str]:
        return list(self.hash_to_path.keys())

    def group_of(self, h: str) -> str:
        return h.split("__", 1)[0]

    def num_frames(self, h: str) -> int:
        return int(self._arr(h, "state").shape[0])

    def embeddings(self, h: str) -> np.ndarray:
        return np.asarray(self._arr(h, EMB_KEY), dtype=np.float32)  # (T, 49152)

    def state(self, h: str, fi: int) -> np.ndarray:
        return np.asarray(self._arr(h, "state")[fi], dtype=np.float32)  # (8,)

    def action_chunk(self, h: str, fi: int, horizon: int) -> np.ndarray:
        """Raw (horizon, 8) action chunk starting at ``fi`` (repeat-pad the last)."""
        acts = np.asarray(self._arr(h, "actions"), dtype=np.float32)  # (T, 8)
        T = acts.shape[0]
        idx = np.clip(np.arange(fi, fi + horizon), 0, T - 1)
        return acts[idx]

    def image(self, h: str, fi: int, cam: str) -> np.ndarray:
        return np.asarray(self._arr(h, _CAM_TO_DROID[cam])[fi])  # (224,224,3) u8

    def prompt(self, h: str) -> str:
        return str(self._arr(h, "prompt").item())


# --------------------------------------------------------------------------- #
# Query dataset
# --------------------------------------------------------------------------- #
class DroidQueryDataset(Dataset):
    """Flat per-frame query dataset over a set of DROID episodes.

    Each item is a flat dict (the RICL collate reads ``episode_hash`` /
    ``frame_idx`` off the top level, then ``annotation_collate`` stacks the rest):
        base_0_rgb / left_wrist_0_rgb / right_wrist_0_rgb : (224,224,3) uint8 HWC
        observations.state.ee_pose : (32,) f32   slot32(quantile_norm(state))
        actions_joint              : (H,32) f32  slot32(quantile_norm(chunk))
        annotations                : [prompt]    (list -> preserved by collate)
        episode_hash, frame_idx
    """

    def __init__(self, corpus: DroidCorpus, hashes: list[str], action_horizon: int):
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
        state = slot32(quantile_norm(c.state(h, fi), "state"))  # (32,)
        chunk = c.action_chunk(h, fi, self.action_horizon)  # (H, 8)
        action = slot32(quantile_norm(chunk, "actions"))  # (H, 32)
        sample = {
            STATE_KEY: state.astype(np.float32),
            AC_KEY: action.astype(np.float32),
            # 8-D normalized state for the *text* State block (the 32-D slot-fill
            # of STATE_KEY feeds the model's continuous proprio; encoding its 24
            # zero pad-dims as tokens just bloats the prompt). See PROMPT_STATE_KEY.
            PROMPT_STATE_KEY: quantile_norm(c.state(h, fi), "state").astype(np.float32),
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
def make_droid_bank_provider(corpus: DroidCorpus, action_horizon: int = 1):
    """Return a ``BankFrameProvider``: (episode_hash, frame_idx) -> retrieved block.

    Returns the raw top image (HWC uint8; the collate scales to [0,1]) plus the
    quantile-normalized 8-D state and the ``(action_horizon, 8)`` retrieved action
    *chunk* (repeat-padded at the episode tail). Carrying the full chunk — not just
    the first step — is what lets a kNN demo's trajectory differ from a random
    one's (the discriminative RICL signal lives in the trajectory shape, cf. #1).
    The collate keeps these 8-D (text-only prompt encoding); the model's own
    continuous inputs are slot-filled to 32 separately.
    """

    def provider(episode_hash: str, frame_idx: int) -> dict:
        fi = int(np.clip(frame_idx, 0, corpus.num_frames(episode_hash) - 1))
        state = quantile_norm(corpus.state(episode_hash, fi), "state")  # (8,)
        action = quantile_norm(
            corpus.action_chunk(episode_hash, fi, action_horizon), "actions"
        )  # (action_horizon, 8)
        return {
            "image": corpus.image(episode_hash, fi, "base_0_rgb"),  # HWC u8
            "state": state.astype(np.float32),
            "action": action.astype(np.float32),
        }

    return provider


# --------------------------------------------------------------------------- #
# Retrieval cache (within-group leave-one-out)
# --------------------------------------------------------------------------- #
def build_droid_retrieval_cache(corpus: DroidCorpus, k: int):
    """Build a within-group leave-one-out ``RetrievalCache`` over all frames.

    For each task group we stack every frame's pre-pooled ``top_image_embeddings``
    once, then for each query episode compute raw-L2 distances (vectorized) to all
    group frames, mask out the query's own frames (leave-one-out), and keep the
    top-k. This reuses one bank per group across its queries and avoids building a
    cKDTree in 49152-D (pathologically slow), while matching the L2 metric and the
    on-disk ``RetrievalCache`` format the collate reads.
    """
    # Distances run in torch: numpy's BLAS on this cluster does ~0.4 GFLOP/s
    # (a single (77,49152)@(49152,1551) took 13.7s); torch (192 threads / CUDA)
    # does the same in milliseconds.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache = RetrievalCache(
        k=k,
        meta={
            "source": "droid",
            "k": k,
            "emb_key": EMB_KEY,
            "loeo": True,
            "device": device,
        },
    )
    for g, hs in corpus.group_to_hashes.items():
        if len(hs) < 2:
            logger.warning("group %s has <2 episodes; skipping (no leave-one-out)", g)
            continue
        vecs, ref_hash, ref_frame = [], [], []
        for h in hs:
            v = corpus.embeddings(h)  # (T, D)
            vecs.append(v)
            ref_hash.append(np.full(v.shape[0], h))
            ref_frame.append(np.arange(v.shape[0], dtype=np.int32))
        ref_hash = np.concatenate(ref_hash)
        ref_frame = np.concatenate(ref_frame)
        bank = torch.from_numpy(np.concatenate(vecs, axis=0)).to(device)  # (N, D)

        for qh in hs:
            qv = torch.from_numpy(corpus.embeddings(qh)).to(device)  # (Tq, D)
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
    corpus: DroidCorpus, k: int, scope: str = "within", seed: int = 0
):
    """Random-retrieval control cache (a `RetrievalCache` with RANDOM neighbors).

    Instead of the k *nearest* bank frames, give each query frame k *random* bank
    frames, so a retrieval-vs-random comparison isolates whether kNN *similarity*
    matters (not just the presence of in-context demos). Two scopes:

      scope="within": random demos from the query's OWN task group (leave-one-out)
                      -> "given the right task bank, does kNN *ranking* matter?"
      scope="bank"  : random demos from ALL groups pooled (leave-one-episode-out)
                      -> "does retrieving the *right task* matter at all?"

    Unlike permuting neighbors within a minibatch, this is independent of loader
    shuffling / batch composition. Same on-disk format the collate reads.
    Deterministic given ``seed``.

    ``dist`` carries the *true* embedding L2 from each query frame to its random
    neighbor (same metric as :func:`build_droid_retrieval_cache`). This matters for
    the distance-weighted interpolation control (#2): a random neighbor sits far
    away -> small ``w=exp(-lamda*dist/dist_max)`` -> the interpolation barely fires
    and the prediction falls back to the model, exactly as it should. (Leaving
    ``dist=0`` here would give random demos full interpolation weight toward a wrong
    action and hand retrieval a meaningless win.)
    """
    rng = np.random.default_rng(seed)
    cache = RetrievalCache(k=k, meta={"source": "droid", "random": scope, "k": k})
    g2h = corpus.group_to_hashes
    h2g = {h: g for g, hs in g2h.items() for h in hs}
    all_h = [h for hs in g2h.values() for h in hs]
    nf = {h: corpus.num_frames(h) for h in all_h}
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
        emb_h = corpus.embeddings(h)  # (T, D) — query embeddings
        for t in range(T):
            e = emb_h[t]
            for j in range(k):
                eh = pool[int(ep[t, j])]
                fj = int(rng.integers(0, nf[eh]))
                bh[t, j] = eh
                bf[t, j] = fj
                diff = e - corpus.embeddings(eh)[fj]  # raw L2, matches kNN metric
                dist[t, j] = float(np.sqrt(diff @ diff))
        cache.add(
            h,
            bh.astype("U"),
            bf.astype(np.int32),
            dist,
            group_id=f"{g}:rand-{scope}:{h}",
        )
    return cache


# --------------------------------------------------------------------------- #
# Norm-stats stub + action converter
# --------------------------------------------------------------------------- #
class DroidNormStats:
    """Minimal norm-stats surface PI's constructor + process_batch touch.

    The shim normalizes in the dataset, so the only methods exercised in the
    train + flow-loss-eval path are the three structural ones; ``unnormalize`` is
    a pass-through in case a sampling eval calls it.
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
    """No-op converter: DROID actions are already slot-filled to 32-D."""

    def to32(self, actions: torch.Tensor) -> torch.Tensor:
        return _pad32(actions)

    def from32(self, actions32: torch.Tensor) -> torch.Tensor:
        return actions32


# --------------------------------------------------------------------------- #
# CLI: build + persist the retrieval cache
# --------------------------------------------------------------------------- #
def _verify_cache(cache, corpus: DroidCorpus) -> None:
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


def main() -> None:
    p = argparse.ArgumentParser(description="Build the DROID RICL retrieval cache")
    p.add_argument("--build-cache", action="store_true")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--out", default=DEFAULT_CACHE_DIR)
    p.add_argument(
        "--groups",
        type=int,
        default=0,
        help="limit to the first N task groups (0 = all)",
    )
    args = p.parse_args()

    groups = None
    if args.groups > 0:
        all_g = sorted(
            d
            for d in os.listdir(DROID_ROOT)
            if os.path.isdir(os.path.join(DROID_ROOT, d))
        )
        groups = all_g[: args.groups]

    corpus = DroidCorpus(groups=groups)
    print(
        f"corpus: {len(corpus.group_to_hashes)} groups, " f"{len(corpus.hashes)} demos"
    )
    if args.build_cache:
        cache = build_droid_retrieval_cache(corpus, k=args.k)
        _verify_cache(cache, corpus)
        os.makedirs(args.out, exist_ok=True)
        cache.save(args.out)
        print(f"wrote cache for {len(cache.query_hashes)} query episodes -> {args.out}")


if __name__ == "__main__":
    main()
