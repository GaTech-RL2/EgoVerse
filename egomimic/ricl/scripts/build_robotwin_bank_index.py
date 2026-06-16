"""Build a consolidated DINOv2 kNN index over a RoboTwin HDF5 demo bank.

The closed-loop eval (``egomimic.ricl.robotwin_policy.PIRiclPolicy``) retrieves
online against a prebuilt index that it loads with
``build_embedding_index.build_retrieval_index`` — i.e. a directory of
``{vectors.npy, refs.npz, manifest.json}`` keyed by ``(episode_hash, frame_idx)``.
The generic ``build_embedding_index.build_index`` consumes *zarr* DINOv2 mirror
stores; RoboTwin's bank is HDF5 (``RoboTwinCorpus``), so this builds the same index
format directly from the corpus.

Critically it uses the SAME per-frame embedding the eval-time retriever uses:
``make_dinov2_embedding_provider`` wraps ``retrieval.DinoV2Embedder().embed`` of the
head view, and ``robotwin_adapter.OnlineRetriever`` embeds the live query with the
same ``DinoV2Embedder().embed`` — so bank vectors and query vectors live in one
space and kNN is meaningful. The refs carry ``RoboTwinCorpus`` episode hashes
(``"{group}__{episode}"``) + frame indices, which ``OnlineRetriever`` feeds straight
back into ``make_robotwin_bank_provider`` to fetch each neighbor's (image, state,
action) block.

Output dir (default ``<root>/_bank_index``):
  - ``vectors.npy``   (N_frames, D) float32  — per-frame head-view descriptors
  - ``refs.npz``      episode_hash (N,) <U, frame_idx (N,) int32
  - ``manifest.json`` {source, embed, embed_dim, n_frames, n_episodes,
                       episodes:[{hash, start, count}]}

Example (GPU, real DINOv2 — matches eval):
    python egomimic/ricl/scripts/build_robotwin_bank_index.py \\
        --root egomimic/ricl/outputs/robotwin_raw/extracted \\
        --out  egomimic/ricl/outputs/robotwin_bank_index --embed dinov2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Callable

import numpy as np

from egomimic.ricl import robotwin_data as R

logger = logging.getLogger(__name__)


def build_bank_index(
    corpus: R.RoboTwinCorpus,
    out_dir: str,
    embeddings_provider: Callable[[str], np.ndarray],
) -> str:
    """Embed every bank frame and write a consolidated index under ``out_dir``.

    ``embeddings_provider`` is a per-episode ``hash -> (T, D)`` callable (e.g.
    ``make_dinov2_embedding_provider``); each row of the stacked matrix is one frame,
    in ``corpus.hashes`` order, with parallel ``(episode_hash, frame_idx)`` refs.
    """
    os.makedirs(out_dir, exist_ok=True)
    hashes = corpus.hashes
    emb = {h: np.asarray(embeddings_provider(h), dtype=np.float32) for h in hashes}

    lengths = [emb[h].shape[0] for h in hashes]
    n_frames = int(sum(lengths))
    embed_dim = int(emb[hashes[0]].shape[1])
    logger.info(
        "n_frames=%d embed_dim=%d over %d episodes", n_frames, embed_dim, len(hashes)
    )

    vectors = np.empty((n_frames, embed_dim), dtype=np.float32)
    ref_hash = np.empty(n_frames, dtype=object)
    ref_frame = np.empty(n_frames, dtype=np.int32)
    episodes_meta: list[dict] = []

    row = 0
    for h in hashes:
        v = emb[h]
        t = v.shape[0]
        if v.shape[1] != embed_dim:
            raise RuntimeError(f"{h}: embed dim {v.shape[1]} != {embed_dim}")
        vectors[row : row + t] = v
        ref_hash[row : row + t] = h
        ref_frame[row : row + t] = np.arange(t, dtype=np.int32)
        episodes_meta.append({"hash": h, "start": row, "count": t})
        row += t

    np.save(os.path.join(out_dir, "vectors.npy"), vectors)
    np.savez(
        os.path.join(out_dir, "refs.npz"),
        episode_hash=np.asarray(ref_hash).astype("U"),
        frame_idx=ref_frame,
    )
    manifest = {
        "source": "robotwin",
        "embed_dim": embed_dim,
        "n_frames": n_frames,
        "n_episodes": len(hashes),
        "episodes": episodes_meta,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        "wrote bank index: %d frames x %d dims over %d episodes -> %s",
        n_frames,
        embed_dim,
        len(hashes),
        out_dir,
    )
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root", required=True, help="RoboTwin bank root (parent of <task>/data)"
    )
    p.add_argument(
        "--out", default=None, help="output index dir (default <root>/_bank_index)"
    )
    p.add_argument(
        "--embed",
        choices=["dinov2", "fake"],
        default="dinov2",
        help="per-frame embedding source (dinov2 = real, matches eval; fake = stub)",
    )
    p.add_argument("--fake-embed-dim", type=int, default=256)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    corpus = R.RoboTwinCorpus(args.root)
    if args.embed == "fake":
        provider = R.make_fake_embedding_provider(corpus, dim=args.fake_embed_dim)
    else:
        provider = R.make_dinov2_embedding_provider(corpus)
    out_dir = args.out or os.path.join(args.root, "_bank_index")
    build_bank_index(corpus, out_dir, provider)


if __name__ == "__main__":
    main()
