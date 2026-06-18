"""Visualize RICL kNN retrieval per frame: query (robot/eva) -> top-k bank (human/aria).

For each sampled query frame this lays out ``[query | nn_1 .. nn_k]`` with the
retrieval distance under each neighbor, so you can eyeball whether the retrieved
human demos actually match the robot frame.

Neighbor source
  * fresh (default): rebuild the kNN index from DINOv2 embeddings and query live
    -- exercises the same ``pool -> index -> query`` path the cache builder uses.
  * cache (``--cache DIR``): read an existing ``RetrievalCache`` (exactly what
    training/eval will see).

Bank/query selection
  * ``--query H --bank H[,H...]``                explicit episodes
  * ``--mode cross_similar [--group-index N]``   pick a pair group from pairs.json
  * ``--cache DIR --query H``                    neighbors straight from a cache

Outputs ``<out>.png`` (rows = sampled query frames) and, with ``--video``,
``<out>.mp4`` sweeping every ``--video-stride``-th query frame. Distances are
raw Euclidean (L2) on the 64-super-patch DINOv2 descriptors, matching the
cache builder (ricl_openpi convention; vectors are NOT unit-normalized).

Run (EgoVerse venv + this repo on PYTHONPATH):
  PYTHONPATH=/storage/project/r-dxu345-0/rco3/EgoVerse2 \\
    /storage/project/r-dxu345-0/rco3/EgoVerse/emimic/bin/python \\
    -m egomimic.ricl.scripts.viz_retrieval --mode cross_similar --group-index 0 \\
    -k 4 --num-frames 8 --out ricl_viz
"""

from __future__ import annotations

import argparse
import io
import os

import numpy as np

from egomimic.ricl.retrieval import (
    groups_from_pairs,
    index_from_episodes,
    load_pairs,
    pool_patch_embeddings,
)

AGAO_DEFAULT = "/storage/project/r-dxu345-0/agao81/pick_place"
DINO_KEY_DEFAULT = "observations.embeddings.dinov2.front_1"  # patch tokens (T,256,768)
IMG_KEY_DEFAULT = "images.front_1"  # JPEG-encoded frames (T,)
TILE_HW = (240, 320)  # per-image tile (h, w) for the montage / video


# ---------------------------------------------------------------------------
# Zarr / image helpers
# ---------------------------------------------------------------------------


def _unwrap(el):
    """Peel nested object-array wrappers down to the raw JPEG bytes."""
    while isinstance(el, np.ndarray) and el.dtype == object:
        el = el.item() if el.ndim == 0 else el[0]
    return bytes(el)


class EpisodeStore:
    """Lazy zarr store for one episode: decodes frames and pools embeddings."""

    def __init__(
        self, path: str, img_key: str, dino_key: str, img_path: str | None = None
    ):
        import zarr

        self.path = path
        self.img_key = img_key
        self.dino_key = dino_key
        self._z = zarr.open(path, mode="r")  # embeddings (dino_key)
        # Images may live in a different store than the embeddings — e.g. the
        # embed step writes a mirror folder (dino_key only) while the images stay
        # in the read-only source store. Default to the same store.
        self._zimg = (
            self._z
            if (img_path is None or img_path == path)
            else zarr.open(img_path, mode="r")
        )

    def __len__(self) -> int:
        return int(self._zimg[self.img_key].shape[0])

    def image(self, idx: int) -> np.ndarray:
        """Decode one frame -> (H, W, 3) uint8 RGB."""
        from PIL import Image

        idx = int(np.clip(idx, 0, len(self) - 1))
        raw = _unwrap(self._zimg[self.img_key][idx])
        return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))

    def pooled_all(self, chunk: int = 256) -> np.ndarray:
        """Pool every frame's patch tokens -> (T, EMBED_DIM), memory-safe.

        Dino arrays are zero-padded to a chunk multiple by the embed step; slice
        to the true length (``total_frames``) so padding rows aren't indexed as
        spurious bank candidates.
        """
        a = self._z[self.dino_key]
        T = int(self._z.attrs.get("total_frames", a.shape[0]))
        out = None
        for i in range(0, T, chunk):
            end = min(i + chunk, T)
            blk = pool_patch_embeddings(a[i:end])
            if out is None:
                out = np.empty((T, blk.shape[-1]), dtype=np.float32)
            out[i:end] = blk
        return out

    def pooled_frames(self, idxs) -> np.ndarray:
        """Pool only the given frames' patch tokens -> (len(idxs), EMBED_DIM)."""
        a = self._z[self.dino_key]
        rows = [a[int(i)] for i in idxs]  # each (N, D)
        return pool_patch_embeddings(np.stack(rows, axis=0))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _tile(img: np.ndarray, hw=TILE_HW) -> np.ndarray:
    import cv2

    return cv2.resize(img, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)


def render_montage(rows: list[dict], k: int, out_png: str, title: str = "") -> None:
    """rows: [{q_frame, q_img, neighbors:[{img,dist,hash,frame,valid}]}]."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nrows = len(rows)
    ncols = 1 + k
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.6 * ncols, 2.4 * nrows), squeeze=False
    )
    for r, row in enumerate(rows):
        ax = axes[r][0]
        ax.imshow(row["q_img"])
        ax.set_ylabel(f"q frame {row['q_frame']}", fontsize=9)
        if r == 0:
            ax.set_title("QUERY (robot)", fontsize=10, color="tab:blue")
        ax.set_xticks([])
        ax.set_yticks([])
        for c in range(k):
            ax = axes[r][c + 1]
            nb = row["neighbors"][c] if c < len(row["neighbors"]) else None
            if nb and nb["valid"]:
                ax.imshow(nb["img"])
                ax.set_title(
                    f"d={nb['dist']:.3f}\n{nb['hash'][:8]}#{nb['frame']}",
                    fontsize=7,
                )
            else:
                ax.imshow(np.zeros((*TILE_HW, 3), np.uint8))
                ax.set_title("(empty)", fontsize=7, color="gray")
            if r == 0 and c == 0:
                ax.text(
                    0.0,
                    1.28,
                    "RETRIEVED (human) ->",
                    transform=ax.transAxes,
                    fontsize=10,
                    color="tab:green",
                )
            ax.set_xticks([])
            ax.set_yticks([])
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98 if title else 1.0))
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"wrote montage -> {out_png}")


def dump_episode(
    store, out_mp4: str, label: str, stride: int = 4, fps: int = 30
) -> None:
    """Decode an episode's raw frames (strided) into an mp4 for inspection."""
    import cv2
    import imageio

    n = len(store)
    idxs = list(range(0, n, max(1, stride)))
    writer = imageio.get_writer(out_mp4, fps=fps, macro_block_size=None)
    for fi in idxs:
        frame = store.image(fi).copy()
        cv2.putText(
            frame,
            f"{label}  f{fi}/{n}",
            (6, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
        writer.append_data(frame)
    writer.close()
    print(f"wrote episode video -> {out_mp4}  ({len(idxs)} frames, stride {stride})")


def render_video(rows: list[dict], k: int, out_mp4: str, fps: int = 4) -> None:
    """One wide frame per query frame: [query | nn_1 .. nn_k] with captions."""
    import cv2
    import imageio

    h, w = TILE_HW
    pad = 6
    label_h = 26
    writer = imageio.get_writer(out_mp4, fps=fps, macro_block_size=None)
    for row in rows:
        cells = []
        q = _tile(row["q_img"])
        q = cv2.copyMakeBorder(q, label_h, 0, 0, 0, cv2.BORDER_CONSTANT, value=0)
        cv2.putText(
            q,
            f"query #{row['q_frame']}",
            (4, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (80, 160, 255),
            1,
            cv2.LINE_AA,
        )
        cells.append(q)
        for c in range(k):
            nb = row["neighbors"][c] if c < len(row["neighbors"]) else None
            if nb and nb["valid"]:
                t = _tile(nb["img"])
                cap = f"d={nb['dist']:.3f} {nb['hash'][:6]}#{nb['frame']}"
                color = (120, 230, 120)
            else:
                t = np.zeros((h, w, 3), np.uint8)
                cap = "(empty)"
                color = (150, 150, 150)
            t = cv2.copyMakeBorder(t, label_h, 0, 0, 0, cv2.BORDER_CONSTANT, value=0)
            cv2.putText(
                t, cap, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA
            )
            cells.append(t)
        sep = np.full((h + label_h, pad, 3), 40, np.uint8)
        frame = cells[0]
        for cell in cells[1:]:
            frame = np.concatenate([frame, sep, cell], axis=1)
        writer.append_data(frame)
    writer.close()
    print(f"wrote video -> {out_mp4} ({len(rows)} frames @ {fps}fps)")


# ---------------------------------------------------------------------------
# Neighbor sources
# ---------------------------------------------------------------------------


def neighbors_fresh(query_store, bank_stores, bank_hashes, q_idxs, k):
    """Build a live kNN index over the bank frames and query the sampled frames."""
    index = index_from_episodes(bank_hashes, lambda h: bank_stores[h].pooled_all())
    qv = query_store.pooled_frames(q_idxs)
    hashes, frames, dists = index.query(qv, k=k)
    return hashes, frames, dists  # each (Q, k)


def neighbors_from_cache(cache_dir, query_hash, q_idxs, k):
    from egomimic.ricl.retrieval import RetrievalCache

    cache = RetrievalCache.load(cache_dir)
    if query_hash not in cache._entries:
        raise SystemExit(
            f"{query_hash} not in cache {cache_dir}; have {cache.query_hashes}"
        )
    hh, ff, dd = [], [], []
    for fi in q_idxs:
        bh, bf, d = cache.neighbors(query_hash, int(fi))
        hh.append(bh[:k])
        ff.append(bf[:k])
        dd.append(d[:k])
    return np.stack(hh), np.stack(ff), np.stack(dd)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--query", help="query (robot/eva) episode hash")
    p.add_argument("--bank", help="comma-separated bank (human/aria) episode hashes")
    p.add_argument(
        "--mode", default=None, choices=["cross_similar"], help="pick from pairs.json"
    )
    p.add_argument(
        "--group-index", type=int, default=0, help="which pair group (with --mode)"
    )
    p.add_argument(
        "--cache", default=None, help="read neighbors from this RetrievalCache dir"
    )
    p.add_argument("-k", type=int, default=4)
    p.add_argument(
        "--num-frames",
        type=int,
        default=8,
        help="evenly-sampled query frames in the montage",
    )
    p.add_argument(
        "--frames",
        default=None,
        help="explicit comma-separated query frames to show (overrides "
        "--num-frames; e.g. mid-segment grasp frames)",
    )
    p.add_argument(
        "--zarr-root",
        default=os.path.join(AGAO_DEFAULT, "{hash}"),
        help="{hash} template -> per-episode zarr store",
    )
    p.add_argument(
        "--img-root",
        default=None,
        help="{hash} template for the IMAGE store when images live in a different "
        "store than --zarr-root (e.g. embeddings in a mirror folder, images in "
        "the source store). Defaults to --zarr-root.",
    )
    p.add_argument("--dino-key", default=DINO_KEY_DEFAULT)
    p.add_argument("--img-key", default=IMG_KEY_DEFAULT)
    p.add_argument("--out", default="ricl_viz", help="output path stem (.png/.mp4)")
    p.add_argument(
        "--dump-episodes",
        action="store_true",
        help="instead of retrieval: write a raw mp4 of the resolved query + bank episodes",
    )
    p.add_argument("--dump-stride", type=int, default=4)
    p.add_argument("--dump-fps", type=int, default=30)
    p.add_argument("--video", action="store_true", help="also render a sweep mp4")
    p.add_argument(
        "--video-stride",
        type=int,
        default=10,
        help="sample every Nth frame for the video",
    )
    p.add_argument("--fps", type=int, default=4)
    args = p.parse_args()

    # ---- resolve query + bank hashes -------------------------------------
    if args.mode:
        groups = groups_from_pairs(load_pairs(), args.mode)
        if not groups:
            raise SystemExit(f"no groups for mode {args.mode}")
        g = groups[args.group_index % len(groups)]
        query_hash = args.query or g.query_hashes[0]
        bank_hashes = args.bank.split(",") if args.bank else list(g.bank_hashes)
        print(f"group {g.group_id}: query={query_hash} bank={bank_hashes}")
    else:
        if not args.query:
            raise SystemExit("need --query (and --bank, or --cache, or --mode)")
        query_hash = args.query
        bank_hashes = args.bank.split(",") if args.bank else []

    def open_store(h):
        img_path = args.img_root.format(hash=h) if args.img_root else None
        return EpisodeStore(
            args.zarr_root.format(hash=h),
            args.img_key,
            args.dino_key,
            img_path=img_path,
        )

    query_store = open_store(query_hash)
    n_q = len(query_store)

    # ---- raw episode dump (no retrieval) ---------------------------------
    if args.dump_episodes:
        dump_episode(
            query_store,
            f"{args.out}_query_{query_hash}.mp4",
            f"QUERY robot {query_hash[:13]}",
            args.dump_stride,
            args.dump_fps,
        )
        for h in bank_hashes:
            dump_episode(
                open_store(h),
                f"{args.out}_bank_{h}.mp4",
                f"BANK human {h[:13]}",
                args.dump_stride,
                args.dump_fps,
            )
        return

    # ---- montage: explicit or evenly-sampled query frames ----------------
    if args.frames:
        samp = sorted(
            {int(np.clip(int(x), 0, n_q - 1)) for x in args.frames.split(",")}
        )
    else:
        samp = np.linspace(0, n_q - 1, num=min(args.num_frames, n_q), dtype=int)
        samp = sorted(set(samp.tolist()))

    if args.cache:
        hashes, frames, dists = neighbors_from_cache(
            args.cache, query_hash, samp, args.k
        )
    else:
        if not bank_hashes:
            raise SystemExit("fresh retrieval needs --bank or --mode")
        bank_stores = {h: open_store(h) for h in bank_hashes}
        hashes, frames, dists = neighbors_fresh(
            query_store, bank_stores, bank_hashes, samp, args.k
        )

    # lazily open whatever bank stores the neighbors reference (for cache mode)
    store_cache: dict = {} if args.cache else bank_stores

    def bank_store(h):
        if h not in store_cache:
            store_cache[h] = open_store(h)
        return store_cache[h]

    def build_rows(q_idxs, hh, ff, dd):
        rows = []
        for r, fi in enumerate(q_idxs):
            nbs = []
            for c in range(args.k):
                h = str(hh[r][c])
                valid = bool(h) and int(ff[r][c]) >= 0 and np.isfinite(dd[r][c])
                img = bank_store(h).image(int(ff[r][c])) if valid else None
                nbs.append(
                    {
                        "img": img,
                        "dist": float(dd[r][c]),
                        "hash": h,
                        "frame": int(ff[r][c]),
                        "valid": valid,
                    }
                )
            rows.append(
                {
                    "q_frame": int(fi),
                    "q_img": query_store.image(int(fi)),
                    "neighbors": nbs,
                }
            )
        return rows

    rows = build_rows(samp, hashes, frames, dists)

    # ---- stdout summary (the "test" view) --------------------------------
    print(
        f"\nquery={query_hash}  frames={n_q}  k={args.k}  "
        f"source={'cache' if args.cache else 'fresh'}"
    )
    for row in rows:
        nb = row["neighbors"][0]
        tag = (
            f"{nb['hash'][:8]}#{nb['frame']} d={nb['dist']:.3f}"
            if nb["valid"]
            else "(none)"
        )
        print(f"  q#{row['q_frame']:>5}  ->  top1 {tag}")

    title = (
        f"RICL retrieval  query={query_hash[:13]}  "
        f"bank={','.join(h[:8] for h in (bank_hashes or list(store_cache)))}  "
        f"({'cache' if args.cache else 'fresh'}, k={args.k})"
    )
    render_montage(rows, args.k, f"{args.out}.png", title=title)

    # ---- optional sweep video --------------------------------------------
    if args.video:
        vidx = list(range(0, n_q, max(1, args.video_stride)))
        if args.cache:
            vh, vf, vd = neighbors_from_cache(args.cache, query_hash, vidx, args.k)
        else:
            vh, vf, vd = neighbors_fresh(
                query_store, bank_stores, bank_hashes, vidx, args.k
            )
        vrows = build_rows(vidx, vh, vf, vd)
        render_video(vrows, args.k, f"{args.out}.mp4", fps=args.fps)


if __name__ == "__main__":
    main()
