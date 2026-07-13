"""Build a SELF-CONTAINED chunkviz.html from a chunkviz_data.npz.

Pure data -> HTML: no GPU, no model, no streamlit. Reads the npz that
``export.py`` writes on the cluster and inlines everything (per-frame
trajectory JPEGs as base64 data URIs + all chunk/PCA arrays as a JSON island)
into one ``chunkviz.html`` you can double-click open in any browser, offline.

    python -m egomimic.eval.explorer.build_html \
        --data chunkviz_data.npz --out chunkviz.html [--frame-stride 1]

Runs fine locally after copying the npz down (numpy-only; no torch).
"""
import argparse
import base64
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "viewer_template.html"
PLACEHOLDER = "/*__CHUNKVIZ_DATA__*/"


def jpeg_uri(buf) -> str | None:
    """object-array entry (bytes / np.bytes_) -> data URI, or None if empty."""
    if buf is None:
        return None
    b = bytes(buf) if not isinstance(buf, (bytes, bytearray)) else buf
    if len(b) == 0:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(b).decode("ascii")


def build(data_path: str, frame_stride: int) -> dict:
    d = np.load(data_path, allow_pickle=True)
    emb_ids = [int(x) for x in d["emb_ids"]]
    out = {"emb_ids": emb_ids, "embs": {}}
    st = max(1, int(frame_stride))
    for emb in emb_ids:
        labels = [str(x) for x in d[f"emb{emb}_labels"]]
        pca = np.round(d[f"emb{emb}_pca_xy"].astype(float), 3).tolist()
        nep = int(d[f"emb{emb}_n_episodes"])
        episodes = []
        for ep in range(nep):
            key = f"emb{emb}_ep{ep}_"
            if key + "cid" not in d:
                continue
            cid = d[key + "cid"][:, ::st].astype(int).tolist()
            prob = np.round(d[key + "prob"][:, ::st].astype(float), 3).tolist()
            topcid = d[key + "topcid"][::st].astype(int).tolist()
            frames = []
            if key + "traj" in d:
                frames = [jpeg_uri(b) for b in d[key + "traj"][::st]]
            episodes.append(
                {"cid": cid, "prob": prob, "topcid": topcid, "frames": frames}
            )
        emb_entry = {"labels": labels, "pca": pca, "episodes": episodes}
        # per-emb start index into the shared PCA array (chunk id c of this emb
        # -> shared row pca_offset + c). Present only on cross-emb exports.
        if f"emb{emb}_pca_offset" in d:
            emb_entry["pca_offset"] = int(d[f"emb{emb}_pca_offset"])
        out["embs"][str(emb)] = emb_entry
        nfr = sum(len(e["frames"]) for e in episodes)
        print(f"emb{emb}: stages={labels} pca={len(pca)} "
              f"episodes={len(episodes)} frames={nfr} (stride {st})")
    # shared cross-embodiment PCA (one basis) — drives the alignment compare view
    if "pca_shared_xy" in d:
        out["pca_shared"] = np.round(d["pca_shared_xy"].astype(float), 3).tolist()
        out["pca_shared_emb"] = d["pca_shared_emb"].astype(int).tolist()
        for _k, _src in (("pca_shared_meandiff", "pca_shared_meandiff_xy"),
                         ("pca_shared_lda", "pca_shared_lda_xy")):
            if _src in d:
                out[_k] = np.round(d[_src].astype(float), 3).tolist()
        print(f"shared PCA: {len(out['pca_shared'])} chunks "
              f"(compare/alignment view enabled)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None,
                    help="single npz (model named 'model'); back-compat")
    ap.add_argument("--model", action="append", default=[], metavar="NAME=PATH",
                    help="named model npz, repeatable -> a model dropdown in the "
                         "viewer: --model cossim=a.npz --model sigmoid=b.npz")
    ap.add_argument("--out", default="chunkviz.html")
    ap.add_argument("--frame-stride", type=int, default=1,
                    help="subsample frames AND strip columns by this factor "
                         "(2 ~halves file size; chunk ids stay aligned)")
    args = ap.parse_args()

    # MODELS = {name: payload}. One entry == the old single-file behaviour.
    models = {}
    if args.model:
        for spec in args.model:
            name, sep, path = spec.partition("=")
            if not sep or not name or not path:
                raise SystemExit(f"--model must be NAME=PATH, got {spec!r}")
            print(f"[{name}] {path}")
            models[name] = build(path, args.frame_stride)
    else:
        models["model"] = build(args.data or "chunkviz_data.npz", args.frame_stride)

    template = TEMPLATE.read_text()
    if PLACEHOLDER not in template:
        raise SystemExit(f"placeholder {PLACEHOLDER!r} not found in {TEMPLATE}")
    html = template.replace(
        PLACEHOLDER, json.dumps(models, separators=(",", ":"))
    )
    Path(args.out).write_text(html)
    mb = Path(args.out).stat().st_size / 1e6
    print(f"wrote {args.out}  ({mb:.1f} MB, {len(models)} model(s): {list(models)})"
          f"  - double-click to open")


if __name__ == "__main__":
    main()
