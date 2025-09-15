#!/usr/bin/env python3
# tsne_viz_app.py
#
# Interactive t-SNE/UMAP viewer with hover/click-to-view images.
# Reuses your existing helpers via imports:
#   - instantiate_model: egomimic.scripts.plot_tsne_single_embodiment
#   - reduce_many:       egomimic.utils.tsne_utils
#
# Two modes:
#  1) Dash server (default): launches a web app with a side preview panel.
#  2) Export HTML (--out_html): writes a single self-contained HTML (no server).
#
# Example (Dash):
#   python tsne_viz_app.py \
#     --ckpt /path/to/model.ckpt \
#     --dataset_paths /data/rl2 /data/eth /data/wang /data/song \
#     --embodiment aria_bimanual --method tsne --viz_cam_key front_img_1 \
#     --batches_per_class 8 --batch_size 16 --host 127.0.0.1 --port 8050
#
# Example (Export HTML):
#   python tsne_viz_app.py \
#     --bundle_pt ./tsne/cup_saucer.pt \
#     --out_html ./tsne/cup_saucer_view.html \
#     --title "Cup on Saucer Embeddings"
#
# Images are embedded as base64 and displayed in a right-hand panel (not overlaid).
# The t-SNE/UMAP scatter uses a square aspect ratio.

import os
import io
import base64
import argparse
import uuid
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from egomimic.scripts.plot_tsne_single_embodiment import instantiate_model
from egomimic.utils.tsne_utils import reduce_many
from rldb.utils import FolderRLDBDataset, get_embodiment_id

# Dash/Plotly (server mode)
from dash import Dash, dcc, html, Input, Output, State, no_update
import plotly.express as px

# Plotly (export mode)
import plotly.graph_objects as go
from plotly.offline import plot as plot_offline


# ----------------------------
# Small utilities
# ----------------------------
def _to_device(obj, device, non_blocking=True):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=non_blocking)
    if isinstance(obj, dict):
        return {k: _to_device(v, device, non_blocking) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(v, device, non_blocking) for v in obj)
    return obj

def _tensor_to_pil_from_maybe_bchw_bhwc(img_t: torch.Tensor):
    """
    Accepts BCHW / BHWC / CHW / HWC tensors, float/uint8.
    Converts to PIL.Image robustly (auto channel/layout handling).
    """
    from PIL import Image
    t = img_t.detach().cpu()

    # Strip batch
    if t.ndim == 4:
        if t.shape[1] in (1, 3):        # BCHW
            t = t[0].permute(1, 2, 0)   # HWC
        elif t.shape[-1] in (1, 3):     # BHWC
            t = t[0]                    # HWC
        else:
            t = t[0]
            if t.ndim == 3 and t.shape[0] in (1, 3):  # CHW
                t = t.permute(1, 2, 0)
    elif t.ndim == 3 and t.shape[0] in (1, 3):        # CHW
        t = t.permute(1, 2, 0)                        # HWC

    arr = t.numpy()

    # float -> uint8
    if np.issubdtype(arr.dtype, np.floating):
        vmin, vmax = float(arr.min()), float(arr.max())
        if 0.0 - 1e-6 <= vmin and vmax <= 1.0 + 1e-6:
            arr = (arr * 255.0).round().clip(0, 255).astype(np.uint8)
        elif 0.0 - 1e-6 <= vmin and vmax <= 255.0 + 1e-6:
            arr = np.clip(arr, 0, 255).round().astype(np.uint8)
        else:
            rng = max(vmax - vmin, 1e-8)
            arr = ((arr - vmin) / rng * 255.0).round().clip(0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    # Grayscale squeeze
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    return Image.fromarray(arr)

def _pil_to_base64(img_pil) -> str:
    bio = io.BytesIO()
    img_pil.save(bio, format="PNG")
    return "data:image/png;base64," + base64.b64encode(bio.getvalue()).decode("utf-8")

def _first_existing(keys: List[str], mapping) -> Optional[str]:
    for k in keys:
        if k in mapping:
            return k
    return None


# ----------------------------
# Embed + thumbnail extraction (reuse your model & dataset APIs)
# ----------------------------
def extract_feats_and_images(
    model,
    dataset_paths: List[str],
    embodiment_name: str,
    batch_size: int,
    num_batches_per_class: int,
    num_workers: int,
    viz_cam_key: Optional[str],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[dict]]]:
    """
    Returns:
      feats_by_class: dict[class] -> Tensor [N, D or SxD]
      meta_by_class:  dict[class] -> list of dicts aligned with feats rows:
        {"image_b64": ..., "info": "...", "idx": global_row, "class": class}
    """
    algo = model.model if hasattr(model, "model") else model
    policy = algo.nets["policy"]
    device = getattr(algo, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # datasets / dataloaders
    datasets = []
    for path in dataset_paths:
        ds = FolderRLDBDataset(
            folder_path=path,
            embodiment=embodiment_name,
            mode="total",
            local_files_only=True,
        )
        datasets.append(ds)

    folder_map = {
        "put_cup_on_saucer_rl2": "rl2",
        "put_cup_on_saucer_eth": "eth",
        "put_cup_on_saucer_wang": "wang",
        "put_cup_on_saucer_song": "song",
    }
    dataloaders = {}
    class_names = []
    for i, ds in enumerate(datasets):
        base = os.path.basename(dataset_paths[i].rstrip("/"))
        cname = folder_map.get(base, base)
        class_names.append(cname)
        dataloaders[cname] = DataLoader(
            ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
        )

    emb_id = get_embodiment_id(embodiment_name)
    cam_keys  = algo.camera_keys[emb_id]
    prop_keys = algo.proprio_keys[emb_id]
    lang_keys = algo.lang_keys[emb_id]
    ac_key    = algo.ac_keys[emb_id]
    aux_keys  = algo.auxiliary_ac_keys.get(embodiment_name, [])

    print(f"[INFO] Known camera keys for '{embodiment_name}': {cam_keys}")

    # Camera key selection preference
    display_cam_key = viz_cam_key if viz_cam_key is not None else (cam_keys[0] if len(cam_keys) > 0 else None)
    if viz_cam_key is not None and viz_cam_key not in cam_keys:
        print(f"[WARN] --viz_cam_key '{viz_cam_key}' not in known cam_keys; will try per-batch fallbacks.")

    feats_by_class: Dict[str, List[torch.Tensor]] = {c: [] for c in class_names}
    meta_by_class:  Dict[str, List[dict]] = {c: [] for c in class_names}
    gidx = 0

    policy.eval()
    with torch.inference_mode():
        for cname, loader in dataloaders.items():
            done = 0
            for batch in loader:
                # Prepare batch
                raw_by_emb = {emb_id: batch}
                processed = algo.process_batch_for_training(raw_by_emb)
                processed[emb_id]["embodiment"] = torch.tensor([emb_id], dtype=torch.int64)

                hpt_data = algo._robomimic_to_hpt_data(
                    processed[emb_id], cam_keys, prop_keys, lang_keys, ac_key, aux_keys
                )
                hpt_data = _to_device(hpt_data, device)

                # Features
                proc_tokens, _ = policy.forward_features(embodiment_name, hpt_data)
                feats_by_class[cname].append(proc_tokens.detach().cpu())

                # --- Choose image tensor: prefer processed[emb_id], then fallback to hpt_data ---
                img_key = None
                imgs = None
                proc_src = processed[emb_id]  # dict

                # A) requested key in PROCESSED
                if display_cam_key and isinstance(proc_src, dict) and display_cam_key in proc_src:
                    candidate = proc_src[display_cam_key]
                    if isinstance(candidate, torch.Tensor) and candidate.ndim in (4, 3):
                        img_key = display_cam_key
                        imgs = candidate

                # B) any known camera key in PROCESSED
                if imgs is None and isinstance(proc_src, dict):
                    for k in cam_keys:
                        if k in proc_src:
                            candidate = proc_src[k]
                            if isinstance(candidate, torch.Tensor) and candidate.ndim in (4, 3):
                                img_key = k
                                imgs = candidate
                                break

                # C) requested key in HPT
                if imgs is None and display_cam_key and display_cam_key in hpt_data:
                    candidate = hpt_data[display_cam_key]
                    if isinstance(candidate, torch.Tensor) and candidate.ndim in (4, 3):
                        img_key = display_cam_key
                        imgs = candidate

                # D) any known camera key in HPT
                if imgs is None:
                    for k in cam_keys:
                        if k in hpt_data:
                            candidate = hpt_data[k]
                            if isinstance(candidate, torch.Tensor) and candidate.ndim in (4, 3):
                                img_key = k
                                imgs = candidate
                                break

                # E) last resort: any 4D image-like tensor
                if imgs is None:
                    for k, v in list(proc_src.items()) + list(hpt_data.items()):
                        if isinstance(v, torch.Tensor) and v.ndim == 4 and (v.shape[1] in (1, 3) or v.shape[-1] in (1, 3)):
                            img_key = k
                            imgs = v
                            break

                if display_cam_key and img_key != display_cam_key:
                    print(f"[WARN] Requested '{display_cam_key}' not found in this batch; using '{img_key or 'N/A'}'.")

                # Log quick stats
                try:
                    if isinstance(imgs, torch.Tensor):
                        vmin = float(imgs.min().cpu().item()); vmax = float(imgs.max().cpu().item())
                        print(f"[INFO] imgs key={img_key} shape={tuple(imgs.shape)} dtype={imgs.dtype} min={vmin:.3f} max={vmax:.3f}")
                except Exception:
                    pass

                # Store thumbnails aligned with embeddings
                B = proc_tokens.shape[0]
                ok_imgs = 0
                for b in range(B):
                    img_b64 = None
                    try:
                        if imgs is not None:
                            img_b64 = _pil_to_base64(_tensor_to_pil_from_maybe_bchw_bhwc(imgs[b]))
                            if img_b64 is not None:
                                ok_imgs += 1
                    except Exception:
                        pass
                    meta_by_class[cname].append({
                        "idx": gidx,
                        "class": cname,
                        "embedding_row": len(meta_by_class[cname]),
                        "image_b64": img_b64,
                        "info": f"{cname} · sample#{gidx} · cam={img_key if img_key else 'N/A'}",
                    })
                    gidx += 1

                print(f"[INFO] {cname}: embedded images this batch = {ok_imgs}/{B}")

                done += 1
                if done >= num_batches_per_class:
                    break

    # Concat per class
    for cname in feats_by_class:
        feats_by_class[cname] = torch.cat(feats_by_class[cname], dim=0) if feats_by_class[cname] else torch.empty(0, 0)

    return feats_by_class, meta_by_class


# ----------------------------
# Dash app (server mode) — square plot, side preview panel
# ----------------------------
def make_app(Z: np.ndarray, y: np.ndarray, names: List[str], meta_by_class: Dict[str, List[dict]], title: str) -> Dash:
    import pandas as pd
    recs = []
    flat_meta = {}
    class_counts = {i: 0 for i in range(len(names))}
    for i in range(Z.shape[0]):
        ci = int(y[i]); cname = names[ci]
        mlist = meta_by_class.get(cname, [])
        m = mlist[class_counts[ci]] if class_counts[ci] < len(mlist) else {"image_b64": None, "info": f"{cname} · idx={i}"}
        class_counts[ci] += 1
        meta_id = uuid.uuid4().hex
        flat_meta[meta_id] = {
            "class": cname,
            "image_b64": m.get("image_b64"),
            "info": m.get("info", f"{cname}"),
            "row_in_class": class_counts[ci] - 1,
            "global_row": i,
        }
        recs.append({"x": Z[i, 0], "y": Z[i, 1], "class": cname, "meta_id": meta_id})

    df = pd.DataFrame(recs)

    fig = px.scatter(df, x="x", y="y", color="class", hover_data=["class", "meta_id"], title=title)
    fig.update_traces(marker={"size": 7, "opacity": 0.85})
    # Square aspect
    fig.update_yaxes(scaleanchor="x", scaleratio=1, constrain="domain")
    fig.update_xaxes(constrain="domain")

    app = Dash(__name__)
    app.layout = html.Div([
        html.Div([
            dcc.Graph(id="scatter", figure=fig, clear_on_unhover=False, style={"height": "80vh"})
        ], style={"width": "65%", "display": "inline-block", "verticalAlign": "top", "padding": "8px"}),

        html.Div([
            html.H3("Preview"),
            html.Div(id="preview-info", style={"marginBottom": "8px", "fontFamily": "monospace", "whiteSpace": "pre-wrap"}),
            html.Img(id="preview-img", style={"maxWidth": "100%", "border": "1px solid #ddd", "borderRadius": "8px"}),
            html.Div(style={"height": "16px"}),
            html.Hr(),
            html.Div("Tip: Hover to preview; click to lock."),
        ], style={"width": "34%", "display": "inline-block", "verticalAlign": "top", "padding": "8px", "borderLeft": "1px solid #eee"}),

        dcc.Store(id="clicked-meta-id", data=None),
        dcc.Store(id="flat-meta-store", data=flat_meta),
    ])

    @app.callback(
        Output("preview-img", "src"),
        Output("preview-info", "children"),
        Input("scatter", "hoverData"),
        State("clicked-meta-id", "data"),
        State("flat-meta-store", "data"),
        prevent_initial_call=True
    )
    def on_hover(hoverData, clicked_meta_id, flat_meta):
        if hoverData is None or clicked_meta_id:
            return no_update, no_update
        pt = hoverData["points"][0]
        meta_id = None
        if "customdata" in pt and isinstance(pt["customdata"], list):
            meta_id = pt["customdata"][1]
        elif "hovertext" in pt:
            meta_id = pt["hovertext"]
        if not meta_id or meta_id not in flat_meta:
            return no_update, no_update
        m = flat_meta[meta_id]
        info = f"{m['info']} (class={m['class']}, row={m['row_in_class']})"
        return m.get("image_b64", None), info

    @app.callback(
        Output("clicked-meta-id", "data"),
        Input("scatter", "clickData"),
        State("clicked-meta-id", "data"),
        prevent_initial_call=True
    )
    def on_click(clickData, clicked_meta_id):
        if clickData is None:
            return clicked_meta_id
        pt = clickData["points"][0]
        meta_id = None
        if "customdata" in pt and isinstance(pt["customdata"], list):
            meta_id = pt["customdata"][1]
        elif "hovertext" in pt:
            meta_id = pt["hovertext"]
        if clicked_meta_id == meta_id:
            return None
        return meta_id

    @app.callback(
        Output("preview-img", "src"),
        Output("preview-info", "children"),
        Input("clicked-meta-id", "data"),
        State("flat-meta-store", "data"),
        prevent_initial_call=True
    )
    def on_click_preview(meta_id, flat_meta):
        if not meta_id:
            return no_update, no_update
        m = flat_meta.get(meta_id)
        if not m:
            return no_update, no_update
        info = f"{m['info']} (class={m['class']}, row={m['row_in_class']})"
        return m.get("image_b64", None), info

    return app


# ----------------------------
# Export HTML (no server) — square plot, side panel
# ----------------------------
def export_html(Z: np.ndarray, y: np.ndarray, names: List[str], meta_by_class: Dict[str, List[dict]], out_html: str, title="Embedding Viewer"):
    xs, ys, cls, imgs, infos = [], [], [], [], []
    class_counts = {i: 0 for i in range(len(names))}
    for i in range(Z.shape[0]):
        ci = int(y[i]); cname = names[ci]
        mlist = meta_by_class.get(cname, [])
        m = mlist[class_counts[ci]] if class_counts[ci] < len(mlist) else {"image_b64": None, "info": f"{cname} · idx={i}"}
        class_counts[ci] += 1
        xs.append(Z[i, 0]); ys.append(Z[i, 1]); cls.append(cname); imgs.append(m.get("image_b64")); infos.append(m.get("info"))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(size=7, opacity=0.85),
        text=cls,
        customdata=np.stack([np.array(cls, dtype=object),
                             np.array(infos, dtype=object),
                             np.array(imgs, dtype=object)], axis=1),
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>",
    ))
    # Square aspect
    fig.update_yaxes(scaleanchor="x", scaleratio=1, constrain="domain")
    fig.update_xaxes(constrain="domain")
    fig.update_layout(title=title, xaxis_title="Component 1", yaxis_title="Component 2")

    # Two-column layout: plot (left) + side image (right)
    html_prefix = """
<div style="display:flex; gap:16px; align-items:flex-start;">
  <div style="flex:2; min-width:400px;">
"""
    html_suffix = """
  </div>
  <div id="sidepanel" style="flex:1; border-left:1px solid #eee; padding-left:12px; position:sticky; top:8px; max-width:520px;">
    <h3>Preview</h3>
    <div id="info" style="font-family:monospace; margin-bottom:8px; white-space:pre-wrap;"></div>
    <img id="img" style="max-width:100%; border:1px solid #ddd; border-radius:8px;" />
    <p style="opacity:0.7;">Hover or click a point to update.</p>
  </div>
</div>
<script>
(function(){
  const gd = document.querySelector('div.js-plotly-plot');
  const info = document.getElementById('info');
  const img = document.getElementById('img');
  function setPreview(pt){
    if(!pt) return;
    const cd = pt.customdata;
    const klass = cd[0], infoText = cd[1], imgB64 = cd[2];
    info.textContent = infoText + " (class=" + klass + ")";
    if(imgB64 && String(imgB64).startsWith('data:image/')) img.src = imgB64; else img.removeAttribute('src');
  }
  gd.on('plotly_hover', function(ev){ if(ev && ev.points && ev.points[0]) setPreview(ev.points[0]); });
  gd.on('plotly_click', function(ev){ if(ev && ev.points && ev.points[0]) setPreview(ev.points[0]); });
})();
</script>
"""
    os.makedirs(os.path.dirname(out_html) or ".", exist_ok=True)
    # Inline plotly.js so the HTML is fully offline/self-contained
    fig_html = plot_offline(fig, include_plotlyjs=True, output_type='div')
    full_html = html_prefix + fig_html + html_suffix
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"[OK] Wrote {out_html} (open it in any browser)")
    return out_html


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, help="Lightning checkpoint (ModelWrapper)")
    parser.add_argument("--dataset_paths", type=str, nargs="*", default=[], help="One or more RLDB folders")
    parser.add_argument("--embodiment", type=str, default="aria_bimanual")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--batches_per_class", type=int, default=64)
    parser.add_argument("--method", type=str, choices=["tsne", "umap"], default="tsne")
    parser.add_argument("--viz_cam_key", type=str, default=None,
                        help="Camera key to visualize (default: first in algo.camera_keys)")
    parser.add_argument("--save_bundle", type=str, default=None, help="Optional .pt to save coords+meta")
    parser.add_argument("--bundle_pt", type=str, default=None, help="Load precomputed bundle instead of recomputing")
    parser.add_argument("--out_html", type=str, default=None, help="If set, export a self-contained HTML instead of running a server")
    parser.add_argument("--title", type=str, default="Embedding Viewer", help="Title for the visualization")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()

    if args.bundle_pt:
        print(f"[INFO] Loading bundle from {args.bundle_pt}")
        obj = torch.load(args.bundle_pt, map_location="cpu")
        Z, y, names, meta_by_class = obj["Z"], obj["y"], obj["names"], obj["meta_by_class"]
    else:
        if not (args.ckpt and args.dataset_paths):
            raise SystemExit("Provide --bundle_pt OR (--ckpt and --dataset_paths ...)")
        print("[INFO] Instantiating model...")
        model = instantiate_model(args.ckpt)
        print("[INFO] Extracting embeddings + images...")
        feats_by_class, meta_by_class = extract_feats_and_images(
            model,
            dataset_paths=args.dataset_paths,
            embodiment_name=args.embodiment,
            batch_size=args.batch_size,
            num_batches_per_class=args.batches_per_class,
            num_workers=4,
            viz_cam_key=args.viz_cam_key,
        )
        print("[INFO] Reducing to 2D with reduce_many(...)")
        Z, y, names = reduce_many(feats_by_class, method=args.method, pool="avg", random_state=42)

        if args.save_bundle:
            out_pt = args.save_bundle if args.save_bundle.endswith(".pt") else (args.save_bundle + ".pt")
            print(f"[INFO] Saving bundle to {out_pt}")
            torch.save({"Z": Z, "y": y, "names": names, "meta_by_class": meta_by_class}, out_pt)

    # Export HTML (no server)
    if args.out_html:
        export_html(Z, y, names, meta_by_class, args.out_html, title=args.title)
        return

    # Dash server
    print("[INFO] Launching app...")
    app = make_app(Z, y, names, meta_by_class, title=args.title)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
