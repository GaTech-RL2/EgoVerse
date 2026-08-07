# Chunkviz Explorer

Interactive explorer for the H-Net dynamic-chunking boundaries + the highest-stage
PCA tokens. Two parts:

- **`export.py`** — runs on the cluster (needs the model). Loads a ckpt, runs the
  boundary-strip + PCA probes, and dumps a portable `chunkviz_data.npz`.
- **`app.py`** — runs locally (no GPU / no model). Streamlit UI: zoomable, linked
  boundary strip + PCA, with embodiment/episode selectors, strip-layer toggles
  (colored chunks / P(boundary) / crisp), and play/scrub controls.

## Usage

On the cluster (e.g. PACE), from the repo root:

    PYTHONPATH=. python -m egomimic.eval.explorer.export \
        --ckpt <ckpt> --config-path <run>/.hydra/config.yaml \
        --out chunkviz_data.npz --n-episodes 6

Copy `chunkviz_data.npz` to your laptop, then locally:

    pip install streamlit plotly numpy
    streamlit run app.py            # expects chunkviz_data.npz in the cwd
    # or: streamlit run app.py  and set the data path in the sidebar

## Notes

- The PCA shows the highest-stage (most-compressed) `ComputeStage` tokens — one
  point per top-level chunk. The red dot **holds for a chunk's duration and hops
  at a new chunk** (it does not slide per-frame). This relies on the
  `_find_inner_main_network` + frame-resolution `boundary_mask` fixes in
  `egomimic/eval/probes/eval_pca_tokens.py`.
- Strip rows are the chunker stages (Stage 0 = frame-level, Stage 1 = inner). In
  the "colored chunks" layer the color flips at every true boundary.
- v2 ideas: link a trajectory frame to the slider; compare multiple ckpts.

## Standalone HTML viewer + SSH tunnel (recommended UI, 2026-06-15)

A polished, dependency-free alternative to the Streamlit `app.py`: one
self-contained `chunkviz.html` (dark theme, canvas video + multi-track boundary
timeline with draggable playhead + clickable chunk list + linked PCA). Frame ↔
playhead ↔ PCA sync is per-frame-exact (canvas frame-flipper, not <video>).

Build (after `export.py` writes the npz) + serve, on a PACE login node:

    bash egomimic/eval/explorer/serve.sh [npz] [port]   # default fixsweep/chunkviz_data.npz : 8765

It prints the exact tunnel command. On your laptop:

    ssh -L 8765:<login-host>:8765 pacerh9      # forwards to the SPECIFIC node (pacerh9 round-robins!)
    open http://localhost:8765/chunkviz.html

`build_html.py` alone (numpy-only, no torch/GPU/streamlit) turns any
`chunkviz_data.npz` into the html — runs locally too if you copy the npz down.
`--frame-stride N` shrinks the file (subsamples frames + strip columns together).

## Compare / alignment mode (chunkviz 2, 2026-06-20)

The HTML viewer now has a **single | compare** toggle in the header.

- **Single** (default, unchanged): one episode — rollout video, boundary timeline
  (per-stage), clickable chunk list (with a stage dropdown), per-emb PCA.
- **Compare (alignment):** pick episode **A** and episode **B** (each from ANY
  embodiment), shown **side by side** — each with its own rollout + high-level
  chunk strip — over **one SHARED PCA** (grey = all chunks from all episodes/
  embodiments; A blue, B orange; faint = full path, bold = up to current phase;
  ringed dot = current). A single **phase slider** (0→1) + play scrubs BOTH
  episodes together (each maps phase→its own frame length), so you watch whether
  aligned episodes trace similar paths through the shared trunk space — the
  primary cross-embodiment alignment probe.

**Requires a cross-emb export.** `export.py` now also writes a **shared PCA
basis** (`pca_shared_xy` — ONE SVD over all embodiments' inner-trunk tokens,
since cotrain embodiments share the ComputeStage) plus per-emb `pca_offset`
(chunk id c of emb e → shared row `pca_offset[e]+c`). Per-emb `pca_xy` is kept
for the single view. Old single-emb npz still load — compare is auto-disabled
(greyed) when no shared PCA is present. To get compare, export from a **cotrain**
ckpt (≥2 embodiments).

"Aligned" pairs are user-selected (A + B dropdowns) — no pairing metadata needed;
pick the two episodes you consider aligned. (A future add could auto-populate
known pairs if pairing metadata exists.)
