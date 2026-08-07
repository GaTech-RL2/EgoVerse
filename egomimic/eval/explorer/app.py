"""Interactive chunkviz explorer (Streamlit) — strip + linked PCA + trajectory.

Reads ``chunkviz_data.npz`` from export.py. Click anywhere on the strip image to
jump to that frame; trajectory image, PCA dot and cursor all follow.

    pip install streamlit plotly numpy pillow streamlit-image-coordinates
    streamlit run app.py
"""
import io
import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    HAVE_CLICK = True
except Exception:
    HAVE_CLICK = False

PALETTE = np.array(
    [
        [31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40],
        [148, 103, 189], [140, 86, 75], [227, 119, 194], [23, 190, 207],
        [188, 189, 34], [0, 0, 0],
    ],
    dtype=np.uint8,
)

st.set_page_config(page_title="Chunkviz Explorer", layout="wide")


@st.cache_data
def load(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


st.sidebar.title("Chunkviz Explorer")
data_path = st.sidebar.text_input("data .npz", "chunkviz_data.npz")
try:
    data = load(data_path)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load {data_path}: {exc}")
    st.stop()

emb_ids = [int(x) for x in data["emb_ids"]]
emb = st.sidebar.selectbox("embodiment", emb_ids)
nep = int(data[f"emb{emb}_n_episodes"])
ep = st.sidebar.selectbox("episode", list(range(nep)))
layer = st.sidebar.radio("strip layer", ["colored chunks", "P(boundary)", "crisp boundary"])

cid = data[f"emb{emb}_ep{ep}_cid"]
prob = data[f"emb{emb}_ep{ep}_prob"]
crisp = data[f"emb{emb}_ep{ep}_crisp"]
topcid = data[f"emb{emb}_ep{ep}_topcid"]
pca = data[f"emb{emb}_pca_xy"]
labels = [str(x) for x in data[f"emb{emb}_labels"]]
traj = data.get(f"emb{emb}_ep{ep}_traj")
n_stages, T = cid.shape

sig = (emb, ep)
if st.session_state.get("_sig") != sig:
    st.session_state._sig = sig
    st.session_state._last_click_xy = None
if "frame" not in st.session_state:
    st.session_state.frame = 0
st.session_state.frame = int(np.clip(st.session_state.frame, 0, T - 1))

cprev, cnext, crst = st.sidebar.columns(3)
if cprev.button("◀ prev"):
    st.session_state.frame -= 1
if cnext.button("next ▶"):
    st.session_state.frame += 1
if crst.button("⟲ 0"):
    st.session_state.frame = 0
st.session_state.frame = int(np.clip(st.session_state.frame, 0, T - 1))

play = st.sidebar.checkbox("▶ play")
fps = st.sidebar.slider("fps", 2, 30, 12)
frame = st.sidebar.slider("frame", 0, T - 1, value=int(st.session_state.frame))
st.session_state.frame = frame

# ---- top row: trajectory + PCA ----
col_traj, col_pca = st.columns([1, 1])
with col_traj:
    st.markdown(f"**Trajectory** — emb {emb}, ep {ep}, frame {frame}/{T-1}")
    if traj is not None and len(traj) > 0:
        fi = min(frame, len(traj) - 1)
        st.image(Image.open(io.BytesIO(traj[fi])), use_container_width=True)
    else:
        st.info("no trajectory frames in this data file")

with col_pca:
    cur_chunk = int(topcid[frame])
    ep_chunks = np.unique(topcid)
    trail = list(dict.fromkeys(int(topcid[f]) for f in range(max(0, frame - 40), frame + 1)))
    figP = go.Figure()
    figP.add_trace(go.Scattergl(x=pca[:, 0], y=pca[:, 1], mode="markers",
                                marker=dict(size=3, color="lightgrey"), name="all chunks"))
    figP.add_trace(go.Scattergl(x=pca[ep_chunks, 0], y=pca[ep_chunks, 1], mode="markers",
                                marker=dict(size=5, color="steelblue"), name="this episode"))
    figP.add_trace(go.Scattergl(x=pca[trail, 0], y=pca[trail, 1], mode="lines+markers",
                                line=dict(color="orange"), marker=dict(size=5, color="orange"),
                                name="recent"))
    figP.add_trace(go.Scattergl(x=[pca[cur_chunk, 0]], y=[pca[cur_chunk, 1]], mode="markers",
                                marker=dict(size=15, color="red"), name=f"chunk {cur_chunk}"))
    figP.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10),
                       title=f"PCA chunk tokens — chunk {cur_chunk}", legend=dict(orientation="h"))
    st.plotly_chart(figP, use_container_width=True, config={"scrollZoom": True})

# ---- strip image (clickable) ----
BAND = 34
LAB_W = 96


def build_strip():
    """Return (rgb image with labels + cursor, lab_w, kx, strip_w_px)."""
    rows = []
    red = np.array([220, 30, 30], np.uint8)
    white = np.array([255, 255, 255], np.uint8)
    for s in range(n_stages):
        if layer == "colored chunks":
            row = PALETTE[cid[s] % len(PALETTE)]
        elif layer == "P(boundary)":
            g = (255 * (1 - np.clip(prob[s], 0, 1))).astype(np.uint8)
            row = np.stack([g, g, g], 1)
        else:
            row = np.where(crisp[s][:, None], red, white).astype(np.uint8)
        rows.append(np.broadcast_to(row[None], (BAND, T, 3)))
        rows.append(np.full((3, T, 3), 40, np.uint8))
    strip = np.concatenate(rows[:-1], 0)               # (H, T, 3)
    H = strip.shape[0]
    kx = max(1, round(1180 / T))
    wide = np.repeat(strip, kx, axis=1)                # (H, T*kx, 3)
    strip_w = wide.shape[1]
    # cursor
    cur_x = int(frame * kx)
    wide[:, max(0, cur_x - 1):cur_x + 2, :] = (255, 215, 0)
    # left label column
    lab = np.full((H, LAB_W, 3), 245, np.uint8)
    pil = Image.fromarray(lab)
    d = ImageDraw.Draw(pil)
    for s, name in enumerate(labels):
        y = s * (BAND + 3) + BAND // 2 - 6
        d.text((6, y), name, fill=(20, 20, 20))
    full = np.concatenate([np.array(pil), wide], axis=1)
    return full, LAB_W, kx, strip_w


strip_img, lab_w, kx, strip_w = build_strip()
st.markdown(f"### Boundary strip — click to jump  (frame {frame}/{T-1})")
if HAVE_CLICK:
    coords = streamlit_image_coordinates(strip_img, key=f"strip_{emb}_{ep}_{layer}")
    if coords is not None:
        cxy = (coords["x"], coords["y"])
        if cxy != st.session_state.get("_last_click_xy"):
            st.session_state._last_click_xy = cxy
            xrel = coords["x"] - lab_w
            if xrel >= 0:
                nf = int(np.clip(round(xrel / max(1, strip_w) * (T - 1)), 0, T - 1))
                st.session_state.frame = nf
                st.rerun()
    st.caption("👆 click the colored strip (right of the labels) to jump to that frame")
else:
    st.image(strip_img, use_container_width=True)
    st.caption("(install streamlit-image-coordinates for click-to-seek; use the slider meanwhile)")

st.caption(
    "Click the strip (or drag the slider / press play) to move through the episode. "
    "Trajectory, PCA dot, and cursor all follow the same frame."
)

if play:
    st.session_state.frame = (frame + 1) % T
    time.sleep(1.0 / fps)
    st.rerun()
