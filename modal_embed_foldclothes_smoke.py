from pathlib import Path
import json
import os

import modal

app = modal.App("egoverse-foldclothes-smoke-embed")

volume = modal.Volume.from_name(
    "egoverse-hackathon-data",
    create_if_missing=True,
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "transformers==4.46.3",
        "opencv-python-headless==4.10.0.84",
        "numpy==1.26.4",
        "pandas==2.2.3",
    )
)

MODEL_ID = "openai/clip-vit-base-patch32"
FRAMES_PER_VIDEO = 8
DATA_ROOT = "/data/input/foldclothes-v1/smoke_mp4s/smoke_mp4s"
OUT_ROOT = "/data/output/foldclothes-smoke-clip-v1"


@app.function(
    image=image,
    gpu="T4",
    timeout=20 * 60,
    retries=2,
    volumes={"/data": volume},
)
def embed_video(filename: str) -> dict:
    import cv2
    import numpy as np
    import torch
    from transformers import CLIPModel, CLIPProcessor

    os.makedirs(OUT_ROOT, exist_ok=True)
    episode_id = Path(filename).stem
    output_path = f"{OUT_ROOT}/{episode_id}.json"

    if os.path.exists(output_path):
        with open(output_path) as f:
            return json.load(f)

    video_path = f"{DATA_ROOT}/{filename}"
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)

    if frame_count <= 0:
        cap.release()
        return {
            "episode_id": episode_id,
            "status": "failed",
            "error": "no_decodable_frames",
        }

    indexes = np.unique(
        np.linspace(0, frame_count - 1, FRAMES_PER_VIDEO, dtype=int)
    )

    frames = []
    for frame_index in indexes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

    if not frames:
        return {
            "episode_id": episode_id,
            "status": "failed",
            "error": "frame_read_failed",
        }

    model = CLIPModel.from_pretrained(MODEL_ID)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    model.eval()

    inputs = processor(images=frames, return_tensors="pt")
    with torch.no_grad():
        vectors = model.get_image_features(**inputs)
        vectors = vectors / vectors.norm(dim=-1, keepdim=True)
        embedding = vectors.mean(dim=0)
        embedding = embedding / embedding.norm()

    result = {
        "episode_id": episode_id,
        "status": "complete",
        "embedding": embedding.cpu().tolist(),
        "frames_sampled": len(frames),
        "frame_count": frame_count,
        "fps": fps,
        "embedding_model": MODEL_ID,
        "embedding_version": "foldclothes_smoke_clip_v1_8frames",
    }

    tmp = f"{output_path}.tmp"
    with open(tmp, "w") as f:
        json.dump(result, f)
    os.replace(tmp, output_path)
    volume.commit()
    return result


@app.local_entrypoint()
def main():
    import pandas as pd

    manifest = Path("artifacts/foldclothes-v1/manifests/smoke_manifest.csv")
    if not manifest.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest}")

    df = pd.read_csv(manifest)
    filenames = [f"{episode_hash}.mp4" for episode_hash in df["episode_hash"]]

    print(f"Embedding {len(filenames)} smoke videos with {MODEL_ID}.")
    for i, result in enumerate(embed_video.map(filenames), start=1):
        print(f"{i}/{len(filenames)}: {result['episode_id']} — {result['status']}")
