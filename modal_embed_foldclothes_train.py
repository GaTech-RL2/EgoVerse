from pathlib import Path
import json
import os

import modal

app = modal.App("egoverse-foldclothes-train-embed")

volume = modal.Volume.from_name("egoverse-hackathon-data", create_if_missing=True)

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
        "boto3",
    )
)

MODEL_ID = "openai/clip-vit-base-patch32"
FRAMES_PER_VIDEO = 8
OUT_ROOT = "/data/output/foldclothes-train-clip-v1"


@app.function(
    image=image,
    gpu="T4",
    timeout=20 * 60,
    retries=2,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("egoverse-r2")],
)
def embed_video(episode_id: str, video_uri: str) -> dict:
    import boto3
    from botocore.config import Config
    import cv2
    import numpy as np
    import torch
    from transformers import CLIPModel, CLIPProcessor
    from urllib.parse import urlparse

    os.makedirs(OUT_ROOT, exist_ok=True)
    output_path = f"{OUT_ROOT}/{episode_id}.json"

    if os.path.exists(output_path):
        with open(output_path) as f:
            return json.load(f)

    parsed = urlparse(video_uri)
    if parsed.scheme != "s3" or parsed.netloc != "rldb":
        raise ValueError(f"Expected s3://rldb URI, got {video_uri!r}")

    tmp_path = f"/tmp/{episode_id}.mp4"

    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=os.environ["R2_ENDPOINT_URL"],
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            region_name="auto",
            config=Config(signature_version="s3v4"),
        )
        s3.download_file("rldb", parsed.path.lstrip("/"), tmp_path)

        cap = cv2.VideoCapture(tmp_path)
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
            "embedding_version": "foldclothes_train_clip_v1_8frames",
            "source_uri": video_uri,
        }

        tmp_out = f"{output_path}.tmp"
        with open(tmp_out, "w") as f:
            json.dump(result, f)
        os.replace(tmp_out, output_path)
        volume.commit()
        return result

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.local_entrypoint()
def main(limit: int = 1304):
    import pandas as pd

    manifest = Path("artifacts/foldclothes-v1/manifests/train_embedding_manifest.csv")
    df = pd.read_csv(manifest).sort_values("episode_hash").head(limit)

    jobs = list(zip(df["episode_hash"].tolist(), df["zarr_mp4_path"].tolist()))
    print(f"Embedding {len(jobs)} training episodes with {MODEL_ID}.")

    for i, result in enumerate(embed_video.starmap(jobs), start=1):
        print(f"{i}/{len(jobs)}: {result['episode_id']} — {result['status']}")
