from pathlib import Path
import json
import os

import modal

app = modal.App("egoverse-embed")

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
        "pyarrow==18.1.0",
        "boto3",
    )
)

MODEL_ID = "openai/clip-vit-base-patch32"
FRAMES_PER_VIDEO = 8
DATA_ROOT = "/data/input/egoverse_mp4s"
OUT_ROOT = "/data/output/clip_v1"


@app.function(
    image=image,
    gpu="T4",
    timeout=20 * 60,
    retries=2,
    volumes={"/data": volume},
)
def unpack():
    import tarfile

    if os.path.isdir(DATA_ROOT) and os.listdir(DATA_ROOT):
        return {"status": "already_unpacked"}

    os.makedirs(DATA_ROOT, exist_ok=True)
    with tarfile.open("/data/input/egoverse_mp4s.tar", "r") as archive:
        archive.extractall("/data/input")
    volume.commit()
    return {"status": "unpacked"}


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

    indexes = np.linspace(0, frame_count - 1, FRAMES_PER_VIDEO).astype(int)
    wanted = set(indexes.tolist())
    frames = []
    frame_i = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_i in wanted:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_i += 1
        if len(frames) == len(indexes):
            break
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
        "embedding_version": "clip_v1_8frames",
    }

    tmp = f"{output_path}.tmp"
    with open(tmp, "w") as f:
        json.dump(result, f)
    os.replace(tmp, output_path)
    volume.commit()
    return result

@app.function(
    image=image,
    timeout=20 * 60,
    retries=2,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("egoverse-r2")],
)
def make_contact_sheet(video_uri: str, frames: int = 12) -> dict:
    import boto3
    import cv2
    import numpy as np
    from botocore.config import Config
    from urllib.parse import urlparse

    out_dir = "/data/output/foldclothes_contact_sheets"
    tmp_dir = "/tmp/foldclothes_mp4s"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    parsed = urlparse(video_uri)
    if parsed.scheme != "s3" or parsed.netloc != "rldb":
        raise ValueError(f"Expected an rldb S3 URI; got {video_uri[:32]!r}")

    episode_id = Path(parsed.path).stem
    output_path = f"{out_dir}/{episode_id}.jpg"
    temp_video = f"{tmp_dir}/{episode_id}.mp4"

    if os.path.exists(output_path):
        return {
            "episode_id": episode_id,
            "status": "exists",
            "remote_path": output_path,
        }

    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=os.environ["R2_ENDPOINT_URL"],
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            region_name="auto",
            config=Config(signature_version="s3v4"),
        )
        s3.download_file("rldb", parsed.path.lstrip("/"), temp_video)

        cap = cv2.VideoCapture(temp_video)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_count <= 0:
            cap.release()
            return {
                "episode_id": episode_id,
                "status": "failed",
                "error": "no_decodable_frames",
            }

        indexes = np.linspace(0, frame_count - 1, frames).astype(int)
        wanted = set(indexes.tolist())
        selected = []
        frame_index = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index in wanted:
                height, width = frame.shape[:2]
                output_width = 320
                output_height = max(1, round(height * output_width / width))
                selected.append(cv2.resize(frame, (output_width, output_height)))

            frame_index += 1
            if len(selected) == len(indexes):
                break

        cap.release()

        if not selected:
            return {
                "episode_id": episode_id,
                "status": "failed",
                "error": "frame_read_failed",
            }

        while len(selected) < frames:
            selected.append(selected[-1].copy())

        sheet_rows = []
        for start in range(0, frames, 4):
            sheet_rows.append(np.hstack(selected[start : start + 4]))

        sheet = np.vstack(sheet_rows)
        cv2.imwrite(
            output_path,
            sheet,
            [int(cv2.IMWRITE_JPEG_QUALITY), 82],
        )
        volume.commit()

        return {
            "episode_id": episode_id,
            "status": "complete",
            "remote_path": output_path,
            "frame_count": frame_count,
        }

    finally:
        if os.path.exists(temp_video):
            os.remove(temp_video)

@app.local_entrypoint()
def main(limit: int = 400):
    unpack.remote()

    from pathlib import Path as LocalPath

    local_manifest = LocalPath("artifacts/hackathon_manifest.csv")
    if not local_manifest.exists():
        raise FileNotFoundError(
            "Run from the repository root with artifacts/hackathon_manifest.csv available."
        )

    import pandas as pd
    df = pd.read_csv(local_manifest)
    filenames = [f"{episode_hash}.mp4" for episode_hash in df["episode_hash"]]
    filenames = filenames[:limit]

    for i, result in enumerate(embed_video.map(filenames), start=1):
        print(f"{i}/{len(filenames)}: {result['episode_id']} — {result['status']}")

@app.local_entrypoint()
def contact_sheets():
    from pathlib import Path as LocalPath
    import pandas as pd

    manifest = LocalPath("artifacts/foldclothes-v1/manifests/candidate_pool.csv")
    if not manifest.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest}")

    df = pd.read_csv(manifest)

    audit_parts = []
    for task_name in sorted(df["task"].dropna().unique()):
        task_rows = df.loc[df["task"] == task_name]
        audit_parts.append(
            task_rows.sample(n=min(4, len(task_rows)), random_state=42)
        )

    audit = pd.concat(audit_parts, ignore_index=True)
    video_uris = audit["zarr_mp4_path"].tolist()

    print("\nAudit counts by task:")
    print(audit["task"].value_counts().sort_index().to_string())

    for i, result in enumerate(make_contact_sheet.map(video_uris), start=1):
        print(f"{i}/{len(video_uris)}: {result['episode_id']} — {result['status']}")
        print(f"  Remote: {result.get('remote_path', result.get('error'))}")