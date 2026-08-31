from pathlib import Path
import json

import pandas as pd

MANIFEST = Path("artifacts/hackathon_manifest.csv")
EMBED_DIR = Path("artifacts/clip_v1")
OUT = Path("artifacts/episode_embeddings.parquet")

manifest = pd.read_csv(MANIFEST)

records = []
for path in EMBED_DIR.rglob("*.json"):
    try:
        item = json.loads(path.read_text())
    except Exception:
        continue
    if item.get("status") != "complete":
        continue
    if not item.get("embedding"):
        continue
    records.append(
        {
            "episode_hash": item["episode_id"],
            "visual_embedding": item["embedding"],
            "frames_sampled": item.get("frames_sampled"),
            "decoded_frame_count": item.get("frame_count"),
            "fps": item.get("fps"),
            "embedding_model": item.get("embedding_model"),
            "embedding_version": item.get("embedding_version"),
        }
    )

embeddings = pd.DataFrame(records)

keep = [
    c for c in [
        "episode_hash", "lab", "task", "task_description",
        "scene", "objects", "embodiment", "num_frames",
    ]
    if c in manifest.columns
]

out = manifest[keep].merge(embeddings, on="episode_hash", how="inner")
out.to_parquet(OUT, index=False)

print(f"Manifest episodes: {len(manifest)}")
print(f"Complete embeddings: {len(embeddings)}")
print(f"Merged rows: {len(out)}")
print(f"Task count: {out['task'].nunique()}")
print(f"Embedding dimensions: {len(out.iloc[0]['visual_embedding']) if len(out) else 0}")
print(f"Wrote: {OUT}")
