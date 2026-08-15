from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from egomimic.utils.aws.aws_data_utils import get_boto3_s3_client

MANIFEST = Path("artifacts/smoke_test_manifest.csv")
OUT_DIR = Path("data/smoke_mp4s")
BUCKET = "rldb"

def s3_key(uri: str) -> str:
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Expected s3:// URI, got {uri!r}")
    if parsed.netloc != BUCKET:
        raise ValueError(f"Expected bucket {BUCKET!r}, got {parsed.netloc!r}")
    return parsed.path.lstrip("/")

def main():
    df = pd.read_csv(MANIFEST)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s3 = get_boto3_s3_client()

    for row in df.itertuples(index=False):
        uri = getattr(row, "zarr_mp4_path")
        key = s3_key(uri)
        dest = OUT_DIR / f"{getattr(row, 'episode_hash')}.mp4"

        if dest.exists() and dest.stat().st_size > 0:
            print(f"SKIP {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
            continue

        print(f"DOWNLOADING {getattr(row, 'episode_hash')} → {dest.name}")
        s3.download_file(BUCKET, key, str(dest))
        print(f"DONE {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")

if __name__ == "__main__":
    main()
