import json
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from egomimic.utils.aws.aws_data_utils import get_boto3_s3_client

MANIFEST = Path("artifacts/foldclothes-v1/manifests/quicktime-five.json")
OUT_DIR = Path("artifacts/foldclothes-v1/contact-sheets")
TMP_DIR = Path("artifacts/foldclothes-v1/tmp-video")
BUCKET = "rldb"
FRAMES = 12
WIDTH = 960

def s3_key(uri: str) -> str:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or parsed.netloc != BUCKET:
        raise ValueError(f"Unexpected S3 URI bucket/scheme for {uri[:24]!r}")
    return parsed.path.lstrip("/")

def make_sheet(video: Path, output: Path) -> None:
    vf = (
        f"fps={FRAMES}/duration,"
        f"scale={WIDTH}:-2:force_original_aspect_ratio=decrease,"
        "tile=4x3:padding=8:margin=8"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video), "-vf", vf, "-frames:v", "1", str(output)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

def main():
    rows = json.loads(MANIFEST.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    s3 = get_boto3_s3_client()

    for i, row in enumerate(rows, start=1):
        episode_id = row["episode_hash"]
        output = OUT_DIR / f"{i:02d}_{episode_id}.jpg"
        video = TMP_DIR / f"{episode_id}.mp4"

        if output.exists() and output.stat().st_size > 0:
            print(f"SKIP {output.name}")
            continue

        try:
            print(f"DOWNLOADING {episode_id}")
            s3.download_file(BUCKET, s3_key(row["zarr_mp4_path"]), str(video))
            print(f"RENDERING    {output.name}")
            make_sheet(video, output)
            print(f"DONE         {output.name} ({output.stat().st_size / 1e6:.1f} MB)")
        finally:
            if video.exists():
                video.unlink()

    try:
        TMP_DIR.rmdir()
    except OSError:
        pass

    print(f"\nReady: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
