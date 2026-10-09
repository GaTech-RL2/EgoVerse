"""Download the pinned release's public assets, retaining exact archive hashes."""

import hashlib
import json
import shutil
import stat
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path("upstream-robocasa/robocasa/models/assets")
LINKS = json.loads((ROOT / "box_links/box_links_assets.json").read_text())
# Generative textures are disabled in the frozen design; no demonstrations.
REGISTRY = {
    "textures": ROOT,
    "fixtures_lightwheel": ROOT,
    "objaverse": ROOT / "objects",
    "aigen_objs": ROOT / "objects",
    "objects_lightwheel": ROOT / "objects",
}


def main():
    downloads = Path("asset-archives")
    downloads.mkdir(exist_ok=False)
    receipts = []
    for name, destination in REGISTRY.items():
        shared = LINKS[name]
        base = shared.split("/s/")[0]
        url = base + "/shared/static/" + shared.rstrip("/").split("/")[-1] + ".zip"
        target = downloads / (name + ".zip")
        print(json.dumps({"asset_download": name}), flush=True)
        for attempt in range(3):
            partial = downloads / f"{name}.attempt-{attempt}.zip"
            try:
                with (
                    urllib.request.urlopen(url, timeout=120) as source,
                    partial.open("xb") as output,
                ):
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                if not zipfile.is_zipfile(partial):
                    raise ValueError("asset_download_is_not_zip")
                partial.rename(target)
                break
            except Exception as error:
                print(
                    json.dumps(
                        {
                            "asset_download_failure": name,
                            "attempt": attempt,
                            "error_class": type(error).__name__,
                        }
                    ),
                    flush=True,
                )
                if attempt == 2:
                    raise
        value = hashlib.sha256()
        with target.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(chunk)
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target) as archive:
            for entry in archive.infolist():
                output = destination / entry.filename
                output.resolve().relative_to(destination.resolve())
                if stat.S_ISLNK(entry.external_attr >> 16):
                    raise ValueError("asset_archive_symlink")
            archive.extractall(destination)
        receipts.append(
            {
                "name": name,
                "url": url,
                "sha256": value.hexdigest(),
                "bytes": target.stat().st_size,
            }
        )
        print(
            json.dumps(
                {
                    "asset_ready": name,
                    "bytes": target.stat().st_size,
                    "sha256": value.hexdigest(),
                }
            ),
            flush=True,
        )
    path = Path("artifacts/runtime/asset-downloads.json")
    with path.open("x") as stream:
        json.dump(
            {
                "source": "pinned upstream box_links_assets.json",
                "license": "CC BY 4.0",
                "demonstrations_downloaded": False,
                "archives": receipts,
            },
            stream,
            indent=2,
        )


if __name__ == "__main__":
    main()
