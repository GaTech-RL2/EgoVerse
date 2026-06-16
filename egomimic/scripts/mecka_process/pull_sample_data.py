import argparse
import json
import os

import requests


def download_files(data, output_dir, per_episode=False):
    os.makedirs(output_dir, exist_ok=True)

    for entry in data:
        episode_id = entry.get("id", "unknown_id")
        urls = entry.get("urls", {})

        target_dir = os.path.join(output_dir, episode_id) if per_episode else output_dir
        os.makedirs(target_dir, exist_ok=True)

        for key, url in urls.items():
            base_url = url.split("?")[0]
            ext = os.path.splitext(base_url)[1] or ".bin"
            filename = f"{key}{ext}" if per_episode else f"{episode_id}_{key}{ext}"
            filepath = os.path.join(target_dir, filename)

            print(f"⬇️  Downloading {key} → {filepath}")
            try:
                with requests.get(url, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    with open(filepath, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
            except Exception as e:
                print(f"❌ Failed to download {key}: {e}")

    print(f"\n✅ All available files saved in: {os.path.abspath(output_dir)}")


def main():
    parser = argparse.ArgumentParser(
        description="Download sample data files from JSON metadata."
    )
    parser.add_argument(
        "--json-path",
        required=True,
        type=str,
        help="Path to the JSON file containing episode metadata.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Directory where downloaded files will be saved.",
    )
    parser.add_argument(
        "--per-episode",
        action="store_true",
        help="Save files into subfolders named by episode ID.",
    )

    args = parser.parse_args()

    with open(args.json_path, "r") as f:
        data = json.load(f)

    download_files(data, args.output_dir, args.per_episode)


if __name__ == "__main__":
    main()
