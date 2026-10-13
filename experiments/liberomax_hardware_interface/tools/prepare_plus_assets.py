"""Install the official, checksum-pinned Plus asset pack in a fresh checkout."""

import argparse
import json
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

from astra_reversal.hardware_interface.common import file_hash

REPOSITORY = "Sylvest/LIBERO-plus"
REVISION = "dd2bd61b7d9a6fef1abc52d606e983b41886a149"
SHA256 = "96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf"
BYTES = 6395849578
# The published archive embeds this build-machine path before assets/.
ARCHIVE_PREFIX = PurePosixPath(
    "inspire/hdd/project/embodied-multimodality/public/syfei/libero_new/"
    "release/dataset/LIBERO-plus-0"
)


def asset_members(archive):
    members, seen = [], set()
    for member in archive.infolist():
        relative = PurePosixPath(member.filename)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or stat.S_ISLNK(member.external_attr >> 16)
        ):
            raise ValueError("asset_archive_member_outside_assets")
        try:
            relative = relative.relative_to(ARCHIVE_PREFIX)
        except ValueError:
            pass
        if not relative.parts or relative.parts[0] != "assets":
            raise ValueError("asset_archive_member_outside_assets")
        if relative in seen:
            raise ValueError("duplicate_asset_archive_destination")
        seen.add(relative)
        members.append((member, relative))
    return members


def prepare(root, receipt):
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            REPOSITORY, "assets.zip", repo_type="dataset", revision=REVISION
        )
    )
    if path.stat().st_size != BYTES or file_hash(path) != SHA256:
        raise ValueError("official_plus_asset_checksum_mismatch")
    destination = Path(root) / "libero/libero"
    with zipfile.ZipFile(path) as archive:
        members = asset_members(archive)
        # This fresh workflow checkout has only ignored runtime assets here.
        # Pinned source, BDDL files and initial-state files remain untouched.
        for member, relative in members:
            target = destination / relative
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    required = (
        destination
        / "assets/scenes/kitchen250/kitchen_table_FabricCarpet001_REFL_3K.xml"
    )
    if not required.is_file():
        raise ValueError("official_plus_asset_pack_missing_required_scene")
    result = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "archive_sha256": SHA256,
        "archive_bytes": BYTES,
        "archive_prefix": str(ARCHIVE_PREFIX),
        "members": len(members),
        "required_scene_sha256": file_hash(required),
        "model_calls": 0,
        "scored": False,
    }
    Path(receipt).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source_root, args.receipt)
