import asyncio
import json
import tempfile
from pathlib import Path

from abstract_upload import Uploader
from projectaria_tools.core import data_provider


def _vrs_metadata(vrs_file: Path) -> dict:
    meta = data_provider.create_vrs_data_provider(str(vrs_file)).get_metadata()
    return {
        "device_serial": meta.device_serial,
        "recording_profile": meta.recording_profile,
        "start_time_epoch_sec": meta.start_time_epoch_sec,
        "source_filename": vrs_file.name,
        "device": "aria_gen2",
    }


class AriaGen2Uploader(Uploader):
    """Uploads Aria Gen 2 recordings (downloaded with `aria_gen2 recording download`).

    Gen 2 downloads have no .vrs.json companion, but SQL ingest expects
    <ts>.json next to <ts>.vrs, so one is written from the VRS metadata.
    run_conversion.py --embodiment aria_gen2 converts from MPS outputs
    (mps_<ts>_vrs/ next to the VRS, as for Gen 1) by default; with
    --gen2-source on_device it uses the VRS's own hand tracking / VIO / gaze
    and needs no MPS step.
    """

    def get_timestamp_name(self, file_path):
        # File birthtime is the download time; the VRS knows when it was recorded.
        if Path(file_path).suffix == ".vrs":
            return _vrs_metadata(Path(file_path))["start_time_epoch_sec"] * 1000
        return super().get_timestamp_name(file_path)


def aria_gen2_uploader(sidecar_dir: Path):
    """Uploader for Gen 2 recordings; <ts>.json sidecars are written to sidecar_dir."""

    def collect_files(local_dir):
        file_paths = []
        for vrs_file in sorted(local_dir.iterdir()):
            if vrs_file.suffix != ".vrs" or not vrs_file.is_file():
                continue
            json_file = sidecar_dir / f"{vrs_file.stem}.json"
            json_file.write_text(json.dumps(_vrs_metadata(vrs_file), indent=2))
            file_paths.append((vrs_file, json_file))
        return file_paths

    return AriaGen2Uploader(
        embodiment="aria_gen2",  # raw_v2/aria_gen2/
        datatype=".vrs",
        collect_files=collect_files,
        defaults={"rig_name": "aria_gen2"},
    )


def main():
    with tempfile.TemporaryDirectory(prefix="aria_gen2_json_") as sidecar_dir:
        uploader = aria_gen2_uploader(Path(sidecar_dir))
        asyncio.run(uploader.run())


if __name__ == "__main__":
    main()
