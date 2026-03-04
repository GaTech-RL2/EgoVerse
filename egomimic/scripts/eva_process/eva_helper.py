from pathlib import Path
from types import SimpleNamespace

from egomimic.scripts.eva_process.eva_to_zarr import main as zarr_main

def zarr_job(
    *,
    raw_path: str | Path,
    output_dir: str | Path,
    dataset_name: str,
    arm: str,
    extrinsics_key: str = "x5Dec13_2",
    fps: int = 30,
    description: str = "",
    chunk_timesteps: int = 100,
    image_compressed: bool = False,
    save_mp4: bool = True,
) -> None:

    args = SimpleNamespace(
        raw_path=raw_path,
        output_dir=output_dir,
        dataset_name=dataset_name,
        arm=arm,
        extrinsics_key=extrinsics_key,
        fps=fps,
        description=description,
        chunk_timesteps=chunk_timesteps,
        image_compressed=image_compressed,
        save_mp4=save_mp4,
    )

    zarr_path, mp4_path = zarr_main(args)
    return zarr_path, mp4_path