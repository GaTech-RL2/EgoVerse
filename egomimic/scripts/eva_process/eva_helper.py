from pathlib import Path
from types import SimpleNamespace

from egomimic.scripts.eva_process.eva_to_zarr import main as zarr_main
from egomimic.scripts.eva_process.eva_to_lerobot import main as lerobot_main

def lerobot_job(
    *,
    raw_path: str | Path,
    output_dir: str | Path,
    dataset_name: str,
    arm: str,
    description: str = "",
    extrinsics_key: str = "x5Dec13_2",
) -> None:

    raw_path = Path(raw_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()

    args = SimpleNamespace(
        name=dataset_name,
        raw_path=raw_path,
        dataset_repo_id=f"rpuns/{dataset_name}",
        fps=30,
        arm=arm,
        extrinsics_key=extrinsics_key,
        description=description,
        image_compressed=False,
        video_encoding=False,
        prestack=True,
        nproc=12,
        nthreads=2,
        output_dir=output_dir,
        push=False,
        private=False,
        license="apache-2.0",
        debug=False,
        save_mp4=True,
    )

    lerobot_main(args)


def zarr_job(
    *,
    raw_path: str | Path,
    output_dir: str | Path,
    dataset_name: str,
    arm: str,
    description: str = "",
    extrinsics_key: str = "x5Dec13_2",
    chunk_timesteps: int = 100,
) -> None:

    raw_path = Path(raw_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()

    args = SimpleNamespace(
        raw_path=raw_path,
        output_dir=output_dir,
        name=dataset_name,
        fps=30,
        arm=arm,
        extrinsics_key=extrinsics_key,
        description=description,
        image_compressed=False,
        prestack=False,
        debug=False,
        save_mp4=True,
        chunk_timesteps=chunk_timesteps,
    )

    zarr_main(args)
