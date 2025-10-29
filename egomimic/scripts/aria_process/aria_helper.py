from pathlib import Path
from types import SimpleNamespace

# import the real entry-point once
from egomimic.scripts.aria_process.aria_to_lerobot import main as aria_main


def lerobot_job(
    *,
    raw_path: str | Path,
    output_dir: str | Path,
    dataset_name: str,
    arm: str,
    description: str = "",
) -> None:
    """
    Convert one <vrs, vrs.json, mps_*> trio to a LeRobot dataset.

    Only the five arguments below are variable; everything else is fixed.
    """
    raw_path   = Path(raw_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()

    args = SimpleNamespace(
        raw_path       = raw_path,
        output_dir     = output_dir,
        name           = dataset_name,
        arm            = arm,
        description    = description,

        # hard-wired defaults you specified
        dataset_repo_id = f"rpuns/{dataset_name}",
        fps             = 30,
        video_encoding  = False,
        push            = False,
        prestack        = True,
        image_compressed= False,
        save_mp4        = True,
        private   = False,
        license   = "apache-2.0",
        nproc     = 10,
        nthreads  = 2,
        debug     = False,
    )

    aria_main(args)
