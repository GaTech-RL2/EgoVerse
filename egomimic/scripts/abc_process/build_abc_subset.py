"""
Build a frame-count-matched ABC-130k subset: download, convert, drop the raw MCAP.

Targets a number of FRAMES rather than a number of episodes. ABC episode lengths
vary by ~3x within a task, and roughly a third of `fold_and_stack_the_t_shirts`
episodes carry no EE pose at all and cannot be converted (see abc_to_zarr), so an
episode count would not land anywhere near a given frame budget. This keeps
converting until the budget is met and records what it used.

    python -m egomimic.scripts.abc_process.build_abc_subset \
        --task fold_and_stack_the_t_shirts \
        --target-frames 316068 \
        --out /coc/flash7/scratch/acheluva3/abc_data

Writes `<out>/zarr/<uuid>.zarr` per episode plus `<out>/manifest.json`. Re-running
is incremental: already-converted episodes count toward the budget and are skipped.
"""

import argparse
import json
import logging
import math
import os
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import zarr
from huggingface_hub import HfApi, hf_hub_download

from egomimic.scripts.abc_process.abc_to_zarr import convert_episode
from egomimic.scripts.abc_process.download_abc import REPO_ID, _tree, list_episodes

logger = logging.getLogger("build_abc_subset")

# egomimic/hydra_configs/data/mecka_foldclothes_*: embodiment=human_bimanual,
# lab=mecka, task=fold_clothes, not deleted, has a zarr -- 1607 episodes.
MECKA_FOLDCLOTHES_FRAMES = 316_068

# fold_and_stack_the_t_shirts: 291.7h over 11009 episodes at 30fps (meta/train_report.txt).
DEFAULT_FRAMES_PER_EPISODE = 2861


def _episode_frames(zarr_path: Path) -> int:
    try:
        return int(zarr.open(str(zarr_path), mode="r", zarr_format=3).attrs["total_frames"])
    except Exception:
        return 0


def _process(args) -> dict:
    """Download one episode, convert it, drop the raw MCAP. Runs in a worker."""
    episode, split, task, raw_root, zarr_dir, embodiment, keep_raw = args
    # hf_hub_download recreates the repo-relative path under local_dir, so
    # local_dir must be the raw ROOT (<out>/raw), not the task directory.
    raw_root = Path(raw_root)
    ep_dir = raw_root / "data" / split / task / episode
    t0 = time.time()
    try:
        for entry in _tree(HfApi(), f"data/{split}/{task}/{episode}"):
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=entry.path,
                local_dir=raw_root,
            )
        out = convert_episode(ep_dir, Path(zarr_dir), embodiment)
        return {
            "episode": episode,
            "ok": True,
            "frames": _episode_frames(out),
            "seconds": round(time.time() - t0, 1),
        }
    except Exception as exc:
        return {
            "episode": episode,
            "ok": False,
            "frames": 0,
            "error": f"{type(exc).__name__}: {exc}"[:200],
            "seconds": round(time.time() - t0, 1),
        }
    finally:
        if not keep_raw:
            shutil.rmtree(ep_dir, ignore_errors=True)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="fold_and_stack_the_t_shirts")
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--out", required=True, help="root; zarr/ and manifest.json go here")
    p.add_argument(
        "--target-frames",
        type=int,
        default=MECKA_FOLDCLOTHES_FRAMES,
        help=f"frame budget (default {MECKA_FOLDCLOTHES_FRAMES}, = mecka fold_clothes)",
    )
    p.add_argument("--embodiment", default="yam_bimanual")
    p.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or (os.cpu_count() or 8),
    )
    p.add_argument("--keep-raw", action="store_true", help="keep the MCAP files")
    args = p.parse_args()

    out = Path(args.out)
    zarr_dir = out / "zarr"
    raw_root = out / "raw"
    zarr_dir.mkdir(parents=True, exist_ok=True)
    (raw_root / "data" / args.split / args.task).mkdir(parents=True, exist_ok=True)

    done = {z.stem: _episode_frames(z) for z in sorted(zarr_dir.glob("*.zarr"))}
    frames = sum(done.values())
    logger.info(
        "already converted: %d episodes / %d frames (target %d)",
        len(done), frames, args.target_frames,
    )

    candidates = [
        e for e in list_episodes(HfApi(), args.split, args.task)
        if e.removeprefix("episode_") not in done
    ]
    logger.info("%d candidate episodes in %s", len(candidates), args.task)

    results = [{"episode": f"episode_{k}", "ok": True, "frames": v} for k, v in done.items()]
    it = iter(candidates)
    inflight: set = set()
    submitted = 0

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        def _window() -> int:
            """How many episodes to keep in flight.

            A running future cannot be cancelled, so everything in flight when
            the budget is met runs to completion. Near the end that is pure
            waste, so the window shrinks to the number of episodes still needed
            (estimated from the mean length seen so far).
            """
            got = [r["frames"] for r in results if r["ok"] and r["frames"] > 0]
            avg = (sum(got) / len(got)) if got else DEFAULT_FRAMES_PER_EPISODE
            needed = max(1, math.ceil((args.target_frames - frames) / max(avg, 1)))
            # A third or so of episodes fail to convert, so ask for a few extra.
            return int(min(args.workers * 2, math.ceil(needed * 1.5)))

        def _fill():
            nonlocal submitted
            while len(inflight) < _window() and frames < args.target_frames:
                try:
                    ep = next(it)
                except StopIteration:
                    return
                inflight.add(
                    ex.submit(
                        _process,
                        (ep, args.split, args.task, str(raw_root), str(zarr_dir),
                         args.embodiment, args.keep_raw),
                    )
                )
                submitted += 1

        _fill()
        while inflight:
            finished, pending = wait(inflight, return_when=FIRST_COMPLETED)
            inflight = set(pending)
            for fut in finished:
                r = fut.result()
                results.append(r)
                if r["ok"]:
                    frames += r["frames"]
                    logger.info(
                        "OK   %s  %5d frames  %5.1fs  |  %d/%d frames (%.1f%%)",
                        r["episode"][:26], r["frames"], r["seconds"],
                        frames, args.target_frames, 100 * frames / args.target_frames,
                    )
                else:
                    logger.warning("SKIP %s  %s", r["episode"][:26], r.get("error"))
            if frames >= args.target_frames:
                for fut in inflight:
                    fut.cancel()
                inflight = {f for f in inflight if not f.cancelled()}
            else:
                _fill()

    ok = [r for r in results if r["ok"]]
    manifest = {
        "task": args.task,
        "split": args.split,
        "embodiment": args.embodiment,
        "target_frames": args.target_frames,
        "total_frames": frames,
        "episodes": len(ok),
        "attempted": submitted,
        "failed": [r for r in results if not r["ok"]],
        "converted": sorted(ok, key=lambda r: r["episode"]),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info(
        "DONE %d episodes / %d frames (target %d) from %d attempts -> %s",
        len(ok), frames, args.target_frames, submitted, out / "manifest.json",
    )


if __name__ == "__main__":
    main()
