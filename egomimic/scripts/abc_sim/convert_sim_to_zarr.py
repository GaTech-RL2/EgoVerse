"""Convert ABC sim_224 teleop episodes (amazon-far/abc) to EgoVerse zarrs.

ABC's ``prepare.py --sim-data <task>`` downloads a task tar and extracts every
episode into ``$ABC_CACHE/{train,val}_sim/<episode>/`` (a plain ``tar -x`` of
the same tar gives ``{train,val}/<episode>/``; both are read): ``states_actions.bin``
((T, 28) float64: 14 recorded + 14 commanded dofs, [left j1..j6, left grip,
right j1..j6, right grip]), ``combined_camera-images-rgb.mp4`` (the cameras
stacked vertically, 224x168 each, 30 fps) and ``episode_metadata.json``. All
tasks share those two folders, so episodes are selected by their
``task_name``.

Each episode becomes one ``eva_bimanual`` zarr with the same keys as the real
ABC zarrs on the Phoenix mirror (``{left,right}.{obs,cmd}_{joints,gripper}``,
``images.{front_1,left_wrist,right_wrist}``, ``annotations``), so
``Eva.get_keymap("joints")`` reads real and sim ABC data alike. Camera map:
top -> front_1, left -> left_wrist, right -> right_wrist. Extra metadata:
``lab: abc_sim``, ``split`` (ABC's own train/val), ``abc_episode``, ``prompt``.

Prompts follow ABC's training rule (``abc_minimal.dataloader``) verbatim: the
episode's ``prompt_timeline`` if it has one, else its ``instruction`` when that
differs from the task name, else the task name with ``_`` -> `` `` (the sim
tasks' names start with ``sim_``). Episodes ABC's loader drops as unlabelled
are skipped.

    python -m egomimic.scripts.abc_sim.convert_sim_to_zarr \
        --src $ABC_CACHE --task sim_put_the_plastic_bottles_in_the_bin \
        --out /storage/project/r-dxu345-0/agao81/abc_sim/zarr/put_bottles
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

log = logging.getLogger("abc_sim.convert")

STATE_DIM = 14
DATA_FPS = 30
DEFAULT_CAMERAS = ("top", "left", "right")
CAMERA_TO_ZARR = {
    "top": "images.front_1",
    "left": "images.left_wrist",
    "right": "images.right_wrist",
}
# abc_sim scenes: every policy camera is a fovy=58 deg pinhole.
CAMERA_FOVY_DEG = 58.0
DONE_MARKER = ".abc_sim_done"


def task_name_to_prompt(task_name: str) -> str:
    """abc_minimal.dit.task_name_to_prompt."""
    return " ".join(task_name.replace("-", " ").replace("_", " ").split())


def prompt_spans(meta: dict, num_steps: int) -> list[tuple[str, int, int]]:
    """``(text, start, end)`` spans covering ``[0, num_steps)``, end exclusive."""
    task_name = str(meta.get("task_name") or "")
    timeline = meta.get("prompt_timeline")
    if timeline:
        entries = sorted((int(e["frame"]), str(e["prompt"])) for e in timeline)
    else:
        instruction = meta.get("instruction")
        if (
            isinstance(instruction, str)
            and instruction.strip()
            and task_name_to_prompt(instruction) != task_name_to_prompt(task_name)
        ):
            text = instruction.strip()
        else:
            text = task_name_to_prompt(task_name)
        entries = [(0, text)]
    spans = []
    for i, (start, text) in enumerate(entries):
        end = entries[i + 1][0] if i + 1 < len(entries) else num_steps
        start = max(0, min(start, num_steps))
        if i == 0:
            start = 0  # the first directive holds from the episode start
        if end > start:
            spans.append((text, start, end))
    return spans


def pinhole_intrinsics(height: int, width: int, fovy_deg: float = CAMERA_FOVY_DEG):
    """3x4 K of a MuJoCo camera (square pixels, principal point centred)."""
    f = 0.5 * height / math.tan(math.radians(fovy_deg) / 2.0)
    return np.array(
        [[f, 0.0, width / 2.0, 0.0], [0.0, f, height / 2.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        dtype=np.float64,
    )


def load_states_actions(ep_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = np.fromfile(ep_dir / "states_actions.bin", dtype=np.float64)
    if raw.size == 0 or raw.size % (2 * STATE_DIM):
        raise ValueError(f"states_actions.bin holds {raw.size} float64s")
    table = raw.reshape(-1, 2 * STATE_DIM)
    return table[:, :STATE_DIM], table[:, STATE_DIM:]


def decode_cameras(ep_dir: Path, cameras: tuple[str, ...]) -> dict[str, np.ndarray]:
    """``{camera: (T, H, W, 3) uint8}`` from the vertically stacked mp4."""
    import av

    with av.open(str(ep_dir / "combined_camera-images-rgb.mp4")) as container:
        stack = np.stack([f.to_ndarray(format="rgb24") for f in container.decode(video=0)])
    h = stack.shape[1] // len(cameras)
    if h * len(cameras) != stack.shape[1]:
        raise ValueError(f"video height {stack.shape[1]} is not {len(cameras)} stacked cameras")
    return {cam: stack[:, i * h : (i + 1) * h] for i, cam in enumerate(cameras)}


def split_arms(x: np.ndarray, kind: str) -> dict[str, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    return {
        f"left.{kind}_joints": x[:, 0:6],
        f"left.{kind}_gripper": x[:, 6:7],
        f"right.{kind}_joints": x[:, 7:13],
        f"right.{kind}_gripper": x[:, 13:14],
    }


def episode_hash(split: str, ep_dir: Path) -> str:
    return f"abcsim_{split}_{ep_dir.name}"


def convert_episode(ep_dir: Path, split: str, out_dir: Path, overwrite: bool = False) -> dict:
    """Write one zarr; returns a report row. Idempotent via a done marker."""
    from egomimic.rldb.zarr.zarr_writer import ZarrWriter

    name = episode_hash(split, ep_dir)
    dst = out_dir / f"{name}.zarr"
    if (dst / DONE_MARKER).exists() and not overwrite:
        return {"episode": name, "status": "skipped"}
    meta = json.loads((ep_dir / "episode_metadata.json").read_text())
    if not meta.get("prompt_timeline") and (meta.get("prompt_source") or {}).get("status") == "excluded":
        return {"episode": name, "status": "skipped_unlabelled"}  # abc_minimal.dataloader drops these
    states, actions = load_states_actions(ep_dir)
    num_steps = len(states)
    if meta.get("num_steps") is not None and int(meta["num_steps"]) != num_steps:
        raise ValueError(f"states_actions.bin has {num_steps} rows, metadata says {meta['num_steps']}")
    cameras = tuple(meta.get("cameras") or DEFAULT_CAMERAS)
    images = decode_cameras(ep_dir, cameras)
    n_frames = min(len(v) for v in images.values())
    if n_frames != num_steps:
        raise ValueError(f"{n_frames} video frames for {num_steps} state rows")
    missing = [c for c in DEFAULT_CAMERAS if c not in images]
    if missing:
        raise ValueError(f"cameras missing: {missing} (have {sorted(images)})")
    h, w = images["top"].shape[1:3]

    tmp = out_dir / f".{name}.zarr.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    spans = prompt_spans(meta, num_steps)
    ZarrWriter.create_and_write(
        tmp,
        numeric_data={**split_arms(states, "obs"), **split_arms(actions, "cmd")},
        image_data={CAMERA_TO_ZARR[c]: images[c] for c in DEFAULT_CAMERAS},
        embodiment="eva_bimanual",
        fps=DATA_FPS,
        task_name=str(meta.get("task_name") or ""),
        task_description=spans[0][0] if spans else "",
        annotations=spans,
        intrinsics={"front_1": pinhole_intrinsics(h, w)},
        metadata_override={
            "lab": "abc_sim",
            "split": split,
            "abc_episode": ep_dir.name,
            "prompt": spans[0][0] if spans else "",
            "source": "abc sim_224",
        },
    )
    (tmp / DONE_MARKER).write_text("ok\n")
    shutil.rmtree(dst, ignore_errors=True)
    os.replace(tmp, dst)
    return {"episode": name, "status": "written", "frames": num_steps}


def find_episodes(src: Path, task: str, splits: list[str]) -> list[tuple[Path, str]]:
    found = []
    for split in splits:
        # prepare.py extracts into {split}_sim/; a plain `tar -x` leaves {split}/
        root = next((src / d for d in (f"{split}_sim", split) if (src / d).is_dir()), None)
        if root is None:
            log.warning("no %s_sim/ or %s/ under %s", split, split, src)
            continue
        for bin_path in sorted(root.glob("*/states_actions.bin")):
            ep_dir = bin_path.parent
            meta_path = ep_dir / "episode_metadata.json"
            if not meta_path.exists():
                continue
            if json.loads(meta_path.read_text()).get("task_name") == task:
                found.append((ep_dir, split))
    return found


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--src", type=Path, required=True, help="ABC cache dir ($ABC_CACHE)")
    p.add_argument("--task", required=True, help="dataset task_name, e.g. sim_pouring_beads")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    episodes = find_episodes(args.src, args.task, args.splits)
    if not episodes:
        log.error("no %s episodes under %s/{%s}_sim", args.task, args.src, ",".join(args.splits))
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    log.info("converting %d episodes of %s -> %s", len(episodes), args.task, args.out)

    rows, failures = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(convert_episode, ep, split, args.out, args.overwrite): ep
            for ep, split in episodes
        }
        for i, fut in enumerate(as_completed(futures), 1):
            ep = futures[fut]
            try:
                rows.append(fut.result())
            except Exception as e:  # one bad episode must not sink the task
                failures.append({"episode": str(ep), "error": f"{type(e).__name__}: {e}"})
                log.warning("FAILED %s: %s", ep, e)
            if i % 100 == 0 or i == len(futures):
                log.info("%d/%d done (%d failed)", i, len(futures), len(failures))

    report = {
        "task": args.task,
        "src": str(args.src),
        "written": sum(r["status"] == "written" for r in rows),
        "skipped": sum(r["status"] == "skipped" for r in rows),
        "skipped_unlabelled": sum(r["status"] == "skipped_unlabelled" for r in rows),
        "failed": failures,
        "frames_written": int(sum(r.get("frames", 0) for r in rows)),
    }
    (args.out / "conversion_report.json").write_text(json.dumps(report, indent=2))
    log.info("report: %s", {k: v for k, v in report.items() if k != "failed"})
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
