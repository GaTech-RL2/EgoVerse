"""EVA uploads: one raw robot demo per delivery, converted as eva_to_lance converts it.

    <id>.hdf5            the demo as the EVA rig records it (no per-frame times)
    <id>_metadata.json   its metadata in the shared keys (egomimic/ingest/metadata.py)

<id> is the demo's start in UTC epoch milliseconds (frame i is at <id> + i / fps). The
metadata may give `fps` (30 or 60; default 30). Recording ids are recording_id("eva",
<id>), as for the backfill (eva_to_lance --stage-to), so a demo that comes in both ways
is committed once.

The rigs' existing files (as add_raw_data_to_table reads them) are taken as they are:
`task` is task_name; `robot_name` is the arm layout (eva_bimanual, eva_left_arm,
eva_right_arm), so it is the embodiment, while their `embodiment` names the robot
(eva, eve) and stands in for rig_name when that is missing; `license` defaults to the
vendor's, set at onboarding. `objects: ["None"]` (how the rigs say "no objects") is
taken as a missing_metadata waiver of objects, flagged for review. device_serial is optional: the rigs do not write it yet and
Lance does not store it; when given it is kept in the review report.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from types import SimpleNamespace

from egomimic.ingest.metadata import (
    check_metadata,
    read_metadata_file,
    rig_metadata,
    start_time_problem,
)
from egomimic.ingest.stage import Prepared, stage_converted

SOURCE = "eva"  # recording-id namespace and the committer's staging prefix
FILES = ("{id}.hdf5", "{id}_metadata.json")
DEFAULT_FPS = 30
# HDF5 image keys each embodiment needs: the front camera and the wrists of its arms.
FRONT, WRISTS = "front_img_1", {"left": "left_wrist_img", "right": "right_wrist_img"}


def upload_metadata(md: dict, vid: str, default_license: str | None = None) -> dict:
    """An EVA metadata file in the shared keys, plus what the recording gives."""
    from egomimic.scripts.eva_process.eva_to_lance import ARM_OF

    return rig_metadata(md, vid, {e: e for e in ARM_OF}, default_license)


# Where an EVA upload's missing keys come from, said with the problem.
HINTS = {
    "embodiment": "give robot_name, the arm layout (eva_bimanual, eva_left_arm or "
    "eva_right_arm)",
    "license": "set it in the metadata, or give the vendor a default at onboarding "
    "(onboard --license)",
}


def expected_cameras(embodiment: str) -> list[str]:
    from egomimic.scripts.eva_process.eva_to_lance import ARM_OF

    arm = ARM_OF[embodiment]
    sides = ("left", "right") if arm == "both" else (arm,)
    return [FRONT, *(WRISTS[s] for s in sides)]


def precheck(raw: Path, vid: str, meta: dict) -> list[str]:
    """What would make the conversion fail, found before paying for it."""
    import h5py

    from egomimic.rldb.zarr.episode_rules import fps_error

    problems = []
    if err := start_time_problem(vid, raw.name):
        problems.append(err)
    if err := fps_error(meta.get("fps", DEFAULT_FPS)):
        problems.append(err)
    try:
        with h5py.File(raw, "r") as f:
            images = f.get("observations/images")
            have = set(images) if images is not None else set()
            for key in ("observations/eepose", "observations/joints"):
                if key not in f:
                    problems.append(f"{raw.name} has no {key}")
    except OSError as e:
        return [*problems, f"{raw.name} is not a readable HDF5 file: {e}"]
    missing = [c for c in expected_cameras(meta["embodiment"]) if c not in have]
    if missing:
        problems.append(
            f"{raw.name} has no {', '.join(missing)}: a {meta['embodiment']} demo "
            f"needs {', '.join(expected_cameras(meta['embodiment']))}"
        )
    return problems


def prepare(
    delivery: Path,
    staged: Path,
    lance_root,
    *,
    fingerprint: str,
    vendor: str,
    ffmpeg: str = "ffmpeg",
    default_license: str | None = None,
) -> Prepared:
    """Check and convert an EVA delivery into `staged` (stage.prepare's contract)."""
    import pyarrow.parquet as pq

    from egomimic.ingest.formats import by_name
    from egomimic.rldb.embodiment.embodiment import END_EFFECTOR
    from egomimic.rldb.lance import commit
    from egomimic.rldb.lance.catalog import episode_task, rows_from_metadata
    from egomimic.rldb.lance.schema import recording_id
    from egomimic.scripts.eva_process.eva_to_lance import ARM_OF, convert
    from egomimic.scripts.lance_commit import check_delivery

    delivery = Path(delivery)
    names = [p.relative_to(delivery).as_posix() for p in delivery.rglob("*")]
    vid = by_name("eva").recording_key(names)
    rid = recording_id(SOURCE, vid)
    md, err = read_metadata_file(delivery / f"{vid}_metadata.json")
    if err:
        return Prepared("rejected", rid, problems=[err])
    meta = upload_metadata(md, vid, default_license)
    problems, waived = check_metadata(
        meta, tuple(ARM_OF), hints=HINTS, optional=frozenset({"device_serial"})
    )
    if not problems:
        problems = precheck(delivery / f"{vid}.hdf5", vid, meta)
    if problems:
        return Prepared("rejected", rid, problems=problems)

    fps = float(meta.get("fps", DEFAULT_FPS))
    work = Path(staged).with_name(f".{Path(staged).name}.work.{uuid.uuid4().hex}")
    try:
        args = SimpleNamespace(
            raw_dir=delivery, out_dir=work, vendor_recording_id=vid, fps=fps,
            ffmpeg=ffmpeg,
        )  # fmt: skip
        try:
            out = convert(args)
        except Exception as e:  # broken HDF5 arrays, ffmpeg, ...
            return Prepared("failed", rid, error=f"{type(e).__name__}: {e}")
        d_dir = work / vid
        steps = pq.read_table(d_dir / "steps.parquet")
        errs = check_delivery(d_dir, out, steps)
        if errs:
            return Prepared("rejected", rid, problems=errs)
        t0, t1 = out["t_start_ns"], out["t_end_ns"]
        recording, tasks, alias = rows_from_metadata(
            meta,
            recording_id=rid,
            end_effector=END_EFFECTOR.GRIPPER.value,
            fps=fps,
            num_frames=out["num_frames"],
            t_start_ns=t0,
            t_end_ns=t1,
            intrinsics=out["intrinsics"],
            extrinsics=out["extrinsics"],
            tasks=[episode_task(meta, t0, t1)],
        )
        videos = {cam: d_dir / f"{cam}.mp4" for cam in out["videos"]}
        converted = SimpleNamespace(
            delivery=commit.Delivery(recording, tasks, alias, steps, videos),
            labels=[],
            label_source=meta["lab"],
        )
        return stage_converted(
            converted,
            staged,
            lance_root,
            format="eva",
            source=SOURCE,
            fingerprint=fingerprint,
            settings={"fps": fps},
            report={
                "valid_frac": out["valid_frac"],
                **(
                    {"device_serial": meta["device_serial"]}
                    if meta.get("device_serial")
                    else {}
                ),
            },
            waived=waived,
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)
