"""Aria uploads: one recording per delivery, converted as aria_to_lance converts it.

    <id>.vrs             the recording from the glasses
    <id>_metadata.json   its metadata (egomimic/ingest/metadata.py; no SQL lookup)
    mps_<id>_vrs/        its MPS outputs; hand_tracking/hand_tracking_results.csv and
                         slam/closed_loop_trajectory.csv are needed, eye_gaze/ is used
                         when present, anything else in the folder is kept as delivered

<id> is the recording's start in UTC epoch milliseconds. Recording ids are
recording_id("aria", <id>), as for the backfill (aria_to_lance_ray), so a recording that
comes in both ways is committed once. The rigs' metadata files are read as they are
(metadata.rig_metadata): robot_name aria_bimanual / aria_left_arm / aria_right_arm is the
human embodiment, the device (aria, also when the file does not say) stands in for a
missing rig_name. device_serial is
optional in the file; when missing it is read from the VRS where the file records it.
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
from egomimic.ingest.stage import Prepared, mano_model_digest, stage_converted

SOURCE = "aria"  # recording-id namespace and the committer's staging prefix
HAND_TRACKING = "hand_tracking/hand_tracking_results.csv"
TRAJECTORY = "slam/closed_loop_trajectory.csv"
FILES = (
    "{id}.vrs",
    "{id}_metadata.json",
    f"mps_{{id}}_vrs/{HAND_TRACKING}",
    f"mps_{{id}}_vrs/{TRAJECTORY}",
)
OPTIONAL = ("mps_{id}_vrs/**",)
SIDES = ("bimanual", "left_arm", "right_arm")
# How the rigs name the arm layout -> the human embodiment it records.
LAYOUTS = {f"aria_{s}": f"human_{s}" for s in SIDES} | {
    f"human_{s}": f"human_{s}" for s in SIDES
}
HINTS = {
    "embodiment": "give robot_name, the arm layout (aria_bimanual, aria_left_arm or "
    "aria_right_arm)",
    "license": "set it in the metadata, or give the vendor a default at onboarding "
    "(onboard --license)",
}


def upload_metadata(md: dict, vid: str, default_license: str | None = None) -> dict:
    """An Aria metadata file in the shared keys, plus what the recording gives."""
    return rig_metadata(md, vid, LAYOUTS, default_license, device="aria")


def vrs_serial(vrs: Path) -> str | None:
    """The glasses' serial as the VRS records it, or None."""
    try:
        from projectaria_tools.core import data_provider

        provider = data_provider.create_vrs_data_provider(str(vrs))
        serial = (
            getattr(provider.get_metadata(), "device_serial", None)
            if provider
            else None
        )
    except Exception:
        return None
    return str(serial) if serial else None


def precheck(delivery: Path, vid: str) -> list[str]:
    """What would make the conversion fail, found before paying for a GPU."""
    problems = []
    if err := start_time_problem(vid, f"{vid}.vrs"):
        problems.append(err)
    mps = delivery / f"mps_{vid}_vrs"
    for rel in (HAND_TRACKING, TRAJECTORY):
        p = mps / rel
        if not p.is_file() or p.stat().st_size == 0:
            problems.append(f"mps_{vid}_vrs/{rel} is missing or empty")
        elif sum(1 for _ in p.open("rb")) < 2:  # a header and no rows
            problems.append(f"mps_{vid}_vrs/{rel} has no rows")
    return problems


def prepare(
    delivery: Path,
    staged: Path,
    lance_root,
    *,
    fingerprint: str,
    vendor: str,
    ffmpeg: str = "ffmpeg",
    mano: bool = True,
    mano_iters: int = 2400,
    mano_model_dir: str | None = None,
    default_license: str | None = None,
) -> Prepared:
    """Check and convert an Aria delivery into `staged` (stage.prepare's contract)."""
    import pyarrow.parquet as pq

    from egomimic.ingest.formats import by_name
    from egomimic.rldb.embodiment.embodiment import END_EFFECTOR
    from egomimic.rldb.lance import commit
    from egomimic.rldb.lance.catalog import episode_task, rows_from_metadata
    from egomimic.rldb.lance.schema import recording_id
    from egomimic.scripts.aria_process.aria_to_lance import ARM_OF, FPS, convert
    from egomimic.scripts.lance_commit import check_delivery

    delivery = Path(delivery)
    names = [p.relative_to(delivery).as_posix() for p in delivery.rglob("*")]
    vid = by_name("aria").recording_key(names)
    rid = recording_id(SOURCE, vid)
    md, err = read_metadata_file(delivery / f"{vid}_metadata.json")
    if err:
        return Prepared("rejected", rid, problems=[err])
    meta = upload_metadata(md, vid, default_license)
    if not meta.get("device_serial") and (
        serial := vrs_serial(delivery / f"{vid}.vrs")
    ):
        meta["device_serial"] = serial
    problems, waived = check_metadata(
        meta, tuple(ARM_OF), hints=HINTS, optional=frozenset({"device_serial"})
    )
    if not problems:
        problems = precheck(delivery, vid)
    if problems:
        return Prepared("rejected", rid, problems=problems)

    settings = {"mano": mano, "mano_iters": mano_iters}
    if mano:  # which model fitted the hands, by content (the folder differs per host)
        settings["mano_model"] = mano_model_digest(mano_model_dir)
    work = Path(staged).with_name(f".{Path(staged).name}.work.{uuid.uuid4().hex}")
    try:
        args = SimpleNamespace(
            raw_dir=delivery, out_dir=work, vendor_recording_id=vid, recording_id=rid,
            embodiment=meta["embodiment"], mano=mano, mano_iters=mano_iters,
            mano_model_dir=mano_model_dir, ffmpeg=ffmpeg, stage_to=None,
        )  # fmt: skip
        try:
            out = convert(args)
        except Exception as e:  # unreadable VRS, MPS that does not cover it, ...
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
            end_effector=END_EFFECTOR.HAND.value,
            fps=FPS,
            num_frames=out["num_frames"],
            t_start_ns=t0,
            t_end_ns=t1,
            intrinsics=out["intrinsics"],
            extrinsics=out["extrinsics"],
            tasks=[episode_task(meta, t0, t1)],
        )
        videos = {
            cam: d_dir / f"{cam}.mp4"
            for cam in out.get("videos", ("front_1", "front_1_480"))
        }
        converted = SimpleNamespace(
            delivery=commit.Delivery(recording, tasks, alias, steps, videos),
            labels=[],
            label_source=meta["lab"],
        )
        report = {
            "valid_frac": out["valid_frac"],
            "filler_frames": out["filler_frames"],
            "gaps": out["gaps"],
        }
        if meta.get("device_serial"):
            report["device_serial"] = meta["device_serial"]
        return stage_converted(
            converted,
            staged,
            lance_root,
            format="aria",
            source=SOURCE,
            fingerprint=fingerprint,
            settings=settings,
            report=report,
            waived=waived,
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)
