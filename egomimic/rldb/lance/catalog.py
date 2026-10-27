"""Catalog rows (recordings, tasks, aliases) of one recording, from its episode metadata.

Every converter that stages a self-describing delivery (commit.stage_delivery) builds
its rows here, so MCAP, EVA and Aria uploads fill the catalog the same way. `meta` uses
the shared metadata keys (egomimic.rldb.zarr.required_metadata plus
egomimic.rldb.mcap.spec.METADATA_KEYS, snake_case); a converter maps its own format's
keys onto them and adds what it derives from the recording (start_time_unix).
"""

from __future__ import annotations

import datetime

from egomimic.rldb.lance.schema import task_instance_id
from egomimic.utils.aws.aws_sql import timestamp_ms_to_episode_hash

# (t_start_ns, t_end_ns, task_name, task_description, objects), half-open.
Task = tuple[int, int, str, str | None, list[str]]


def objects_list(v) -> list[str]:
    """`objects` as a list: comma-separated text (MCAP, zarr attrs) or a list; None when
    waived."""
    if v is None:
        return []
    if isinstance(v, str):
        return [o for o in (x.strip() for x in v.split(",")) if o]
    return [str(o) for o in v]


def episode_task(meta: dict, t_start_ns: int, t_end_ns: int) -> Task:
    """The episode's own task over [t_start_ns, t_end_ns), for recordings that carry no
    per-segment tasks."""
    return (
        int(t_start_ns),
        int(t_end_ns),
        meta["task_name"],
        meta.get("task_description"),
        objects_list(meta.get("objects")),
    )


def rows_from_metadata(
    meta: dict,
    *,
    recording_id: str,
    end_effector: str,
    fps: float,
    num_frames: int,
    t_start_ns: int,
    t_end_ns: int,
    intrinsics: dict,
    extrinsics: dict,
    tasks: list[Task],
) -> tuple[dict, list[dict], dict]:
    """(recording, tasks, alias) rows; the recording row without video_path and
    content_digest, which the commit fills. Waived scene stays None."""
    start_ms = int(meta["start_time_unix"])
    recording = dict(
        recording_id=recording_id,
        vendor_recording_id=meta["vendor_recording_id"],
        lab=meta["lab"],
        embodiment=meta["embodiment"],
        end_effector=end_effector,
        rig_name=meta["rig_name"],
        operator=meta["operator"],
        scene=meta.get("scene"),
        captured_at=datetime.datetime.fromtimestamp(
            start_ms / 1000, tz=datetime.timezone.utc
        ),
        fps=fps,
        num_frames=num_frames,
        intrinsics=sorted(intrinsics.items()),
        extrinsics=sorted(extrinsics.items()),
        license=meta.get("license"),
    )
    task_rows = [
        dict(
            task_instance_id=task_instance_id(recording_id, name, t0),
            recording_id=recording_id,
            t_start_ns=t0,
            t_end_ns=t1,
            task_name=name,
            task_description=desc,
            objects=objs,
        )
        for t0, t1, name, desc, objs in tasks
    ]
    alias = dict(
        episode_hash=timestamp_ms_to_episode_hash(start_ms),
        recording_id=recording_id,
        t_start_ns=int(t_start_ns),
        t_end_ns=int(t_end_ns),
    )
    return recording, task_rows, alias
