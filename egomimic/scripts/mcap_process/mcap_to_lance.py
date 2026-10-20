"""Convert vendor MCAP episodes (spec 0.1, egomimic/rldb/mcap) into the Lance tables.

    python -m egomimic.scripts.mcap_process.mcap_to_lance --lance <root> episode_<id>/ ...

Per episode: validate (any problem rejects it), check its labels against the tasks it
will have, build one steps row per frame of the steps-clock camera, write a video per
camera, and commit the recording, its tasks and its labels with
egomimic.rldb.lance.commit. Everything comes from the MCAP itself: the episode info
replaces the SQL lookup of the Aria pilot (lance_pilot/commit.py).

The per-frame math follows aria_to_lance: nearest sample within MATCH_TOL_NS of each
frame (a true nearest search; projectaria's helper can land one sample off), filler rows
for dropped frames, head pose as the camera pose turned upright, hands in the world
frame, MANO keypoints from the Aria fit, validity from tracking, timing and pose jumps.
Timestamps stay absolute Unix ns, as delivered. Human embodiments only: Lance has no
robot steps schema yet.
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
from scipy.spatial.transform import Rotation

from egomimic.rldb.lance import commit
from egomimic.rldb.lance.schema import (
    ANNOTATIONS,
    HUMAN_STEPS,
    annotation_id,
    task_instance_id,
)
from egomimic.rldb.lance.timeline import build_timeline, pose_outliers
from egomimic.rldb.mcap import schemas, spec
from egomimic.rldb.mcap.validate import frames_in_message, validate_episode
from egomimic.utils.aws.aws_sql import timestamp_ms_to_episode_hash

MATCH_TOL_NS = 50_000_000  # as aria_to_lance: further than this from a frame = missing
IDENTITY_POSE = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32)  # xyz + wxyz
# The steps camera's pinhole outputs (h, w), as aria_to_lance.
FRONT = {"front_1": (360, 480), "front_1_480": (480, 640)}
# Aria's output focal at 480 rows (ARIA_INTRINSICS). Other sources keep their own field
# of view: their native focal, scaled to the output.
ARIA_FOCAL_480 = 133.25430222 * 2
# Converter-side conventions shared with the Aria path (aria_utils).
T_ROT_CAM = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
ENCODER_OPTS = {
    "libsvtav1": ["-preset", "12", "-crf", "30"],  # aria_to_lance's choice
    "libaom-av1": ["-cpu-used", "8", "-crf", "30", "-b:v", "0"],
    "libx264": ["-preset", "veryfast", "-crf", "23"],
}
# The only topics the conversion reads besides the cameras and their calibrations.
NEEDED = {
    "/tf_static",
    "/head-pose",
    "/head-pose-quality",
    "/left-hand-keypoints",
    "/right-hand-keypoints",
    "/eye-gaze",
    "/task",
    "/subtask-annotation",
}


class Rejected(Exception):
    """A validated episode Lance would still refuse (e.g. labels outside its tasks)."""

    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


# ---------------------------------------------------------------- reading


@dataclass
class _Series:
    t: np.ndarray  # log times, Unix ns, sorted
    msgs: list


@dataclass
class _Camera:
    topic: str
    t: np.ndarray  # frame times, Unix ns, sorted
    fmt: str  # h264 / h265


def _read(paths: list[Path]):
    """The needed topics decoded, every camera's frame times and codec, and the episode
    info. Cameras and info come from episode.mcap (paths[0]) only, as validated."""
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    topics: dict[str, list] = defaultdict(list)
    cameras: dict[str, _Camera] = {}
    meta: dict[str, str] = {}
    for k, path in enumerate(paths):
        with open(path, "rb") as f:
            r = make_reader(f, decoder_factories=[DecoderFactory()])
            summary = r.get_summary()
            if summary is not None:
                names = {c.topic for c in summary.channels.values()}
            else:  # unindexed file: one scan finds the topics
                names = {ch.topic for _, ch, _ in r.iter_messages()}
            wanted = sorted(
                t
                for t in names
                if t in NEEDED
                or t.endswith((spec.CALIBRATION_SUFFIX, spec.CAMERA_EXTRA_SUFFIX))
            )
            if wanted:
                for _, ch, msg, dec in r.iter_decoded_messages(
                    topics=wanted, log_time_order=True
                ):
                    topics[ch.topic].append((msg.log_time, dec))
            if k:
                continue
            meta = {m.name: dict(m.metadata) for m in r.iter_metadata()}.get(
                spec.METADATA_RECORD, {}
            )
            cams = sorted(t for t in names if t.endswith(spec.CAMERA_SUFFIX))
            times: dict[str, list[int]] = defaultdict(list)
            for _, ch, msg in r.iter_messages(topics=cams) if cams else ():
                times[ch.topic].append(msg.log_time)
            for cam in cams:
                _, _, _, first = next(r.iter_decoded_messages(topics=[cam]))
                cameras[cam] = _Camera(
                    cam, np.sort(np.array(times[cam], dtype=np.int64)), first.format
                )
    series = {
        k: _Series(np.array([t for t, _ in v], dtype=np.int64), [m for _, m in v])
        for k, v in topics.items()
    }
    return series, cameras, meta


def _packets(path: Path, topic: str):
    """One camera's frames as bytes, streamed in log-time order."""
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    with open(path, "rb") as f:
        r = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, _, _, dec in r.iter_decoded_messages(
            topics=[topic], log_time_order=True
        ):
            yield bytes(dec.data)


def _mat(position, orientation) -> np.ndarray:
    T = np.eye(4)
    o = orientation
    T[:3, :3] = Rotation.from_quat([o.x, o.y, o.z, o.w]).as_matrix()
    T[:3, 3] = (position.x, position.y, position.z)
    return T


def _xyz_wxyz(T: np.ndarray) -> np.ndarray:
    q = Rotation.from_matrix(T[:3, :3]).as_quat()  # xyzw
    return np.r_[T[:3, 3], q[3], q[:3]]


class _Frames:
    """World poses of named frames: /tf_static plus the moving world -> head link."""

    def __init__(self, series: dict[str, _Series]):
        self.edges: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
        self._cache: dict[tuple[str, str], np.ndarray | None] = {}
        for msg in series["/tf_static"].msgs:
            for x in msg.transforms:
                T = _mat(x.translation, x.rotation)
                self.edges[x.parent_frame_id][x.child_frame_id] = T
                self.edges[x.child_frame_id][x.parent_frame_id] = np.linalg.inv(T)

    def static(self, root: str, frame: str) -> np.ndarray | None:
        """T_root_frame through /tf_static only, or None if unconnected (cached)."""
        if (root, frame) not in self._cache:
            self._cache[root, frame] = self._search(root, frame)
        return self._cache[root, frame]

    def on_head(self, frame: str) -> bool:
        """Whether placing `frame` in the world needs the head pose."""
        return (
            frame != spec.WORLD_FRAME
            and self.static(spec.HEAD_FRAME, frame) is not None
        )

    def _search(self, root: str, frame: str) -> np.ndarray | None:
        todo, seen = [(root, np.eye(4))], {root}
        while todo:
            name, T = todo.pop()
            if name == frame:
                return T
            for nxt, step in self.edges[name].items():
                if nxt not in seen:
                    seen.add(nxt)
                    todo.append((nxt, T @ step))
        return None

    def world(self, frame: str, T_world_head: np.ndarray) -> np.ndarray:
        """T_world_frame at a time whose head pose is T_world_head."""
        if frame == spec.WORLD_FRAME:
            return np.eye(4)
        T = self.static(spec.HEAD_FRAME, frame)
        if T is not None:
            return T_world_head @ T
        T = self.static(spec.WORLD_FRAME, frame)
        if T is None:
            raise ValueError(f"frame {frame!r} is not linked to world or head")
        return T


def _nearest(times: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index of the nearest sample to each time, and whether it is within MATCH_TOL_NS
    (the index form of timeline.nearest_dt)."""
    last = len(times) - 1
    j = np.clip(np.searchsorted(times, t), 0, last)
    prev = np.clip(j - 1, 0, last)
    pick = np.where(np.abs(times[prev] - t) <= np.abs(times[j] - t), prev, j)
    return pick, np.abs(times[pick] - t) <= MATCH_TOL_NS


# ---------------------------------------------------------------- hands


def _aria_slots() -> list[str]:
    """Joint name per slot of Aria's 21-landmark layout, which the MANO fit
    (aria_utils.convert_aria_keypoints_to_mano) takes as input."""
    from egomimic.scripts.aria_process.aria_utils import (
        ARIA_TO_OTAHERI,
        OTAHERI_TO_CANONICAL,
    )

    names = ["palm_center"] * 21  # the one slot the fit ignores
    for aria, otaheri in ARIA_TO_OTAHERI.items():
        names[aria] = spec.MANO_JOINTS[OTAHERI_TO_CANONICAL.index(otaheri)]
    return names


def _reorder(kp: np.ndarray, joints: list[str], slots: list[str]) -> np.ndarray:
    """(T, J*3) in `joints` order -> (T, len(slots)*3) in `slots` order; absent joints NaN."""
    out = np.full((len(kp), len(slots), 3), np.nan)
    src = kp.reshape(len(kp), len(joints), 3)
    for i, name in enumerate(slots):
        if name in joints:
            out[:, i] = src[:, joints.index(name)]
    return out.reshape(len(kp), -1)


def _hand(
    series: _Series,
    cam_t: np.ndarray,
    T_world_head: list[np.ndarray],
    frames: _Frames,
    joints: list[str],
    head_ok: np.ndarray,
    *,
    is_right: bool,
    mano: bool,
    mano_iters: int,
) -> dict[str, np.ndarray]:
    """Per camera frame: wrist pose, ee pose, MANO keypoints, Aria keypoints, confidence, ok."""
    from egomimic.scripts.aria_process.aria_utils import (
        convert_aria_keypoints_to_mano,
        mano_keypoints_to_cartesian,
    )

    n = len(cam_t)
    pick, close = _nearest(series.t, cam_t)
    raw = np.full((n, len(joints) * 3), np.nan)
    wrist = np.full((n, 7), np.nan)
    conf = np.full(n, -1.0, dtype=np.float32)
    tracked = np.zeros(n, dtype=bool)
    for i, (p, ok) in enumerate(zip(pick, close)):
        m = series.msgs[p]
        if not (ok and m.valid):
            continue
        if frames.on_head(m.frame_id) and not head_ok[i]:
            continue  # placed with a stale or lost head pose (aria_to_lance: pose_close)
        T = frames.world(m.frame_id, T_world_head[i])
        conf[i] = m.confidence
        tracked[i] = True
        if len(m.positions):
            pts = np.asarray(m.positions, dtype=np.float64).reshape(-1, 3)
            raw[i] = (T[:3, :3] @ pts.T + T[:3, 3:4]).T.ravel()
        if m.HasField("wrist_pose"):
            Tw = T @ _mat(m.wrist_pose.position, m.wrist_pose.orientation)
            Tw[:3, :3] = Tw[:3, :3] @ T_ROT_CAM
            wrist[i] = _xyz_wxyz(Tw)

    has_kp = np.isfinite(raw).all(1)
    aria_layout = joints == _aria_slots()
    # MANO order by joint name; joints the device lacks stay NaN.
    canonical = _reorder(raw, joints, list(spec.MANO_JOINTS))
    if tuple(joints) == spec.MANO_JOINTS:
        keypoints = raw  # already MANO: no fit
    elif mano and has_kp.any():
        aria = _reorder(raw, joints, _aria_slots())
        aria = np.where(np.isfinite(aria) & (tracked & has_kp)[:, None], aria, 1e9)
        keypoints = convert_aria_keypoints_to_mano(
            aria, is_rhand=is_right, n_iters=mano_iters, lr=0.05
        )
    elif aria_layout:  # --no-mano on Aria data, as aria_to_lance --no-mano
        keypoints = raw
    else:
        keypoints = canonical
    # The ee pose needs wrist + 4 MCPs: from the fit, or the vendor's joints by name.
    ee_src = keypoints if (mano or tuple(joints) == spec.MANO_JOINTS) else canonical
    ee = mano_keypoints_to_cartesian(np.nan_to_num(ee_src, nan=1e9), is_rhand=is_right)
    ee_ok = (np.abs(ee) < 1e8).all(1)

    wrist_ok = np.isfinite(wrist).all(1)
    if "wrist" in joints:  # no wrist pose sent: the wrist joint, oriented as the ee
        w = raw[:, 3 * joints.index("wrist") : 3 * joints.index("wrist") + 3]
        fill = ~wrist_ok & ee_ok
        wrist[fill] = np.c_[w[fill], ee[fill, 3:]]
    no_kp = tracked & ~has_kp  # wrist-only device: the wrist pose is the ee
    ee[no_kp & wrist_ok] = wrist[no_kp & wrist_ok]
    ee_ok |= no_kp & wrist_ok

    kp_ok = np.isfinite(keypoints).all(1) | no_kp
    ok = tracked & np.isfinite(wrist).all(1) & kp_ok & ee_ok
    ok &= ~pose_outliers(np.nan_to_num(wrist[:, :3]), ok)
    return dict(
        wrist=wrist,
        ee=ee,
        kp=keypoints,
        # Lance's aria_keypoints holds Aria landmarks only, never another layout.
        aria=raw if aria_layout else np.zeros((n, 63)),
        conf=conf,
        ok=ok,
    )


# ---------------------------------------------------------------- video


def _rot_z(deg_cw: int) -> np.ndarray:
    """Camera-frame rotation matching a clockwise image turn (90 -> aria_utils.ROTATION_MATRIX)."""
    return Rotation.from_euler("z", -deg_cw, degrees=True).as_matrix()


def _upright(extra) -> int:
    return int(extra.upright_rotation_cw_deg) if extra else 0


def _undistorter(calib, extra, size_hw: tuple[int, int], focal: float):
    """raw frame -> upright pinhole frame of `size_hw` with `focal`, centred."""
    import cv2

    h, w = size_hw
    rot = _upright(extra)
    pre_h, pre_w = (w, h) if rot % 180 else (h, w)  # size before turning upright
    model = (calib.distortion_model or "plumb_bob").lower()
    k_rot = (-rot // 90) % 4

    if model == "fisheye624":
        import projectaria_tools.core.sophus as sp
        from projectaria_tools.core import calibration

        params = np.array([calib.K[0], calib.K[2], calib.K[5], *calib.D])
        radius = float(extra.valid_radius) if extra and extra.valid_radius else None
        src = calibration.CameraCalibration(
            "camera",
            calibration.CameraModelType.FISHEYE624,
            params,
            sp.SE3.from_matrix(np.eye(4)),
            int(calib.width),
            int(calib.height),
            radius,
            np.pi,
            "",
        )
        dst = calibration.get_linear_camera_calibration(pre_w, pre_h, focal, "camera")

        def remap(raw):
            return calibration.distort_by_calibration(raw, dst, src)

    else:
        K = np.asarray(calib.K, dtype=np.float64).reshape(3, 3)
        D = np.asarray(calib.D, dtype=np.float64)
        K_new = np.array(
            [[focal, 0, (pre_w - 1) / 2], [0, focal, (pre_h - 1) / 2], [0, 0, 1]]
        )
        if model == "kannala_brandt":
            m1, m2 = cv2.fisheye.initUndistortRectifyMap(
                K, D[:4], np.eye(3), K_new, (pre_w, pre_h), cv2.CV_32FC1
            )
        elif model in ("plumb_bob", "rational_polynomial"):
            m1, m2 = cv2.initUndistortRectifyMap(
                K, D, np.eye(3), K_new, (pre_w, pre_h), cv2.CV_32FC1
            )
        else:
            raise ValueError(f"unsupported distortion_model {model!r}")

        def remap(raw):
            return cv2.remap(raw, m1, m2, cv2.INTER_LINEAR)

    def f(raw):
        return np.ascontiguousarray(np.rot90(remap(raw), k=k_rot))

    return f


def _encoder(path: Path, h: int, w: int, ffmpeg: str, codec: str, fps: float):
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "-",
        "-c:v", codec, "-g", "2", *ENCODER_OPTS.get(codec, []), "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(path),
    ]  # fmt: skip
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def _write_camera(packets, cam: _Camera, rows, outputs, render, ffmpeg, codec, fps):
    """Decode one camera once, streaming, and write one frame per steps row: rows[i] is
    the camera frame shown on row i (non-decreasing; repeats for filler rows).
    `render(raw)` gives one frame per (path, (h, w)) in `outputs`. Every message must be
    exactly one frame: the validator checks a sample, this checks all of them."""
    import av

    ctx = av.CodecContext.create("hevc" if cam.fmt == "h265" else "h264", "r")
    seen = 0

    def frames():
        nonlocal seen
        for data in packets:
            seen += 1
            units = frames_in_message(data, cam.fmt)
            if units != 1:
                raise ValueError(
                    f"{cam.topic} message {seen} holds {units} frames, not 1"
                )
            yield from (
                f.to_ndarray(format="rgb24") for f in ctx.decode(av.Packet(data))
            )
        yield from (f.to_ndarray(format="rgb24") for f in ctx.decode(None))

    enc = [_encoder(p, h, w, ffmpeg, codec, fps) for p, (h, w) in outputs]
    try:
        decoded, have, shown = frames(), 0, None
        for want in rows:
            while have <= want:
                raw = next(decoded, None)
                if raw is None:
                    raise ValueError(f"{cam.topic} ends after {have} frames")
                shown = render(raw)
                have += 1
            for p, img in zip(enc, shown):
                p.stdin.write(img.tobytes())
        total = have + sum(1 for _ in decoded)
        if total != len(cam.t):
            raise ValueError(
                f"{cam.topic}: {len(cam.t)} messages decoded to {total} frames"
            )
        for p in enc:
            p.stdin.close()
            if p.wait() != 0:
                raise RuntimeError(f"ffmpeg ({codec}) failed")
    finally:
        for p in enc:
            if p.poll() is None:
                p.kill()
                p.wait()


def _video_name(topic: str) -> str:
    """/left-wrist-camera -> left_wrist (the zarr images.* names)."""
    return topic.strip("/").removesuffix(spec.CAMERA_SUFFIX).replace("-", "_")


# ---------------------------------------------------------------- labels


def _label_rows(series, meta, rid: str, period: float) -> list[dict]:
    """/subtask-annotation as Lance annotation rows; a point becomes one frame wide."""
    rows = []
    if "/subtask-annotation" not in series:
        return rows
    s = series["/subtask-annotation"]
    for t0, m in zip(s.t, s.msgs):
        row = dict(
            op="add",
            recording_id=rid,
            t_start_ns=int(t0),
            t_end_ns=int(m.end_time_unix_ns) or int(t0 + period),
            annotation_type=m.annotation_type or "language",
            text=m.data,
            annotation_vendor=m.annotation_vendor or meta["vendor"],
            quality_score=None,
            graded_by=None,
            rubric_version=None,
        )
        row["annotation_id"] = annotation_id(
            rid, row["annotation_vendor"], row["annotation_type"],
            row["t_start_ns"], row["t_end_ns"], row["text"],
        )  # fmt: skip
        rows.append(row)
    return rows


def _label_errors(rows: list[dict], spans: list[tuple[int, int]]) -> list[str]:
    """Lance's own annotation review rule (annotations._add_errors) against these tasks."""
    from egomimic.rldb.lance.annotations import _add_errors

    if not rows:
        return []
    tasks = {rows[0]["recording_id"]: spans}
    return [
        f"label {r['text']!r} [{r['t_start_ns']}, {r['t_end_ns']}): {e}"
        for r in rows
        for e in _add_errors(r, tasks)
    ]


def _sync_labels(root: Path, rid: str, rows: list[dict], vendors: set[str]):
    """Commit `rows`; the vendors' committed labels of `rid` not among them are deleted
    (a delivery's labels are authoritative for its vendors). Returns (added, deleted)."""
    import lance

    uri = commit.annotations_uri(root)
    new_ids = {r["annotation_id"] for r in rows}
    stale = []
    if commit._exists(uri):
        old = lance.dataset(uri).to_table(
            columns=["annotation_id", "annotation_vendor"],
            filter=f"recording_id = '{rid}'",
        )
        stale = [
            r["annotation_id"]
            for r in old.to_pylist()
            if r["annotation_vendor"] in vendors and r["annotation_id"] not in new_ids
        ]
    adds = pa.Table.from_pylist(
        [{k: r.get(k) for k in ANNOTATIONS.names} for r in rows], schema=ANNOTATIONS
    )
    if not adds.num_rows and not stale:
        return 0, 0
    out = commit.apply_annotations(root, adds, stale)
    return out["added"], out["deleted"]


# ---------------------------------------------------------------- convert


@dataclass
class Result:
    """What one episode came to; what a watcher or dashboard records."""

    episode: str
    recording_id: str = ""
    lab: str = ""
    status: str = ""  # rejected | failed | committed | replaced | skipped
    problems: list[str] = field(default_factory=list)  # why it was rejected
    error: str = ""  # why a validated episode could not be converted
    num_frames: int = 0
    filler_frames: int = 0
    tasks: int = 0
    videos: list[str] = field(default_factory=list)
    annotations_added: int = 0
    annotations_deleted: int = 0
    valid_frac: dict[str, float] = field(default_factory=dict)


@dataclass
class Converted:
    delivery: commit.Delivery
    labels: list[dict]
    vendor: str
    stats: dict


def convert(
    episode: Path,
    out_dir: Path,
    *,
    mano: bool = True,
    mano_iters: int = 2400,
    ffmpeg: str = "ffmpeg",
    codec: str = "libsvtav1",
) -> Converted:
    """One validated episode -> its Lance rows and MP4s (under out_dir). Raises Rejected
    when Lance would refuse its labels, before any video is written."""
    episode = Path(episode)
    files = [episode / spec.EPISODE_FILE]
    if (episode / spec.ANNOTATION_FILE).exists():
        files.append(episode / spec.ANNOTATION_FILE)
    series, cameras, meta = _read(files)
    embodiment = meta["embodiment"]
    if spec.profile_for(embodiment) is not spec.HUMAN:
        raise NotImplementedError(
            "robot recordings: Lance has no robot steps schema yet"
        )
    step_clock = spec.HUMAN.step_clock
    cam_t = cameras[step_clock].t
    period = float(np.median(np.diff(cam_t)))
    fps = round(1e9 / period, 3)
    ts, src, filler, after_gap = build_timeline(cam_t, fps=fps)
    n = len(ts)
    rid = spec.recording_id(meta["lab"], meta["vendor-recording-id"])

    # Tasks and labels first: a label Lance would refuse rejects the episode up front.
    if "/task" in series:
        tasks = [
            (int(t0), int(m.end_time_unix_ns), m.name, m.description, list(m.objects))
            for t0, m in zip(series["/task"].t, series["/task"].msgs)
        ]
    else:  # the episode info over the whole recording, half-open
        objects = [o for o in meta.get("objects", "").split(",") if o]
        tasks = [
            (
                int(ts[0]),
                int(ts[-1] + period),
                meta["task"],
                meta["instruction"],
                objects,
            )
        ]
    labels = _label_rows(series, meta, rid, period)
    errors = _label_errors(labels, [(t0, t1) for t0, t1, *_ in tasks])
    if errors:
        raise Rejected(errors)

    # Head: pose nearest each camera frame; SLAM resets break validity like gaps.
    frames = _Frames(series)
    hp = series["/head-pose"]
    pick, pose_close = _nearest(hp.t, cam_t)
    T_world_head = [_mat(hp.msgs[p].pose.position, hp.msgs[p].pose.orientation) for p in pick]  # fmt: skip
    calib = series[step_clock + spec.CALIBRATION_SUFFIX].msgs[0]
    extra_s = series.get(step_clock + spec.CAMERA_EXTRA_SUFFIX)
    extra = extra_s.msgs[0] if extra_s else None
    R4 = np.eye(4)
    R4[:3, :3] = _rot_z(_upright(extra))
    T_head_front = frames.static(spec.HEAD_FRAME, calib.frame_id) @ R4
    head = np.array([_xyz_wxyz(T @ T_head_front) for T in T_world_head])
    q = series["/head-pose-quality"]
    qi, q_close = _nearest(q.t, cam_t)
    state = np.array([q.msgs[i].state for i in qi])
    Q = schemas.TrackingQuality
    head_ok = pose_close & q_close & (state != Q.LOST)
    head_usable = head_ok.copy()  # what hands placed via the head need
    head_ok &= ~pose_outliers(head[:, :3], head_ok)
    # A relocalization between a frame and the one before it breaks that frame.
    reloc = q.t[[m.state == Q.RELOCALIZED for m in q.msgs]]
    seen = np.searchsorted(reloc, cam_t, side="right")
    reset = np.r_[False, seen[1:] > seen[:-1]]
    brk = after_gap | (reset[src] & ~filler)
    head_valid = head_ok[src] & ~filler & ~brk

    def col(x, valid, fill):
        x = np.asarray(x, dtype=np.float32)[src]
        x[~valid] = fill
        x = np.nan_to_num(x)
        return pa.FixedSizeListArray.from_arrays(
            pa.array(x.reshape(-1)), x.shape[1]
        ).cast(pa.list_(pa.float32(), x.shape[1]))

    cols = {
        "recording_id": pa.array([rid] * n),
        "timestamp_ns": pa.array(ts),
        "head_pose": col(head, head_valid, IDENTITY_POSE),
    }
    joints = meta["hand-joints"].split(",")
    valid = {}
    for side in ("left", "right"):
        topic = f"/{side}-hand-keypoints"
        if topic in series:
            h = _hand(
                series[topic], cam_t, T_world_head, frames, joints, head_usable,
                is_right=side == "right", mano=mano, mano_iters=mano_iters,
            )  # fmt: skip
        else:  # single-arm embodiment
            z7, z63 = np.zeros((len(cam_t), 7)), np.zeros((len(cam_t), 63))
            h = dict(
                wrist=z7, ee=z7, kp=z63, aria=z63,
                conf=np.full(len(cam_t), -1.0), ok=np.zeros(len(cam_t), dtype=bool),
            )  # fmt: skip
        v = h["ok"][src] & ~filler & ~brk
        valid[side] = v
        cols[side] = pa.StructArray.from_arrays(
            [
                col(h["wrist"], v, IDENTITY_POSE),
                col(h["ee"], v, IDENTITY_POSE),
                col(h["kp"], v, 0.0),
                col(h["aria"], v, 0.0),
                pa.array(np.where(filler, -1.0, h["conf"][src]).astype(np.float32)),
            ],
            fields=list(HUMAN_STEPS.field(side).type),
        )
    cols["left_valid"] = pa.array(valid["left"])
    cols["right_valid"] = pa.array(valid["right"])
    cols["head_pose_valid"] = pa.array(head_valid)
    eye = np.zeros((len(cam_t), 3))
    eye_ok = np.zeros(len(cam_t), dtype=bool)
    if "/eye-gaze" in series:
        g = series["/eye-gaze"]
        gi, g_close = _nearest(g.t, cam_t)
        for i, (k, close) in enumerate(zip(gi, g_close)):
            m = g.msgs[k]
            eye[i] = (m.yaw, m.pitch, m.depth)
            eye_ok[i] = close and m.valid and np.isfinite(eye[i]).all()
    cols["eye_gaze"] = col(eye, eye_ok[src] & ~filler, 0.0)
    steps = pa.table(cols, schema=HUMAN_STEPS)

    # Videos: the steps camera as the pinhole front outputs, every other camera as its
    # own video aligned to the steps rows. Extrinsics are relative to head_pose (the
    # front camera, upright).
    import cv2

    out = Path(out_dir) / rid
    out.mkdir(parents=True, exist_ok=True)
    intrinsics, extrinsics, videos = {}, {}, {}
    big_h, big_w = FRONT["front_1_480"]
    pre_w = big_h if _upright(extra) % 180 else big_w
    aria = meta["lab"] == "aria"
    focal = ARIA_FOCAL_480 if aria else float(calib.K[0]) * pre_w / float(calib.width)
    undistort = _undistorter(calib, extra, (big_h, big_w), focal)
    small_h, small_w = FRONT["front_1"]

    def render_front(raw):
        big = undistort(raw)
        return big, cv2.resize(big, (small_w, small_h), interpolation=cv2.INTER_AREA)

    outputs = [(out / f"{k}.mp4", FRONT[k]) for k in ("front_1_480", "front_1")]
    _write_camera(
        _packets(files[0], step_clock), cameras[step_clock], src, outputs,
        render_front, ffmpeg, codec, fps,
    )  # fmt: skip
    for k, (vh, vw) in FRONT.items():
        f = focal * vh / big_h
        # Aria keeps aria_to_lance's ARIA_INTRINSICS (centre at w/2, h/2).
        cx, cy = (vw / 2, vh / 2) if aria else ((vw - 1) / 2, (vh - 1) / 2)
        intrinsics[k] = [f, 0.0, cx, 0.0, f, cy, 0.0, 0.0, 1.0]
        extrinsics[k] = np.eye(4).ravel().tolist()
        videos[k] = out / f"{k}.mp4"

    for topic, cam in sorted(cameras.items()):
        if topic == step_clock:
            continue
        name = _video_name(topic)
        c = series[topic + spec.CALIBRATION_SUFFIX].msgs[0]
        x_s = series.get(topic + spec.CAMERA_EXTRA_SUFFIX)
        x = x_s.msgs[0] if x_s else None
        h, w = (
            (int(c.width), int(c.height))
            if _upright(x) % 180
            else (int(c.height), int(c.width))
        )
        f = float(c.K[0])  # native size and focal: the camera keeps its field of view
        rows, _ = _nearest(cam.t, ts)
        render = _undistorter(c, x, (h, w), f)
        _write_camera(
            _packets(files[0], topic), cam, rows, [(out / f"{name}.mp4", (h, w))],
            lambda raw, r=render: (r(raw),), ffmpeg, codec, fps,
        )  # fmt: skip
        Rc = np.eye(4)
        Rc[:3, :3] = _rot_z(_upright(x))
        T_head_cam = frames.static(spec.HEAD_FRAME, c.frame_id)
        intrinsics[name] = [f, 0.0, (w - 1) / 2, 0.0, f, (h - 1) / 2, 0.0, 0.0, 1.0]
        if T_head_cam is not None:  # a camera fixed in the world has no head extrinsic
            extrinsics[name] = (
                (np.linalg.inv(T_head_front) @ T_head_cam @ Rc).ravel().tolist()
            )
        videos[name] = out / f"{name}.mp4"

    # Catalog rows.
    start_ms = int(meta["start-time-unix"])
    recording = dict(
        recording_id=rid,
        vendor_recording_id=meta["vendor-recording-id"],
        lab=meta["lab"],
        site=meta["site"],
        embodiment=embodiment,
        rig_name=meta["rig-name"],
        operator=meta["operator-id"],
        scene=meta["scene"],
        captured_at=datetime.datetime.fromtimestamp(
            start_ms / 1000, tz=datetime.timezone.utc
        ),
        fps=fps,
        num_frames=n,
        intrinsics=sorted(intrinsics.items()),
        extrinsics=sorted(extrinsics.items()),
        license=meta["license"],
    )
    task_rows = [
        dict(
            task_instance_id=task_instance_id(rid, name, t0),
            recording_id=rid,
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
        recording_id=rid,
        t_start_ns=int(ts[0]),
        t_end_ns=int(ts[-1]),
    )
    stats = dict(
        filler_frames=int(filler.sum()),
        valid_frac={
            "head": float(head_valid.mean()),
            "left": float(valid["left"].mean()),
            "right": float(valid["right"].mean()),
        },
    )
    delivery = commit.Delivery(recording, task_rows, alias, steps, videos)
    return Converted(delivery, labels, meta["vendor"], stats)


def _labels_of_committed(episode: Path, root: Path, rid: str) -> tuple[list, str]:
    """Label rows of an episode whose recording is already committed, with errors
    against its committed tasks."""
    import lance

    files = [episode / spec.EPISODE_FILE]
    if (episode / spec.ANNOTATION_FILE).exists():
        files.append(episode / spec.ANNOTATION_FILE)
    series, cameras, meta = _read(files)
    period = float(np.median(np.diff(cameras[spec.HUMAN.step_clock].t)))
    rows = _label_rows(series, meta, rid, period)
    spans = [
        (t["t_start_ns"], t["t_end_ns"])
        for t in lance.dataset(Path(root) / "tasks.lance")
        .to_table(filter=f"recording_id = '{rid}'")
        .to_pylist()
    ]
    errors = _label_errors(rows, spans)
    if errors:
        raise Rejected(errors)
    return rows, meta["vendor"]


def convert_and_commit(
    episode: Path,
    lance_root: Path,
    *,
    replace: bool = False,
    work_dir: Path | None = None,
    **convert_kw,
) -> Result:
    """Validate, convert and commit one episode; safe to re-run. A committed recording
    is skipped, but its labels are brought up to date (labels delivered later)."""
    res = Result(episode=str(episode))
    try:
        v = validate_episode(Path(episode))
    except Exception as e:
        res.status, res.error = "failed", f"{type(e).__name__}: {e}"
        return res
    res.recording_id = v.recording_id
    if v.problems:
        res.status = "rejected"
        res.problems = [str(p) for p in v.problems]
        return res
    have = commit.committed_ids(lance_root)
    if v.recording_id in have and not replace:
        res.status = "skipped"
        try:
            rows, vendor = _labels_of_committed(
                Path(episode), lance_root, v.recording_id
            )
        except Rejected as e:
            res.problems = e.problems
            return res
        vendors = {vendor, *(r["annotation_vendor"] for r in rows)}
        res.annotations_added, res.annotations_deleted = _sync_labels(
            lance_root, v.recording_id, rows, vendors
        )
        return res
    with tempfile.TemporaryDirectory(dir=work_dir) as tmp:
        try:
            c = convert(Path(episode), Path(tmp), **convert_kw)
        except Rejected as e:  # validated, but Lance would refuse it
            res.status, res.problems = "rejected", e.problems
            return res
        except Exception as e:  # unsupported embodiment / lens, broken video, ...
            res.status, res.error = "failed", f"{type(e).__name__}: {e}"
            return res
        d = c.delivery
        res.lab = d.recording["lab"]
        if d.recording_id in have:
            commit.replace(lance_root, d, lab=res.lab)
            res.status = "replaced"
        else:
            commit.append(lance_root, [d], lab=res.lab)
            res.status = "committed"
    res.num_frames = d.recording["num_frames"]
    res.tasks = len(d.tasks)
    res.videos = sorted(d.videos)
    res.filler_frames = c.stats["filler_frames"]
    res.valid_frac = c.stats["valid_frac"]
    vendors = {c.vendor, *(r["annotation_vendor"] for r in c.labels)}
    res.annotations_added, res.annotations_deleted = _sync_labels(
        lance_root, d.recording_id, c.labels, vendors
    )
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("episodes", nargs="+", type=Path, help="episode_<id>/ dirs")
    ap.add_argument("--lance", type=Path, required=True, help="Lance root")
    ap.add_argument("--replace", action="store_true", help="re-commit changed episodes")
    ap.add_argument("--tag", help="tag every table's resulting version")
    ap.add_argument("--mano", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--mano-iters", type=int, default=2400)
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--codec", default="libsvtav1", choices=sorted(ENCODER_OPTS))
    ap.add_argument("--work-dir", type=Path, help="scratch for videos (default: tmp)")
    args = ap.parse_args(argv)

    results = []
    for ep in args.episodes:
        r = convert_and_commit(
            ep,
            args.lance,
            replace=args.replace,
            work_dir=args.work_dir,
            mano=args.mano,
            mano_iters=args.mano_iters,
            ffmpeg=args.ffmpeg,
            codec=args.codec,
        )
        results.append(r)
        print(json.dumps(r.__dict__))
    labs = {r.lab for r in results if r.status in ("committed", "replaced")}
    if args.tag and labs:
        if len(labs) > 1:
            raise SystemExit(f"--tag covers one steps table; got labs {sorted(labs)}")
        versions = commit.tag(args.lance, args.tag, lab=labs.pop())
        print(json.dumps({"tag": args.tag, "versions": versions}))
    # Rejected / failed episodes and refused labels all need a look.
    bad = [r for r in results if r.status in ("rejected", "failed") or r.problems]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
