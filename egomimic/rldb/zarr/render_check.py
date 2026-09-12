"""Render a local episode's keypoint overlay and write a JSON review report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np
import zarr

from egomimic.rldb.zarr.camera_coverage import camera_coverage_report
from egomimic.rldb.zarr.overlay import (
    OverlayUnavailable,
    decode_frame,
    episode_length,
    render_keypoints,
)
from egomimic.rldb.zarr.validate import validate_episode


def render_episode(
    path, out, *, camera="front_1", horizon=1, start=0, max_frames=None, step=1
):
    """Stream a preview; missing overlays remain visible and are reported.

    Video frames stop at total_frames. The adjacent .json carries validation
    findings and projection counts; a preview does not certify calibration.
    """
    path, out = Path(path), Path(out)
    if out.resolve().is_relative_to(path.resolve()):
        raise ValueError("write the preview outside the source episode")
    group = zarr.open_group(path, mode="r")
    total = episode_length(group)
    if (
        step < 1
        or horizon < 1
        or not 0 <= start < total
        or (max_frames is not None and max_frames < 1)
    ):
        raise ValueError(
            "start, step, horizon and max_frames must select a nonempty episode range"
        )
    frames = range(start, total, step)
    if max_frames is not None:
        frames = frames[:max_frames]
    first = decode_frame(group, start, camera)
    height, width = first.shape[:2]
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "episode": str(path.resolve()),
        "camera": camera,
        "horizon": horizon,
        "total_frames": total,
        "camera_coverage": camera_coverage_report(group, start),
        "provenance": group.attrs.get("preview_provenance"),
        "validation": validate_episode(path).to_jsonable(),
        "frames": [],
    }
    with av.open(str(out), mode="w") as container:
        from fractions import Fraction

        stream = container.add_stream(
            "libx264", rate=Fraction(int(group.attrs.get("fps", 30)), step)
        )
        stream.width, stream.height, stream.pix_fmt = (
            width + width % 2,
            height + height % 2,
            "yuv420p",
        )
        stream.options = {"crf": "20", "preset": "fast"}
        for frame in frames:
            image = first if frame == start else decode_frame(group, frame, camera)
            try:
                image, diagnostic = render_keypoints(
                    group, frame, image=image, horizon=horizon, camera=camera
                )
                diagnostic["available"] = True
                label = None
                if diagnostic["coverage"]["estimated"]:
                    label = "ESTIMATED CAMERA / FK KEYPOINTS - analysis only"
                elif any(
                    f["level"] != "ok"
                    and f["check"]
                    in (
                        "pose_degeneracy",
                        "calibration_degeneracy",
                        "intrinsics_signature",
                    )
                    for f in report["validation"]["findings"]
                ):
                    label = "Calibration/pose diagnostics - review JSON"
                if label:
                    cv2.rectangle(image, (0, 0), (width, 26), (0, 0, 0), -1)
                    cv2.putText(
                        image,
                        label,
                        (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        min(0.42, width / 1200),
                        (255, 210, 60),
                        1,
                    )
            except (OverlayUnavailable, ValueError, KeyError) as exc:
                diagnostic = {"available": False, "reason": str(exc)}
                cv2.rectangle(image, (0, 0), (width, 32), (0, 0, 0), -1)
                cv2.putText(
                    image,
                    "Overlay unavailable - see JSON report",
                    (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 200, 70),
                    1,
                )
            report["frames"].append({"frame": frame, **diagnostic})
            if image.shape[:2] != (stream.height, stream.width):
                image = cv2.copyMakeBorder(
                    image,
                    0,
                    stream.height - height,
                    0,
                    stream.width - width,
                    cv2.BORDER_CONSTANT,
                )
            for packet in stream.encode(
                av.VideoFrame.from_ndarray(np.ascontiguousarray(image), format="rgb24")
            ):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    report["overlay_frames"] = sum(item["available"] for item in report["frames"])
    out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--camera", default="front_1")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--step", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        report = render_episode(
            args.episode,
            args.out,
            camera=args.camera,
            horizon=args.horizon,
            start=args.start,
            max_frames=args.max_frames,
            step=args.step,
        )
    except (ValueError, OSError) as exc:
        parser.exit(1, f"render_check: {exc}\n")
    print(
        f"{args.out}: overlays on {report['overlay_frames']}/{len(report['frames'])} frames; report {args.out.with_suffix('.json')}"
    )
    return 0 if report["overlay_frames"] == len(report["frames"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
