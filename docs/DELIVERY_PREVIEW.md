# Delivery previews

Install the locked environment and open a folder of local episodes:

```bash
uv sync --locked
uv run python -m egomimic.scripts.data_visualization.latent_inspector \
  --dataset-path /path/to/episodes --host 127.0.0.1 --port 8653
```

Open the displayed address. Search by episode name or annotation, select an
episode, play/pause, and scrub the frame slider. Choose **Cartesian (xyz path)**,
**Orientation (rot axes)**, or **Keypoints**. Set the horizon in frames and toggle
annotations or the existing 3D view. Episode info shows validation findings and
status eligibility; unavailable overlays retain RGB and show the reason on the
image. Use `--image-key images.left_wrist` to browse another recorded stream.
Geometry uses the stream actually displayed, independently of the coordinate
reference. If the server runs remotely, forward its port to your computer.

Export an MP4 plus a detailed JSON report using the same geometry:

```bash
uv run python -m egomimic.rldb.zarr.render_check /path/to/episode.zarr \
  --mode keypoint --camera front_1 --out outputs/episode-keypoints.mp4
uv run python -m egomimic.rldb.zarr.render_check /path/to/episode.zarr \
  --mode cartesian --horizon 30 --out outputs/episode-trajectory.mp4
```

| Inspector choice | Artifact `--mode` | Required supplied data |
|---|---|---|
| Cartesian (xyz path) | `cartesian` | Observed end-effector poses |
| Orientation (rot axes) | `orientation` | Observed end-effector poses; shows current axes |
| Keypoints | `keypoint` (default) | Human `obs_keypoints`, legacy `obs_aria_keypoints`, or robot `obs_hand_keypoints` |
| None | `none` | RGB only |

Projection modes also need intrinsics and a usable transform for the selected
camera. Pose-only Human episodes remain supported. Explicit keypoint mode
reports missing points; it never creates them. Raw Aria aliases retain Aria's
finger topology, while canonical Human and robot arrays use MANO21.

Use `--start 100 --max-frames 90` for a clip and `--step 3` to subsample playback.
Cartesian/keypoint horizons use future observations, capped at `total_frames`.
Every future point uses the optical camera pose at the displayed frame. Padded
tails are excluded. Both surfaces use the same left/right colors, finger edges,
and registered validity masks. The 3D view shows supplied coordinates in their
stored frame and preserves missing slots without connecting unrelated landmarks.

The CLI prints validation errors/warnings, status eligibility, unavailable-overlay
reasons, and estimated-data limitations. JSON retains the per-frame details and
projection counts. Exit status is 0 when all selected frames render the requested
mode, 2 when some overlays are unavailable, and 1 for an input/output failure.
**Render success is separate from delivery validation and data eligibility.**
Use the [complete dexterous validation command](CONTRIBUTING_DEXTEROUS_HANDS.md#verify-before-delivery)
before delivering robot-hand data.

## Camera coverage and estimates

Ego and wrist camera coverage are independent: missing wrist calibration does
not prevent an ego overlay. Moving ego images use `obs_head_pose`; fixed cameras
can use a valid `ref_T_cam`. Invalid rotations, short trajectories, resolution
mismatches, and unsupported unrectified optics make that view unavailable.
Hands may leave the field of view; projection counts are diagnostics.

Legacy EVA camera-pose constants remain available. Old `aria_*` names at the
established 640×480 or 320×240 rectified sizes can use the existing Aria intrinsics
when calibration is absent. The report identifies these fallbacks. Unknown
Human sources are not assigned Aria intrinsics.

The normal overlay decodes images, transforms supplied points, projects them,
and applies masks. It performs no FK generation, fitting, smoothing, or retargeting.
A plausible skeleton cannot independently prove physical calibration, joint
column meaning, or synchronization accuracy.

## Optional Sharpa sample analysis

Only for the supported 65-column structural-sample format, create a separate
analysis copy with manual landmarks from **your own source recording**:

```bash
uv run python -m egomimic.rldb.zarr.sharpa_preview \
  /path/to/sharpa-structural-sample.zarr \
  --out outputs/sharpa-analysis.zarr \
  --landmarks /path/to/manual-landmarks.json
uv run python -m egomimic.rldb.zarr.render_check outputs/sharpa-analysis.zarr \
  --out outputs/sharpa-analysis.mp4
```

The output directory must not exist. The optional producer computes palm poses
and keypoints from URDFs and fits a constant camera mount and focal lengths to
manual landmarks, assuming a centered principal point and zero distortion.
The [example landmark file](examples/sharpa_preview_landmarks.json) describes one
particular recording; do not reuse its picks on a different recording.

`preview_provenance` records the source, models, landmarks, fit residuals, bounds,
and assumptions. These are estimates, not measured delivery evidence. The copy
remains `structural_sample`, its video is labeled, and staging/resolvers exclude
it. This optional analysis is not an ingestion prerequisite. Obtain measured
camera calibration and synchronized supplied arrays for complete deliveries.
