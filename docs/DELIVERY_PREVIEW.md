# Delivery previews

Create a review artifact from a local episode:

```bash
UV_PROJECT_ENVIRONMENT=ev uv run python -m egomimic.rldb.zarr.render_check episode.zarr --out outputs/check.mp4
```

The command streams decoded RGB frames into an MP4 and writes `check.json`
alongside it. The JSON includes validation findings, per-frame overlay availability
and counts of projected keypoints in front of the camera and inside the image.
It reads only the retained `total_frames`; padded tails are not shown.

Use `--horizon 30` to overlay future observed keypoints on each current image,
`--start 100 --max-frames 90` for a clip, and `--step 3` to subsample playback.
Every point in a horizon uses the camera pose at the displayed observation.
Human `obs_keypoints` and dexterous `obs_hand_keypoints` use the same renderer
and the resolved per-hand validity masks. The inspector's keypoint overlay calls
this same episode overlay implementation.

An unavailable overlay produces a labelled RGB preview and an explanation in the
report. Exit status is 0 when all requested frames can be overlaid, 2 when some
overlays are unavailable, and 1 for an invalid input or output failure. Validation
findings are included for review and do not prevent inspecting a structural sample.
The preview is a calibration review aid; an on-screen skeleton does not prove that
camera calibration or reconstructed keypoints are accurate.

## Camera coverage and estimates

The JSON report lists ego and wrist camera coverage independently. Missing wrist
calibration does not prevent an ego overlay. A moving ego view uses the optical
`obs_head_pose` at the displayed frame, including for the whole future horizon.
Supplied static camera/arm-base transforms remain supported. A malformed declared
pose, a short trajectory, a resolution mismatch or an unsupported unrectified
camera model makes that view unavailable; raw fisheye is not projected as pinhole.

Legacy EVA camera-pose fallbacks remain available. The old `aria_*` names at the
established 640×480 or 320×240 rectified resolution can use the existing Aria
intrinsics constants when episode calibration is absent. Each fallback is recorded
in the coverage limitations. Unknown human camera sources are not assigned Aria K.

For the supplied Sharpa structural sample, create a separate analysis copy:

```bash
UV_PROJECT_ENVIRONMENT=ev uv run python -m egomimic.rldb.zarr.sharpa_preview \
  /nethome/jni66/EV-sharpa-sample/2026-05-02-06-10-27-614455.zarr \
  --out outputs/dexterous-previews/sharpa-analysis.zarr \
  --landmarks docs/examples/sharpa_preview_landmarks.json
UV_PROJECT_ENVIRONMENT=ev uv run python -m egomimic.rldb.zarr.render_check \
  outputs/dexterous-previews/sharpa-analysis.zarr \
  --out outputs/dexterous-previews/sharpa-keypoints.mp4
```

The output directory must not exist. The producer copies the sample and adds the
arm/hand/aux split, palm poses and MANO21 keypoints from the shipped URDFs. It fits
a constant optical-camera mount and two focal lengths to the provided manual image
landmarks, fixing the principal point at image centre and assuming zero distortion.
The resulting `obs_head_pose` includes observed lower-body and neck motion. Joint
commands do not drive the observed camera trajectory.

`preview_provenance` preserves the source attributes and metadata hash, model hashes,
all manual picks, fitting errors, active fitting bounds, assumptions and derived
array names. The video labels the estimated camera and computed keypoints. The
copy remains `structural_sample` and excluded from training; original timestamp,
annotation and unknown tactile-mapping limitations are retained. Agreement with
the same URDF establishes internal consistency, not independent measurement.

The example manual landmarks are approximate. The fitted mount can approach the
parameter bounds and is not uniquely established by this short motion. Inspect
the video and residual report; obtain measured camera calibration before using the
result as calibrated training data. This tool is an optional analysis producer,
not a repair step run by the training loader or an ingestion prerequisite.
