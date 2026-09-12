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
