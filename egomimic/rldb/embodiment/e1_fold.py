"""Keymap and transform list for the E1 fold speed-spread rows (mecka human_bimanual).

Three variants share one pipeline (world → head frame → each wrist's own frame,
euler rotation, zero gripper pad → (T, 14)) and differ only in the target:

  time     (100, 14)  raw 30 Hz chunk, 3.3 s — the protocol's Time row
  arcmean  (101, 14)  the branch's arc token, velocity row re-normed to PATH speed
  arcvel   (100, 16)  Arc+Vel: waypoints + per-arm speed profile, integral clock
  arclogdur (100, 16) tempo ablation (#1+#2): waypoints + per-arm log mean
                      slowness (row 0) and log relative segment durations
                      (rows 1..); with ``progress_smooth_hz`` set, arc length
                      is accumulated on 3 Hz low-passed positions (#4)

Every variant also carries ``actions_time`` = the first ``time_rows`` rows of the
un-tokenized chunk, which is what the E1 evaluator scores against.

Why not ``Human.get_keymap('arc_tokenizer_cartesian')``: that keymap reads a
600-frame raw window that ``InterpolatePose`` squeezes to 100 samples, so the
tokenizer's ``dt`` no longer matches the sample spacing (6x off for mecka at
stride 1). Here the raw window equals ``chunk_length`` and the interpolation is
the identity, so a sample is 1/30 s throughout.
"""

from __future__ import annotations

from egomimic.rldb.embodiment.human import (
    Human,
    _build_human_cartesian_eef_frame_transform_list,
    _pad_human_cartesian_gripper,
)
from egomimic.rldb.zarr.e1_arc_tokenizer import CopyKeyRows, TokenizeBimanualArcLengthE1

VARIANTS = ("time", "arcmean", "arcvel", "arclogdur")
VELOCITY_MODES = {"arcmean": "mean", "arcvel": "profile", "arclogdur": "logdur"}


def get_keymap(horizon: int, keymap_mode: str = "cartesian", **kwargs):
    """Plain cartesian keymap with every action key's raw window set to ``horizon``."""
    key_map = Human.get_keymap(keymap_mode, **kwargs)
    for spec in key_map.values():
        if "horizon" in spec:
            spec["horizon"] = int(horizon)
    return key_map


def get_transform_list(
    variant: str,
    chunk_length: int,
    time_rows: int = 100,
    min_distance_unit: float = 0.40,
    resampled_vector_length: int = 100,
    stride: int = 1,
    rotation_mode: str = "euler",
    speed_smooth_frames: int = 7,
    velocity_norm: str = "path",
    progress_smooth_hz: float | None = None,
):
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    tl = _build_human_cartesian_eef_frame_transform_list(
        stride=int(stride), rotation_mode=rotation_mode, chunk_length=int(chunk_length)
    )
    tl = _pad_human_cartesian_gripper(tl, rotation_mode=rotation_mode)
    tl.append(CopyKeyRows("actions_cartesian", "actions_time", int(time_rows)))
    if variant != "time":
        tl.append(
            TokenizeBimanualArcLengthE1(
                action_key="actions_cartesian",
                output_action_key="actions_cartesian",
                min_distance_unit=float(min_distance_unit),
                resampled_vector_length=int(resampled_vector_length),
                dt=float(stride) / 30.0,
                velocity_norm=velocity_norm,
                velocity_mode=VELOCITY_MODES[variant],
                speed_smooth_frames=int(speed_smooth_frames),
                progress_smooth_hz=progress_smooth_hz,
            )
        )
    return tl
