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
  arcdur   (100, 16) the port of Ryan's duration codec (EgoVerse-graph
                      6d1b5f93): the same timing content as arclogdur, carried
                      as the ABSOLUTE elapsed seconds of each waypoint interval
                      instead of a log mean slowness plus a log relative
                      profile

Every variant also carries ``actions_time`` = the first ``time_rows`` rows of the
un-tokenized chunk, which is what the E1 evaluator scores against.

``embodiment="yam"`` runs the same three rows on the ABC / YAM robot episodes:
the Yam ``cartesian`` keymap without the two wrist cameras (not transferred to
ICE; E1 uses the front image only, as on mecka) and the Yam eef-frame transform
list, which already yields the (T, 14) [xyz ypr grip] x {L, R} layout with a
real gripper — no padding step.

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

VARIANTS = ("time", "arcmean", "arcvel", "arclogdur", "arcdur")
VELOCITY_MODES = {"arcmean": "mean", "arcvel": "profile", "arclogdur": "logdur", "arcdur": "dur"}


def get_keymap(horizon: int, keymap_mode: str = "cartesian", embodiment: str = "human", drop_wrist_images: bool = True, **kwargs):
    """Plain cartesian keymap with every action key's raw window set to ``horizon``."""
    if embodiment == "human":
        key_map = Human.get_keymap(keymap_mode, **kwargs)
    elif embodiment == "yam":
        from egomimic.rldb.embodiment.yam import Yam

        key_map = Yam.get_keymap(keymap_mode, **kwargs)
        if drop_wrist_images:
            key_map = {k: v for k, v in key_map.items() if v.get("zarr_key") not in ("images.left_wrist", "images.right_wrist")}
    else:
        raise ValueError(f"embodiment must be 'human' or 'yam', got {embodiment!r}")
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
    embodiment: str = "human",
    fixed_spacing: bool = False,
):
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    if embodiment == "human":
        tl = _build_human_cartesian_eef_frame_transform_list(
            stride=int(stride), rotation_mode=rotation_mode, chunk_length=int(chunk_length)
        )
        tl = _pad_human_cartesian_gripper(tl, rotation_mode=rotation_mode)
    elif embodiment == "yam":
        from egomimic.rldb.embodiment.yam import _build_yam_bimanual_eef_frame_transform_list

        tl = _build_yam_bimanual_eef_frame_transform_list(
            stride=int(stride), rotation_mode=rotation_mode, chunk_length=int(chunk_length)
        )
    else:
        raise ValueError(f"embodiment must be 'human' or 'yam', got {embodiment!r}")
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
                fixed_spacing=bool(fixed_spacing),
            )
        )
    return tl
