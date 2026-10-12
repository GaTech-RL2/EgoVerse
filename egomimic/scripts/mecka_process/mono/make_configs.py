"""Write the private operator-scaling data + train configs (1/3/6/10 operators).

Held out per private operator: 5 episodes nearest that operator's median length,
among converted episodes with no glitch / frozen-camera / missing-hand frames
(an operator with fewer than 5 clean ones falls back to its least-glitched).
The video pin per operator is the nearest-to-median of those 5.
Unseen operators: 5 SQL mecka folding_clothes operators the user picked, 5
clean episodes each (the user's named episodes always included).

  python -m egomimic.scripts.mecka_process.mono.make_configs {fold_tshirt,fold_no_tshirt}

Inputs and picks live in $EGOVERSE_MECKA_PRIVATE_DIR/scaling; the configs' IDs go
to $EGOVERSE_MECKA_PRIVATE_DIR/splits (private_ids.py).
"""

import json
import sys
from pathlib import Path

import pandas as pd

from egomimic.scripts.mecka_process.mono.private_ids import (
    ZARR_DIR,
    private_dir,
    write_config,
)

PRIV = private_dir()
HERE = PRIV / "scaling"
SPLITS = PRIV / "splits"
REPO = Path(__file__).resolve().parents[4]
CFG = REPO / "egomimic/hydra_configs"
SQL_PKL = HERE / "eps_sql_20260930.pkl"  # episodes table snapshot
N_HELD = 5
SCALES = (1, 3, 6, 10)

# The user's picks (2026-09-30), in opscale_inputs.json per task: unseen_viz, the viz
# episode of each of three operators; unseen_named, the other two operators, each
# with an episode that must be included (None = any). epochs: 100-step epochs per
# scale, max(60k steps, ~4 passes over the train split); launch_pair.sbatch derives
# the WSD phases from them.
TASKS = {
    "fold_tshirt": dict(
        slug="tshirt",
        order="fold_tshirt_operator_order.json",
        picks="opscale_picks.json",
        unseen_no_tshirt=False,
        epochs={1: 600, 3: 600, 6: 1000, 10: 1650},
    ),
    # A different task, so its own unseen set: episodes whose description names no
    # t-shirt. Only one of the four checkered-cloth operators (one shared room).
    "fold_no_tshirt": dict(
        slug="notshirt",
        order="fold_no_tshirt_operator_order.json",
        picks="opscale_picks_notshirt.json",
        unseen_no_tshirt=True,
        epochs={1: 600, 3: 600, 6: 850, 10: 1400},
    ),
}
TASK = sys.argv[1] if len(sys.argv) > 1 else "fold_tshirt"
SPEC = TASKS[TASK] | json.loads((HERE / "opscale_inputs.json").read_text())[TASK]
SLUG = SPEC["slug"]


def nearest_median(df: pd.DataFrame, length: str, n: int, must=()) -> list[str]:
    med = df[length].median()
    df = df.assign(d=(df[length] - med).abs()).sort_values(["d", "episode_hash"])
    must = [h for h in must if h is not None]
    if missing := set(must) - set(df.episode_hash):
        raise ValueError(f"named episodes filtered out: {sorted(missing)}")
    picks = list(must)
    picks += [h for h in df.episode_hash if h not in picks][: n - len(picks)]
    if len(picks) < n:
        raise ValueError(f"only {len(picks)} eligible episodes")
    return picks


def private_rows() -> pd.DataFrame:
    """From zarr/_episode_index.json (build_index.py): opening every zarr on NFS
    is exactly the small-op load that takes the GPU nodes down."""
    index = json.loads((PRIV / "zarr" / "_episode_index.json").read_text())
    rows = []
    for episode_hash, a in sorted(index.items()):
        if a.get("task") != TASK:
            continue
        qa = a["qa"]
        rows.append(
            {
                "episode_hash": episode_hash,
                "user_id": a["user_id"],
                "frames": a["total_frames"],
                "glitch": qa["glitch_frames"],
                "usable": qa["frozen_cam_frames"] == 0 and qa["zero_hand_frames"] == 0,
                "clean": qa["glitch_frames"] == 0
                and qa["frozen_cam_frames"] == 0
                and qa["zero_hand_frames"] == 0,
            }
        )
    return pd.DataFrame(rows)


def unseen_picks() -> dict[str, list[str]]:
    df = pd.read_pickle(SQL_PKL)
    bl = set(
        (REPO / "egomimic/resources/blocklists/mecka_bad_egomotion.txt")
        .read_text()
        .split()
    )
    f = df[
        (df.lab.str.lower() == "mecka")
        & (df.task.str.lower() == "folding_clothes")
        & ~df.is_deleted.fillna(False).astype(bool)
        & ~df.episode_hash.isin(bl)
    ]
    if SPEC["unseen_no_tshirt"]:
        desc = f.task_description.fillna("t-shirt").str.lower()
        f = f[~desc.str.contains(r"t-?shirt|tee|polo")]
    out = {}
    for op, named in {**SPEC["unseen_viz"], **SPEC["unseen_named"]}.items():
        out[op] = nearest_median(
            f[f.operator == op], "num_frames", N_HELD, must=[named]
        )
    return out


HIST_ANCHOR = "        annotation_key: annotations\n"
# The private heads share the train resolver (history settings come from the
# train config); the unseen head builds its own key_map and must match them.
HIST_LINES = "".join(
    f"        {k}: ${{data.train_datasets.human_bimanual.resolver.key_map.{k}}}\n"
    for k in ("proprio_history", "history_stride", "image_history_gap_s")
)


def with_mask(text: str) -> str:
    """Every dataset node drops the start frames that read a camera-pose jump."""
    return text.replace(
        "\n    mode: total\n",
        "\n    pose_glitch_mask: {step_m: 0.05, rot_deg: 10.0}\n    mode: total\n",
    )


def ylist(items, comments=None, indent=2) -> str:
    pad = " " * indent
    comments = comments or {}
    return "\n".join(
        f"{pad}- {x}" + (f" # {comments[x]}" if x in comments else "") for x in items
    )


def held_out(rows: pd.DataFrame, op: str) -> list[str]:
    g = rows[(rows.user_id == op) & rows.clean]
    if len(g) < N_HELD:
        g = rows[(rows.user_id == op) & rows.usable]
        g = g[g.glitch <= g.glitch.nsmallest(N_HELD).max()]
    return nearest_median(g, "frames", N_HELD)


def main() -> None:
    order = json.load(open(HERE / SPEC["order"]))["order"]
    rows = private_rows()
    held, viz = {}, {}
    for op in order:
        held[op] = held_out(rows, op)
        viz[op] = held[op][0]
    all_held = [h for op in order for h in held[op]]
    op1 = order[0]
    op1_train = rows[
        (rows.user_id == op1) & rows.clean & ~rows.episode_hash.isin(all_held)
    ]
    train_viz_pin = nearest_median(op1_train, "frames", 1)[0]
    unseen = unseen_picks()
    unseen_hashes = [h for op in unseen for h in unseen[op]]
    first_of = {held[op][0]: f"op{i + 1} {op}" for i, op in enumerate(order)}
    first_unseen = {unseen[op][0]: op for op in unseen}

    stats = rows.groupby("user_id").agg(
        eps=("episode_hash", "size"), clean=("clean", "sum")
    )
    summary = {
        "order": order,
        "held_out": held,
        "viz": viz,
        "train_viz_pin": train_viz_pin,
        "unseen": unseen,
        "unseen_viz": list(SPEC["unseen_viz"].values()),
        "converted_eps_per_op": stats.loc[order].to_dict("index"),
    }
    json.dump(summary, open(HERE / SPEC["picks"], "w"), indent=1)

    km_tl = """      key_map:
        _target_: egomimic.rldb.embodiment.human.Human.get_keymap
        keymap_mode: keypoints
        annotation_key: annotations
      transform_list:
        _target_: egomimic.rldb.embodiment.human.Human.get_transform_list
        mode: keypoints_wristframe_6d
        stride: 1
        fix_left_wrist_convention: {fix}
        pad_proprio_gripper: false"""

    base = f"""# PRIVATE Mecka mono {TASK}, operator scaling (1/3/6/10 operators), the
# 1-operator run. Generated by egomimic/scripts/mecka_process/mono/make_configs.py;
# do not edit by hand. IDs: $EGOVERSE_MECKA_PRIVATE_DIR/splits (private to the lab).
#   train            = the first N operators of the fixed order, minus every
#                      operator's 5 held-out episodes ({TASK} only)
#   valid            = operator 1's 5 held-out episodes (seen_op1_valid)
#   unseen_op_valid  = 5 SQL mecka folding_clothes operators x 5 episodes{" (no t-shirt)" if SPEC["unseen_no_tshirt"] else ""}
# Operator 1 and the other held-out picks: scaling/{SPEC["picks"]} in the private dir.
# The private zarrs are post-fix left wrist; the SQL mecka zarrs are not.
_target_: egomimic.pl_utils.pl_data_utils.MultiDataModuleWrapper

train_operators:
{ylist(order[:1])}

# 5 per operator for all 10, so every scale trains on the same episodes of the
# operators it shares.
held_out_hashes:
{ylist(all_held, first_of)}

valid_prefix: seen_op1_valid

train_datasets:
  human_bimanual:
    _target_: egomimic.rldb.zarr.zarr_dataset_multi.MultiDataset._from_resolver
    resolver:
      _target_: egomimic.rldb.zarr.zarr_dataset_multi.LocalEpisodeResolver
      folder_path: {ZARR_DIR}
{km_tl.format(fix="false")}
    filters:
      _target_: egomimic.rldb.filters.DatasetFilter
      filter_lambdas:
        - "lambda row: str(row.get('task', '')) == '{TASK}'"
        - "lambda row: str(row.get('user_id', '')) in ${{data.train_operators}}"
        - "lambda row: str(row.get('episode_hash', '')) not in ${{data.held_out_hashes}}"
    mode: total

# Eval heads on other episodes widen the bounds to 10x each cell's quantile range
# (few operators' stats; see mecka_freeform_fold_kp_op2_holdout). train_viz is
# the stats' own split and keeps the default.
valid_datasets:
  human_bimanual:
    _target_: ${{data.train_datasets.human_bimanual._target_}}
    resolver: ${{data.train_datasets.human_bimanual.resolver}}
    bounds_rel_slack: 10.0
    filters:
      _target_: egomimic.rldb.filters.DatasetFilter
      episode_hashes:
{ylist(held[op1], indent=8)}
    mode: total

train_viz_datasets:
  human_bimanual:
    _target_: ${{data.train_datasets.human_bimanual._target_}}
    resolver: ${{data.train_datasets.human_bimanual.resolver}}
    filters: ${{data.train_datasets.human_bimanual.filters}}
    mode: total

unseen_op_valid_datasets:
  human_bimanual:
    _target_: ${{data.train_datasets.human_bimanual._target_}}
    bounds_rel_slack: 10.0
    resolver:
      _target_: egomimic.rldb.zarr.zarr_dataset_multi.S3EpisodeResolver
      folder_path: ${{paths.dataset_dir}}
{km_tl.format(fix="true").replace(HIST_ANCHOR, HIST_ANCHOR + HIST_LINES)}
    filters:
      _target_: egomimic.rldb.filters.DatasetFilter
      filter_lambdas:
        - "lambda row: str(row.get('lab', '')).lower() == 'mecka'"
      episode_hashes:
{ylist(unseen_hashes, first_unseen, indent=8)}
    mode: total

train_dataloader_params:
  human_bimanual:
    batch_size: 32
    num_workers: 12

valid_dataloader_params:
  human_bimanual:
    batch_size: 32
    num_workers: 8
    shuffle: false

train_viz_dataloader_params:
  human_bimanual:
    batch_size: 32
    num_workers: 4
    shuffle: false

unseen_op_valid_dataloader_params:
  human_bimanual:
    batch_size: 32
    num_workers: 8
    shuffle: false

metric_frames_per_episode:
  valid: auto
  train_viz: auto
  unseen_op_valid: auto

video_episodes:
  valid:
    - {viz[op1]}
  train_viz:
    - {train_viz_pin}
  unseen_op_valid:
{ylist(SPEC["unseen_viz"].values(), indent=4)}
"""
    write_config(
        CFG / f"data/mecka_private_{SLUG}_opscale_1.yaml", with_mask(base), SPLITS
    )

    three = f"""# mecka_private_{SLUG}_opscale_1 on operators 1-3, plus a second seen head:
# seen_op123_valid (extra_valid) = the 5 held-out episodes of each of operators
# 1-3. Its videos are operators 2 and 3 (operator 1's is on valid). Generated;
# see mecka_private_{SLUG}_opscale_1.
defaults:
  - mecka_private_{SLUG}_opscale_1
  - _self_

train_operators:
{ylist(order[:3])}

extra_valid_prefix: seen_op123_valid

extra_valid_datasets:
  human_bimanual:
    _target_: ${{data.train_datasets.human_bimanual._target_}}
    resolver: ${{data.train_datasets.human_bimanual.resolver}}
    bounds_rel_slack: 10.0
    filters:
      _target_: egomimic.rldb.filters.DatasetFilter
      episode_hashes:
{ylist([h for op in order[:3] for h in held[op]], first_of, indent=8)}
    mode: total

extra_valid_dataloader_params:
  human_bimanual:
    batch_size: 32
    num_workers: 8
    shuffle: false

metric_frames_per_episode:
  extra_valid: auto

video_episodes:
  extra_valid:
    - {viz[order[1]]}
    - {viz[order[2]]}
"""
    write_config(
        CFG / f"data/mecka_private_{SLUG}_opscale_3.yaml", with_mask(three), SPLITS
    )
    for n in (6, 10):
        write_config(
            CFG / f"data/mecka_private_{SLUG}_opscale_{n}.yaml",
            f"""# mecka_private_{SLUG}_opscale_3 on operators 1-{n}; the same val heads.
# Generated; see mecka_private_{SLUG}_opscale_1.
defaults:
  - mecka_private_{SLUG}_opscale_3
  - _self_

train_operators:
{ylist(order[:n])}
""",
            SPLITS,
        )
    for n in SCALES:
        epochs = SPEC["epochs"][n]
        write_config(
            CFG / f"train_zarr_mecka_private_{SLUG}_opscale_{n}.yaml",
            f"""# RDT-1B keypoint recipe of train_zarr_mecka_freeform_fold_kp_rdt_op2 (4 GPUs x
# 32, WSD LR from the launcher, EMA on) on {n} private {TASK} operator(s).
# {epochs} x 100 steps: max(60k, ~4 passes over the train split). Generated; see
# data/mecka_private_{SLUG}_opscale_1.
defaults:
  - train_zarr_mecka_freeform_fold_kp_rdt_op2
  - override data: mecka_private_{SLUG}_opscale_{n}
  - _self_

model:
  ema:
    enabled: true

trainer:
  max_epochs: {epochs}
  min_epochs: {epochs}

# Every 20% (~23 GB each with EMA); the 80% one is the decay start, for resuming with more steps.
callbacks:
  model_checkpoint:
    every_n_epochs: {epochs // 5}
""",
            SPLITS,
        )
    print(
        json.dumps(
            {k: summary[k] for k in ("viz", "train_viz_pin", "converted_eps_per_op")},
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
