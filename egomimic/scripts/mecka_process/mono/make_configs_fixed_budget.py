"""Write the fixed-budget operator-scaling ladder for the private fold episodes.

Every rung trains on the same total hours B, split equally over its k operators
(B/k each), so the ladder varies operator diversity at constant data.

  pool      on-task fold_tshirt + fold_no_tshirt episodes (manifest.csv) that
            converted cleanly (no frozen camera / missing hands), minus the
            egomotion blocklist and, with --strict-fold (default), captions that
            name a non-folding action (categorize.NON_FOLDING_ACTIONS). Pose
            glitches stay in, masked by pose_glitch_mask as in the other ladders.
  homes     one operator per same-home group (scaling/fixed_budget_inputs.json):
            the member with both categories, else the most hours. Episodes filmed
            in another ladder operator's home are dropped. Same-home operators
            would make "operators" and "homes" different counts.
  holdout   N_HELD seeded-random episodes per (operator, category), clean and of
            the dominant task_id where possible, out of every rung.
  mix       the t-shirt share of every rung is --tshirt-share (default 0.5). Most
            operators fold only one category, so the share is set per rung: the
            operators that fold both absorb the imbalance of the single-category
            ones. Their per-category hours never grow from one rung to the next.
  order     seeded permutations of the operators, kept if every rung's mix is
            reachable and nested; the one with the largest budget wins, ties to
            the one keeping the two-category operators' own mix nearest the share.
  budget    B = min over rungs, operators and categories of k x available / share.
  packing   per (operator, category), from the largest k (smallest share) down:
            whole episodes up to the target, longest first (fill); each smaller k
            keeps the previous pick and tops up from the whole cell, so picks nest.
            Rare task_ids are packed first while they fit, so they survive into
            every rung the operator is in.

Val heads: seen_op1_valid = operator 1's holdouts (operator 1 is in every rung);
seen_ops_valid (k > 1) = the holdouts of the rung's operators; unseen_op_valid =
the union of the t-shirt and no-t-shirt opscale SQL unseen-operator sets.

  python -m egomimic.scripts.mecka_process.mono.make_configs_fixed_budget [--plan-only]

Picks go to scaling/fixed_budget_picks.json, config IDs to the private splits dir.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from egomimic.scripts.mecka_process.mono.categorize import non_folding_actions
from egomimic.scripts.mecka_process.mono.private_ids import (
    ZARR_DIR,
    private_dir,
    write_config,
)

REPO = Path(__file__).resolve().parents[4]
CFG = REPO / "egomimic/hydra_configs"
TSHIRT, NO_TSHIRT = "fold_tshirt", "fold_no_tshirt"
CATS = (TSHIRT, NO_TSHIRT)
FPS = 30
RUNGS = (1, 2, 4, 8, 14)
N_HELD = 5
SEED = 0
TRIES = 20000
# 100-step epochs, equal on every rung since the data budget is.
EPOCHS = 1000
GLOBAL_BATCH = 128


@dataclass(frozen=True)
class Ep:
    id: str
    op: str
    cat: str
    frames: int
    task_id: str = "folding_clothes"
    clean: bool = True


def hours(frames: float) -> float:
    return frames / FPS / 3600


def fill(pool: list[Ep], target: int, required: set[str] = frozenset()) -> list[Ep]:
    """Whole episodes summing to at most ``target``: ``required``, then greedy longest-first.

    No unpicked episode fits in what is left. PR #710 follows this with one-for-one
    swaps to a longer episode, but after a longest-first pass none can fit: an
    episode skipped for being too long is longer than the room left plus anything
    picked after it.
    """
    chosen = [e for e in pool if e.id in required]
    left = target - sum(e.frames for e in chosen)
    if left < 0:
        raise ValueError("required episodes exceed the target")
    for e in sorted(pool, key=lambda e: (-e.frames, e.id)):
        if e.id not in required and e.frames <= left:
            chosen.append(e)
            left -= e.frames
    return sorted(chosen, key=lambda e: e.id)


def collapse_same_home(
    eps: list[Ep], groups: list[list[str]]
) -> tuple[list[Ep], list[str]]:
    """Keep one operator per same-home group; return the kept episodes and dropped ops."""
    cats, frames = defaultdict(set), Counter()
    for e in eps:
        cats[e.op].add(e.cat)
        frames[e.op] += e.frames
    dropped = []
    for group in groups:
        present = [op for op in group if op in frames]
        if len(present) < 2:
            continue
        keep = max(present, key=lambda op: (len(cats[op]), frames[op], op))
        dropped += [op for op in present if op != keep]
    return [e for e in eps if e.op not in dropped], dropped


def pick_holdouts(eps: list[Ep], n: int, seed: int) -> dict[str, list[Ep]]:
    """``n`` seeded-random episodes per (operator, category), clean + common first."""
    common = Counter(e.task_id for e in eps).most_common(1)[0][0]
    cells = defaultdict(list)
    for e in eps:
        cells[(e.op, e.cat)].append(e)
    held = defaultdict(list)
    for (op, cat), cell in sorted(cells.items()):
        cell = sorted(cell, key=lambda e: e.id)
        rng = random.Random(f"{seed}:{op}:{cat}")
        rng.shuffle(cell)
        cell.sort(key=lambda e: (not e.clean, e.task_id != common))
        if len(cell) <= n:
            raise ValueError(f"{op} {cat}: {len(cell)} episodes, holdout needs > {n}")
        held[op] += cell[:n]
    return dict(held)


def rung_fractions(
    order: list[str], rungs, op_cats: dict[str, set], share: float
) -> dict[int, dict[str, dict[str, float]]] | None:
    """Per rung, each operator's fraction of its B/k hours per category, or None.

    Single-category operators give all of it to their category; the two-category
    ones split theirs to bring the rung to ``share`` t-shirt hours.
    """
    out = {}
    for k in rungs:
        ops = order[:k]
        n_t = sum(op_cats[op] == {TSHIRT} for op in ops)
        n_d = sum(len(op_cats[op]) == 2 for op in ops)
        if n_d:
            x = (share * k - n_t) / n_d
            if not -1e-9 <= x <= 1 + 1e-9:
                return None
            x = min(max(x, 0.0), 1.0)
        elif abs(n_t - share * k) > 1e-9:
            return None
        fr = {}
        for op in ops:
            if len(op_cats[op]) == 2:
                fr[op] = {TSHIRT: x, NO_TSHIRT: 1 - x}
            else:
                (cat,) = op_cats[op]
                fr[op] = {cat: 1.0}
        out[k] = fr
    return out


def nested(fracs: dict, rungs) -> bool:
    """Per operator and category, the hours never grow with k (picks can nest)."""
    prev = {}
    for k in rungs:
        for op, fr in fracs[k].items():
            for cat in CATS:
                h = fr.get(cat, 0.0) / k
                if h > prev.get((op, cat), float("inf")) + 1e-12:
                    return False
                prev[(op, cat)] = h
    return True


def budget(fracs: dict, avail: dict[str, Counter]) -> float:
    return min(
        k * avail[op][cat] / f
        for k, fr in fracs.items()
        for op, cats in fr.items()
        for cat, f in cats.items()
        if f > 0
    )


def choose_order(
    avail: dict[str, Counter], rungs, share: float, seed: int, tries: int = TRIES
) -> tuple[list[str], dict, float]:
    ops = sorted(avail)
    if len(ops) < max(rungs):
        raise ValueError(f"{len(ops)} operators, the top rung needs {max(rungs)}")
    op_cats = {op: {c for c in CATS if avail[op][c] > 0} for op in ops}
    rng = random.Random(seed)
    best = None
    for _ in range(tries):
        order = ops[:]
        rng.shuffle(order)
        fracs = rung_fractions(order, rungs, op_cats, share)
        if fracs is None or not nested(fracs, rungs):
            continue
        dev = sum(
            abs(fracs[k][op][TSHIRT] - share)
            for k in rungs
            for op in fracs[k]
            if len(op_cats[op]) == 2
        )
        score = (round(budget(fracs, avail)), -dev)
        if best is None or score > best[0]:
            best = (score, order[: max(rungs)], fracs)
    if best is None:
        raise ValueError("no operator order reaches the t-shirt share on every rung")
    return best[1], best[2], best[0][0]


def plan(eps: list[Ep], rungs=RUNGS, share=0.5, seed=SEED, n_held=N_HELD) -> dict:
    """The ladder: operator order, holdouts, budget and nested picks per rung."""
    held = pick_holdouts(eps, n_held, seed)
    held_ids = {e.id for v in held.values() for e in v}
    pool = defaultdict(list)
    for e in eps:
        if e.id not in held_ids:
            pool[(e.op, e.cat)].append(e)
    avail = defaultdict(Counter)
    for (op, cat), cell in pool.items():
        avail[op][cat] = sum(e.frames for e in cell)
    order, fracs, b = choose_order(avail, rungs, share, seed)
    common = Counter(e.task_id for e in eps).most_common(1)[0][0]

    picks = {k: defaultdict(list) for k in rungs}
    for op in order:
        for cat in CATS:
            cell = pool.get((op, cat), [])
            rare = sorted(
                (e for e in cell if e.task_id != common), key=lambda e: (e.frames, e.id)
            )
            req: set[str] = set()
            # Largest k (smallest target) first; each smaller k keeps that pick and
            # tops up from the whole cell, so every rung packs from the full pool.
            for k in sorted(rungs, reverse=True):
                f = fracs[k].get(op, {}).get(cat, 0.0)
                if f == 0:
                    continue
                target = int(b * f / k)
                used = sum(e.frames for e in cell if e.id in req)
                for e in rare:
                    if e.id not in req and used + e.frames <= target:
                        req.add(e.id)
                        used += e.frames
                cur = fill(cell, target, req)
                req = {e.id for e in cur}
                picks[k][op] += cur
    return {
        "order": order,
        "budget_frames": b,
        "held_out": held,
        "fractions": fracs,
        "rungs": {k: dict(v) for k, v in picks.items()},
    }


def plan_rows(p: dict) -> list[dict]:
    rows = []
    for k, per_op in p["rungs"].items():
        eps = [e for v in per_op.values() for e in v]
        f = sum(e.frames for e in eps)
        op_h = [hours(sum(e.frames for e in v)) for v in per_op.values()]
        rows.append(
            {
                "rung": k,
                "ops": len(per_op),
                "hours": round(hours(f), 3),
                "episodes": len(eps),
                "tshirt_share": round(
                    sum(e.frames for e in eps if e.cat == TSHIRT) / f, 3
                ),
                "per_op_hours": f"{min(op_h):.2f}-{max(op_h):.2f}",
                "frames": f,
            }
        )
    return rows


def load_pool(priv: Path, strict_fold: bool) -> tuple[list[Ep], dict]:
    index = json.loads((priv / "zarr" / "_episode_index.json").read_text())
    ok = {}
    for log in sorted((priv / "zarr_logs").glob("convert_log*.jsonl")):
        for line in log.read_text().splitlines():
            r = json.loads(line)
            ok[r["episode_id"]] = ok.get(r["episode_id"], False) or bool(r.get("ok"))
    blocklist = set(
        (REPO / "egomimic/resources/blocklists/mecka_bad_egomotion.txt")
        .read_text()
        .split()
    )
    inputs = json.loads((priv / "scaling" / "fixed_budget_inputs.json").read_text())
    foreign = set(inputs["foreign_home_episodes"])
    drops = Counter()
    eps = []
    with open(priv / "manifest.csv") as f:
        for r in csv.DictReader(f):
            if r["category"] not in CATS or r["off_task"] != "0":
                continue
            h = r["episode_id"]
            a = index.get(h)
            qa = (a or {}).get("qa", {})
            if a is None or not ok.get(h):
                drops["not converted"] += 1
            elif qa["frozen_cam_frames"] or qa["zero_hand_frames"]:
                drops["frozen camera / no hands"] += 1
            elif h in blocklist:
                drops["blocklist"] += 1
            elif h in foreign:
                drops["filmed in another operator's home"] += 1
            elif strict_fold and non_folding_actions(r["task_description"]):
                drops["strict fold"] += 1
            else:
                eps.append(
                    Ep(
                        id=h,
                        op=r["user_id"],
                        cat=r["category"],
                        frames=int(a["total_frames"]),
                        task_id=r["task_id"],
                        clean=qa["glitch_frames"] == 0,
                    )
                )
    eps, dropped_ops = collapse_same_home(eps, inputs["same_home"])
    drops["same-home operators"] = len(dropped_ops)
    return eps, dict(drops)


def unseen_sets(scaling: Path) -> tuple[list[str], list[str]]:
    hashes, viz = [], []
    for name in ("opscale_picks.json", "opscale_picks_notshirt.json"):
        picks = json.loads((scaling / name).read_text())
        hashes += [h for v in picks["unseen"].values() for h in v if h not in hashes]
        viz.append(picks["unseen_viz"][0])
    return hashes, viz


def _ylist(items, indent: int) -> str:
    return "\n".join(f"{' ' * indent}- {x}" for x in items)


def _head(name: str, hashes: list[str]) -> str:
    return f"""{name}:
  human_bimanual:
    _target_: ${{data.train_datasets.human_bimanual._target_}}
    resolver: ${{data.train_datasets.human_bimanual.resolver}}
    bounds_rel_slack: 10.0
    filters:
      _target_: egomimic.rldb.filters.DatasetFilter
      episode_hashes:
{_ylist(hashes, 8)}
    pose_glitch_mask: {{step_m: 0.05, rot_deg: 10.0}}
    mode: total
"""


def _loader(name: str, workers: int, shuffle: bool = False) -> str:
    tail = "" if shuffle else "\n    shuffle: false"
    return f"""{name}:
  human_bimanual:
    batch_size: 32
    num_workers: {workers}{tail}
"""


def data_config(k, rows, train, seen1, seen_k, videos, unseen, unseen_viz) -> str:
    km_tl = """      key_map:
        _target_: egomimic.rldb.embodiment.human.Human.get_keymap
        keymap_mode: keypoints
        annotation_key: annotations{hist}
      transform_list:
        _target_: egomimic.rldb.embodiment.human.Human.get_transform_list
        mode: keypoints_wristframe_6d
        stride: 1
        fix_left_wrist_convention: {fix}
        pad_proprio_gripper: false"""
    hist = "".join(
        f"\n        {k_}: ${{data.train_datasets.human_bimanual.resolver.key_map.{k_}}}"
        for k_ in ("proprio_history", "history_stride", "image_history_gap_s")
    )
    extra = extra_metric = extra_video = ""
    if seen_k:
        extra_metric = "\n  extra_valid: auto"
        extra_video = "\n  extra_valid:\n" + _ylist(videos["extra_valid"], 4)
        extra = (
            "\nextra_valid_prefix: seen_ops_valid\n\n"
            + _head("extra_valid_datasets", seen_k)
            + "\n"
            + _loader("extra_valid_dataloader_params", 8)
        )
    return f"""# PRIVATE Mecka mono fold (t-shirt + no-t-shirt), fixed-budget operator ladder,
# {k} operator(s): {rows["hours"]} h, {rows["episodes"]} episodes, t-shirt share {rows["tshirt_share"]}.
# Every rung has the same hours, split equally over its operators, with nested
# picks. Generated by egomimic/scripts/mecka_process/mono/make_configs_fixed_budget.py;
# do not edit by hand. IDs: $EGOVERSE_MECKA_PRIVATE_DIR/splits (private to the lab).
#   train            = this rung's packed episodes
#   valid            = operator 1's held-out episodes (seen_op1_valid, every rung)
#   extra_valid      = the held-out episodes of this rung's operators (k > 1)
#   unseen_op_valid  = the t-shirt + no-t-shirt opscale SQL unseen-operator sets
# The private zarrs are post-fix left wrist; the SQL mecka zarrs are not.
_target_: egomimic.pl_utils.pl_data_utils.MultiDataModuleWrapper

valid_prefix: seen_op1_valid

train_datasets:
  human_bimanual:
    _target_: egomimic.rldb.zarr.zarr_dataset_multi.MultiDataset._from_resolver
    resolver:
      _target_: egomimic.rldb.zarr.zarr_dataset_multi.LocalEpisodeResolver
      folder_path: {ZARR_DIR}
{km_tl.format(hist="", fix="false")}
    filters:
      _target_: egomimic.rldb.filters.DatasetFilter
      episode_hashes:
{_ylist(train, 8)}
    pose_glitch_mask: {{step_m: 0.05, rot_deg: 10.0}}
    mode: total

# Eval heads on other episodes widen the bounds to 10x each cell's quantile range.
{_head("valid_datasets", seen1)}
train_viz_datasets:
  human_bimanual:
    _target_: ${{data.train_datasets.human_bimanual._target_}}
    resolver: ${{data.train_datasets.human_bimanual.resolver}}
    filters: ${{data.train_datasets.human_bimanual.filters}}
    pose_glitch_mask: {{step_m: 0.05, rot_deg: 10.0}}
    mode: total

unseen_op_valid_datasets:
  human_bimanual:
    _target_: ${{data.train_datasets.human_bimanual._target_}}
    bounds_rel_slack: 10.0
    resolver:
      _target_: egomimic.rldb.zarr.zarr_dataset_multi.S3EpisodeResolver
      folder_path: ${{paths.dataset_dir}}
{km_tl.format(hist=hist, fix="true")}
    filters:
      _target_: egomimic.rldb.filters.DatasetFilter
      filter_lambdas:
        - "lambda row: str(row.get('lab', '')).lower() == 'mecka'"
      episode_hashes:
{_ylist(unseen, 8)}
    pose_glitch_mask: {{step_m: 0.05, rot_deg: 10.0}}
    mode: total
{extra}
{_loader("train_dataloader_params", 12, shuffle=True)}
{_loader("valid_dataloader_params", 8)}
{_loader("train_viz_dataloader_params", 4)}
{_loader("unseen_op_valid_dataloader_params", 8)}
metric_frames_per_episode:
  valid: auto
  train_viz: auto
  unseen_op_valid: auto{extra_metric}

video_episodes:
  valid:
{_ylist(videos["valid"], 4)}
  train_viz:
{_ylist(videos["train_viz"], 4)}
  unseen_op_valid:
{_ylist(unseen_viz, 4)}{extra_video}
"""


def train_config(k: int, epochs: int, passes: float) -> str:
    return f"""# RDT-1B keypoint recipe of the opscale ladders (train_zarr_mecka_freeform_fold_kp_rdt_op2:
# 4 GPUs x 32, WSD LR decaying over the last 20%, EMA on) on fixed-budget rung {k}.
# {epochs} x 100 steps on every rung (~{passes:.0f} passes over the budget). Generated by
# make_configs_fixed_budget.py; see data/mecka_private_fixedbudget_op{k}.
defaults:
  - train_zarr_mecka_freeform_fold_kp_rdt_op2
  - override data: mecka_private_fixedbudget_op{k}
  - _self_

model:
  ema:
    enabled: true
  scheduler:
    stable: {epochs * 100 - 1000 - epochs * 20}
    decay: {epochs * 20}

trainer:
  max_epochs: {epochs}
  min_epochs: {epochs}
  check_val_every_n_epoch: 10

# Every 20% (~23 GB each with EMA); the 80% one is the decay start.
callbacks:
  model_checkpoint:
    every_n_epochs: {epochs // 5}
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--strict-fold", action=argparse.BooleanOptionalAction, default=True
    )
    ap.add_argument("--tshirt-share", type=float, default=0.5)
    ap.add_argument("--rungs", type=int, nargs="+", default=list(RUNGS))
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    priv = private_dir()
    eps, drops = load_pool(priv, args.strict_fold)
    p = plan(eps, tuple(args.rungs), args.tshirt_share, args.seed)
    rows = plan_rows(p)
    passes = args.epochs * 100 * GLOBAL_BATCH / p["budget_frames"]

    print(f"pool: {len(eps)} episodes, {len({e.op for e in eps})} operators")
    print("dropped:", json.dumps(drops))
    print(f"budget {hours(p['budget_frames']):.3f} h, {args.epochs * 100} steps")
    print(f"~{passes:.1f} passes")
    cols = ("rung", "ops", "hours", "episodes", "tshirt_share", "per_op_hours")
    print(" ".join(f"{c:>13s}" for c in cols))
    for r in rows:
        print(" ".join(f"{r[c]!s:>13s}" for c in cols))
    if args.plan_only:
        return

    order = p["order"]
    held = p["held_out"]
    top = max(p["rungs"])
    op1 = order[0]
    op1_top = sorted(p["rungs"][top][op1], key=lambda e: e.frames)
    train_viz_pin = op1_top[len(op1_top) // 2].id
    seen1 = [e.id for e in held[op1]]
    first_per_cat = {}
    for e in held[op1]:
        first_per_cat.setdefault(e.cat, e.id)
    unseen, unseen_viz = unseen_sets(priv / "scaling")
    ladder_ops = set(order)
    sql_ops = set()
    for name in ("opscale_picks.json", "opscale_picks_notshirt.json"):
        sql_ops |= set(json.loads((priv / "scaling" / name).read_text())["unseen"])
    if ladder_ops & sql_ops:
        raise ValueError("an unseen SQL operator is a ladder operator")

    json.dump(
        {
            "args": vars(args),
            "order": order,
            "budget_frames": p["budget_frames"],
            "held_out": {op: [e.id for e in v] for op, v in held.items()},
            "train_viz_pin": train_viz_pin,
            "rungs": {
                k: {op: [e.id for e in v] for op, v in per_op.items()}
                for k, per_op in p["rungs"].items()
            },
            "plan": rows,
        },
        open(priv / "scaling" / "fixed_budget_picks.json", "w"),
        indent=1,
    )
    splits = priv / "splits"
    for r in rows:
        k = r["rung"]
        train = sorted(e.id for v in p["rungs"][k].values() for e in v)
        seen_k = [e.id for op in order[:k] for e in held[op]] if k > 1 else []
        videos = {
            "valid": list(first_per_cat.values()),
            "train_viz": [train_viz_pin],
            "extra_valid": [held[op][0].id for op in order[1 : min(k, 3)]],
        }
        write_config(
            CFG / f"data/mecka_private_fixedbudget_op{k}.yaml",
            data_config(k, r, train, seen1, seen_k, videos, unseen, unseen_viz),
            splits,
        )
        write_config(
            CFG / f"train_zarr_mecka_private_fixedbudget_op{k}.yaml",
            train_config(k, args.epochs, passes),
            splits,
        )


if __name__ == "__main__":
    main()
