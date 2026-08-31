"""Pull control-mode success rates from wandb into one comparable table.

Prints SR on the four SEEN controller modes and the two HELD-OUT ones, per arm
and capacity, alongside the parameter count — because a capacity-unmatched
comparison measures capacity, and the reader should be able to see that it was
matched.

Rows are grouped so the comparison the study exists to make is adjacent:

    arm2_causal_bidir     bidirectional | the CONTROL
    arm3_state_action_ar  causal        | arm3 - arm2 IS the result

arm1 uses flow matching AND a different attention pattern, so a gap between
arm1 and arm3 mixes the objective with the mask. It is an absolute reference
point, not the control.

Usage:
    python scripts/control_modes/collect_results.py [--entity E] [--project P]
                                                    [--step-matched]
"""

from __future__ import annotations

import argparse
import re

SEEN = ["tight", "laggy", "loose", "sticky"]
UNSEEN = ["ideal", "jittery"]
ARM_ORDER = ["arm1_dp_flow", "arm2_causal_bidir", "arm3_state_action_ar",
             "arm4_state_idm"]
ARM_LABEL = {
    "arm1_dp_flow": "arm1 dp_flow      (bidir, flow)",
    "arm2_causal_bidir": "arm2 causal_bidir (bidir, MSE)  <- CONTROL",
    "arm3_state_action_ar": "arm3 state_ar     (causal, MSE)",
    "arm4_state_idm": "arm4 state_idm    (causal, IDM)",
}
PARAMS_M = {("large", "arm1_dp_flow"): 309.12, ("large", "arm2_causal_bidir"): 313.63,
            ("large", "arm3_state_action_ar"): 313.63, ("large", "arm4_state_idm"): 313.70,
            ("small", "arm1_dp_flow"): 50.01, ("small", "arm2_causal_bidir"): 49.09,
            ("small", "arm3_state_action_ar"): 49.09, ("small", "arm4_state_idm"): 49.16}


def parse_run_name(name: str):
    m = re.match(r"ctrlmode_(.+?)_(large|small)_\d+", name)
    return (m.group(1), m.group(2)) if m else (None, None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="rl2-group")
    ap.add_argument("--project", default="zarr_test")
    ap.add_argument("--step-matched", action="store_true",
                    help="report at the largest step ALL runs have reached, "
                         "instead of each run's last value; wall-clock "
                         "comparisons are unfair when runs start at "
                         "different times")
    args = ap.parse_args()

    import wandb

    api = wandb.Api()
    try:
        runs = [r for r in api.runs(f"{args.entity}/{args.project}")
                if r.name.startswith("ctrlmode_")]
    except ValueError as e:
        # The runs log to the entity the NODE's key can reach (rl2-group, from
        # Secrets Manager). A personal key with a different default entity
        # cannot see them, and wandb reports that as a missing PROJECT rather
        # than a permissions error, which is misleading.
        teams = [t if isinstance(t, str) else getattr(t, "name", str(t))
                 for t in (getattr(api.viewer, "teams", None) or [])]
        reachable = sorted({e for e in (api.default_entity, *teams) if e})
        print(f"could not read {args.entity}/{args.project}: {e}")
        print(f"this key ({api.viewer.username}) can reach: {reachable}")
        if args.entity not in reachable:
            print(f"\n{args.entity!r} is not among them. The runs are logged by "
                  f"the node's key, not this one. Either export a WANDB_API_KEY "
                  f"with {args.entity!r} access, or read them in the browser:")
            print(f"  https://wandb.ai/{args.entity}/{args.project}")
        else:
            print(f"\nentity is reachable, so the project name is likely wrong. "
                  f"Try --project with one of that entity's projects.")
        return 2
    if not runs:
        print(f"no ctrlmode_* runs in {args.entity}/{args.project} yet")
        return 1

    keys = ([f"Valid/seen_{m}_sim_success_rate" for m in SEEN]
            + [f"Valid/unseen_{m}_sim_success_rate" for m in UNSEEN])

    table = {}
    common_step = None
    histories = {}
    for run in runs:
        arm, cap = parse_run_name(run.name)
        if arm is None:
            continue
        hist = run.history(keys=keys + ["trainer/global_step"], pandas=True)
        if hist is None or len(hist) == 0:
            continue
        histories[(cap, arm)] = hist
        last = hist["trainer/global_step"].dropna()
        if len(last):
            s = int(last.iloc[-1])
            common_step = s if common_step is None else min(common_step, s)

    for (cap, arm), hist in histories.items():
        if args.step_matched and common_step is not None:
            sub = hist[hist["trainer/global_step"] <= common_step]
            row = sub.iloc[-1] if len(sub) else hist.iloc[-1]
        else:
            row = hist.iloc[-1]
        table[(cap, arm)] = {
            k: (float(row[k]) if k in row and row[k] == row[k] else None)
            for k in keys
        }
        table[(cap, arm)]["_step"] = (
            int(row["trainer/global_step"])
            if row.get("trainer/global_step") == row.get("trainer/global_step")
            else -1)

    if args.step_matched:
        print(f"# step-matched at global_step <= {common_step}\n")
    else:
        print("# each run's LAST logged eval (steps may differ)\n")

    hdr = (f"{'arm':<38}{'params':>8}{'step':>8}"
           + "".join(f"{m:>9}" for m in SEEN)
           + f"{'SEEN':>9}" + "".join(f"{'*'+m:>10}" for m in UNSEEN))
    for cap in ("small", "large"):
        rows = [(a, table[(cap, a)]) for a in ARM_ORDER if (cap, a) in table]
        if not rows:
            continue
        print(f"=== {cap.upper()} capacity ===")
        print(hdr)
        print("-" * len(hdr))
        for arm, vals in rows:
            seen_vals = [vals[f"Valid/seen_{m}_sim_success_rate"] for m in SEEN]
            got = [v for v in seen_vals if v is not None]
            mean_seen = sum(got) / len(got) if got else None
            line = (f"{ARM_LABEL[arm]:<38}"
                    f"{PARAMS_M.get((cap, arm), float('nan')):>7.1f}M"
                    f"{vals['_step']:>8}")
            for v in seen_vals:
                line += f"{'--' if v is None else f'{100*v:.0f}%':>9}"
            line += f"{'--' if mean_seen is None else f'{100*mean_seen:.0f}%':>9}"
            for m in UNSEEN:
                v = vals[f"Valid/unseen_{m}_sim_success_rate"]
                line += f"{'--' if v is None else f'{100*v:.0f}%':>10}"
            print(line)
        print()

    print("* = held out. ideal (noise 0.0) is BELOW the training range "
          "(0.3-0.8); jittery (2.5) is 3x above it.")
    print("The result is arm3 - arm2 on *jittery, at matched capacity. "
          "arm1 differs in objective too.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
