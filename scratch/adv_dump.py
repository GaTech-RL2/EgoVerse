"""ADVERSARIAL independent re-dumper for phase-2 config identity gate.

Written from scratch (does NOT import the agents' dump_phase2_configs). It
re-derives the SAME documented method per file family, dumps resolved (or
rawload-fallback) JSON sorted-keys indent=2, then the caller diffs against the
phase-2 baseline at scratch/config_phase2_baseline/resolved/.

Method (mirrors methods.json documentation, re-implemented independently):
  models      : compose entry (pi entry for pi0.5* else cartesian) with model=<name>,
                dump cfg.model resolve=True. If that raises, retry with a sequence of
                tsimulation* data groups. If all fail, rawload the yaml (resolve=False).
  groups      : compose cartesian with <group>=<name>, dump cfg.<group> resolve=True;
                on failure rawload (resolve=False).
  hydra, evaluator/viz : rawload every yaml (resolve=False).
  entries     : compose <name>, dump full cfg resolve=True; on failure rawload.
"""
import json
import os
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = "/coc/flash7/paphiwetsa3/projects/EgoVerse2"
CONFIG_DIR = os.path.join(REPO, "egomimic/hydra_configs")
OUT_DIR = sys.argv[1]
ENTRY = "train_zarr_cartesian"
ENTRY_PI = "train_zarr_cartesian_pi"
TSIM = ("tsimulation_hpt", "tsimulation", "tsimulation_hpt_fast",
        "tsimulation_hpt_causal", "tsimulation_delta")


def stems(subdir):
    d = os.path.join(CONFIG_DIR, subdir)
    return sorted(fn[:-5] for fn in os.listdir(d)
                  if fn.endswith(".yaml") and not fn.startswith("._")
                  and os.path.isfile(os.path.join(d, fn)))


def dump(relkey, container):
    p = os.path.join(OUT_DIR, relkey + ".json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(container, f, sort_keys=True, indent=2)
        f.write("\n")


def rawload(group, name):
    node = OmegaConf.load(os.path.join(CONFIG_DIR, group, name + ".yaml"))
    return OmegaConf.to_container(node, resolve=False)


def do_model(name):
    relkey = f"model/{name}"
    entries = [ENTRY_PI, ENTRY] if name.startswith("pi0.5") else [ENTRY]
    # build attempt list: (entry, overrides)
    attempts = []
    for e in entries:
        attempts.append((e, [f"model={name}"]))
    if not name.startswith("pi0.5"):
        for dg in TSIM:
            attempts.append((ENTRY, [f"model={name}", f"data={dg}"]))
    for e, ovr in attempts:
        try:
            cfg = compose(config_name=e, overrides=ovr)
            container = OmegaConf.to_container(cfg.model, resolve=True)
            dump(relkey, container)
            return ("compose", e, ovr)
        except Exception:
            continue
    dump(relkey, rawload("model", name))
    return ("rawload", None, None)


def do_group(group):
    res = {}
    for name in stems(group):
        relkey = f"{group}/{name}"
        try:
            cfg = compose(config_name=ENTRY, overrides=[f"{group}={name}"])
            container = OmegaConf.to_container(getattr(cfg, group), resolve=True)
            dump(relkey, container)
            res[relkey] = "compose"
        except Exception:
            dump(relkey, rawload(group, name))
            res[relkey] = "rawload"
    return res


def do_rawdir(subdir):
    base = os.path.join(CONFIG_DIR, subdir)
    for root, _d, files in os.walk(base):
        for fn in sorted(files):
            if fn.startswith("._") or not fn.endswith(".yaml"):
                continue
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, CONFIG_DIR)[:-5]
            node = OmegaConf.load(full)
            dump(rel, OmegaConf.to_container(node, resolve=False))


def do_entry(name):
    relkey = f"entry_{name}"
    try:
        cfg = compose(config_name=name, overrides=[])
        dump(relkey, OmegaConf.to_container(cfg, resolve=True))
    except Exception:
        node = OmegaConf.load(os.path.join(CONFIG_DIR, name + ".yaml"))
        dump(relkey, OmegaConf.to_container(node, resolve=False))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    n = 0
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        for name in stems("model"):
            do_model(name); n += 1
        for g in ("data", "evaluator", "trainer", "callbacks",
                  "logger", "paths", "data_schematic"):
            r = do_group(g); n += len(r)
        do_rawdir("hydra")
        do_rawdir("evaluator/viz")
        for name in ("train_zarr_cartesian", "train_zarr_cartesian_pi",
                     "train_zarr_keypoints", "train_zarr_keypoint_wrist",
                     "viz_language"):
            do_entry(name); n += 1
    print(f"DUMPED model+group+entry families to {OUT_DIR}")


if __name__ == "__main__":
    main()
