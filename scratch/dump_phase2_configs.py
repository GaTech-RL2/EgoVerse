"""Phase-2 baseline dumper: resolved (or raw-load) JSON for EVERY hydra config.

Extends scratch/dump_baseline_configs.py to all groups + entry yamls + full model set.

Proof bar: each dump is the canonical baseline a later refactor must diff EMPTY
against (resolve=True, sort_keys, indent=2). Files that cannot be composed/resolved
via the default entry are handled with a documented fallback and recorded in
methods.json so the identity gate knows exactly how each was produced.

Method tiers (recorded per file in OUT_DIR/methods.json):
  - "compose:<entry>[ +overrides]"  -> hydra compose, dump resolved cfg.<group>/cfg.model/full cfg
  - "rawload"                        -> OmegaConf.load(file), resolve=False  (non-composable groups)
"""
import json
import os
import sys
import traceback

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = "/coc/flash7/paphiwetsa3/projects/EgoVerse2"
CONFIG_DIR = os.path.join(REPO, "egomimic/hydra_configs")
OUT_DIR = os.path.join(REPO, "scratch/config_phase2_baseline/resolved")
ENTRY = "train_zarr_cartesian"
ENTRY_PI = "train_zarr_cartesian_pi"

methods = {}   # relpath(without .json) -> method string
ok, fail = [], []


def list_yamls(subdir):
    """All real yaml stems in a group dir (skip ._ resource forks, skip nested dirs)."""
    d = os.path.join(CONFIG_DIR, subdir)
    out = []
    for fn in sorted(os.listdir(d)):
        if fn.startswith("._") or not fn.endswith(".yaml"):
            continue
        if os.path.isfile(os.path.join(d, fn)):
            out.append(fn[:-5])
    return out


def write_json(relkey, container):
    out_path = os.path.join(OUT_DIR, relkey + ".json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(container, f, sort_keys=True, indent=2)
        f.write("\n")
    return out_path


def try_compose(entry, overrides):
    cfg = compose(config_name=entry, overrides=overrides, return_hydra_config=False)
    return cfg


def dump_model(name):
    """model/<name>.json. Try default entry; fall back through documented overrides."""
    relkey = f"model/{name}"
    # candidate (entry, extra_overrides, label) attempts, in order
    attempts = []
    if name.startswith("pi0.5"):
        # pi models need the pi entry (pi data/trainer/evaluator); model override on top
        attempts.append((ENTRY_PI, [f"model={name}"], f"compose:{ENTRY_PI} model={name}"))
        attempts.append((ENTRY, [f"model={name}"], f"compose:{ENTRY} model={name}"))
    else:
        attempts.append((ENTRY, [f"model={name}"], f"compose:{ENTRY} model={name}"))
        # pushshapes_sim HPT/flow/hnet models reference ${data.action_horizon} /
        # ${data.dataset.data_schematic}; default data=eva lacks them. Retry with a
        # pushshapes data group that provides both.
        for dgrp in ("tsimulation_hpt", "tsimulation", "tsimulation_hpt_fast",
                     "tsimulation_hpt_causal", "tsimulation_delta"):
            attempts.append(
                (ENTRY, [f"model={name}", f"data={dgrp}"],
                 f"compose:{ENTRY} model={name} data={dgrp}"))
    last_err = None
    for entry, ovr, label in attempts:
        try:
            cfg = try_compose(entry, ovr)
            container = OmegaConf.to_container(cfg.model, resolve=True)
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
        write_json(relkey, container)
        methods[relkey] = label
        ok.append(relkey)
        print(f"OK   {relkey}  [{label}]", flush=True)
        return
    # No compose path resolved cfg.model (e.g. hpt_bc_flow_pushshapes references
    # ${data.dataset.data_schematic}, which no data group provides -> orphan/drift
    # config). Fall back to raw-load; identity gate = byte-stable raw content.
    try:
        node = OmegaConf.load(os.path.join(CONFIG_DIR, "model", name + ".yaml"))
        container = OmegaConf.to_container(node, resolve=False)
        write_json(relkey, container)
        methods[relkey] = f"rawload (model compose unresolvable: {last_err})"
        ok.append(relkey)
        print(f"OK   {relkey}  [rawload, compose unresolvable]", flush=True)
    except Exception as e2:  # noqa: BLE001
        fail.append(relkey)
        methods[relkey] = f"FAIL: {e2}"
        print(f"FAIL {relkey}: {e2}", flush=True)


def dump_group(group):
    """<group>/<name>.json for each yaml: compose entry with <group>=<name>, dump cfg.<group> resolved."""
    for name in list_yamls(group):
        relkey = f"{group}/{name}"
        try:
            cfg = try_compose(ENTRY, [f"{group}={name}"])
            sub = getattr(cfg, group)
            container = OmegaConf.to_container(sub, resolve=True)
            write_json(relkey, container)
            methods[relkey] = f"compose:{ENTRY} {group}={name} -> cfg.{group}"
            ok.append(relkey)
            print(f"OK   {relkey}  [compose {group}={name}]", flush=True)
        except Exception as e:  # noqa: BLE001
            # fall back to raw load (resolve=False); identity gate = byte-stable raw content
            try:
                node = OmegaConf.load(os.path.join(CONFIG_DIR, group, name + ".yaml"))
                container = OmegaConf.to_container(node, resolve=False)
                write_json(relkey, container)
                methods[relkey] = f"rawload (compose failed: {e})"
                ok.append(relkey)
                print(f"OK   {relkey}  [rawload, compose failed]", flush=True)
            except Exception as e2:  # noqa: BLE001
                fail.append(relkey)
                methods[relkey] = f"FAIL: {e2}"
                print(f"FAIL {relkey}: {e2}", flush=True)


def dump_rawload_dir(subdir):
    """Recursively raw-load every yaml under subdir (non-composable: hydra/, evaluator/viz/)."""
    base = os.path.join(CONFIG_DIR, subdir)
    for root, _dirs, files in os.walk(base):
        for fn in sorted(files):
            if fn.startswith("._") or not fn.endswith(".yaml"):
                continue
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, CONFIG_DIR)[:-5]  # strip .yaml
            try:
                node = OmegaConf.load(full)
                container = OmegaConf.to_container(node, resolve=False)
                write_json(rel, container)
                methods[rel] = "rawload (non-composable group)"
                ok.append(rel)
                print(f"OK   {rel}  [rawload]", flush=True)
            except Exception as e:  # noqa: BLE001
                fail.append(rel)
                methods[rel] = f"FAIL: {e}"
                print(f"FAIL {rel}: {e}", flush=True)


def dump_entry(name):
    """entry_<name>.json: full resolved cfg with default groups."""
    relkey = f"entry_{name}"
    try:
        cfg = try_compose(name, [])
        container = OmegaConf.to_container(cfg, resolve=True)
        write_json(relkey, container)
        methods[relkey] = f"compose:{name} (full cfg, default groups)"
        ok.append(relkey)
        print(f"OK   {relkey}  [full cfg]", flush=True)
    except Exception as e:  # noqa: BLE001
        # viz_language has different default groups (no model/trainer); raw-load fallback
        try:
            node = OmegaConf.load(os.path.join(CONFIG_DIR, name + ".yaml"))
            container = OmegaConf.to_container(node, resolve=False)
            write_json(relkey, container)
            methods[relkey] = f"rawload (compose failed: {e})"
            ok.append(relkey)
            print(f"OK   {relkey}  [rawload, compose failed]", flush=True)
        except Exception as e2:  # noqa: BLE001
            fail.append(relkey)
            methods[relkey] = f"FAIL: {e2}"
            print(f"FAIL {relkey}: {e2}", flush=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        # 1. models
        for name in list_yamls("model"):
            dump_model(name)
        # 2. composable groups
        for group in ("data", "evaluator", "trainer", "callbacks",
                      "logger", "paths", "data_schematic"):
            dump_group(group)
        # 3. non-composable / nested: raw-load
        dump_rawload_dir("hydra")
        dump_rawload_dir("evaluator/viz")
        # 4. entry yamls
        for name in ("train_zarr_cartesian", "train_zarr_cartesian_pi",
                     "train_zarr_keypoints", "train_zarr_keypoint_wrist",
                     "viz_language"):
            dump_entry(name)

    with open(os.path.join(OUT_DIR, "methods.json"), "w") as f:
        json.dump(methods, f, sort_keys=True, indent=2)
        f.write("\n")
    print(f"\nSUMMARY ok={len(ok)} fail={len(fail)}", flush=True)
    if fail:
        print("FAILED: " + ", ".join(fail), flush=True)


if __name__ == "__main__":
    main()
