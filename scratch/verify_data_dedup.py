"""Round-2 data/ dedup verifier — BOTH gates.

Gate 1 (identity): for each target data config, compose train_zarr_cartesian
data=<name>, OmegaConf.to_container(cfg.data, resolve=True), sort_keys+indent2
JSON, diff against scratch/config_phase2_baseline/resolved/data/<name>.json.
EMPTY diff required (exact baseline dump method, dump_phase2_configs.py).

Gate 2 (override-compat): for each dotted override in the gate set that targets
a touched file, run a real compose with that override appended (representative
value). Must compose without ConfigCompositionException.

Usage: python verify_data_dedup.py <name1> <name2> ...
       (no args -> all tsimulation* + pickplace_qwen cluster)
"""
import json
import os
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = "/coc/flash7/paphiwetsa3/projects/EgoVerse2"
CONFIG_DIR = os.path.join(REPO, "egomimic/hydra_configs")
BASELINE = os.path.join(REPO, "scratch/config_phase2_baseline/resolved/data")
ENTRY = "train_zarr_cartesian"

NC3 = "/coc/flash7/paphiwetsa3/datasets/new_circle_3"
KM = "egomimic.rldb.embodiment.pushshapes.get_keymap_eval"

# THE OVERRIDE GATE SET (data group). Each entry: (path, representative_value, is_plus_override)
# All paths target pushshapes_sim. The +data.* ones are append-overrides (the
# leaf may not declare that key, so launcher uses '+').
GATE = [
    ("data.train_datasets.pushshapes_sim.resolver.folder_path", NC3, False),
    ("data.valid_datasets.pushshapes_sim.resolver.folder_path", NC3, False),
    ("data.valid_datasets.pushshapes_sim.resolver.key_map._target_", KM, False),
    ("data.train_datasets.pushshapes_sim.resolver.key_map._target_", KM, False),
    ("data.train_dataloader_params.pushshapes_sim.batch_size", "16", False),
    ("data.valid_dataloader_params.pushshapes_sim.batch_size", "16", False),
]
# The fullhist_ratio launcher (train_bc_rnn_hnet_chunk8_fullhist_ratio.sh, 1
# script, data=tsimulation): DELETES the whole valid node (~) then REBUILDS it
# explicitly (+). The ~ delete REQUIRES valid_datasets.pushshapes_sim to be a
# concrete node (works on concrete; would fail/misbehave on an interpolation).
# This is the exact verbatim sequence from the launcher.
RATIO_BUNDLE = [
    "~data.valid_datasets.pushshapes_sim",
    "+data.valid_datasets.pushshapes_sim._target_=egomimic.rldb.zarr.zarr_dataset_packed.ZarrEpisodePackedDataset.from_resolver",
    "+data.valid_datasets.pushshapes_sim.resolver._target_=egomimic.rldb.zarr.zarr_dataset_multi.LocalEpisodeResolver",
    "+data.valid_datasets.pushshapes_sim.resolver.folder_path=" + NC3,
    "+data.valid_datasets.pushshapes_sim.resolver.key_map._target_=" + KM,
    "+data.valid_datasets.pushshapes_sim.resolver.key_map.action_horizon=1024",
    "+data.valid_datasets.pushshapes_sim.resolver.transform_list=null",
    "+data.valid_datasets.pushshapes_sim.chunking=none",
    "+data.valid_datasets.pushshapes_sim.min_seq_len=64",
    "+data.valid_datasets.pushshapes_sim.max_seq_len=null",
]

PUSHSHAPES_FILES = {
    "tsimulation", "tsimulation_delta", "tsimulation_hpt",
    "tsimulation_hpt_fast", "tsimulation_hpt_causal",
}


def dump_resolved(name):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name=ENTRY, overrides=[f"data={name}"])
        container = OmegaConf.to_container(cfg.data, resolve=True)
    return json.dumps(container, sort_keys=True, indent=2) + "\n"


def identity_gate(name):
    base_path = os.path.join(BASELINE, name + ".json")
    if not os.path.exists(base_path):
        return False, f"no baseline json {base_path}"
    with open(base_path) as f:
        baseline = f.read()
    try:
        got = dump_resolved(name)
    except Exception as e:  # noqa: BLE001
        return False, f"compose/dump raised: {repr(e)[:200]}"
    if got == baseline:
        return True, "EMPTY diff"
    # produce a short diff summary
    import difflib
    diff = list(difflib.unified_diff(
        baseline.splitlines(), got.splitlines(),
        fromfile="baseline", tofile="current", lineterm=""))
    return False, "DIFF:\n" + "\n".join(diff[:60])


def override_gate(name):
    """Run each applicable gate-set override against this file. Returns list of (label, ok, err)."""
    results = []
    # which overrides apply? pushshapes_sim overrides only matter for pushshapes
    # data files. For pickplace_qwen files (no pushshapes_sim node), the gate set
    # has NO entries -> trivially pass (record as N/A).
    if name not in PUSHSHAPES_FILES:
        results.append(("(no pushshapes_sim gate paths target this file)", True, "N/A"))
        return results
    for path, val, _is_plus in GATE:
        ovr = [f"data={name}", f"{path}={val}"]
        ok, err = _try_compose(ovr)
        results.append((path + "=" + str(val), ok, err))
    # ratio launcher (~delete + re-add) — only emitted with data=tsimulation, but
    # the ~ delete works on any concrete valid node, so test it on every
    # pushshapes file as a strong concreteness check.
    ok, err = _try_compose([f"data={name}"] + RATIO_BUNDLE)
    results.append(("~delete+re-add valid (fullhist_ratio launcher)", ok, err))
    return results


def _try_compose(overrides):
    try:
        with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
            compose(config_name=ENTRY, overrides=overrides)
        return True, None
    except Exception as e:  # noqa: BLE001
        return False, repr(e).split("\n")[0][:240]


def main():
    names = sys.argv[1:]
    if not names:
        names = sorted(PUSHSHAPES_FILES) + [
            "cotrain_pickplace_qwen", "cotrain_pickplace_qwen_wrist",
            "cotrain_pickplace_qwen_objgen", "cotrain_pi_pickplace_qwen",
            "bc_pickplace_eva_qwen",
        ]
    all_ok = True
    for name in names:
        print(f"\n========== {name} ==========")
        ok1, msg1 = identity_gate(name)
        print(f"  IDENTITY : {'PASS' if ok1 else 'FAIL'}  {msg1 if not ok1 else msg1}")
        if not ok1:
            all_ok = False
        ov = override_gate(name)
        ov_ok = all(o for _, o, _ in ov)
        print(f"  OVERRIDE : {'PASS' if ov_ok else 'FAIL'}")
        for label, o, err in ov:
            tag = "ok" if o else "FAIL"
            extra = "" if o else f"  -> {err}"
            print(f"      [{tag}] {label}{extra}")
        if not ov_ok:
            all_ok = False
    print("\n==================== VERDICT ====================")
    print("ALL PASS" if all_ok else "SOME FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
