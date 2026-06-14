"""ADVERSARIAL re-dump: resolve parent + leaf configs NOW, independent of implementer.

Checks:
 1. parent (..._hnet_chunk8_fullhist) resolves byte-identical to the saved baseline JSON.
 2. leaf  (..._hnet_chunk8_fullhist_ratio) resolves with collect_ratio_loss == True
    and core_net.target_compression_ratio == 2.0, ratio_loss_weight == 0.03.
 3. parent resolved has NO collect_ratio_loss==True (must be absent or False).
"""
import json
import os
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = "/coc/flash7/paphiwetsa3/projects/EgoVerse2"
CONFIG_DIR = os.path.join(REPO, "egomimic/hydra_configs")
CONFIG_NAME = "train_zarr_cartesian"
BASELINE = os.path.join(
    REPO,
    "scratch/config_refactor_baseline/resolved/"
    "bc_rnn_pushshapes_paperexact_hnet_chunk8_fullhist.json",
)

PARENT = "bc_rnn_pushshapes_paperexact_hnet_chunk8_fullhist"
LEAF = "bc_rnn_pushshapes_paperexact_hnet_chunk8_fullhist_ratio"


def resolve(name):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name=CONFIG_NAME, overrides=[f"model={name}"])
        return OmegaConf.to_container(cfg.model, resolve=True)


def main():
    # 1. parent vs baseline
    parent = resolve(PARENT)
    parent_str = json.dumps(parent, sort_keys=True, indent=2) + "\n"
    with open(BASELINE) as f:
        baseline_str = f.read()
    parent_match = parent_str == baseline_str
    print(f"[dump] PARENT resolves == saved baseline JSON : {parent_match}")
    if not parent_match:
        # show a compact diff
        import difflib
        b = baseline_str.splitlines()
        p = parent_str.splitlines()
        diff = list(difflib.unified_diff(b, p, "baseline", "parent_now", lineterm=""))
        print("\n".join(diff[:60]))

    # parent must not carry an active ratio knob
    rm = parent.get("robomimic_model", parent)
    parent_crl = rm.get("collect_ratio_loss", "<absent>")
    print(f"[dump] PARENT collect_ratio_loss : {parent_crl}")

    # 2. leaf
    leaf = resolve(LEAF)
    lrm = leaf.get("robomimic_model", leaf)
    leaf_crl = lrm.get("collect_ratio_loss", "<absent>")
    core = lrm.get("core_net", {})
    tcr = core.get("target_compression_ratio", "<absent>")
    rlw = core.get("ratio_loss_weight", "<absent>")
    print(f"[dump] LEAF collect_ratio_loss : {leaf_crl}")
    print(f"[dump] LEAF target_compression_ratio : {tcr}")
    print(f"[dump] LEAF ratio_loss_weight : {rlw}")

    # 3. leaf minus the knob must equal parent (single-knob diff). Compare with
    #    collect_ratio_loss removed and core_net ratio fields normalized.
    import copy
    leaf_norm = copy.deepcopy(leaf)
    lrm2 = leaf_norm.get("robomimic_model", leaf_norm)
    lrm2.pop("collect_ratio_loss", None)
    parent_norm = copy.deepcopy(parent)
    prm2 = parent_norm.get("robomimic_model", parent_norm)
    prm2.pop("collect_ratio_loss", None)
    leaf_minus_knob_eq_parent = (
        json.dumps(leaf_norm, sort_keys=True) == json.dumps(parent_norm, sort_keys=True)
    )
    print(f"[dump] LEAF (minus collect_ratio_loss) == PARENT : {leaf_minus_knob_eq_parent}")

    ok = (
        parent_match
        and (parent_crl in ("<absent>", False))
        and (leaf_crl is True)
        and (float(tcr) == 2.0)
        and (float(rlw) == 0.03)
        and leaf_minus_knob_eq_parent
    )
    print(f"[dump] OVERALL_CONFIG_OK : {ok}")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
