"""Self-gate for the hnet_pushshapes cluster refactor.

Re-composes each TOUCHED leaf the baseline way and diffs the resolved JSON vs the
phase-2 baseline dump. Identity bar: every touched leaf must diff EMPTY.
"""
import json
import os

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = "/coc/flash7/paphiwetsa3/projects/EgoVerse2"
CONFIG_DIR = os.path.join(REPO, "egomimic/hydra_configs")
BASELINE = os.path.join(REPO, "scratch/config_phase2_baseline/resolved/model")
ENTRY = "train_zarr_cartesian"

# leaves I touched (must diff EMPTY). The two _common bases are abstract and
# have no baseline dump; they're validated transitively via the leaves.
TOUCHED = [
    "hnet_pushshapes",
    "hnet_pushshapes_goal",
    "hnet_pushshapes_chunktoken",
    "hnet_pushshapes_chunktoken_hptfull",
]
# leaves left STANDALONE (not edited) -- compose them too as a sanity check that
# the new abstract bases didn't break sibling resolution.
UNTOUCHED = [
    "hnet_pushshapes_big",
    "hnet_pushshapes_recipe",
    "hnet_pushshapes_crossattn",
]


def flat(d, p=""):
    o = {}
    if isinstance(d, dict):
        for k, v in d.items():
            o.update(flat(v, p + "." + str(k)))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            o.update(flat(v, p + "[" + str(i) + "]"))
    else:
        o[p] = d
    return o


def resolved(name):
    cfg = compose(config_name=ENTRY, overrides=[f"model={name}"])
    return OmegaConf.to_container(cfg.model, resolve=True)


def diff(name):
    cur = resolved(name)
    base = json.load(open(os.path.join(BASELINE, name + ".json")))
    fc, fb = flat(cur), flat(base)
    keys = sorted(set(fc) | set(fb))
    bad = [(k, fb.get(k, "<MISS>"), fc.get(k, "<MISS>"))
           for k in keys if fc.get(k, "<MISS>") != fb.get(k, "<MISS>")]
    return bad


def main():
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        print("=== TOUCHED (must be EMPTY) ===")
        all_clean = True
        for n in TOUCHED:
            try:
                bad = diff(n)
            except Exception as e:  # noqa: BLE001
                print(f"  {n}: COMPOSE-ERROR {e}")
                all_clean = False
                continue
            if bad:
                all_clean = False
                print(f"  {n}: {len(bad)} DIFFS")
                for k, b, c in bad:
                    print(f"      {k}: baseline={b}  current={c}")
            else:
                print(f"  {n}: EMPTY (identity OK)")
        print("=== UNTOUCHED siblings (sanity: still EMPTY) ===")
        for n in UNTOUCHED:
            try:
                bad = diff(n)
            except Exception as e:  # noqa: BLE001
                print(f"  {n}: COMPOSE-ERROR {e}")
                all_clean = False
                continue
            print(f"  {n}: {'EMPTY' if not bad else str(len(bad)) + ' DIFFS'}")
            for k, b, c in bad:
                print(f"      {k}: baseline={b}  current={c}")
        print("\nGATE:", "PASS" if all_clean else "FAIL")


if __name__ == "__main__":
    main()
