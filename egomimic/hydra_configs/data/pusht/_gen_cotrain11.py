"""Emit the 11-embodiment co-train data configs: dense h16 baseline + arc grid.

Held out: circle_small (interpolation probe -- mechanically near-identical to
circle) and suction (extrapolation probe -- the one non-pushing mechanism).
Regenerate with:  python _gen_cotrain11.py
"""
import pathlib

ALL = ["L","chain_gripper","circle","circle_small","flipper","gripper","scoop",
       "spring","stick","suction","triangle","u_socket","umi"]
HELDOUT = ["circle_small", "suction"]
TRAIN = [e for e in ALL if e not in HELDOUT]
ROOT = "${oc.env:PUSHSHAPES_ROOT,/workspace/pushshapes_dedup_gen}/ideal"
OUT = pathlib.Path(__file__).parent

HEAD = """_target_: egomimic.pl_utils.pl_data_utils.MultiDataModuleWrapper

# {note}
#
# 11-embodiment PushShapes co-train. HELD OUT: {held}.
#   circle_small is mechanically near-identical to circle (2-D pusher, similar
#   path length) -> interpolation probe, expected to transfer.
#   suction is the one non-pushing mechanism (grip latch, ~1067-frame
#   episodes) -> extrapolation probe, expected not to.
# Effectors emit 2, 3 or 4 native action channels; every recipe here widens
# them to a shared [x, y, cos, sin, grip] so one action head serves all 11 and
# the dense/arc comparison is not confounded by a change of action space.
"""

def block(emb, mode, horizon, factory, params):
    p = "".join(f"        {k}: {v}\n" for k, v in params.items())
    return f"""  pushshapes_sim_{emb}:
    _target_: egomimic.rldb.zarr.zarr_dataset_multi.MultiDataset._from_resolver
    resolver:
      _target_: egomimic.rldb.zarr.zarr_dataset_multi.LocalEpisodeResolverWithEmbodimentOverride
      folder_path: {ROOT}/{emb}/T
      embodiment_override: pushshapes_sim_{emb}
      key_map:
        _target_: egomimic.rldb.embodiment.pushshapes.get_keymap_hpt
        action_horizon: {horizon}
      transform_list:
        _target_: egomimic.rldb.embodiment.pushshapes.{factory}
{p}    mode: {mode}
    valid_ratio: 0.02
    bounds_check: false
"""

def loaders(name, bs, nw, anchor):
    s = f"{name}:\n"
    for i, e in enumerate(TRAIN):
        s += (f"  pushshapes_sim_{e}: &{anchor}\n    batch_size: {bs}\n"
              f"    num_workers: {nw}\n    pin_memory: true\n"
              f"    persistent_workers: true\n    prefetch_factor: 3\n") if i == 0 \
             else f"  pushshapes_sim_{e}: *{anchor}\n"
    return s

def emit(name, horizon, factory, params, note):
    body = HEAD.format(note=note, held=", ".join(HELDOUT))
    body += "\ntrain_datasets:\n" + "".join(block(e,"train",horizon,factory,params) for e in TRAIN)
    body += "\nvalid_datasets:\n" + "".join(block(e,"valid",horizon,factory,params) for e in TRAIN)
    body += "\n" + loaders("train_dataloader_params", 16, 6, "train_loader")
    body += "\n" + loaders("valid_dataloader_params", 16, 4, "valid_loader")
    (OUT / f"{name}.yaml").write_text(body)
    return name

made = [emit("cotrain11_dense_h16", 16, "get_planar_dense_transform_list", {},
             "BASELINE: dense 16-timestep chunks, no arc tokenization.")]
for D in (50, 100, 200):
    for M in (50, 100, 200):
        made.append(emit(f"cotrain11_arc_D{D}_M{M}", 200,
            "get_planar_arc_length_transform_list",
            {"min_distance_unit": f"{float(D)}", "resampled_vector_length": M,
             "dt": 0.03333333333333333, "rotation_radius": 40.0},
            f"ARC: planar SE(2)+grip tokens, D={D} M={M} -> ({M}+1, 5) per chunk."))
print("\n".join(made))
print(f"\ntrain ({len(TRAIN)}): {', '.join(TRAIN)}")
print(f"held out ({len(HELDOUT)}): {', '.join(HELDOUT)}")
