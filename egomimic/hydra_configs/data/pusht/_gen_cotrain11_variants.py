import pathlib, yaml, copy
WT = pathlib.Path('/Users/rpunamiya/Desktop/GEAR/sim_run/wt_sweep')
TRAIN = ["L","chain_gripper","circle","flipper","gripper","scoop","spring","stick","triangle","u_socket","umi"]
doms = [f"pushshapes_sim_{e}" for e in TRAIN]
ROOT = "${oc.env:PUSHSHAPES_ROOT,/workspace/pushshapes}/ideal"
DD = WT/'egomimic/hydra_configs/data/pusht'
MD = WT/'egomimic/hydra_configs/model/bf'
base_model = yaml.safe_load((MD/'bf_pipeline_sampler_usocket_chain_points_dense_medium_h16.yaml').read_text())

# (name, D, M, rotation_radius, velocity_layout, note, hybrid_rotation_unit)
VARIANTS = [
 ("arc_D10_M16_append_r0",  10, 16, 0.0,  "append",
  "M matched to the h16 baseline's width with a much smaller D, so the token spans a short arc at baseline resolution."),
 ("arc_D10_M16_concat_r0",  10, 16, 0.0,  "concat",
  "Same as append but the velocity rides every waypoint instead of a trailing row."),
 ("arc_D10_M16_append_r30", 10, 16, 30.0, "append",
  "Rotation enters the distance function: lambda=2sqrt(2)*30, so rotating costs what translating 30 units costs."),
 ("arc_D25_M16_append_r0",  25, 16, 0.0,  "append",
  "Width matched to h16, D between the fine 10 and the coarse 50."),
 ("arc_D10_M16_hybrid_rm", 10, 16, 0.0, "append",
  "HYBRID rate-matched: rotation gets its own budget and the token spans "
  "min(D/v_trans, D_rot/v_rot). D_rot=0.044 is D times the measured median "
  "ratio 0.0044 rad/unit, i.e. Cor 6.9's D_s ~ v_s, so both streams exhaust "
  "together and this should behave like translation-only -- the control.", 0.044),
 ("arc_D10_M16_hybrid_tight", 10, 16, 0.0, "append",
  "HYBRID rotation-limited: D_rot=0.015 is ~3x tighter than rate-matched, so "
  "rotation binds and truncates the token whenever the effector turns -- the "
  "case where independent budgets differ from one translational clock.", 0.015),
]

def data_cfg(name, D, M, rr, lay, hyb=None):
    hybline = f"        hybrid_rotation_unit: {float(hyb)}\n" if hyb is not None else ""
    def block(e, mode):
        return f"""  pushshapes_sim_{e}:
    _target_: egomimic.rldb.zarr.zarr_dataset_multi.MultiDataset._from_resolver
    resolver:
      _target_: egomimic.rldb.zarr.zarr_dataset_multi.LocalEpisodeResolverWithEmbodimentOverride
      folder_path: {ROOT}/{e}/T
      embodiment_override: pushshapes_sim_{e}
      key_map:
        _target_: egomimic.rldb.embodiment.pushshapes.get_keymap_hpt
        action_horizon: 200
      transform_list:
        _target_: egomimic.rldb.embodiment.pushshapes.get_planar_arc_length_transform_list
        min_distance_unit: {float(D)}
        resampled_vector_length: {M}
        dt: 0.03333333333333333
        rotation_radius: {float(rr)}
        velocity_mode: mean_scalar
        velocity_layout: {lay}
{hybline}    mode: {mode}
    valid_ratio: 0.02
    bounds_check: false
"""
    def loaders(nm, bs, nw, anch):
        s = f"{nm}:\n"
        for i, e in enumerate(TRAIN):
            s += (f"  pushshapes_sim_{e}: &{anch}\n    batch_size: {bs}\n    num_workers: {nw}\n"
                  f"    pin_memory: true\n    persistent_workers: true\n    prefetch_factor: 3\n") if i == 0 \
                 else f"  pushshapes_sim_{e}: *{anch}\n"
        return s
    body = f"""_target_: egomimic.pl_utils.pl_data_utils.MultiDataModuleWrapper

# ARC D={D} M={M} rotation_radius={rr} velocity_layout={lay}
# 11-embodiment cotrain; HELD OUT circle_small, suction.
# Effectors emit 2, 3 or 4 native channels, all widened to a shared
# [x, y, cos, sin, grip] so one head serves all 11.

train_datasets:
""" + "".join(block(e, "train") for e in TRAIN) + "\nvalid_datasets:\n" \
    + "".join(block(e, "valid") for e in TRAIN) + "\n" \
    + loaders("train_dataloader_params", 16, 6, "train_loader") + "\n" \
    + loaders("valid_dataloader_params", 16, 4, "valid_loader")
    (DD/f"cotrain11_{name}.yaml").write_text(body)

def model_cfg(name, M, lay, note):
    d = copy.deepcopy(base_model); m = d['robomimic_model']
    m['domains'] = list(doms); m['ac_keys'] = {x: 'actions' for x in doms}
    m.pop('rollout_adapters', None)
    dim = 5 if lay == "append" else 6      # concat appends the velocity channel
    hor = M + 1 if lay == "append" else M  # append adds a trailing velocity row
    for st in m.get('stages', []):
        if isinstance(st, dict):
            if 'action_dims' in st: st['action_dims'] = {x: dim for x in doms}
            if 'schedule_anchor_domain' in st: st['schedule_anchor_domain'] = 'pushshapes_sim_circle'
    def seth(o):
        if isinstance(o, dict):
            for k, v in o.items():
                # act_seq is CrossTransformer's positional-embedding length; if it
                # does not track action_horizon the first forward pass dies with
                # 'size of tensor a (17) must match tensor b (16)' -- AFTER the
                # R2 pull and norm-stats, i.e. ~2.5h of staging burned.
                if k in ('action_horizon', 'act_seq') and isinstance(v, int): o[k] = hor
                else: seth(v)
        elif isinstance(o, list):
            for v in o: seth(v)
    seth(d)
    hdr = (f"# {note}\n# layout={lay} -> action_horizon={hor}, action_dims={dim}\n"
           f"# 11-embodiment cotrain; HELD OUT circle_small, suction.\n")
    (MD/f"bf_cotrain11_{name}.yaml").write_text(hdr + yaml.safe_dump(d, sort_keys=False, default_flow_style=False))

for v in VARIANTS:
    name, D, M, rr, lay, note = v[:6]; hyb = v[6] if len(v) > 6 else None
    data_cfg(name, D, M, rr, lay, hyb); model_cfg(name, M, lay, note)
    print(f"  {name:26} D={D:>3} M={M} r={rr:>4} layout={lay:6} horizon={M+1 if lay=='append' else M} dims={5 if lay=='append' else 6}")
