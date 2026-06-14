"""ADVERSARIAL ratio-loss runtime probe (independent re-derivation).

Build mirrors egomimic.trainHydra.train() VERBATIM (datasets -> datamodule ->
norm_stats with per-dataset keymap norm_mode -> ModelWrapper). For BOTH configs
(PARENT=knob false, LEAF=knob true) on the SAME fixed-seed real new_circle_3
batch, exercises the REAL forward_training + compute_losses + pl_model contract.

Checks: P0 default-path identity, P1 ratio finite nonzero, P2 no double-weight,
P3 action_loss invariant, P4 grad flow to router (ON asserted; OFF reported).
"""
import copy
import os
import sys

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = "/coc/flash7/paphiwetsa3/projects/EgoVerse2"
CONFIG_DIR = os.path.join(REPO, "egomimic/hydra_configs")
CONFIG_NAME = "train_zarr_cartesian"
SEED = 12345
PARENT = "bc_rnn_pushshapes_paperexact_hnet_chunk8_fullhist"
LEAF = "bc_rnn_pushshapes_paperexact_hnet_chunk8_fullhist_ratio"

sys.path.insert(0, REPO)
import hydra  # noqa: E402
import egomimic.utils.hydra_resolvers  # noqa: E402,F401  -- registers OmegaConf resolvers
from egomimic.pl_utils.pl_model import ModelWrapper  # noqa: E402
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset  # noqa: E402
from egomimic.trainHydra import _build_model_config_tree  # noqa: E402
from egomimic.rldb.zarr.utils import set_global_seed  # noqa: E402
from egomimic.utils.aws.aws_data_utils import load_env  # noqa: E402


def build(name):
    """Replicate trainHydra.train()'s dataset+norm+model build for `model=name`."""
    set_global_seed(SEED)
    load_env()
    NC3 = "/coc/flash7/paphiwetsa3/datasets/new_circle_3"
    KM = "egomimic.rldb.embodiment.pushshapes.get_keymap_eval"
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(
            config_name=CONFIG_NAME,
            overrides=[
                f"model={name}",
                "data=tsimulation",
                "norm_stats.norm_mode=minmax",
            ],
        )
    # Apply the dataset path/keymap/batch overrides via direct assignment (the
    # tsimulation valid leaf resolves resolver lazily so CLI-style overrides on
    # those keys are rejected at compose time; this is exactly what the launcher
    # passes, applied post-compose with struct relaxed).
    OmegaConf.set_struct(cfg, False)
    for split in ("train_datasets", "valid_datasets"):
        node = cfg.data[split]["pushshapes_sim"]
        node.resolver.folder_path = NC3
        node.resolver.key_map._target_ = KM
    cfg.data.train_dataloader_params.pushshapes_sim.batch_size = 16
    cfg.data.valid_dataloader_params.pushshapes_sim.batch_size = 16
    OmegaConf.set_struct(cfg, True)

    train_datasets = {}
    for dn in cfg.data.train_datasets:
        train_datasets[dn] = hydra.utils.instantiate(cfg.data.train_datasets[dn])
    valid_datasets = {}
    for dn in cfg.data.valid_datasets:
        valid_datasets[dn] = hydra.utils.instantiate(cfg.data.valid_datasets[dn])

    datamodule = hydra.utils.instantiate(
        cfg.data, train_datasets=train_datasets, valid_datasets=valid_datasets
    )

    norm_stats = MultiDataset(
        state={},
        norm_mode=OmegaConf.select(cfg, "norm_stats.norm_mode", default="quantile"),
    )
    norm_stats.populate_from_datasets(datamodule.train_datasets)
    for dn, dataset in datamodule.train_datasets.items():
        norm_stats.infer_shapes_from_batch(dataset[0])
        inst_copy = copy.deepcopy(cfg.data.train_datasets[dn])
        km = OmegaConf.to_container(inst_copy.resolver.key_map, resolve=False)
        km["norm_mode"] = True
        inst_copy.resolver.key_map = km
        norm_dataset = hydra.utils.instantiate(inst_copy)
        norm_stats.infer_norm_from_dataset(
            norm_dataset,
            dn,
            sample_frac=OmegaConf.select(cfg, "norm_stats.sample_frac", default=1.0),
            num_workers=OmegaConf.select(cfg, "norm_stats.num_workers", default=4),
            precomputed_norm_path=OmegaConf.select(
                cfg, "norm_stats.precomputed_norm_path", default=None
            ),
        )
    for ds in datamodule.train_datasets.values():
        ds.set_norm_stats_from(norm_stats)
    for ds in datamodule.valid_datasets.values():
        ds.set_norm_stats_from(norm_stats)

    model = ModelWrapper(
        config_tree=_build_model_config_tree(cfg),
        norm_stats_state=norm_stats.to_state(),
        scheduler_interval=cfg.model.get("scheduler_interval", "step"),
    )
    return cfg, datamodule, model


def _to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(v, device) for v in obj)
    return obj


def first_batch(datamodule):
    set_global_seed(SEED)
    dl = datamodule.train_dataloader()
    b = next(iter(dl))
    # the packed train_dataloader yields (batch_dict, idx, idx); Lightning's
    # loop consumes element 0 as the batch. Extract the dict.
    if isinstance(b, (list, tuple)):
        b = b[0]
    return b


def fwd(model, batch, device, force_knob=None, reseed=None):
    algo = model.model
    if force_knob is not None:
        algo.collect_ratio_loss = force_knob
    if reseed is not None:
        # Re-seed immediately before the stochastic forward (random image crop +
        # GMM low_noise sampling) so two forwards we want to COMPARE share RNG.
        torch.manual_seed(reseed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(reseed)
    pb = algo.process_batch_for_training(batch)
    pb = _to_device(pb, algo.device)
    preds = algo.forward_training(pb)
    losses = algo.compute_losses(preds, pb)
    return preds, losses, algo, pb


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[probe] device={device}")

    cfgP, dmP, modelP = build(PARENT)
    algoP = modelP.model
    print(f"[probe] PARENT collect_ratio_loss = {getattr(algoP,'collect_ratio_loss','<absent>')}  core={type(algoP.nets['policy'].lstm).__name__}")
    batchP = first_batch(dmP)
    predsP, lossesP, algoP, pbP = fwd(modelP, batchP, device)

    cfgL, dmL, modelL = build(LEAF)
    algoL = modelL.model
    core = algoL.nets["policy"].lstm
    print(f"[probe] LEAF collect_ratio_loss = {getattr(algoL,'collect_ratio_loss','<absent>')}  core={type(core).__name__}  w={core.ratio_loss_weight}  tcr={core.target_compression_ratio}")
    batchL = first_batch(dmL)
    # DETERMINISTIC pair on the SAME LEAF model + SAME batch + SAME RNG seed:
    # knob TRUE vs knob FALSE. The model has stochastic ops (random image crop,
    # GMM low_noise sampling), so we re-seed identically before each so the ONLY
    # difference is the collect_ratio_loss branch. This isolates the knob effect.
    RS = 777
    predsL, lossesL, algoL, pbL = fwd(modelL, batchL, device, force_knob=True, reseed=RS)
    predsLfalse, lossesLfalse, _, _ = fwd(modelL, batchL, device, force_knob=False, reseed=RS)
    algoL.collect_ratio_loss = True

    emb_ids = list(pbP.keys())
    print(f"[probe] emb_ids={emb_ids}  len(batch)={len(pbL)}")

    # P0 default-path identity. On the SAME LEAF model + batch + RNG seed, the
    # knob=False path must reproduce EXACTLY the legacy placeholder behavior:
    #   (a) ratio_loss == 0.0 placeholder, and
    #   (b) the optimized action_loss is byte-identical to the knob=True run's
    #       action component (the ratio term is the ONLY delta). We assert this
    #       via the per-emb action_loss (deterministic), since the scalar
    #       action_loss bundles +ratio on the True run.
    # Also PARENT(false) emits the 0.0 placeholder (independent build).
    p0 = True
    for e in emb_ids:
        rlP = predsP[f"{e}_ratio_loss"]
        zP = bool(torch.equal(rlP, torch.zeros_like(rlP)))
        rlLf = predsLfalse[f"{e}_ratio_loss"]
        zLf = bool(torch.equal(rlLf, torch.zeros_like(rlLf)))
        print(f"[P0] PARENT emb{e} ratio_loss==0.0 : {zP} (val={rlP.item():.10f})")
        print(f"[P0] LEAF(knob=false) emb{e} ratio_loss==0.0 : {zLf} (val={rlLf.item():.10f})")
        p0 = p0 and zP and zLf
    # action component byte-identical on the same model+seed (only the ratio term differs)
    for e in emb_ids:
        aT = predsL[f"{e}_action_loss"].item()
        aF = predsLfalse[f"{e}_action_loss"].item()
        same = abs(aT - aF) < 1e-9
        print(f"[P0] LEAF same-seed action_loss True={aT:.10f} False={aF:.10f} byte-identical : {same}")
        p0 = p0 and same
    # and the knob=False optimized scalar == its own action component (no +ratio)
    sf = abs(lossesLfalse['action_loss'].item() - predsLfalse[f"{emb_ids[0]}_action_loss"].item()) < 1e-7
    print(f"[P0] LEAF(false) optimized action_loss == emb action_loss (no ratio added) : {sf}")
    p0 = p0 and sf

    # P1
    p1 = True
    for e in emb_ids:
        rl = predsL[f"{e}_ratio_loss"]
        fin = bool(torch.isfinite(rl).all()); nz = bool(rl.abs().item() > 0)
        print(f"[P1] LEAF emb{e} ratio_loss={rl.item():.10f} finite={fin} nonzero={nz}")
        p1 = p1 and fin and nz

    # P2
    n = len(pbL); hand = 0.0
    for e in emb_ids:
        hand += predsL[f"{e}_action_loss"].item() + predsL[f"{e}_ratio_loss"].item()
    hand /= max(n, 1)
    rep = lossesL['action_loss'].item()
    p2a = abs(hand - rep) < 1e-5
    print(f"[P2] by-hand (a+r)/{n}={hand:.10f} vs compute_losses action_loss={rep:.10f} match={p2a}")
    w = core.ratio_loss_weight
    for e in emb_ids:
        r = predsL[f"{e}_ratio_loss"].item(); raw = r / w if w else float('nan')
        ratio = (r / raw) if raw else float('nan')
        print(f"[P2] emb{e} weighted={r:.10f} weight={w} raw_unweighted={raw:.10f} weighted/raw={ratio:.6f}(==weight)")

    # P3 action_loss invariant: the knob=True action component == knob=False
    # action component on the SAME model+seed (ratio added separately, action
    # path untouched). This is the same-seed deterministic comparison.
    p3 = True
    for e in emb_ids:
        at = predsL[f"{e}_action_loss"].item(); af = predsLfalse[f"{e}_action_loss"].item()
        inv = abs(at - af) < 1e-9
        print(f"[P3] emb{e} action_loss ON={at:.10f} OFF={af:.10f} invariant={inv}")
        p3 = p3 and inv

    # P4 grad flow. Re-seed identically before ON and OFF forwards so the ONLY
    # difference in the router gradient is the ratio term's contribution.
    def rgrad(algo, knob, rs):
        algo.collect_ratio_loss = knob
        for p in algo.nets.parameters():
            p.grad = None
        torch.manual_seed(rs)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(rs)
        pb = _to_device(algo.process_batch_for_training(batchL), algo.device)
        preds = algo.forward_training(pb)
        losses = algo.compute_losses(preds, pb)
        losses['action_loss'].backward()
        g = {}
        for nme, p in algo.nets.named_parameters():
            if 'routing_module' in nme and ('q_proj' in nme or 'k_proj' in nme):
                g[nme] = None if p.grad is None else float(p.grad.abs().sum().item())
        return g, losses['action_loss'].item()
    GS = 4242
    gON, lON = rgrad(algoL, True, GS)
    gOFF, lOFF = rgrad(algoL, False, GS)
    print(f"[P4] knob ON total={lON:.8f}")
    for k, v in gON.items():
        print(f"[P4]   ON  {k}: grad_abs_sum={v}")
    print(f"[P4] knob OFF total={lOFF:.8f}")
    for k, v in gOFF.items():
        print(f"[P4]   OFF {k}: grad_abs_sum={v}")
    p4 = len(gON) > 0 and all(v is not None and v > 0 for v in gON.values())
    diffr = any(gON.get(k) is not None and gOFF.get(k) is not None and abs(gON[k]-gOFF[k]) > 1e-9 for k in gON)
    print(f"[P4] router grad present+nonzero ON : {p4}")
    print(f"[P4] router grad DIFFERS ON vs OFF : {diffr}")

    overall = p0 and p1 and p2a and p3 and p4 and diffr
    print("\n[probe] ===== SUMMARY =====")
    print(f"[probe] P0 default-path identity : {p0}")
    print(f"[probe] P1 ratio finite nonzero  : {p1}")
    print(f"[probe] P2 no double-weight      : {p2a}")
    print(f"[probe] P3 action_loss invariant : {p3}")
    print(f"[probe] P4 grad ON + differs      : {p4 and diffr}")
    print(f"[probe] OVERALL_RUNTIME_OK : {overall}")
    sys.exit(0 if overall else 3)


if __name__ == "__main__":
    main()
