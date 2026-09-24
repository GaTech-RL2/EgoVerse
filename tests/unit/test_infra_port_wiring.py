"""Wiring checks for the hand-keypoint default: recipes, stems, the mecka
left-wrist fix in keypoint modes, the annotation-cutoff span filter and the
train_viz second val loader."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from lightning import LightningModule
from scipy.spatial.transform import Rotation as R

import egomimic
from egomimic.rldb.embodiment.human import Human

CONFIG_DIR = Path(egomimic.__file__).parent / "hydra_configs"
TRAIN_TOP_LEVEL = sorted(p.stem for p in CONFIG_DIR.glob("train_*.yaml"))
HUMAN = "human_bimanual"

# The all-cartesian cam-frame cotrain recipe is cotrain_pi_lang plus these
# overrides (model/pi0.5_cotrain_eva_aria_6d.yaml); there is no data config
# that differs from its parent only in the transform mode.
CAM_FRAME_COTRAIN = [
    "data.train_datasets.eva_bimanual.resolver.transform_list.mode=cartesian_6d",
    f"data.train_datasets.{HUMAN}.resolver.key_map.keymap_mode=cartesian",
    f"data.train_datasets.{HUMAN}.resolver.key_map.include_ee_pose=false",
    f"data.train_datasets.{HUMAN}.resolver.transform_list.mode=cartesian_6d",
    f"data.train_datasets.{HUMAN}.resolver.transform_list.include_ee_pose=false",
]


def _human_dataset_nodes(cfg):
    for split in (
        "train_datasets",
        "valid_datasets",
        "train_viz_datasets",
        "unseen_op_valid_datasets",
    ):
        node = (cfg.data.get(split) or {}).get(HUMAN)
        if node is not None and node.get("resolver") is not None:
            yield split, node.resolver


@pytest.mark.parametrize("top", TRAIN_TOP_LEVEL)
def test_human_action_width_matches_data_mode(top, compose_resolve):
    """A model declaring the 144-D human action must be fed the
    keypoints_*_6d data (and vice versa); a stale ypr/cartesian mode would
    only fail at the first forward pass."""
    cfg = compose_resolve(top, [])
    rm = cfg.model.robomimic_model
    if HUMAN not in (rm.get("domains") or []):
        pytest.skip(f"{top}: no human domain")
    dims = (rm.get("dims") or {}).get(HUMAN)
    ac_key = rm.ac_keys[HUMAN]
    for split, resolver in _human_dataset_nodes(cfg):
        mode = resolver.transform_list.mode
        keymap_mode = resolver.key_map.keymap_mode
        if ac_key == "actions_keypoints":
            assert keymap_mode == "keypoints", (top, split, keymap_mode)
            assert mode.startswith("keypoints_") and mode.endswith("_6d"), (
                top,
                split,
                mode,
            )
            if dims is not None:
                assert dims.action == 144, (top, split, dims)
        else:
            assert keymap_mode == "cartesian", (top, split, keymap_mode)
            assert mode.startswith("cartesian"), (top, split, mode)


@pytest.mark.parametrize(
    "model,data,extra",
    [
        ("hpt_bc_flow_aria", "aria", []),
        ("hpt_bc_flow_mecka", "mecka", []),
        ("hpt_bc_flow_scale", "scale", []),
        ("hpt_bc_flow_human", "human", []),
        ("pi0.5_bc_aria", "aria", []),
        ("pi0.5_bc_mecka", "mecka", []),
        ("pi0.5_bc_scale", "scale", []),
        ("pi0.5_cotrain_eva_aria", "cotrain_pi_base", []),
        ("pi0.5_bc_mecka_6d", "mecka_all_6d", []),
        ("pi0.5_cotrain_eva_aria_6d", "cotrain_pi_lang", CAM_FRAME_COTRAIN),
        ("hpt_bc_mecka_6d_300M", "mecka_fold_flagship_opsplit_hpt_6d", []),
        (
            "hpt_bc_keypoints_wrist_300M",
            "mecka_fold_freeform_opsplit_hpt_keypoints",
            [],
        ),
    ],
)
def test_vendor_pairings_agree_on_the_human_action(model, data, extra, compose_resolve):
    cfg = compose_resolve(
        "train_zarr_cartesian", [f"model={model}", f"data={data}", *extra]
    )
    rm = cfg.model.robomimic_model
    ac_key = rm.ac_keys[HUMAN]
    resolver = cfg.data.train_datasets[HUMAN].resolver
    mode = resolver.transform_list.mode
    if ac_key == "actions_keypoints":
        assert mode.endswith("_6d") and mode.startswith("keypoints_"), (model, mode)
        if rm.get("dims"):
            assert rm.dims[HUMAN].action == 144
        if "pi0.5" in model:
            assert rm.config.model.action_dim == 144
            assert "Keypoints" in rm.action_converters.rules.HUMAN_BIMANUAL._target_
            # the pi0.5 prompt state is the cartesian ee_pose
            assert resolver.key_map.get("include_ee_pose") is True
            assert resolver.transform_list.get("include_ee_pose") is True
    else:
        assert mode.startswith("cartesian"), (model, mode)
        if "pi0.5" in model:
            assert mode.endswith("_6d"), (model, mode)
            assert rm.config.model.action_dim == 32


def test_hpt_stems_follow_the_human_action(compose_resolve):
    """HPT human stems: keypoint models expose exactly ``state_keypoints``
    (a base's ``state_ee_pose`` is nulled out, not inherited)."""
    from egomimic.algo.hpt import HPTModel

    for model in ("hpt_bc_flow_mecka", "hpt_cotrain_flow_seperate_head"):
        cfg = compose_resolve("train_zarr_cartesian", [f"model={model}"])
        stems = cfg.model.robomimic_model.stem_specs[HUMAN]
        live = {k for k, v in stems.items() if v is not None}
        assert live == {"state_keypoints"}, (model, live)
    cfg = compose_resolve(
        "train_zarr_cartesian", ["model=hpt_cotrain_flow_shared_head"]
    )
    stems = cfg.model.robomimic_model.stem_specs[HUMAN]
    assert {k for k, v in stems.items() if v is not None} == {"state_ee_pose"}

    # the model side drops null stems instead of registering None
    m = HPTModel.__new__(HPTModel)
    torch.nn.Module.__init__(m)
    m.stem_spec, m.modalities, m.stems = {}, {}, {}
    m.init_domain_stem("human_bimanual", {"state_ee_pose": None, "x": object()})
    assert m.modalities["human_bimanual"] == ["x"]
    assert set(m.stems) == {"human_bimanual_x"}


# ------------------------------------------------------ mecka left-wrist fix
def _pose(rng):
    q = R.random(random_state=int(rng.integers(1 << 31))).as_quat()
    return np.concatenate([rng.uniform(-1, 1, 3), q[[3, 0, 1, 2]]])


def _chunk(rng, start, n):
    out = np.zeros((n, 7))
    p, r = start[:3].copy(), R.from_quat(start[[4, 5, 6, 3]])
    for t in range(n):
        if t:
            p = p + rng.normal(0, 0.01, 3)
            r = R.from_rotvec(rng.normal(0, 0.05, 3)) * r
        out[t] = np.concatenate([p, r.as_quat()[[3, 0, 1, 2]]])
    return out


def _rz180(pose7):
    """What the fixed converter would have written: the same pose with its
    local axes relabelled by Rz(180 deg)."""
    out = np.array(pose7, dtype=np.float64, copy=True)
    q = R.from_quat(out[..., [4, 5, 6, 3]]) * R.from_euler("z", np.pi)
    out[..., 3:7] = q.as_quat()[..., [3, 0, 1, 2]]
    return out


def _apply(tl, s):
    s = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in s.items()}
    for t in tl:
        s = t.transform(s)
    return s


def test_fix_left_wrist_convention_equals_reconverting_in_keypoint_mode():
    rng = np.random.default_rng(31)
    H = Human.ACTION_HORIZON
    raw = {"obs_head_pose": _pose(rng)}
    for side in ("left", "right"):
        wrist = _pose(rng)
        raw[f"{side}.obs_wrist_pose"] = wrist
        raw[f"{side}.action_wrist_pose"] = _chunk(rng, wrist, H)
        kp = raw[f"{side}.action_wrist_pose"][:, None, :3] + rng.uniform(
            -0.1, 0.1, (H, 21, 3)
        )
        raw[f"{side}.action_keypoints"] = kp.reshape(H, 63)
        raw[f"{side}.obs_keypoints"] = kp[0].reshape(63)
        ee = _pose(rng)
        raw[f"{side}.obs_ee_pose"] = ee
        raw[f"{side}.action_ee_pose"] = _chunk(rng, ee, H)
    # "reconverted" twin: the fixed converter relabels the LEFT hand's axes on
    # every pose it writes (wrist_pose and ee_pose share the rotation).
    fixed = dict(raw)
    for k in (
        "left.obs_wrist_pose",
        "left.action_wrist_pose",
        "left.obs_ee_pose",
        "left.action_ee_pose",
    ):
        fixed[k] = _rz180(raw[k])

    kw = dict(stride=1, include_ee_pose=True, pad_proprio_gripper=True)
    with_flag = _apply(
        Human.get_transform_list(
            "keypoints_wristframe_6d", fix_left_wrist_convention=True, **kw
        ),
        raw,
    )
    reconverted = _apply(
        Human.get_transform_list("keypoints_wristframe_6d", **kw), fixed
    )
    assert set(with_flag) == set(reconverted)
    for k in with_flag:
        np.testing.assert_allclose(with_flag[k], reconverted[k], atol=1e-9, err_msg=k)
    # and the flag really changes the left hand's frame
    unfixed = _apply(Human.get_transform_list("keypoints_wristframe_6d", **kw), raw)
    assert (
        np.abs(unfixed["actions_keypoints"] - with_flag["actions_keypoints"]).max()
        > 1e-3
    )
    # the same equivalence without the ee_pose side (plain HPT keypoint data)
    plain = {k: v for k, v in raw.items() if "ee_pose" not in k}
    plain_fixed = {k: v for k, v in fixed.items() if "ee_pose" not in k}
    a = _apply(
        Human.get_transform_list(
            "keypoints_wristframe_6d", stride=1, fix_left_wrist_convention=True
        ),
        plain,
    )
    b = _apply(
        Human.get_transform_list("keypoints_wristframe_6d", stride=1), plain_fixed
    )
    for k in a:
        np.testing.assert_allclose(a[k], b[k], atol=1e-9, err_msg=k)


# --------------------------------------------------- annotation-cutoff filter
def test_episode_has_annotation_spans():
    from egomimic.rldb.zarr.zarr_dataset_multi import _episode_has_annotation_spans

    def ds(anns):
        return SimpleNamespace(_load_annotations=lambda: anns)

    assert _episode_has_annotation_spans(
        ds([{"text": "a", "start_idx": 0, "end_idx": 5}])
    )
    assert not _episode_has_annotation_spans(ds([]))
    assert not _episode_has_annotation_spans(ds([{"text": "a"}]))  # span-less
    assert not _episode_has_annotation_spans(ds([{"start_idx": 5, "end_idx": 5}]))

    def boom():
        raise OSError("corrupt")

    assert not _episode_has_annotation_spans(SimpleNamespace(_load_annotations=boom))


def test_annotation_cutoff_resolver_defaults_to_requiring_spans():
    from egomimic.rldb.zarr.zarr_dataset_multi import S3AnnotationCutoffEpisodeResolver

    r = S3AnnotationCutoffEpisodeResolver.__new__(S3AnnotationCutoffEpisodeResolver)
    sig = inspect.signature(S3AnnotationCutoffEpisodeResolver.__init__)
    assert sig.parameters["require_annotations"].default is True
    good = SimpleNamespace(_load_annotations=lambda: [{"start_idx": 0, "end_idx": 3}])
    bad = SimpleNamespace(_load_annotations=lambda: [])
    r.require_annotations = True
    # bypass the S3 base resolve: patch it on the instance's class chain
    import egomimic.rldb.zarr.zarr_dataset_multi as m

    orig = m.S3EpisodeResolver.resolve
    m.S3EpisodeResolver.resolve = lambda self, filters=None, expected_embodiment=None: {
        "g": good,
        "b": bad,
    }
    try:
        kept = r.resolve(filters=None, expected_embodiment="human_bimanual")
        assert set(kept) == {"g"}
        r.require_annotations = False
        assert set(r.resolve()) == {"g", "b"}
        m.S3EpisodeResolver.resolve = (
            lambda self, filters=None, expected_embodiment=None: {"b": bad}
        )
        r.require_annotations = True
        with pytest.raises(ValueError, match="no resolved episodes"):
            r.resolve()
    finally:
        m.S3EpisodeResolver.resolve = orig


# ------------------------------------------------------ train_viz second loader
class _Dicts:
    def __init__(self, n, tag):
        self.n, self.tag = n, tag

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {"x": np.float32(i), "tag": [self.tag]}


def test_val_dataloader_returns_two_loaders_with_train_viz():
    from lightning.pytorch.utilities.combined_loader import CombinedLoader

    from egomimic.pl_utils.pl_data_utils import MultiDataModuleWrapper

    params = {"human_bimanual": {"batch_size": 2, "num_workers": 0}}
    dm = MultiDataModuleWrapper(
        train_datasets={"human_bimanual": _Dicts(4, "train")},
        valid_datasets={"human_bimanual": _Dicts(4, "valid")},
        train_dataloader_params=params,
        valid_dataloader_params=params,
        held_out_operators=["op1"],
    )
    assert isinstance(dm.val_dataloader(), CombinedLoader)
    dm = MultiDataModuleWrapper(
        train_datasets={"human_bimanual": _Dicts(4, "train")},
        valid_datasets={"human_bimanual": _Dicts(4, "valid")},
        train_dataloader_params=params,
        valid_dataloader_params=params,
        train_viz_datasets={"human_bimanual": _Dicts(4, "viz")},
        train_viz_dataloader_params={
            "human_bimanual": {**params["human_bimanual"], "shuffle": False}
        },
    )
    loaders = dm.val_dataloader()
    assert isinstance(loaders, list) and len(loaders) == 2
    batch, _, _ = next(iter(loaders[1]))
    assert batch["human_bimanual"]["tag"] == [["viz"], ["viz"]]


def test_validation_step_routes_by_dataloader_idx():
    from egomimic.pl_utils.pl_model import ModelWrapper

    calls = []

    class _Ev:
        def __init__(self, name):
            self.name = name

        def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
            calls.append((self.name, batch, batch_idx, dataloader_idx))

    w = ModelWrapper.__new__(ModelWrapper)
    LightningModule.__init__(w)  # global_rank etc. without a trainer
    w.model = SimpleNamespace(process_batch_for_training=lambda b: {"processed": b})
    w.evaluator = _Ev("valid")
    w.train_viz_evaluator = _Ev("train_viz")
    # Lightning hands the outer sequential loader's inner triple as `batch`
    w.validation_step(({"k": 1}, 0, 0), 0, dataloader_idx=0)
    w.validation_step(({"k": 2}, 0, 1), 3, dataloader_idx=1)
    w.validation_step({"k": 3}, 1, dataloader_idx=0)
    assert calls == [
        ("valid", {"processed": {"k": 1}}, 0, 0),
        ("train_viz", {"processed": {"k": 2}}, 3, 1),
        ("valid", {"processed": {"k": 3}}, 1, 0),
    ]
    w.train_viz_evaluator = None
    w.validation_step({"k": 4}, 0, dataloader_idx=1)  # no second head: ignored
    assert len(calls) == 3


def test_unseen_op_valid_third_loader_and_routing():
    """unseen_op_valid_datasets adds a loader after train_viz; without train_viz it
    takes idx 1, and ModelWrapper routes by val_loader_names, not position."""
    from egomimic.pl_utils.pl_data_utils import MultiDataModuleWrapper
    from egomimic.pl_utils.pl_model import ModelWrapper

    params = {"human_bimanual": {"batch_size": 2, "num_workers": 0}}
    common = dict(
        train_datasets={"human_bimanual": _Dicts(4, "train")},
        valid_datasets={"human_bimanual": _Dicts(4, "valid")},
        train_dataloader_params=params,
        valid_dataloader_params=params,
        unseen_op_valid_datasets={"human_bimanual": _Dicts(4, "unseen")},
        unseen_op_valid_dataloader_params=params,
    )
    dm = MultiDataModuleWrapper(
        **common,
        train_viz_datasets={"human_bimanual": _Dicts(4, "viz")},
        train_viz_dataloader_params=params,
    )
    assert dm.val_loader_names() == ["valid", "train_viz", "unseen_op_valid"]
    loaders = dm.val_dataloader()
    tags = [next(iter(ld))[0]["human_bimanual"]["tag"][0][0] for ld in loaders]
    assert tags == ["valid", "viz", "unseen"]

    dm = MultiDataModuleWrapper(**common)
    assert dm.val_loader_names() == ["valid", "unseen_op_valid"]

    calls = []

    class _Ev:
        def __init__(self, name):
            self.name = name

        def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
            calls.append((self.name, dataloader_idx))

    w = ModelWrapper.__new__(ModelWrapper)
    LightningModule.__init__(w)
    w.model = SimpleNamespace(process_batch_for_training=lambda b: b)
    w.evaluator = _Ev("valid")
    w.train_viz_evaluator = None
    w.unseen_op_valid_evaluator = _Ev("unseen_op_valid")
    w.val_loader_names = dm.val_loader_names()
    w.validation_step({"k": 1}, 0, dataloader_idx=0)
    w.validation_step({"k": 2}, 0, dataloader_idx=1)
    assert calls == [("valid", 0), ("unseen_op_valid", 1)]


def test_flagship_opsplit_val_heads(compose_resolve):
    """Flagship opsplit: valid (prefixed seen_op_valid) = seen operators'
    held-out episodes (complement of train), train_viz = the train split (no
    explicit datasets), unseen_op_valid = the held-out operators. The topop
    twin inherits the same layout."""
    for data, train_op in (
        ("mecka_fold_flagship_opsplit_hpt_6d", "not in"),
        ("mecka_fold_flagship_topop_hpt_6d", "== '6903686e0e94ce070afd1f24'"),
    ):
        cfg = compose_resolve("train_zarr_mecka_flagship_6d_hpt", [f"data={data}"])
        d = cfg.data
        train, valid = d.train_datasets[HUMAN], d.valid_datasets[HUMAN]
        unseen = d.unseen_op_valid_datasets[HUMAN]
        assert d.get("train_viz_datasets") is None
        assert d.valid_prefix == "seen_op_valid"
        assert (train.mode, valid.mode, unseen.mode) == ("train", "valid", "total")
        assert valid.valid_ratio == train.valid_ratio
        assert list(valid.filters.filter_lambdas) == list(train.filters.filter_lambdas)
        assert train_op in train.filters.filter_lambdas[-1]
        ops = list(d.held_out_operators)
        assert unseen.filters.filter_lambdas[-1].endswith(f" in {ops}")


def test_bounds_quantiles_include_the_observed_extremes():
    """With fewer than 10k samples a linear 0.01 / 99.99 percentile lands
    strictly inside the sample range, so the stats' own extreme frames were
    rejected at train time; the bounds quantiles must snap to samples."""
    from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset

    X = np.random.default_rng(0).normal(size=(500, 4)).astype(np.float32)
    st = MultiDataset._compute_stats_for_array(X)
    np.testing.assert_array_equal(st["quantile_0_01"], X.min(axis=0))
    np.testing.assert_array_equal(st["quantile_99_99"], X.max(axis=0))
    md = MultiDataset.__new__(MultiDataset)
    md.norm_mode = "quantile"
    md.norm_stats = {0: {"actions_keypoints": st}}
    md.zarr_keys = {0: {"actions_keypoints": "actions_keypoints"}}
    md._warned_violations = set()
    for row in (X.min(axis=0), X.max(axis=0)):
        assert (
            md._check_bounds(
                {"embodiment": 0, "actions_keypoints": row[None]}, None, 0, "ep"
            )
            is None
        )


def test_pi_loss_is_reduced_over_the_packed_width():
    from egomimic.algo.pi import PI
    from egomimic.rldb.embodiment.embodiment import EMBODIMENT
    from egomimic.utils.action_utils import (
        ConverterRegistry,
        HumanBimanualKeypoints,
        RobotBimanualCartesian6D,
    )

    pi = PI.__new__(PI)
    pi.action_registry = ConverterRegistry()
    pi._packed_widths = {}
    eva, human = EMBODIMENT.EVA_BIMANUAL.value, EMBODIMENT.HUMAN_BIMANUAL.value
    pi.action_registry.register(eva, "actions_cartesian", RobotBimanualCartesian6D())
    pi.action_registry.register(human, "actions_keypoints", HumanBimanualKeypoints())
    losses = torch.zeros(2, 5, 144)
    losses[..., 32:] = 1.0  # error only in the zero-padded slots
    eva_loss = pi._reduce_loss(losses, torch.zeros(2, 5, 20), eva, "actions_cartesian")
    assert eva_loss.item() == 0.0
    kp_loss = pi._reduce_loss(
        losses, torch.zeros(2, 5, 144), human, "actions_keypoints"
    )
    assert abs(kp_loss.item() - 112 / 144) < 1e-6
    # already-reduced losses pass through unchanged
    assert (
        pi._reduce_loss(torch.tensor([1.0, 3.0]), None, eva, "actions_cartesian").item()
        == 2.0
    )


def test_pi_process_batch_keeps_the_per_sample_intrinsics():
    from egomimic.algo.pi import PI
    from egomimic.rldb.embodiment.embodiment import get_embodiment_id

    class _NormStats:
        def zarr_key_to_keyname(self, key, embodiment_id):
            return key if key == "actions_keypoints" else None

    pi = PI.__new__(PI)
    pi.norm_stats = _NormStats()
    pi.device = "cpu"
    pi._build_prompts = lambda _batch, name, B: [""] * B
    pi._tokenize_prompts = lambda prompts: {}
    emb = "human_bimanual"
    emb_id = get_embodiment_id(emb)
    pi.ac_keys = {emb_id: "actions_keypoints"}
    K = torch.eye(3).expand(2, 3, 3).clone()
    out = PI.process_batch_for_training(
        pi,
        {emb: {"actions_keypoints": torch.zeros(2, 5, 144), "intrinsics": K}},
    )[emb_id]
    torch.testing.assert_close(out["intrinsics"], K)


def test_bounds_check_has_relative_slack_but_catches_corrupt_values():
    """Per-cell bounds tolerate frames moderately beyond the stats sample's
    range (valid extreme motion) and still reject values orders of magnitude
    off (fill constants, wrong-frame data)."""
    from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset

    md = MultiDataset.__new__(MultiDataset)
    md.norm_mode = "quantile"
    lo, hi = np.full(18, -1.0, np.float32), np.full(18, 1.0, np.float32)
    md.norm_stats = {0: {"actions_cartesian": {"quantile_1": lo, "quantile_99": hi}}}
    md.zarr_keys = {0: {"actions_cartesian": "actions_cartesian"}}
    md._warned_violations = set()

    def check(v):
        arr = np.zeros((3, 18), np.float32)
        arr[1, 0] = v  # an xyz channel
        return md._check_bounds(
            {"embodiment": 0, "actions_cartesian": arr}, None, 0, "e"
        )

    assert check(1.0 + 0.4 * 2.0) is None  # within 50 % of the [-1, 1] range
    assert check(1.0 + 0.6 * 2.0) is not None
    assert check(1e9) is not None


# ------------------------------------------------------ train_viz on by default
def _viz_cfg(**top):
    from omegaconf import OmegaConf

    data = top.pop("data", {})
    cfg = {
        "mode": "train",
        "evaluator": {
            "_target_": "egomimic.eval.eval_hpt.HPTEvalVideo",
            "viz_every_n_epochs": 7,
        },
        "data": {
            "valid_dataloader_params": {HUMAN: {"batch_size": 4, "num_workers": 3}},
            **data,
        },
        **top,
    }
    return OmegaConf.create(cfg)


def test_train_viz_defaults_to_the_train_datasets():
    import egomimic.trainHydra as th

    train = {HUMAN: object()}
    viz, params = th._train_viz_datasets(_viz_cfg(), train, instantiate=None)
    assert viz[HUMAN] is train[HUMAN], "reuses the train split, no second resolve"
    # valid loader params, never shuffled (same leading slice every val)
    assert params == {HUMAN: {"batch_size": 4, "num_workers": 3, "shuffle": False}}


def test_train_viz_explicit_datasets_and_opt_outs():
    import egomimic.trainHydra as th

    train = {HUMAN: object()}
    explicit = _viz_cfg(
        data={
            "train_viz_datasets": {HUMAN: {"tag": "explicit"}},
            "train_viz_dataloader_params": {HUMAN: {"batch_size": 32}},
        }
    )
    viz, params = th._train_viz_datasets(
        explicit, train, instantiate=lambda node, dataset_name: ("built", node.tag)
    )
    assert viz == {HUMAN: ("built", "explicit")}
    assert params is None, "the data config's own params are left to Hydra"

    for off in (
        _viz_cfg(train_viz=False),
        _viz_cfg(mode="eval"),
        _viz_cfg(evaluator=None),
    ):
        assert th._train_viz_datasets(off, train, instantiate=None) == ({}, None)


def test_train_viz_evaluator_wraps_the_canonical_evaluator():
    import egomimic.trainHydra as th
    from egomimic.eval.eval_train_viz import TrainVizEvalVideo

    cfg = _viz_cfg()
    ev = th._build_train_viz_evaluator(cfg)
    assert isinstance(ev, TrainVizEvalVideo)
    assert ev.viz_every_n_epochs == 7
    assert th._build_train_viz_evaluator(_viz_cfg(train_viz=False)) is None
