"""The freeform fold-clothes OPERATOR-SCALING ladder holds its shape.

mecka_fold_freeform_opscale{1,2,4} vary ONE thing: how many operators the
training data comes from. Each arm trains on ALL the annotated data its
operators have, minus the shared held-out val episodes. Everything that would
confound the comparison -- the seen-op val, the unseen-op val, the pipeline --
must be identical across the three rungs, and the operator sets must be nested.

The train split is an OPERATOR FILTER, not an episode pin, so these tests drive
the real DatasetFilter with synthetic rows rather than counting hashes.
"""

from pathlib import Path

import hydra
import pytest
import yaml
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

import egomimic.hydra_configs as _cfg_pkg
from egomimic.rldb.filters import DatasetFilter

DATA_DIR = Path(_cfg_pkg.__file__).parent / "data"
ARMS = (1, 2, 4)
VARIANTS = ("6d", "hpt_keypoints")

# The 4 train operators, ranked by ANNOTATED hours (SQL app.episodes, 2026-09-17).
# Arm k trains on the first k of these. NOT the same order as raw hours.
OPERATORS = [
    "6964d38183a9fdf2d8663792",
    "68e0b875c33a1abcb8fc55b9",
    "695d09fc83a9fdf2d84d9c11",
    "69439ad5a2a8b5aee76602f5",
]
# The only episodes withheld from training: 3 episodes of operator #1.
SEEN_VAL = [
    "696bcbe0e13d37dd4293f58b",
    "696bcbe86377d0871b6a9253",
    "696bcc71ed8ed158122ec794",
]
# Episodes per arm as measured 2026-09-17. The filter is LIVE, so these are
# documentation, not a guarantee; the metric budget is derived from the resolved
# split at runtime (`auto`), so nothing has to be kept in step with them.
TRAIN_EPISODES = {1: 9, 2: 24, 4: 48}
SEEN_VAL_EPISODES = 3
UNSEEN_OPERATORS = 31
SPLITS = ("train_datasets", "valid_datasets", "unseen_op_valid_datasets")

# A row that clears every non-operator lambda: mecka, freeform, fold-clothes,
# annotated, and not one of the withheld episodes.
BASE_ROW = {
    "lab": "mecka",
    "task": "folding_clothes",
    "zarr_processed_path": "s3://rldb/processed_v3/mecka/freeform/abc.zarr",
    "episode_hash": "an_episode_not_in_the_val_set",
    "segments": [{"label": "fold shirt", "start_seconds": 0, "end_seconds": 9}],
}


def _data(name: str):
    with initialize_config_module(
        config_module="egomimic.hydra_configs", version_base=None
    ):
        cfg = compose(config_name="train_zarr_cartesian", overrides=[f"data={name}"])
    cfg._set_flag("allow_objects", True)
    OmegaConf.resolve(cfg.data)
    return cfg.data


def _name(k: int, variant: str) -> str:
    return f"mecka_fold_freeform_opscale{k}_{variant}"


def _filter(data, split: str) -> DatasetFilter:
    node = data[split].human_bimanual
    return DatasetFilter(
        filter_lambdas=list(node.filters.filter_lambdas),
        episode_hashes=list(node.filters.get("episode_hashes") or []),
    )


@pytest.fixture(scope="module")
def ladder():
    return {(k, v): _data(_name(k, v)) for k in ARMS for v in VARIANTS}


@pytest.mark.parametrize("k", ARMS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_train_accepts_exactly_the_first_k_operators(ladder, k, variant):
    """Nesting, stated as behaviour: arm k takes operators 1..k and no others,
    so each rung only ADDS an operator to the one below it."""
    f = _filter(ladder[(k, variant)], "train_datasets")
    for i, op in enumerate(OPERATORS):
        row = {**BASE_ROW, "operator": op}
        assert f.matches(row) is (i < k), f"operator #{i + 1} in the k={k} arm"
    assert not f.matches({**BASE_ROW, "operator": "some_other_operator"})


@pytest.mark.parametrize("k", ARMS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_train_withholds_the_seen_val_episodes(ladder, k, variant):
    """The seen-op val is the ONLY thing kept out of training, so the exclusion
    must be present in every arm -- including k=1, whose single operator owns
    all three of those episodes."""
    f = _filter(ladder[(k, variant)], "train_datasets")
    for h in SEEN_VAL:
        row = {**BASE_ROW, "operator": OPERATORS[0], "episode_hash": h}
        assert not f.matches(row), f"k={k}: train accepted seen-val episode {h}"


@pytest.mark.parametrize("k", ARMS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_seen_val_pin_matches_what_train_excludes(ladder, k, variant):
    """The val pin and the train exclusion are written separately; if they drift,
    an episode is either trained on AND validated, or dropped entirely."""
    data = ladder[(k, variant)]
    pin = list(data.valid_datasets.human_bimanual.filters.episode_hashes)
    assert sorted(pin) == sorted(SEEN_VAL)
    assert len(pin) == SEEN_VAL_EPISODES
    train = _filter(data, "train_datasets")
    # every pinned episode is refused by train, and a non-pinned one is not
    assert all(
        not train.matches({**BASE_ROW, "operator": OPERATORS[0], "episode_hash": h})
        for h in pin
    )
    assert train.matches({**BASE_ROW, "operator": OPERATORS[0]})


@pytest.mark.parametrize("variant", VARIANTS)
def test_val_sets_are_identical_across_the_ladder(ladder, variant):
    seen = {
        k: list(
            ladder[(k, variant)].valid_datasets.human_bimanual.filters.episode_hashes
        )
        for k in ARMS
    }
    unseen = {k: list(ladder[(k, variant)].held_out_operators) for k in ARMS}
    assert len({tuple(v) for v in seen.values()}) == 1, "seen-op val differs by rung"
    assert len({tuple(v) for v in unseen.values()}) == 1, "unseen-op list differs"


@pytest.mark.parametrize("k", ARMS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_train_operators_are_never_in_the_unseen_split(ladder, k, variant):
    data = ladder[(k, variant)]
    assert len(data.held_out_operators) == UNSEEN_OPERATORS
    assert not (set(OPERATORS) & set(data.held_out_operators))
    # and the unseen filter refuses a train operator outright
    unseen = _filter(data, "unseen_op_valid_datasets")
    assert not unseen.matches({**BASE_ROW, "operator": OPERATORS[0]})
    assert unseen.matches({**BASE_ROW, "operator": data.held_out_operators[0]})


def test_unseen_operators_match_the_opsplit_config():
    """Reusing mecka_fold_freeform_opsplit_6d's held-out list verbatim is what
    keeps this ladder's unseen_op_valid/ numbers comparable with the existing
    freeform opsplit checkpoints. Do not re-cut it."""
    raw = yaml.safe_load(
        (DATA_DIR / "mecka_fold_freeform_opsplit_6d.yaml").read_text(encoding="utf-8")
    )
    assert sorted(_data(_name(4, "6d")).held_out_operators) == sorted(
        raw["held_out_operators"]
    )


# --- language annotations -------------------------------------------------
# 112 of the 462 freeform episodes have segments = NULL -> an empty `annotations`
# track -> _annotation_text_for_frame returns [] for every frame -> _build_prompts
# substitutes `default_prompt` ("" unless overridden). Every split of every rung
# must filter those out, or a third of training runs on an empty instruction.


@pytest.mark.parametrize("k", ARMS)
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("split", SPLITS)
def test_every_split_requires_language_annotations(ladder, k, variant, split):
    data = ladder[(k, variant)]
    f = _filter(data, split)
    operator = (
        data.held_out_operators[0]
        if split == "unseen_op_valid_datasets"
        else OPERATORS[0]
    )
    good = {**BASE_ROW, "operator": operator}
    if split == "valid_datasets":
        good["episode_hash"] = SEEN_VAL[0]  # that split is pinned by hash
    for bad in (None, []):
        assert not f.matches(
            {**good, "segments": bad}
        ), f"{split}: accepted an episode with segments={bad!r}"
    assert f.matches(good), f"{split}: guard rejects an annotated row"


@pytest.mark.parametrize("k", ARMS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_annotation_key_is_wired(ladder, k, variant):
    """Without annotation_key the keymap emits no annotation entry at all, and
    the prompt is `default_prompt` everywhere regardless of the guard."""
    km = ladder[(k, variant)].train_datasets.human_bimanual.resolver.key_map
    assert km.annotation_key == "annotations"


@pytest.mark.parametrize("k", ARMS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_no_rung_asks_for_proprio_history_alone(ladder, k, variant):
    """proprio_history is half a setting: the keymap decides how many steps a
    sample carries and the model's stem how many tokens consume them, and
    HPTModel.stem_process raises when they disagree. Every model in the repo is
    history_len 1, so a rung shipping K > 1 is a rung nothing can train -- which
    is what happened when the ladder outlived the one model that wanted K=4."""
    km = ladder[(k, variant)].train_datasets.human_bimanual.resolver.key_map
    assert km.get("proprio_history", 1) == 1


# --- pipeline and loaders -------------------------------------------------


@pytest.mark.parametrize("k", ARMS)
def test_keypoint_twin_shares_its_parents_split(ladder, k):
    """The HPT twin may change only the keymap/transform; if its filters drift
    from the 6d parent the two model families stop being comparable."""
    for split in SPLITS:
        assert list(
            ladder[(k, "hpt_keypoints")][split].human_bimanual.filters.filter_lambdas
        ) == list(ladder[(k, "6d")][split].human_bimanual.filters.filter_lambdas), split


@pytest.mark.parametrize("k", ARMS)
def test_pipeline_per_variant(ladder, k):
    for variant, keymap, mode, pad in (
        ("6d", "cartesian", "cartesian_wristframe_6d", True),
        ("hpt_keypoints", "keypoints", "keypoints_wristframe_6d", False),
    ):
        res = ladder[(k, variant)].train_datasets.human_bimanual.resolver
        assert res.key_map.keymap_mode == keymap
        assert res.transform_list.mode == mode
        assert res.transform_list.pad_proprio_gripper is pad
        # processed_v3 mecka zarrs have the LEFT wrist frame double-mirrored.
        # Instantiated, not just read off the node: a renamed kwarg would keep
        # the assertion green while the config could no longer be built.
        assert res.transform_list.fix_left_wrist_convention is True
        assert hydra.utils.instantiate(res.transform_list)


@pytest.mark.parametrize("k", ARMS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_filters_are_the_whole_split(ladder, k, variant):
    """`mode: total` everywhere: the filters ARE the split, so no seeded episode
    split may run on top of them and quietly drop a fraction of the data."""
    data = ladder[(k, variant)]
    for split in SPLITS:
        assert data[split].human_bimanual.mode == "total", split
    assert data.valid_prefix == "seen_op_valid"


@pytest.mark.parametrize("k", ARMS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_metric_budget_is_derived_from_the_resolved_split(ladder, k, variant):
    """Every head is `auto`, so K follows the live filters; a hard-coded K would
    push K x episodes past the limit_val_batches window as SQL grows. At the
    documented counts `auto` reproduces floor(5120 / episodes)."""
    from egomimic import trainHydra as th

    data = ladder[(k, variant)]
    assert dict(data.metric_frames_per_episode) == {
        "valid": "auto",
        "unseen_op_valid": "auto",
        "train_viz": "auto",
    }
    cfg = OmegaConf.create(
        {
            "data": {"metric_frames_per_episode": dict(data.metric_frames_per_episode)},
            "trainer": {"limit_val_batches": 80, "devices": 1},
        }
    )
    bs = data.valid_dataloader_params.human_bimanual.batch_size
    for head, n in (("train_viz", TRAIN_EPISODES[k]), ("valid", SEEN_VAL_EPISODES)):
        kk = th._metric_frames_per_episode(cfg, head, n_episodes=n, batch_size=bs)
        assert kk == 80 * bs // n
        assert kk * n <= 80 * bs


@pytest.mark.parametrize("k", ARMS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_video_pins_belong_to_their_own_split(ladder, k, variant):
    """A video pin is taken out of that head's already-resolved split; a pin from
    elsewhere would make the loader re-resolve or render nothing."""
    data = ladder[(k, variant)]
    vids = data.video_episodes
    assert set(vids) == {"valid", "unseen_op_valid", "train_viz"}
    assert set(vids.valid) <= set(SEEN_VAL)
    assert len(vids.unseen_op_valid) == 1 and len(vids.train_viz) == 1
    # the train_viz pin is an operator #1 episode, so every arm -- including
    # k=1 -- must actually accept it, which is why the children can inherit it
    train = _filter(data, "train_datasets")
    assert train.matches(
        {**BASE_ROW, "operator": OPERATORS[0], "episode_hash": vids.train_viz[0]}
    ), f"k={k}: the inherited train_viz video pin is not in this arm's split"
