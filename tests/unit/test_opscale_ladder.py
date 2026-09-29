"""The freeform fold-clothes OPERATOR-SCALING ladder holds its shape.

mecka_fold_freeform_opscale{1,2,4} vary ONE thing: how many operators the
training data comes from. Each arm trains on ALL the annotated data its
operators have, minus the shared held-out val episodes. Everything that would
confound the comparison -- the seen-op val, the unseen-op val, the pipeline --
must be identical across the three rungs, and the operator sets must be nested.

The train split is an OPERATOR FILTER, not an episode pin, so these tests drive
the real DatasetFilter with synthetic rows rather than counting hashes.
"""

import pytest
from test_data_configs_compose import compose_data

from egomimic.rldb.filters import DatasetFilter

ARMS = (1, 2, 4)
# 6d: the rung's data config; kp: the keypoint recipe over the same rung.
VARIANTS = {"6d": "train_zarr_cartesian", "kp": "train_zarr_mecka_kp_wrist_hpt"}

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


def _filter(data, split: str) -> DatasetFilter:
    node = data[split].human_bimanual
    return DatasetFilter(
        filter_lambdas=list(node.filters.filter_lambdas),
        episode_hashes=list(node.filters.get("episode_hashes") or []),
    )


@pytest.fixture(scope="module")
def ladder():
    return {
        (k, v): compose_data(f"mecka_fold_freeform_opscale{k}_6d", config_name=top).data
        for k in ARMS
        for v, top in VARIANTS.items()
    }


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
def test_seen_val_pin_matches_what_train_excludes(ladder, k, variant):
    """The val pin and the train exclusion are written separately; if they drift,
    an episode is either trained on AND validated, or dropped entirely."""
    data = ladder[(k, variant)]
    pin = list(data.valid_datasets.human_bimanual.filters.episode_hashes)
    assert sorted(pin) == sorted(SEEN_VAL)
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
    assert not (set(OPERATORS) & set(data.held_out_operators))
    # and the unseen filter refuses a train operator outright
    unseen = _filter(data, "unseen_op_valid_datasets")
    assert not unseen.matches({**BASE_ROW, "operator": OPERATORS[0]})
    assert unseen.matches({**BASE_ROW, "operator": data.held_out_operators[0]})


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


# --- pipeline and loaders -------------------------------------------------


@pytest.mark.parametrize("k", ARMS)
def test_keypoint_recipe_shares_the_rungs_split(ladder, k):
    """The keypoint recipe may change only the keymap/transform; if its filters
    drift from the 6d rung the two model families stop being comparable."""
    for split in SPLITS:
        assert list(
            ladder[(k, "kp")][split].human_bimanual.filters.filter_lambdas
        ) == list(ladder[(k, "6d")][split].human_bimanual.filters.filter_lambdas), split


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
