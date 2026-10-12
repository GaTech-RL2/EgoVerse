"""The fixed-budget operator ladder (make_configs_fixed_budget) on synthetic episodes."""

import random

import pytest

from egomimic.scripts.mecka_process.mono.categorize import non_folding_actions
from egomimic.scripts.mecka_process.mono.make_configs_fixed_budget import (
    NO_TSHIRT,
    TSHIRT,
    Ep,
    collapse_same_home,
    fill,
    plan,
    plan_rows,
)

RUNGS = (1, 2, 4, 8, 14)


def _ops(spec, seed=0):
    """spec: op -> {cat: n_episodes}; lengths 40-140 s at 30 fps."""
    rng = random.Random(seed)
    eps = []
    for op, cats in spec.items():
        for cat, n in cats.items():
            for i in range(n):
                eps.append(
                    Ep(
                        id=f"{op}-{cat}-{i:03d}",
                        op=op,
                        cat=cat,
                        frames=rng.randint(1200, 4200),
                        task_id="folding_socks" if i == 0 else "folding_clothes",
                        clean=i % 4 != 1,
                    )
                )
    return eps


def _pool():
    spec = {f"d{i}": {TSHIRT: 120, NO_TSHIRT: 120} for i in range(3)}
    spec |= {f"t{i}": {TSHIRT: 120} for i in range(5)}
    spec |= {f"n{i}": {NO_TSHIRT: 120} for i in range(4)}
    spec |= {"n_small0": {NO_TSHIRT: 40}, "n_small1": {NO_TSHIRT: 30}}
    return _ops(spec)


@pytest.fixture(scope="module")
def ladder():
    eps = _pool()
    return eps, plan(eps, RUNGS, share=0.5, seed=0, n_held=5)


def test_fill_stays_under_target_and_leaves_no_fitting_episode():
    eps = _ops({"a": {TSHIRT: 60}}, seed=3)
    for target in (5_000, 37_123, 100_000):
        got = fill(eps, target)
        left = target - sum(e.frames for e in got)
        assert left >= 0
        unused = [e for e in eps if e not in got]
        assert all(e.frames > left for e in unused)


def test_fill_keeps_required_episodes():
    eps = _ops({"a": {TSHIRT: 30}}, seed=1)
    short = min(eps, key=lambda e: e.frames)
    got = fill(eps, 20_000, {short.id})
    assert short in got
    with pytest.raises(ValueError):
        fill(eps, short.frames - 1, {short.id})


def test_every_rung_has_the_budget(ladder):
    eps, p = ladder
    longest = max(e.frames for e in eps)
    for row in plan_rows(p):
        assert row["frames"] <= p["budget_frames"]
        # each (operator, category) cell leaves less than one episode unpacked
        assert p["budget_frames"] - row["frames"] < 2 * row["rung"] * longest
        assert row["ops"] == row["rung"]


def test_operators_get_equal_shares(ladder):
    eps, p = ladder
    longest = max(e.frames for e in eps)
    for k, per_op in p["rungs"].items():
        for picked in per_op.values():
            assert abs(sum(e.frames for e in picked) - p["budget_frames"] / k) < (
                2 * longest
            )


def test_rungs_are_nested(ladder):
    _, p = ladder
    ks = sorted(p["rungs"])
    assert p["order"][: ks[0]] == list(p["rungs"][ks[0]])
    for small, big in zip(ks, ks[1:]):
        assert set(p["rungs"][small]) < set(p["rungs"][big])
        for op, picked in p["rungs"][small].items():
            assert {e.id for e in p["rungs"][big][op]} <= {e.id for e in picked}


def test_tshirt_share_is_constant(ladder):
    _, p = ladder
    for row in plan_rows(p):
        assert row["tshirt_share"] == pytest.approx(0.5, abs=0.02)


def test_holdouts_are_out_of_every_rung_and_seeded(ladder):
    eps, p = ladder
    held = {e.id for v in p["held_out"].values() for e in v}
    assert len(held) == 5 * sum(1 for _ in {(e.op, e.cat) for e in eps})
    for per_op in p["rungs"].values():
        assert not held & {e.id for v in per_op.values() for e in v}
    again = plan(eps, RUNGS, share=0.5, seed=0, n_held=5)
    assert again["held_out"] == p["held_out"]
    assert again["order"] == p["order"]
    assert all(e.clean and e.task_id == "folding_clothes" for e in held_eps(p))


def held_eps(p):
    return [e for v in p["held_out"].values() for e in v]


def test_rare_task_ids_survive_into_every_rung(ladder):
    _, p = ladder
    top = max(p["rungs"])
    for op, picked in p["rungs"][top].items():
        assert any(e.task_id == "folding_socks" for e in picked), op


def test_first_operator_folds_both_categories(ladder):
    _, p = ladder
    assert {e.cat for e in p["rungs"][1][p["order"][0]]} == {TSHIRT, NO_TSHIRT}


def test_no_mix_reachable_raises():
    eps = _ops({f"t{i}": {TSHIRT: 50} for i in range(4)})
    with pytest.raises(ValueError):
        plan(eps, (1, 2, 4), share=0.5, seed=0, n_held=2)


def test_same_home_group_keeps_one_operator():
    eps = _ops({"a": {TSHIRT: 10}, "b": {TSHIRT: 10, NO_TSHIRT: 3}, "c": {TSHIRT: 5}})
    kept, dropped = collapse_same_home(eps, [["a", "b"], ["c", "absent"]])
    assert dropped == ["a"]
    assert {e.op for e in kept} == {"b", "c"}


def test_same_home_collapse_feeds_the_ladder():
    eps = _pool() + _ops({"d0_twin": {TSHIRT: 120, NO_TSHIRT: 120}}, seed=9)
    kept, dropped = collapse_same_home(eps, [["d0", "d0_twin"]])
    assert len(dropped) == 1
    p = plan(kept, RUNGS, share=0.5, seed=0, n_held=5)
    assert len({"d0", "d0_twin"} & set(p["order"])) == 1


@pytest.mark.parametrize(
    "caption, hits",
    [
        ("Synthetic: folds shirts on an ironing board.", []),
        ("Synthetic: folds t-shirts and stacks them.", []),
        ("Synthetic: folds clothes and packs them into a suitcase.", ["packing"]),
        ("Synthetic: organizes the clothes in a drawer.", ["organizing"]),
        ("Synthetic: irons the shirt, then folds it.", ["ironing"]),
        ("Synthetic: puts a jacket on hangers.", ["hanging"]),
        ("Synthetic: sorts through laundry and folds it.", ["sorting"]),
        ("Synthetic: washes socks.", ["washing"]),
    ],
)
def test_strict_fold_actions(caption, hits):
    assert non_folding_actions(caption) == hits
