"""Fixed-budget ladder: 4/8/12/16 operators, one total above 30 h."""

import json
from collections import defaultdict
from pathlib import Path

from egomimic.scripts.data_download.fixed_budget_opscale import (
    OP4_OPERATORS,
    TIERS,
    plan,
    to_cs,
)

MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "egomimic/hydra_configs/experiments/fixed_budget_opscale/manifest.json"
)
# Longest folding_clothes episode in the master JSON is under two minutes.
MAX_EPISODE_CS = 13000


def _episode(episode_id: str, user_id: str, hours: float) -> dict:
    return {
        "episode_id": episode_id,
        "user_id": user_id,
        "duration": hours * 3600,
        "duration_cs": to_cs(hours * 3600),
    }


def _synthetic() -> list[dict]:
    rows = []
    # op4 (user00-03) each have 8 h, 7.9 h after the 1-episode holdout, and that
    # binds the budget at 31.6 h. Later operators can still fill their caps.
    for index in range(20):
        if index < 4:
            n, hours = 80, 0.1
        elif index < 8:
            n, hours = 45, 0.1
        elif index < 12:
            n, hours = 30, 0.1
        elif index < 14:
            n, hours = 25, 0.1
        elif index < 16:
            # Below an equal 16-way share, so the last rung keeps all of their data.
            n, hours = 12 - (index - 14) * 4, 0.1
        else:
            n, hours = 2, 0.1
        if 4 <= index < 16:
            n += 30  # reserve the requested per-operator validation holdout
        for j in range(n):
            rows.append(_episode(f"u{index:02d}e{j:04d}", f"user{index:02d}", hours))
    return rows


def _per_operator(episodes: list[dict]) -> dict[str, int]:
    totals = defaultdict(int)
    for episode in episodes:
        totals[episode["user_id"]] += episode["duration_cs"]
    return totals


def _assert_ladder(spec: dict, max_episode_cs: int) -> None:
    assert spec["budget_cs"] == spec["binding_tier"] * spec["marginal_cs"]
    assert spec["budget_hours"] > 30
    assert list(spec["tiers"]["op4"]["operators"]) == list(spec["op4_operators"])
    seen_by_op = defaultdict(list)
    for episode in spec["seen_val"]["episodes"]:
        seen_by_op[episode["user_id"]].append(episode["episode_id"])
    assert set(seen_by_op) == set(spec["op4_operators"])
    assert all(len(ids) == 1 for ids in seen_by_op.values())
    seen_ids = {episode_id for ids in seen_by_op.values() for episode_id in ids}
    unseen_ops = set(spec["unseen_val"]["operators"])
    assert unseen_ops.isdisjoint(spec["op4_operators"])
    assert unseen_ops.isdisjoint(
        op for tier in spec["tiers"].values() for op in tier["operators"]
    )

    sets = {}
    opsets = {}
    for k in TIERS:
        tier = spec["tiers"][f"op{k}"]
        assert len(tier["operators"]) == k
        per_op = _per_operator(tier["episodes"])
        assert set(per_op) == set(tier["operators"])
        short = set(tier["short_operators"])
        if k == TIERS[-1]:
            assert short
            for op, total in per_op.items():
                if op in short:
                    assert total == tier["short_cs"][op]
                else:
                    assert tier["cap_cs"] - max_episode_cs <= total <= tier["cap_cs"]
        else:
            assert tier["target_cs"] == tier["cap_cs"] * k
            assert abs(tier["target_cs"] - spec["budget_cs"]) < 100
            assert not short
            for total in per_op.values():
                assert tier["cap_cs"] - max_episode_cs <= total <= tier["cap_cs"]
        assert not seen_ids & {e["episode_id"] for e in tier["episodes"]}
        sets[k] = {
            op: {e["episode_id"] for e in tier["episodes"] if e["user_id"] == op}
            for op in tier["operators"]
        }
        opsets[k] = list(tier["operators"])

    assert opsets[4] == opsets[8][:4] == opsets[12][:4] == opsets[16][:4]
    assert opsets[8] == opsets[12][:8] == opsets[16][:8]
    assert opsets[12] == opsets[16][:12]
    for op in opsets[4]:
        assert sets[16][op] <= sets[12][op] <= sets[8][op] <= sets[4][op]
    for op in opsets[8][4:]:
        assert sets[16][op] <= sets[12][op] <= sets[8][op]
    for op in opsets[12][8:]:
        assert sets[16][op] <= sets[12][op]


def test_synthetic_budget_is_fixed_and_operators_nest():
    core = tuple(f"user{i:02d}" for i in range(4))
    spec = plan(_synthetic(), op4_operators=core)
    assert spec["op4_operators"] == list(core)
    assert spec["binding_tier"] == 4
    assert spec["marginal_operator"] == "user03"
    assert spec["budget_hours"] == to_cs(4 * 7.9 * 3600) / 100 / 3600
    _assert_ladder(spec, max_episode_cs=to_cs(0.1 * 3600))


def test_open_loop_pins_are_one_episode_per_validation_set():
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(
        MANIFEST.parents[2] / "data" / "mecka_fold_fixed_budget_op4_6d.yaml"
    )
    spec = json.loads(MANIFEST.read_text())
    seen = [str(h) for h in cfg.open_loop_episodes.seen]
    assert seen == [spec["seen_val"]["video_episode"]]
    assert cfg.open_loop_stride == 30
    assert [str(h) for h in cfg.open_loop_episodes.unseen] == [
        spec["unseen_val"]["video_episode"],
        spec["flagship_unseen_val"]["video_episode"],
    ]


def test_manifest_op4_is_the_pinned_operators():
    spec = json.loads(MANIFEST.read_text())
    assert spec["task_id"] == "folding_clothes"
    assert spec["op4_operators"] == list(OP4_OPERATORS)
    assert spec["binding_tier"] == 8
    assert spec["marginal_operator"] == spec["tiers"]["op8"]["operators"][-1]
    assert 32 < spec["budget_hours"] < 33
    _assert_ladder(spec, MAX_EPISODE_CS)
    marginal_eps = [
        e
        for e in spec["tiers"]["op8"]["episodes"]
        if e["user_id"] == spec["marginal_operator"]
    ]
    assert sum(e["duration_cs"] for e in marginal_eps) == spec["marginal_cs"]
    achieved = [spec["tiers"][f"op{k}"]["achieved_cs"] for k in TIERS]
    assert max(achieved) - min(achieved) < to_cs(0.6 * 3600)
