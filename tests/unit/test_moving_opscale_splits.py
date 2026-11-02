"""Pure CPU checks for the controlled moving-budget split design."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def builder():
    path = (
        Path(__file__).resolve().parents[2]
        / "experiments/fold_clothes_moving_opscale/build_splits.py"
    )
    spec = importlib.util.spec_from_file_location("moving_split_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def population(builder):
    table, episodes = {}, {}
    for i in range(160):
        uid = f"operator_{i:03d}"
        hours = (
            30
            if i < 4
            else 15
            if i < 8
            else 8
            if i < 16
            else 4
            if i < 32
            else 2
            if i < 48
            else 1.15
        )
        table[uid] = {
            "user_id": uid,
            "location_group": "unknown" if i % 3 == 0 else f"place_{i:03d}",
        }
        for j in range(80):
            eid = f"{uid}_episode_{j:03d}"
            episodes[eid] = {
                "episode_id": eid,
                "user_id": uid,
                "task_id": builder.TASK,
                "duration": hours * 3600 / 80,
            }
    return episodes, table


@pytest.fixture(scope="module")
def generated(builder, population):
    episodes, table = population
    return builder.create_design(episodes, table, {"complete": True}, 42026)


def test_all_36_configs_keep_nested_operators_and_budgets(builder, generated):
    design, splits = generated
    checks = builder.validate(design, splits, set(), set())
    assert checks["configurations"] == 36
    assert checks["operator_nesting"] and checks["cross_budget_identity"]
    for scale in builder.SCALES:
        previous = set()
        for budget in builder.BUDGETS:
            s = splits[f"fold_moving_b{budget}_op{scale}"]
            assert s["operators"] == design["operator_order"][:scale]
            assert previous <= set(s["train"])
            previous = set(s["train"])


def test_validation_is_disjoint_fixed_and_complete(generated):
    design, splits = generated
    baseline = splits["fold_moving_b24_op4"]
    for s in splits.values():
        assert s["seen_matched"] == baseline["seen_matched"]
        assert s["unseen"] == baseline["unseen"]
        assert len(s["seen_matched"]) == 40
        assert len(s["unseen"]) == 400
        assert len(s["seen_perlevel"]) == 3 * len(s["operators"])
        parts = [
            set(s[k]) for k in ("train", "seen_matched", "seen_perlevel", "unseen")
        ]
        assert len(set.union(*parts)) == sum(map(len, parts))
        for label, count in (
            ("seen_matched", 4),
            ("seen_perlevel", min(5, len(s["operators"]))),
            ("unseen", 5),
        ):
            ids = s[f"video_{label}"]
            assert len(ids) == count
            assert len({design["episodes"][e]["user_id"] for e in ids}) == count


def test_unknown_locations_remain_independent_but_known_pair_does_not(
    builder, population
):
    episodes, table = population
    table = copy.deepcopy(table)
    assert builder.identity("operator_000", table) != builder.identity(
        "operator_003", table
    )
    table["operator_001"]["location_group"] = "shared_place"
    table["operator_002"]["location_group"] = "shared_place"
    design, splits = builder.create_design(episodes, table, {"complete": True}, 42026)
    selected = set(design["operator_order"] + design["unseen_operators"])
    assert len(selected & {"operator_001", "operator_002"}) == 1
    assert len({design["identity_by_operator"][u] for u in selected}) == len(selected)


def test_capped_allocation_preserves_full_budget(generated):
    design, _ = generated
    allocation = design["budgets"]["fold_moving_b96_op64"]
    assert not allocation["balanced_possible"]
    assert abs(sum(allocation["allocated_target_hours"].values()) - 96) < 1e-8
    assert abs(allocation["actual_hours"] - 96) <= allocation["max_episode_hours"]
    assert any(v < 1.5 for v in allocation["allocated_target_hours"].values())
    assert any(v > 1.5 for v in allocation["allocated_target_hours"].values())


@pytest.mark.parametrize(
    "defect", ["training_leak", "operator_order", "duplicate_pin", "banned_episode"]
)
def test_integrity_checker_rejects_broken_manifests(builder, generated, defect):
    design, original = generated
    splits = copy.deepcopy(original)
    split = splits["fold_moving_b24_op4"]
    banned = set()
    if defect == "training_leak":
        split["train"].append(split["seen_matched"][0])
    elif defect == "operator_order":
        split["operators"].reverse()
    elif defect == "duplicate_pin":
        split["video_seen_matched"][0] = split["video_seen_matched"][1]
    else:
        banned.add(split["train"][0])
    with pytest.raises(ValueError):
        builder.validate(design, splits, banned, set())


def test_missing_metadata_fails_before_split_creation(builder, tmp_path, monkeypatch):
    main = tmp_path / "main.json"
    main.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "episode_id": "one",
                        "user_id": "operator",
                        "duration": 1800,
                        "task_id": builder.TASK,
                    }
                ]
            }
        )
    )
    table = tmp_path / "operators.csv"
    table.write_text(
        "task_id,user_id,hours,episodes,location_group\nfolding_clothes,operator,1.0,2,unknown\n"
    )
    banned_eps, banned_users = tmp_path / "banned_eps", tmp_path / "banned_users"
    banned_eps.write_text("excluded_episode\n")
    banned_users.write_text("excluded_user\n")
    private = tmp_path / "private"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_splits",
            "--main-manifest",
            str(main),
            "--operator-csv",
            str(table),
            "--banned-episodes",
            str(banned_eps),
            "--banned-users",
            str(banned_users),
            "--private-dir",
            str(private),
        ],
    )
    with pytest.raises(ValueError, match="Incomplete authoritative CSV coverage"):
        builder.main()
    assert not (private / "splits").exists()
    assert not json.loads((private / "moving_opscale_coverage.json").read_text())[
        "complete"
    ]


def test_duplicate_source_requires_matching_identity_and_duration(builder, tmp_path):
    # Unit-level duplicate identity disagreement is exercised via complete inputs.
    main, replacement = tmp_path / "main.json", tmp_path / "replacement.json"
    episode = {
        "episode_id": "same",
        "user_id": "operator",
        "duration": 1800.004,
        "task_id": builder.TASK,
    }
    main.write_text(json.dumps({"episodes": [episode]}))
    replacement.write_text(json.dumps({"episodes": [dict(episode, duration=1800.0)]}))
    table = tmp_path / "operators.csv"
    table.write_text(
        "task_id,user_id,hours,episodes,location_group\nfolding_clothes,operator,0.5,1,unknown\n"
    )
    ban, users = tmp_path / "ban", tmp_path / "users"
    ban.write_text("excluded_episode\n")
    users.write_text("excluded_user\n")
    eps, _, _, _, report = builder.load_sources(
        [main], [replacement], table, ban, users
    )
    assert report["complete"] and len(eps) == 1
    assert eps["same"]["duration"] == 1800.004
    replacement.write_text(
        json.dumps({"episodes": [dict(episode, user_id="different")]})
    )
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        builder.load_sources([main], [replacement], table, ban, users)
