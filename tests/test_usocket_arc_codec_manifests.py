import csv
import json

import build_topk_codec_manifests as subject


def test_builds_unique_topk_and_deterministic_episode_superset(tmp_path):
    ranking = tmp_path / "ranking.csv"
    with ranking.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["rank", "index", "candidate", "D", "M", "R_deg"]
        )
        writer.writeheader()
        writer.writerows([
            {"rank": 1, "index": 7, "candidate": "D80_M56_R26deg", "D": 80, "M": 56, "R_deg": 26},
            {"rank": 2, "index": 8, "candidate": "D80_M56_R18deg", "D": 80, "M": 56, "R_deg": 18},
        ])
    split = tmp_path / "split.json"
    split.write_text(json.dumps({
        "domains": {
            "u_socket": {
                "folder_path": "/dataset",
                "valid_ids": ["valid-a", "valid-b"],
                "train_ids": ["train-a", "train-b", "train-c"],
            }
        }
    }))

    candidates = subject.build_candidate_manifest(ranking, 2)
    first = subject.build_episode_manifest(split, 4, "fixed")
    second = subject.build_episode_manifest(split, 4, "fixed")

    assert [row["candidate"] for row in candidates["candidates"]] == [
        "D80_M56_R26deg", "D80_M56_R18deg"
    ]
    ids = first["domains"]["u_socket"]["valid_ids"]
    assert ids[:2] == ["valid-a", "valid-b"]
    assert len(ids) == len(set(ids)) == 4
    assert first == second
    assert first["selection"]["supplemental_training_count"] == 2
