"""Finite scalars cannot bypass genuine tracking coverage and video timing."""

import copy

import pytest

from experiments.fold_clothes_moving_opscale.run_preflight import (
    EVALUATION_POLICY,
    verify_tracking_status,
)


def evidence():
    cohorts = ("seen_matched", "seen_perlevel", "unseen")
    counts = {"visited": 2, "scored": 1, "skipped": 1}
    return {
        "evaluation_policy": EVALUATION_POLICY,
        "tracking_coverage": {
            name: {**counts, "episodes": {name: dict(counts)}} for name in cohorts
        },
        "videos": [
            {
                "episode": "seen_matched",
                "evaluation_policy": EVALUATION_POLICY,
                "tracking": dict(counts),
                "source_frames": 60,
                "unscored_frames": 30,
                "unscored_spans": [{"start_frame": 30, "frame_count": 30}],
                "timing": "original_source_frames_with_labeled_unscored_spans",
            }
        ],
    }


def test_valid_and_skipped_counts_retain_exact_episode_coverage_and_video_time():
    status = evidence()
    counts = dict.fromkeys(status["tracking_coverage"], 1)
    assert (
        verify_tracking_status(
            status,
            counts,
            split={key: [key] for key in counts},
            canonical_index={"seen_matched": {"total_frames": 60}},
        )
        == status["tracking_coverage"]
    )


@pytest.mark.parametrize(
    "fault",
    [
        "old_policy",
        "unscored_episode",
        "bad_totals",
        "wrong_episode",
        "lost_video_time",
        "overlapping_spans",
        "fake_zero_skips",
    ],
)
def test_incomplete_or_misrepresented_scoring_never_passes(fault):
    status = copy.deepcopy(evidence())
    counts = dict.fromkeys(status["tracking_coverage"], 1)
    if fault == "old_policy":
        status.pop("evaluation_policy")
    elif fault == "unscored_episode":
        status["tracking_coverage"]["unseen"]["episodes"]["unseen"]["scored"] = 0
    elif fault == "bad_totals":
        status["tracking_coverage"]["seen_matched"]["visited"] += 1
    elif fault == "wrong_episode":
        episodes = status["tracking_coverage"]["unseen"]["episodes"]
        episodes["other"] = episodes.pop("unseen")
    elif fault == "lost_video_time":
        status["videos"][0]["source_frames"] = 61
    elif fault == "overlapping_spans":
        status["videos"][0]["unscored_spans"].append(
            {"start_frame": 40, "frame_count": 10}
        )
    else:
        status["videos"][0]["unscored_frames"] = 0
    with pytest.raises(RuntimeError):
        verify_tracking_status(
            status,
            counts,
            split={key: [key] for key in counts},
            canonical_index={"seen_matched": {"total_frames": 60}},
        )
