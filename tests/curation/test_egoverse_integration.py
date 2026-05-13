"""
Integration test with real EgoVerse data.

This test exercises the full curation pipeline on a small slice of actual
EgoVerse data downloaded from S3.  It is intentionally excluded from the
standard CI test suite because it requires:

  1. Network access to S3 (or pre-downloaded data at DATA_DIR below).
  2. A valid EgoVerse SQL database connection (for S3EpisodeResolver).
  3. A GPU is helpful but not required.

How to run manually:
    # 1. Make sure the test data is available locally.  Either download via
    #    sync_s3.py or point DATA_DIR at an existing local zarr cache:
    DATA_DIR=/coc/flash7/scratch/egoverseDebugDatasets/aria

    # 2. Run with pytest, opting in to the slow + requires_data marks:
    pytest tests/curation/test_egoverse_integration.py \\
        -v -m "slow and requires_data" \\
        --tb=short

    # 3. Alternatively, run as a plain script for interactive debugging:
    cd /path/to/Egoversedev
    source emimic/bin/activate
    python tests/curation/test_egoverse_integration.py

GPU requirement (for real-data runs):
    Request a GPU via salloc before running the integration test if you want
    to use image embeddings:
        salloc --gres=gpu:1 --mem=32G --time=1:00:00

Marks used:
    @pytest.mark.slow         — skipped in normal CI (add -m slow to run)
    @pytest.mark.requires_data — skipped if DATA_DIR does not exist
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

# ---------------------------------------------------------------------------
# Configuration — override via env vars for CI / different environments
# ---------------------------------------------------------------------------

# Root directory of local EgoVerse zarr cache
DATA_DIR = Path(
    os.environ.get(
        "EGOVERSE_TEST_DATA_DIR",
        "/coc/flash7/scratch/egoverseDebugDatasets/aria",
    )
)

# Maximum episodes to load (keeps the test fast)
MAX_EPISODES = int(os.environ.get("EGOVERSE_TEST_MAX_EPISODES", "10"))


# ---------------------------------------------------------------------------
# Pytest marks
# ---------------------------------------------------------------------------

pytestmark = [
    pytest.mark.slow,
    pytest.mark.requires_data,
]


# ---------------------------------------------------------------------------
# Skip guard — fail gracefully when data is absent
# ---------------------------------------------------------------------------


def _data_available() -> bool:
    return DATA_DIR.is_dir() and any(DATA_DIR.iterdir())


skip_if_no_data = pytest.mark.skipif(
    not _data_available(),
    reason=(
        f"Test data not available at {DATA_DIR}. "
        "Set EGOVERSE_TEST_DATA_DIR env var or download data first. "
        "See test docstring for instructions."
    ),
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_episodes():
    """
    Load up to MAX_EPISODES real EgoVerse episodes from DATA_DIR.

    Uses load_episodes_from_dir() (no SQL / S3 required) so the test runs
    on any machine with the zarr data present locally.
    """
    from egomimic.curation.utils import load_episodes_from_dir

    episodes = load_episodes_from_dir(
        folder_path=DATA_DIR,
        max_episodes=MAX_EPISODES,
    )
    if not episodes:
        pytest.skip(f"No episodes loaded from {DATA_DIR}")
    return episodes


@pytest.fixture(scope="module")
def curation_result(real_episodes, tmp_path_factory):
    """Run the full curation pipeline on real episodes."""
    from egomimic.curation.curator import DemInfCurator

    tmp = tmp_path_factory.mktemp("integration_output")
    curator = DemInfCurator(
        filter_ratio=0.3,
        state_mode="proprioceptive",
        latent_dim=16,
        k_range=(3, 5),
        pause_epsilon=1e-3,
        action_clip=True,
        min_trajectory_length=10,
        cross_embodiment_mode="independent",
    )
    result = curator.curate(real_episodes)
    return curator, result, tmp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@skip_if_no_data
class TestRealDataLoading:
    """Verify that real EgoVerse episodes load without errors."""

    def test_loads_nonzero_episodes(self, real_episodes):
        assert len(real_episodes) > 0, "Expected at least one episode"

    def test_episode_fields_populated(self, real_episodes):
        for ep in real_episodes:
            assert ep.episode_hash, "episode_hash must be non-empty"
            assert (
                ep.observations.ndim == 2
            ), f"observations must be 2D, got {ep.observations.shape}"
            assert ep.actions.ndim == 2, f"actions must be 2D, got {ep.actions.shape}"
            assert len(ep.observations) == len(ep.actions), "obs/action length mismatch"
            assert ep.embodiment, "embodiment must be non-empty"

    def test_observations_finite(self, real_episodes):
        for ep in real_episodes:
            assert np.isfinite(
                ep.observations
            ).all(), f"Non-finite values in observations for {ep.episode_hash}"

    def test_actions_finite(self, real_episodes):
        for ep in real_episodes:
            assert np.isfinite(
                ep.actions
            ).all(), f"Non-finite values in actions for {ep.episode_hash}"


@skip_if_no_data
class TestFullCurationPipeline:
    """End-to-end curation on real episodes."""

    def test_all_episodes_scored(self, curation_result, real_episodes):
        """Every episode (not filtered by preprocessing) gets a score."""
        _, result, _ = curation_result
        # Scored = total_input - pre_filter_removed
        expected_scored = result.stats["scored"]
        assert (
            len(result.scores) == expected_scored
        ), f"Expected {expected_scored} scores, got {len(result.scores)}"

    def test_scores_finite_and_varied(self, curation_result):
        """Scores must be finite and not all identical (would indicate a bug)."""
        _, result, _ = curation_result
        scored_values = list(result.scores.values())
        if len(scored_values) < 2:
            pytest.skip("Need ≥ 2 scored episodes to check variance")

        assert all(np.isfinite(v) for v in scored_values), "Non-finite MI scores"
        assert (
            np.std(scored_values) > 0.0
        ), "All MI scores are identical — check pipeline"

    def test_kept_hashes_subset_of_input(self, curation_result, real_episodes):
        all_hashes = {ep.episode_hash for ep in real_episodes}
        _, result, _ = curation_result
        assert set(result.kept_hashes).issubset(all_hashes)

    def test_removed_and_kept_partition(self, curation_result, real_episodes):
        """kept + all_removed = total input (no episodes silently lost)."""
        _, result, _ = curation_result
        n_in = len(real_episodes)
        n_accounted = len(result.kept_hashes) + len(result.all_removed_hashes)
        assert n_accounted == n_in, (
            f"kept ({len(result.kept_hashes)}) + removed ({len(result.all_removed_hashes)}) "
            f"≠ input ({n_in})"
        )

    def test_stats_keys_present(self, curation_result):
        _, result, _ = curation_result
        required = {
            "total_input",
            "pre_filter_removed",
            "scored",
            "kept",
            "low_mi_removed",
            "filter_ratio",
            "threshold_score",
            "mi_mean",
            "mi_std",
            "mi_median",
        }
        assert required.issubset(
            result.stats.keys()
        ), f"Missing keys: {required - result.stats.keys()}"


@skip_if_no_data
class TestExportedFilter:
    """Verify that the exported YAML filter is loadable and correct."""

    def test_yaml_filter_exported(self, curation_result, tmp_path):
        curator, result, _ = curation_result
        out = tmp_path / "test_filter.yaml"
        curator.export_sql_filter(out, format="yaml")
        assert out.exists(), "YAML filter not written"

    def test_yaml_filter_loads_with_omegaconf(self, curation_result, tmp_path):
        curator, result, _ = curation_result
        out = tmp_path / "test_filter.yaml"
        curator.export_sql_filter(out, format="yaml")

        loaded = OmegaConf.load(str(out))
        assert loaded is not None

    def test_yaml_filter_has_kept_hashes(self, curation_result, tmp_path):
        curator, result, _ = curation_result
        out = tmp_path / "test_filter.yaml"
        curator.export_sql_filter(out, format="yaml")

        loaded = OmegaConf.to_container(OmegaConf.load(str(out)), resolve=True)
        assert "kept_hashes" in loaded, "YAML filter missing 'kept_hashes'"
        assert set(loaded["kept_hashes"]) == set(
            result.kept_hashes
        ), "kept_hashes in YAML does not match result"

    def test_yaml_filter_has_metadata(self, curation_result, tmp_path):
        curator, result, _ = curation_result
        out = tmp_path / "test_filter.yaml"
        curator.export_sql_filter(out, format="yaml")

        loaded = OmegaConf.to_container(OmegaConf.load(str(out)), resolve=True)
        assert "metadata" in loaded, "YAML filter missing 'metadata'"
        meta = loaded["metadata"]
        assert "timestamp" in meta
        assert "filter_ratio_applied" in meta

    def test_json_filter_exported(self, curation_result, tmp_path):
        curator, result, _ = curation_result
        out = tmp_path / "test_filter.json"
        curator.export_sql_filter(out, format="json")
        payload = json.loads(out.read_text())
        assert "kept_hashes" in payload
        assert set(payload["kept_hashes"]) == set(result.kept_hashes)

    def test_sql_filter_exported(self, curation_result, tmp_path):
        curator, result, _ = curation_result
        out = tmp_path / "test_filter.sql"
        curator.export_sql_filter(out, format="sql")
        sql = out.read_text()
        assert "WHERE episode_hash IN" in sql
        # All kept hashes appear in the SQL
        for h in result.kept_hashes[:3]:  # spot-check first few
            assert h in sql


# ---------------------------------------------------------------------------
# Standalone runner (for interactive debugging)
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not _data_available():
        print(f"[ERROR] No data at {DATA_DIR}")
        print("Set EGOVERSE_TEST_DATA_DIR or download data first.")
        raise SystemExit(1)

    from egomimic.curation.curator import DemInfCurator
    from egomimic.curation.utils import load_episodes_from_dir

    print(f"Loading up to {MAX_EPISODES} episodes from {DATA_DIR} …")
    eps = load_episodes_from_dir(DATA_DIR, max_episodes=MAX_EPISODES)
    print(f"Loaded {len(eps)} episodes")

    curator = DemInfCurator(
        filter_ratio=0.3,
        latent_dim=16,
        k_range=(3, 5),
        cross_embodiment_mode="independent",
    )
    result = curator.curate(eps)
    print(f"Kept: {len(result.kept_hashes)} / {len(eps)}")
    print(f"MI mean: {result.stats.get('mi_mean', 'N/A'):.4f}")

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        out = curator.export_sql_filter(f"{td}/filter.yaml")
        loaded = OmegaConf.load(str(out))
        print(f"Filter YAML loaded OK: {len(loaded.get('kept_hashes', []))} hashes")
    print("Integration test passed.")
