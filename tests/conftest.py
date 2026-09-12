"""Root pytest config for EgoVerse.

``tests/unit`` is hermetic (CPU, no network, no cluster paths) and always runs.
``tests/integration`` needs cluster data, credentials, a GPU, or optional extras;
every test under it is marked ``integration`` and skipped unless ``--integration``
is passed.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).parent)
)  # makes `fixtures.*` importable from any test dir

INTEGRATION_DIR = Path(__file__).parent / "integration"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="run tests/integration (cluster data, credentials, GPU, optional extras)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    run_integration = config.getoption("--integration")
    skip = pytest.mark.skip(reason="integration test; pass --integration to run")
    for item in items:
        if not Path(item.path).is_relative_to(INTEGRATION_DIR):
            continue
        item.add_marker(pytest.mark.integration)
        if not run_integration:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _reset_hydra_config_singleton():
    """compose_recipe installs a HydraConfig; without a reset a later test could
    silently resolve ${hydra:runtime.output_dir} to a previous test's tmp_path."""
    yield
    from hydra.core.hydra_config import HydraConfig

    HydraConfig.instance().cfg = None
