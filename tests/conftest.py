"""Root pytest config for EgoVerse.

``tests/unit`` is hermetic (CPU, no network, no cluster paths) and always runs.
``tests/integration`` needs cluster data, credentials, a GPU, or optional extras;
every test under it is marked ``integration`` and skipped unless ``--integration``
is passed.
"""

from pathlib import Path

import pytest

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
